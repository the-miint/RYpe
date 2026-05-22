//! End-to-end Plan 1.3 ↔ Plan 1.2 ↔ Plan 1.1 closure.
//!
//! Builds a deterministic target, derives three queries (mutated copy,
//! reverse-complement fragment, disjoint genome), and runs the full
//! `.ryci` write → load → extract → join → chain pipeline. Asserts:
//!   - The mutated copy produces a strong fwd chain.
//!   - The rc fragment produces a strong rc chain (no significant fwd chain).
//!   - The disjoint genome produces no chain on either strand.
//!
//! This is the load-bearing Plan 1.3 integration test: the chain DP
//! consumes positions that survived the Plans 1.1 + 1.2 round-trip.

use rype::parquet_cluster_index::{
    create_cluster_parquet_index, ClusterBucketData, ClusterParquetIndex,
};
use rype::{
    chain_anchors, compute_anchors_into, extract_dual_strand_into_with_positions,
    extract_into_with_positions, pairs_into_cluster_bucket_arrays, ChainParams, ChainWorkspace,
    MinimizerWorkspace,
};
use std::path::PathBuf;
use tempfile::TempDir;

/// LCG-based deterministic DNA generator (same approach as
/// src/cluster/edges.rs — kept here for test self-containment).
fn seq_from_seed(len: usize, seed: u64) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            bases[((s >> 56) & 0b11) as usize]
        })
        .collect()
}

/// Deterministic mutation: roll an LCG-driven die at every base; flip
/// the base when the die comes up `0 mod inv_rate`. `inv_rate=20` ≈ 5%
/// rate, `inv_rate=100` ≈ 1%. Unlike a regular stride, this Poisson-like
/// pattern doesn't alias with `k` — some k-mers escape mutation entirely
/// even when `k > inv_rate`.
fn mutate_random(seq: &[u8], inv_rate: u64, seed: u64) -> Vec<u8> {
    let mut out = seq.to_vec();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for byte in out.iter_mut() {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        if s % inv_rate == 0 {
            *byte = match *byte {
                b'A' => b'C',
                b'C' => b'G',
                b'G' => b'T',
                b'T' => b'A',
                b => b,
            };
        }
    }
    out
}

fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            _ => b'N',
        })
        .collect()
}

/// Run `extract_dual_strand_into_with_positions` + `pairs_into_cluster_bucket_arrays`
/// on each strand, returning hash-sorted, hash-deduped parallel arrays.
fn extract_query_positions(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
) -> ((Vec<u64>, Vec<u32>), (Vec<u64>, Vec<u32>)) {
    let mut ws = MinimizerWorkspace::new();
    extract_dual_strand_into_with_positions(seq, k, w, salt, &mut ws)
        .expect("dual-strand extraction must succeed on a No-N test seq");

    let mut fwd_h = std::mem::take(&mut ws.buffer);
    let mut fwd_p = std::mem::take(&mut ws.positions_fwd);
    pairs_into_cluster_bucket_arrays(&mut fwd_h, &mut fwd_p);

    let mut rc_h = std::mem::take(&mut ws.rc_buffer);
    let mut rc_p = std::mem::take(&mut ws.positions_rc);
    pairs_into_cluster_bucket_arrays(&mut rc_h, &mut rc_p);

    ((fwd_h, fwd_p), (rc_h, rc_p))
}

/// Build a single-bucket `.ryci` from a target sequence using the
/// **forward strand only** — matches the existing cluster::edges convention
/// (target = single strand, query = dual strand).
fn build_target_index(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    out_dir: &std::path::Path,
) -> (Vec<u64>, Vec<u32>) {
    let mut ws = MinimizerWorkspace::new();
    extract_into_with_positions(seq, k, w, salt, &mut ws).expect("extract must succeed");
    let mut t_h = std::mem::take(&mut ws.buffer);
    let mut t_p = std::mem::take(&mut ws.positions_fwd);
    pairs_into_cluster_bucket_arrays(&mut t_h, &mut t_p);

    let bucket = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "target".into(),
        sources: vec!["target.fa".into()],
        minimizers: t_h.clone(),
        positions: t_p.clone(),
    };
    bucket.validate().expect("target bucket must validate");

    create_cluster_parquet_index(out_dir, vec![bucket], k, w, salt).expect("create cluster index");

    (t_h, t_p)
}

fn scratch_dir(prefix: &str) -> TempDir {
    let scratch_base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scratch");
    std::fs::create_dir_all(&scratch_base).unwrap();
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(&scratch_base)
        .unwrap()
}

// k=32 keeps the random-match rate near zero in RY space (2^32 ≈ 4B hashes),
// so the disjoint test is robust against spurious chains from chance matches.
// w=5 keeps anchor density high enough that 3-anchor chains form readily on
// 5kb test sequences.
const K: usize = 32;
const W: usize = 5;
const SALT: u64 = 0x5555_5555_5555_5555;

fn small_params() -> ChainParams {
    ChainParams::starting_for_w(W)
}

/// Read the target's `(hash, position)` arrays back from the `.ryci`
/// reader instead of trusting the in-memory copies. This closes the
/// Plan 1.1 round-trip half of the test.
fn load_target_from_disk(index_path: &std::path::Path) -> (Vec<u64>, Vec<u32>) {
    let index = ClusterParquetIndex::open(index_path).unwrap();
    assert_eq!(index.num_shards(), 1, "single-bucket builds single shard");
    let triples = index.load_shard(0).unwrap();
    // The reader returns triples sorted by (minimizer, bucket_id). For a
    // single-bucket index that's hash-sorted — directly usable as
    // merge-join input.
    triples.iter().map(|&(h, _b, p)| (h, p)).unzip()
}

/// WHY (headline test): a query that is a 5%-mutated copy of the target
/// should chain on the FORWARD strand. This is the load-bearing Plan 1.3
/// integration: positions survive the .ryci round-trip and the chain DP
/// finds a strong colinear chain on the right strand. If positions drifted
/// during write/read, the chain wouldn't form. If the fwd/rc strand flag
/// were inverted somewhere in the pipeline, the chain would form on the
/// wrong strand.
#[test]
fn cluster_chain_roundtrip_mutated_copy_chains_on_fwd() {
    let tmp = scratch_dir("chain-mutated.");
    let target_seq = seq_from_seed(5_000, 1);
    let query_seq = mutate_random(&target_seq, 20, 17); // ~5% mutation

    let target_dir = tmp.path().join("target.ryci");
    let _ = build_target_index(&target_seq, K, W, SALT, &target_dir);
    let (target_h, target_p) = load_target_from_disk(&target_dir);

    let ((q_fwd_h, q_fwd_p), (q_rc_h, q_rc_p)) = extract_query_positions(&query_seq, K, W, SALT);

    let mut anchors_fwd = Vec::new();
    compute_anchors_into(&q_fwd_h, &q_fwd_p, &target_h, &target_p, &mut anchors_fwd);
    let mut anchors_rc = Vec::new();
    compute_anchors_into(&q_rc_h, &q_rc_p, &target_h, &target_p, &mut anchors_rc);

    let params = small_params();
    let mut ws = ChainWorkspace::new();
    let r_fwd = chain_anchors(&mut anchors_fwd, false, &params, &mut ws);
    let r_rc = chain_anchors(&mut anchors_rc, true, &params, &mut ws);

    let fwd = r_fwd.expect("mutated copy must chain on fwd strand");
    assert!(
        fwd.anchors >= 10,
        "expected ≥10-anchor fwd chain on a 5%-mutated copy, got {}",
        fwd.anchors
    );
    // RC may produce a small chain from chance palindromic matches but
    // must be substantially weaker than fwd.
    if let Some(rc) = r_rc {
        assert!(
            rc.anchors < fwd.anchors,
            "rc chain ({} anchors) should be weaker than fwd chain ({} anchors)",
            rc.anchors,
            fwd.anchors,
        );
    }
}

/// WHY: a query that is the reverse-complement of a target SLICE should
/// chain on the RC strand of the dual-strand query (and NOT the fwd
/// strand). This pins the strand-flag plumbing all the way from
/// `extract_dual_strand_into_with_positions`'s `rc_buffer`/`positions_rc`
/// through `compute_anchors_into` to `chain_anchors(.., is_rc=true)`.
/// A flipped flag anywhere in the chain would direct the match to fwd.
#[test]
fn cluster_chain_roundtrip_rc_fragment_chains_on_rc() {
    let tmp = scratch_dir("chain-rc.");
    let target_seq = seq_from_seed(5_000, 42);
    // 2000-bp slice, reverse-complemented. Query's RC strand recovers the
    // original target-slice minimizers; query's FWD strand sees the RC of
    // those k-mers, which (mostly) don't match the target's fwd-only bucket.
    let query_seq = reverse_complement(&target_seq[1_000..3_000]);

    let target_dir = tmp.path().join("target.ryci");
    let _ = build_target_index(&target_seq, K, W, SALT, &target_dir);
    let (target_h, target_p) = load_target_from_disk(&target_dir);

    let ((q_fwd_h, q_fwd_p), (q_rc_h, q_rc_p)) = extract_query_positions(&query_seq, K, W, SALT);

    let mut anchors_fwd = Vec::new();
    compute_anchors_into(&q_fwd_h, &q_fwd_p, &target_h, &target_p, &mut anchors_fwd);
    let mut anchors_rc = Vec::new();
    compute_anchors_into(&q_rc_h, &q_rc_p, &target_h, &target_p, &mut anchors_rc);

    let params = small_params();
    let mut ws = ChainWorkspace::new();
    let r_fwd = chain_anchors(&mut anchors_fwd, false, &params, &mut ws);
    let r_rc = chain_anchors(&mut anchors_rc, true, &params, &mut ws);

    let rc = r_rc.expect("rc fragment must chain on rc strand");
    assert!(
        rc.anchors >= 10,
        "expected ≥10-anchor rc chain on a 2kb rc fragment, got {}",
        rc.anchors
    );
    if let Some(fwd) = r_fwd {
        assert!(
            fwd.anchors < rc.anchors,
            "fwd chain ({} anchors) should be weaker than rc chain ({} anchors)",
            fwd.anchors,
            rc.anchors,
        );
    }
}

/// WHY: false-positive control. Two genomes from different seeds share
/// essentially zero k-mers at k=32 in RY space (random match rate ≈ 1
/// in 4B). The chain DP must return None on both strands — any chain
/// reported here would indicate the algorithm finds spurious colinearity
/// in noise.
#[test]
fn cluster_chain_roundtrip_disjoint_genomes_no_chain() {
    let tmp = scratch_dir("chain-disjoint.");
    let target_seq = seq_from_seed(5_000, 1);
    let query_seq = seq_from_seed(5_000, 99_999);

    let target_dir = tmp.path().join("target.ryci");
    let _ = build_target_index(&target_seq, K, W, SALT, &target_dir);
    let (target_h, target_p) = load_target_from_disk(&target_dir);

    let ((q_fwd_h, q_fwd_p), (q_rc_h, q_rc_p)) = extract_query_positions(&query_seq, K, W, SALT);

    let mut anchors_fwd = Vec::new();
    compute_anchors_into(&q_fwd_h, &q_fwd_p, &target_h, &target_p, &mut anchors_fwd);
    let mut anchors_rc = Vec::new();
    compute_anchors_into(&q_rc_h, &q_rc_p, &target_h, &target_p, &mut anchors_rc);

    let params = small_params();
    let mut ws = ChainWorkspace::new();
    let r_fwd = chain_anchors(&mut anchors_fwd, false, &params, &mut ws);
    let r_rc = chain_anchors(&mut anchors_rc, true, &params, &mut ws);

    assert!(
        r_fwd.is_none(),
        "disjoint genomes must NOT chain on fwd; got {:?}",
        r_fwd
    );
    assert!(
        r_rc.is_none(),
        "disjoint genomes must NOT chain on rc; got {:?}",
        r_rc
    );
}

/// WHY: `ChainParams::starting_for_w(w)` ships uncalibrated research-doc
/// starting values. This test exercises them with `w=50` on a 50kb,
/// 1%-mutated copy — if the defaults are unusably tight (chain doesn't
/// form on realistic-scale input), we catch it here rather than at
/// Plan 1.4 integration time. Light assertion: the chain forms with at
/// least `min_anchors`. Plan 1.6 will tighten this once calibration
/// produces blessed values.
#[test]
fn cluster_chain_roundtrip_uses_research_doc_starting_params() {
    let tmp = scratch_dir("chain-w50.");
    const LONG_K: usize = 32;
    const LONG_W: usize = 50;
    let target_seq = seq_from_seed(50_000, 7);
    let query_seq = mutate_random(&target_seq, 100, 23); // ~1% mutation

    let target_dir = tmp.path().join("target.ryci");
    let _ = build_target_index(&target_seq, LONG_K, LONG_W, SALT, &target_dir);
    let (target_h, target_p) = load_target_from_disk(&target_dir);

    let ((q_fwd_h, q_fwd_p), _) = extract_query_positions(&query_seq, LONG_K, LONG_W, SALT);

    let mut anchors_fwd = Vec::new();
    compute_anchors_into(&q_fwd_h, &q_fwd_p, &target_h, &target_p, &mut anchors_fwd);

    let params = ChainParams::starting_for_w(LONG_W);
    let mut ws = ChainWorkspace::new();
    let r_fwd = chain_anchors(&mut anchors_fwd, false, &params, &mut ws)
        .expect("starting_for_w(50) must produce a chain on a 1%-mutated 50kb copy");
    assert!(
        r_fwd.anchors >= params.min_anchors,
        "expected ≥{} chained anchors, got {}",
        params.min_anchors,
        r_fwd.anchors,
    );
}
