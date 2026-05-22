//! End-to-end integration tests for the clustering API.

use rype::cluster::{cluster_contigs, ClusterConfig, ContigInput};
use rype::ChainParams;

/// Deterministic pseudo-random DNA, seeded so tests are reproducible.
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

fn cfg_relaxed() -> ClusterConfig {
    // Smaller k/w and lower thresholds let the test work on small synthetic
    // contigs without needing megabase-scale inputs. Production defaults live
    // in ClusterConfig::strain_default().
    ClusterConfig {
        k: 32,
        w: 20,
        salt: 0x5555_5555_5555_5555,
        min_length: 1_000,
        threshold: 0.80,
        min_shared: 50,
        chain_params: None,
        min_chain_containment: None,
    }
}

#[test]
fn one_full_genome_plus_two_fragments_plus_one_unrelated() {
    let a = seq_from_seed(20_000, 1);
    // Two non-overlapping fragments of A
    let b = a[2_000..10_000].to_vec();
    let d = a[12_000..18_000].to_vec();
    let c = seq_from_seed(20_000, 2);

    let inputs = vec![
        ContigInput {
            id: "A".to_string(),
            source_mag: Some("mag1".to_string()),
            sequence: a,
        },
        ContigInput {
            id: "B".to_string(),
            source_mag: Some("mag2".to_string()),
            sequence: b,
        },
        ContigInput {
            id: "C".to_string(),
            source_mag: Some("mag3".to_string()),
            sequence: c,
        },
        ContigInput {
            id: "D".to_string(),
            source_mag: Some("mag2".to_string()),
            sequence: d,
        },
    ];

    let result = cluster_contigs(inputs, &cfg_relaxed()).unwrap();

    // Complete partition: one row per input
    assert_eq!(result.rows.len(), 4);

    // A is the longest, so it should be a representative and the two
    // fragments (B and D) should be absorbed into it.
    let reps: Vec<&str> = result
        .rows
        .iter()
        .filter(|r| r.rep_contig == r.member_contig)
        .map(|r| r.rep_contig.as_str())
        .collect();
    assert!(
        reps.contains(&"A"),
        "expected A as representative, got reps {:?}",
        reps
    );
    assert!(
        reps.contains(&"C"),
        "expected C (unrelated) as its own representative, got reps {:?}",
        reps
    );
    assert_eq!(reps.len(), 2, "expected exactly 2 reps, got {:?}", reps);

    // B and D should both be members of A
    for member in ["B", "D"] {
        let row = result
            .rows
            .iter()
            .find(|r| r.member_contig == member && r.rep_contig != r.member_contig)
            .unwrap_or_else(|| panic!("expected {} to be absorbed", member));
        assert_eq!(row.rep_contig, "A");
        assert!(row.containment >= 0.80);
    }
}

#[test]
fn all_unrelated_become_singleton_clusters() {
    let inputs: Vec<ContigInput> = (0..4)
        .map(|i| ContigInput {
            id: format!("U{}", i),
            source_mag: None,
            sequence: seq_from_seed(15_000, 100 + i),
        })
        .collect();

    let result = cluster_contigs(inputs, &cfg_relaxed()).unwrap();

    assert_eq!(result.rows.len(), 4);
    for row in &result.rows {
        assert_eq!(row.rep_contig, row.member_contig);
        assert_eq!(row.containment, 1.0);
    }
}

#[test]
fn length_floor_excludes_short_contigs_entirely() {
    let inputs = vec![
        ContigInput {
            id: "long".to_string(),
            source_mag: None,
            sequence: seq_from_seed(15_000, 7),
        },
        ContigInput {
            id: "short".to_string(),
            source_mag: None,
            sequence: seq_from_seed(500, 8), // below min_length=1000
        },
    ];

    let result = cluster_contigs(inputs, &cfg_relaxed()).unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].rep_contig, "long");
}

#[test]
fn empty_input_returns_empty_result() {
    let result = cluster_contigs(Vec::new(), &cfg_relaxed()).unwrap();
    assert!(result.rows.is_empty());
}

/// Plan 1.4 phase 3 — load-bearing test that justifies the chain feature.
///
/// Construct a query genome `B` whose minimizer **set** is almost identical
/// to representative `A` (high set-containment) but whose minimizer
/// **positions** are scrambled relative to `A` (low chain containment).
///
/// WHY this is the right test for Plan 1.4: set-containment alone cannot
/// distinguish "real syntenic match" from "shared minimizers scattered
/// without colinearity" (chaining-research.md §5.1). Without the chain
/// gate, `B` is absorbed into `A`. With the gate set, the colinearity
/// check rejects the absorption.
///
/// Construction: take 40 non-overlapping 200-bp chunks of `A` and emit
/// them in a deterministic shuffled order. Each chunk's minimizers are
/// internally colinear (yielding short within-chunk chains) but
/// inter-chunk anchors are off-diagonal (the chain DP breaks at every
/// boundary).
#[test]
fn cluster_chain_gate_filters_random_match() {
    let a = seq_from_seed(10_000, 42);

    // Build B by permuting 200-bp chunks of A. Deterministic Fisher-Yates
    // shuffle via a seeded LCG so the test is reproducible.
    let chunk_size = 200usize;
    let num_chunks = 40usize; // B = 8000 bp (shorter than A so A wins length sort)
    let mut order: Vec<usize> = (0..num_chunks).collect();
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_F00D;
    for i in (1..num_chunks).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (rng_state >> 33) as usize % (i + 1);
        order.swap(i, j);
    }
    let mut b = Vec::with_capacity(num_chunks * chunk_size);
    for &idx in &order {
        b.extend_from_slice(&a[idx * chunk_size..(idx + 1) * chunk_size]);
    }
    assert_eq!(b.len(), num_chunks * chunk_size);

    let inputs = || {
        vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: Some("mag_a".to_string()),
                sequence: a.clone(),
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: Some("mag_b".to_string()),
                sequence: b.clone(),
            },
        ]
    };

    // Baseline: chain enabled but gate disabled. Set-containment alone
    // absorbs B into A — the failure mode the gate exists to prevent.
    let cfg_no_gate = ClusterConfig {
        k: 32,
        w: 20,
        salt: 0x5555_5555_5555_5555,
        min_length: 1_000,
        threshold: 0.80,
        min_shared: 50,
        chain_params: Some(ChainParams::starting_for_w(20)),
        min_chain_containment: None,
    };
    let result_no_gate = cluster_contigs(inputs(), &cfg_no_gate).unwrap();
    let absorbed_no_gate = result_no_gate
        .rows
        .iter()
        .any(|r| r.member_contig == "B" && r.rep_contig == "A");
    assert!(
        absorbed_no_gate,
        "without chain gate, set-containment should absorb the scrambled B \
         into A; rows: {:#?}",
        result_no_gate.rows
    );

    // With chain gate at 0.5: the scrambled B has no long colinear chain
    // into A, so its `chain.containment` falls below the gate and the
    // absorption is rejected.
    let cfg_gated = ClusterConfig {
        min_chain_containment: Some(0.5),
        ..cfg_no_gate
    };
    let result_gated = cluster_contigs(inputs(), &cfg_gated).unwrap();
    assert_eq!(
        result_gated.rows.len(),
        2,
        "expected two singleton clusters with the chain gate on; got {:#?}",
        result_gated.rows
    );
    for row in &result_gated.rows {
        assert_eq!(
            row.rep_contig, row.member_contig,
            "every row must be its own rep when chain gate rejects the only \
             candidate edge; got {:?}",
            row
        );
    }
}

#[test]
fn strain_default_config_has_expected_field_values() {
    let cfg = ClusterConfig::strain_default();
    assert_eq!(cfg.k, 64);
    assert_eq!(cfg.w, 50);
    assert_eq!(cfg.min_length, 10_000);
    assert!((cfg.threshold - 0.85).abs() < 1e-12);
    assert_eq!(cfg.min_shared, 500);
}
