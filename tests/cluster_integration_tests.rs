//! End-to-end integration tests for the clustering API.

use rype::cluster::{cluster_contigs, ClusterConfig, ContigInput};

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

#[test]
fn strain_default_config_has_expected_field_values() {
    let cfg = ClusterConfig::strain_default();
    assert_eq!(cfg.k, 64);
    assert_eq!(cfg.w, 50);
    assert_eq!(cfg.min_length, 10_000);
    assert!((cfg.threshold - 0.85).abs() < 1e-12);
    assert_eq!(cfg.min_shared, 500);
}
