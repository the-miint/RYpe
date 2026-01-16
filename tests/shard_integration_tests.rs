//! Integration tests for sharded index classification.
//!
//! These tests verify that sharded and non-sharded index configurations
//! produce identical classification results.

use std::collections::HashMap;

use anyhow::Result;
use tempfile::tempdir;

use rype::{
    classify_batch, classify_batch_sharded_main, classify_batch_sharded_merge_join,
    classify_batch_sharded_sequential, HitResult, Index, IndexMetadata, InvertedIndex,
    MainIndexManifest, MainIndexShard, MinimizerWorkspace, QueryRecord, ShardManifest,
    ShardedInvertedIndex, ShardedMainIndex,
};

/// Test helper to sort results for comparison
fn sort_results(mut results: Vec<HitResult>) -> Vec<HitResult> {
    results.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));
    results
}

/// Test helper to compare two result sets
fn compare_results(name_a: &str, results_a: &[HitResult], name_b: &str, results_b: &[HitResult]) {
    let a = sort_results(results_a.to_vec());
    let b = sort_results(results_b.to_vec());

    if a.len() != b.len() {
        eprintln!("\n=== RESULT COUNT MISMATCH ===");
        eprintln!("{}: {} results", name_a, a.len());
        eprintln!("{}: {} results", name_b, b.len());

        // Show what's different
        for r in &a {
            if !b
                .iter()
                .any(|br| br.query_id == r.query_id && br.bucket_id == r.bucket_id)
            {
                eprintln!(
                    "  In {} only: query={}, bucket={}, score={:.4}",
                    name_a, r.query_id, r.bucket_id, r.score
                );
            }
        }
        for r in &b {
            if !a
                .iter()
                .any(|ar| ar.query_id == r.query_id && ar.bucket_id == r.bucket_id)
            {
                eprintln!(
                    "  In {} only: query={}, bucket={}, score={:.4}",
                    name_b, r.query_id, r.bucket_id, r.score
                );
            }
        }
    }

    assert_eq!(
        a.len(),
        b.len(),
        "{} has {} results, {} has {} results",
        name_a,
        a.len(),
        name_b,
        b.len()
    );

    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.query_id, rb.query_id, "Query ID mismatch");
        assert_eq!(ra.bucket_id, rb.bucket_id, "Bucket ID mismatch");
        let diff = (ra.score - rb.score).abs();
        assert!(
            diff < 0.001,
            "Score mismatch for query {} bucket {}: {} ({}) vs {} ({}) diff={}",
            ra.query_id,
            ra.bucket_id,
            ra.score,
            name_a,
            rb.score,
            name_b,
            diff
        );
    }
}

/// Create test sequences that produce distinct minimizer sets
fn create_test_sequences() -> Vec<Vec<u8>> {
    // Create sequences with different patterns that will hash to different minimizers
    vec![
        // Sequence 0: ACGT repeat
        (0..200)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            })
            .collect(),
        // Sequence 1: AT repeat
        (0..200)
            .map(|i| if i % 2 == 0 { b'A' } else { b'T' })
            .collect(),
        // Sequence 2: GC repeat
        (0..200)
            .map(|i| if i % 2 == 0 { b'G' } else { b'C' })
            .collect(),
        // Sequence 3: Different pattern
        (0..200)
            .map(|i| match i % 6 {
                0 => b'A',
                1 => b'A',
                2 => b'C',
                3 => b'G',
                4 => b'T',
                _ => b'T',
            })
            .collect(),
        // Sequence 4: Yet another pattern
        (0..200)
            .map(|i| match i % 3 {
                0 => b'A',
                1 => b'C',
                _ => b'T',
            })
            .collect(),
    ]
}

/// Create a main index with multiple buckets from test sequences
fn create_main_index(seqs: &[Vec<u8>], k: usize, w: usize, salt: u64) -> Index {
    let mut index = Index::new(k, w, salt).unwrap();
    let mut ws = MinimizerWorkspace::new();

    for (i, seq) in seqs.iter().enumerate() {
        let bucket_id = (i + 1) as u32;
        let name = format!("Bucket{}", bucket_id);
        let source = format!("seq{}", i);
        index.add_record(bucket_id, &source, seq, &mut ws);
        index.bucket_names.insert(bucket_id, name);
        index.finalize_bucket(bucket_id);
    }

    index
}

/// Create query records from sequences
fn create_query_records<'a>(seqs: &'a [Vec<u8>]) -> Vec<QueryRecord<'a>> {
    seqs.iter()
        .enumerate()
        .map(|(i, seq)| (i as i64, seq.as_slice(), None))
        .collect()
}

/// Print diagnostic information about an index
fn print_index_diagnostics(name: &str, index: &Index) {
    eprintln!("\n=== {} ===", name);
    eprintln!("K={}, W={}, salt={:#x}", index.k, index.w, index.salt);
    eprintln!("Buckets: {}", index.buckets.len());
    for (id, mins) in &index.buckets {
        eprintln!(
            "  Bucket {}: {} minimizers, range [{:#x}, {:#x}]",
            id,
            mins.len(),
            mins.first().copied().unwrap_or(0),
            mins.last().copied().unwrap_or(0)
        );
    }
}

/// Print diagnostic information about an inverted index
fn print_inverted_diagnostics(name: &str, inv: &InvertedIndex) {
    eprintln!("\n=== {} ===", name);
    eprintln!("K={}, W={}, salt={:#x}", inv.k, inv.w, inv.salt);
    eprintln!("Unique minimizers: {}", inv.num_minimizers());
    eprintln!("Bucket entries: {}", inv.num_bucket_entries());
}

/// Print diagnostic information about a sharded inverted index
fn print_sharded_inverted_diagnostics(name: &str, sharded: &ShardedInvertedIndex) {
    let manifest = sharded.manifest();
    eprintln!("\n=== {} ===", name);
    eprintln!(
        "K={}, W={}, salt={:#x}",
        manifest.k, manifest.w, manifest.salt
    );
    eprintln!("Total shards: {}", manifest.shards.len());
    eprintln!("Total minimizers: {}", manifest.total_minimizers);
    eprintln!("Total bucket entries: {}", manifest.total_bucket_ids);
    for shard in &manifest.shards {
        eprintln!(
            "  Shard {}: {} minimizers, {} bucket entries, range [{:#x}, {:#x}], is_last={}",
            shard.shard_id,
            shard.num_minimizers,
            shard.num_bucket_ids,
            shard.min_start,
            shard.min_end,
            shard.is_last_shard
        );
    }
}

#[test]
fn test_baseline_non_sharded_classification() -> Result<()> {
    eprintln!("\n\n========== BASELINE: NON-SHARDED CLASSIFICATION ==========");

    let seqs = create_test_sequences();
    let index = create_main_index(&seqs, 32, 10, 0x1234);
    let records = create_query_records(&seqs);

    print_index_diagnostics("Main Index", &index);

    let inverted = InvertedIndex::build_from_index(&index);
    print_inverted_diagnostics("Inverted Index", &inverted);

    let threshold = 0.1;

    // Test direct classification
    let results_direct = classify_batch(&index, None, &records, threshold);

    eprintln!("\n=== Results ===");
    eprintln!("Direct: {} results", results_direct.len());

    // Verify we get at least some results
    assert!(
        !results_direct.is_empty(),
        "Should have classification results"
    );

    // Each query sequence should match its own bucket with high score
    for r in &results_direct {
        eprintln!(
            "  Query {} -> Bucket {}: {:.4}",
            r.query_id, r.bucket_id, r.score
        );
    }

    Ok(())
}

#[test]
fn test_sharded_inverted_from_non_sharded_main() -> Result<()> {
    eprintln!("\n\n========== SHARDED INVERTED FROM NON-SHARDED MAIN ==========");

    let dir = tempdir()?;
    let inverted_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let index = create_main_index(&seqs, 32, 10, 0x1234);
    let records = create_query_records(&seqs);

    print_index_diagnostics("Main Index", &index);

    // Build inverted index
    let inverted = InvertedIndex::build_from_index(&index);
    print_inverted_diagnostics("Inverted", &inverted);

    // Save as single shard (like we do for non-sharded main index)
    eprintln!("\nSaving as 1 shard...");
    let shard_path = ShardManifest::shard_path(&inverted_path, 0);
    let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

    let manifest = ShardManifest {
        k: inverted.k,
        w: inverted.w,
        salt: inverted.salt,
        source_hash: 0,
        total_minimizers: inverted.num_minimizers(),
        total_bucket_ids: inverted.num_bucket_entries(),
        has_overlapping_shards: true,
        shards: vec![shard_info],
    };
    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    manifest.save(&manifest_path)?;

    eprintln!("Created {} shards", manifest.shards.len());
    for shard in &manifest.shards {
        eprintln!(
            "  Shard {}: {} minimizers, range [{:#x}, {:#x}], is_last={}",
            shard.shard_id,
            shard.num_minimizers,
            shard.min_start,
            shard.min_end,
            shard.is_last_shard
        );
    }

    // Load sharded index
    let sharded = ShardedInvertedIndex::open(&inverted_path)?;
    print_sharded_inverted_diagnostics("Sharded Inverted", &sharded);

    let threshold = 0.1;

    // Compare results: direct vs sharded inverted
    let results_direct = classify_batch(&index, None, &records, threshold);
    let results_sequential =
        classify_batch_sharded_sequential(&sharded, None, &records, threshold)?;
    let results_merge = classify_batch_sharded_merge_join(&sharded, None, &records, threshold)?;

    eprintln!("\n=== Results ===");
    eprintln!("Direct: {} results", results_direct.len());
    eprintln!("Sharded sequential: {} results", results_sequential.len());
    eprintln!("Sharded merge-join: {} results", results_merge.len());

    compare_results(
        "Direct",
        &results_direct,
        "Sharded-sequential",
        &results_sequential,
    );
    compare_results("Direct", &results_direct, "Sharded-merge", &results_merge);

    Ok(())
}

#[test]
fn test_sharded_main_index_classification() -> Result<()> {
    eprintln!("\n\n========== SHARDED MAIN INDEX CLASSIFICATION ==========");

    let dir = tempdir()?;
    let main_path = dir.path().join("test.ryidx");

    let seqs = create_test_sequences();
    let index = create_main_index(&seqs, 32, 10, 0x1234);
    let records = create_query_records(&seqs);

    print_index_diagnostics("Main Index", &index);

    // Save main index as sharded (small budget to force multiple shards)
    eprintln!("\nSaving main index as sharded (budget=100 bytes)...");
    index.save_sharded(&main_path, 100)?;

    // Load sharded main index
    let sharded_main = ShardedMainIndex::open(&main_path)?;
    eprintln!("Loaded sharded main: {} shards", sharded_main.num_shards());
    for shard_info in &sharded_main.manifest().shards {
        eprintln!(
            "  Shard {}: {} buckets, {} minimizers",
            shard_info.shard_id,
            shard_info.bucket_ids.len(),
            shard_info.num_minimizers
        );
    }

    let threshold = 0.1;

    // Compare results
    let results_direct = classify_batch(&index, None, &records, threshold);
    let results_sharded = classify_batch_sharded_main(&sharded_main, None, &records, threshold)?;

    eprintln!("\n=== Results ===");
    eprintln!("Direct (non-sharded): {} results", results_direct.len());
    eprintln!("Sharded main: {} results", results_sharded.len());

    compare_results("Direct", &results_direct, "Sharded-main", &results_sharded);

    Ok(())
}

#[test]
fn test_inverted_from_sharded_main() -> Result<()> {
    eprintln!("\n\n========== INVERTED INDEX FROM SHARDED MAIN (BUG TEST) ==========");

    let dir = tempdir()?;
    let main_path = dir.path().join("test.ryidx");
    let inverted_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let index = create_main_index(&seqs, 32, 10, 0x1234);
    let records = create_query_records(&seqs);

    print_index_diagnostics("Main Index", &index);

    // Build ground truth: inverted from non-sharded main
    let inverted_ground_truth = InvertedIndex::build_from_index(&index);
    print_inverted_diagnostics("Ground Truth Inverted", &inverted_ground_truth);

    // Compute source hash for manifest
    let metadata = IndexMetadata {
        k: index.k,
        w: index.w,
        salt: index.salt,
        bucket_names: index.bucket_names.clone(),
        bucket_sources: index.bucket_sources.clone(),
        bucket_minimizer_counts: index.buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
    };
    let source_hash = InvertedIndex::compute_metadata_hash(&metadata);

    // Save main index as sharded
    eprintln!("\nSaving main index as sharded (budget=100 bytes)...");
    index.save_sharded(&main_path, 100)?;

    // Load sharded main and build inverted shards 1:1
    let main_manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(&main_path))?;
    eprintln!("Main index has {} shards", main_manifest.shards.len());

    let mut inv_shards = Vec::new();
    let mut total_minimizers = 0usize;
    let mut total_bucket_ids = 0usize;
    let num_shards = main_manifest.shards.len();

    for (idx, shard_info) in main_manifest.shards.iter().enumerate() {
        let shard_path = MainIndexManifest::shard_path(&main_path, shard_info.shard_id);
        let main_shard = MainIndexShard::load(&shard_path)?;

        eprintln!(
            "\nMain shard {}: {} buckets",
            shard_info.shard_id,
            main_shard.buckets.len()
        );
        for (bid, mins) in &main_shard.buckets {
            eprintln!("  Bucket {}: {} minimizers", bid, mins.len());
        }

        // Build inverted from this shard
        let inverted = InvertedIndex::build_from_shard(&main_shard);
        eprintln!(
            "Inverted shard: {} unique minimizers, {} bucket entries",
            inverted.num_minimizers(),
            inverted.num_bucket_entries()
        );

        // Save as inverted shard
        let inv_shard_path = ShardManifest::shard_path(&inverted_path, shard_info.shard_id);
        let is_last = idx == num_shards - 1;
        let inv_shard_info = inverted.save_shard(
            &inv_shard_path,
            shard_info.shard_id,
            0,
            inverted.num_minimizers(),
            is_last,
        )?;

        eprintln!(
            "Saved shard: min_start={:#x}, min_end={:#x}, is_last={}",
            inv_shard_info.min_start, inv_shard_info.min_end, inv_shard_info.is_last_shard
        );

        total_minimizers += inv_shard_info.num_minimizers;
        total_bucket_ids += inv_shard_info.num_bucket_ids;
        inv_shards.push(inv_shard_info);
    }

    // Create manifest (bucket-partitioned since built from main shards)
    let manifest = ShardManifest {
        k: main_manifest.k,
        w: main_manifest.w,
        salt: main_manifest.salt,
        source_hash,
        total_minimizers,
        total_bucket_ids,
        has_overlapping_shards: true,
        shards: inv_shards,
    };

    eprintln!("\n=== Manifest ===");
    eprintln!("Total minimizers: {}", manifest.total_minimizers);
    eprintln!("Total bucket IDs: {}", manifest.total_bucket_ids);
    for shard in &manifest.shards {
        eprintln!(
            "  Shard {}: min_start={:#x}, min_end={:#x}, is_last={}, minimizers={}, bucket_ids={}",
            shard.shard_id,
            shard.min_start,
            shard.min_end,
            shard.is_last_shard,
            shard.num_minimizers,
            shard.num_bucket_ids
        );
    }

    // Save manifest
    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    manifest.save(&manifest_path)?;
    eprintln!("Saved manifest to {:?}", manifest_path);

    // Try to load the sharded inverted index
    eprintln!("\nLoading sharded inverted index...");
    let load_result = ShardedInvertedIndex::open(&inverted_path);

    match load_result {
        Ok(sharded_inv) => {
            eprintln!("Successfully loaded sharded inverted index");
            print_sharded_inverted_diagnostics("Loaded Sharded Inverted", &sharded_inv);

            let threshold = 0.1;

            // Compare results using direct classification (ground truth)
            let results_ground_truth = classify_batch(&index, None, &records, threshold);

            eprintln!("\nClassifying with sharded sequential...");
            let results_sharded_seq =
                classify_batch_sharded_sequential(&sharded_inv, None, &records, threshold)?;

            eprintln!("Classifying with sharded merge-join...");
            let results_sharded_merge =
                classify_batch_sharded_merge_join(&sharded_inv, None, &records, threshold)?;

            eprintln!("\n=== Results ===");
            eprintln!("Ground truth: {} results", results_ground_truth.len());
            eprintln!("Sharded sequential: {} results", results_sharded_seq.len());
            eprintln!(
                "Sharded merge-join: {} results",
                results_sharded_merge.len()
            );

            // Print detailed results
            eprintln!("\nGround truth results:");
            for r in &results_ground_truth {
                eprintln!(
                    "  Query {} -> Bucket {}: {:.4}",
                    r.query_id, r.bucket_id, r.score
                );
            }

            eprintln!("\nSharded sequential results:");
            for r in &results_sharded_seq {
                eprintln!(
                    "  Query {} -> Bucket {}: {:.4}",
                    r.query_id, r.bucket_id, r.score
                );
            }

            compare_results(
                "Ground-truth",
                &results_ground_truth,
                "Sharded-sequential",
                &results_sharded_seq,
            );
            compare_results(
                "Ground-truth",
                &results_ground_truth,
                "Sharded-merge",
                &results_sharded_merge,
            );
        }
        Err(e) => {
            eprintln!("\n!!! FAILED TO LOAD SHARDED INVERTED INDEX !!!");
            eprintln!("Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

#[test]
fn test_minimizer_distribution_across_shards() -> Result<()> {
    eprintln!("\n\n========== MINIMIZER DISTRIBUTION ANALYSIS ==========");

    let dir = tempdir()?;
    let main_path = dir.path().join("test.ryidx");

    let seqs = create_test_sequences();
    let index = create_main_index(&seqs, 32, 10, 0x1234);

    print_index_diagnostics("Main Index", &index);

    // Build ground truth inverted index
    let inverted_full = InvertedIndex::build_from_index(&index);

    // Collect all minimizers by bucket
    let mut minimizers_by_bucket: HashMap<u32, std::collections::HashSet<u64>> = HashMap::new();
    for (bucket_id, mins) in &index.buckets {
        minimizers_by_bucket.insert(*bucket_id, mins.iter().copied().collect());
    }

    // Save main index as sharded
    index.save_sharded(&main_path, 100)?;

    // Analyze each main shard
    let main_manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(&main_path))?;

    let mut all_inverted_minimizers: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    let mut minimizers_in_multiple_shards: Vec<u64> = Vec::new();
    let mut minimizers_per_shard: Vec<std::collections::HashSet<u64>> = Vec::new();

    for shard_info in &main_manifest.shards {
        let shard_path = MainIndexManifest::shard_path(&main_path, shard_info.shard_id);
        let main_shard = MainIndexShard::load(&shard_path)?;

        // Collect all unique minimizers in this shard across all its buckets
        let mut shard_mins: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for (_, bucket_mins) in &main_shard.buckets {
            shard_mins.extend(bucket_mins.iter().copied());
        }

        eprintln!(
            "\nShard {}: {} unique minimizers",
            shard_info.shard_id,
            shard_mins.len()
        );

        // Check for overlap with existing minimizers
        let overlap: Vec<u64> = shard_mins
            .intersection(&all_inverted_minimizers)
            .copied()
            .collect();
        if !overlap.is_empty() {
            eprintln!(
                "  !!! {} minimizers overlap with previous shards !!!",
                overlap.len()
            );
            minimizers_in_multiple_shards.extend(overlap);
        }

        all_inverted_minimizers.extend(&shard_mins);
        minimizers_per_shard.push(shard_mins);
    }

    eprintln!("\n=== Analysis Summary ===");
    eprintln!(
        "Full inverted index: {} unique minimizers",
        inverted_full.num_minimizers()
    );
    eprintln!(
        "Sum across shards: {} minimizers (with duplication)",
        minimizers_per_shard.iter().map(|s| s.len()).sum::<usize>()
    );
    eprintln!(
        "Union across shards: {} unique minimizers",
        all_inverted_minimizers.len()
    );
    eprintln!(
        "Minimizers in multiple shards: {}",
        minimizers_in_multiple_shards.len()
    );

    if !minimizers_in_multiple_shards.is_empty() {
        eprintln!("\n!!! OVERLAPPING MINIMIZERS DETECTED !!!");
        eprintln!("This is the root cause of the sharding bug.");
        eprintln!("When a minimizer appears in multiple main shards,");
        eprintln!("it will be in multiple inverted shards too,");
        eprintln!("but the manifest validation assumes non-overlapping ranges.");

        eprintln!("\nFirst 10 overlapping minimizers:");
        for (i, &min) in minimizers_in_multiple_shards.iter().take(10).enumerate() {
            let in_shards: Vec<usize> = minimizers_per_shard
                .iter()
                .enumerate()
                .filter(|(_, s)| s.contains(&min))
                .map(|(i, _)| i)
                .collect();
            eprintln!("  {}: {:#x} in shards {:?}", i, min, in_shards);
        }
    }

    Ok(())
}

/// Test the exact scenario from the bug report: classification with 1:1 inverted shards
#[test]
fn test_user_scenario_sharded_inverted_from_sharded_main() -> Result<()> {
    eprintln!("\n\n========== USER SCENARIO: SHARDED INVERTED FROM SHARDED MAIN ==========");
    eprintln!("This test simulates the user's workflow:");
    eprintln!("  1. rype index from-config --max-shard-size N");
    eprintln!("  2. rype index invert");
    eprintln!("  3. rype classify run --use-inverted --merge-join");

    let dir = tempdir()?;
    let main_path = dir.path().join("test.ryidx");
    let inverted_path = dir.path().join("test.ryxdi");

    // Create a more substantial index
    let k = 32;
    let w = 20;
    let salt = 0x5555555555555555u64;

    let seqs: Vec<Vec<u8>> = (0..10)
        .map(|i| {
            // Create unique sequences
            let pattern = match i % 5 {
                0 => vec![b'A', b'C', b'G', b'T'],
                1 => vec![b'T', b'A', b'T', b'A'],
                2 => vec![b'G', b'C', b'G', b'C'],
                3 => vec![b'A', b'A', b'C', b'C'],
                _ => vec![b'G', b'G', b'T', b'T'],
            };
            (0..300).map(|j| pattern[j % pattern.len()]).collect()
        })
        .collect();

    let index = create_main_index(&seqs, k, w, salt);

    eprintln!("\nCreated index with {} buckets:", index.buckets.len());
    let mut total_mins = 0;
    for (id, mins) in &index.buckets {
        eprintln!("  Bucket {}: {} minimizers", id, mins.len());
        total_mins += mins.len();
    }
    eprintln!("Total minimizers across all buckets: {}", total_mins);

    // Build ground truth
    let inverted_truth = InvertedIndex::build_from_index(&index);
    eprintln!(
        "\nGround truth inverted: {} unique minimizers, {} bucket entries",
        inverted_truth.num_minimizers(),
        inverted_truth.num_bucket_entries()
    );

    // Save as sharded main index
    eprintln!("\nSaving as sharded main index (budget=500 bytes)...");
    index.save_sharded(&main_path, 500)?;

    let main_manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(&main_path))?;
    eprintln!("Created {} main shards:", main_manifest.shards.len());
    for shard in &main_manifest.shards {
        eprintln!(
            "  Shard {}: buckets {:?}, {} minimizers",
            shard.shard_id, shard.bucket_ids, shard.num_minimizers
        );
    }

    // Compute source hash for manifest
    let metadata = IndexMetadata {
        k: index.k,
        w: index.w,
        salt: index.salt,
        bucket_names: index.bucket_names.clone(),
        bucket_sources: index.bucket_sources.clone(),
        bucket_minimizer_counts: index.buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
    };
    let source_hash = InvertedIndex::compute_metadata_hash(&metadata);

    // Build inverted shards from main shards (as the CLI does)
    let mut inv_shards = Vec::new();
    let mut total_minimizers = 0usize;
    let mut total_bucket_ids = 0usize;
    let num_shards = main_manifest.shards.len();

    eprintln!("\nBuilding inverted shards...");
    for (idx, shard_info) in main_manifest.shards.iter().enumerate() {
        let shard_path = MainIndexManifest::shard_path(&main_path, shard_info.shard_id);
        let main_shard = MainIndexShard::load(&shard_path)?;
        let inverted = InvertedIndex::build_from_shard(&main_shard);

        eprintln!(
            "  Shard {}: {} minimizers, {} bucket entries",
            shard_info.shard_id,
            inverted.num_minimizers(),
            inverted.num_bucket_entries()
        );

        let inv_shard_path = ShardManifest::shard_path(&inverted_path, shard_info.shard_id);
        let is_last = idx == num_shards - 1;
        let inv_shard_info = inverted.save_shard(
            &inv_shard_path,
            shard_info.shard_id,
            0,
            inverted.num_minimizers(),
            is_last,
        )?;

        total_minimizers += inv_shard_info.num_minimizers;
        total_bucket_ids += inv_shard_info.num_bucket_ids;
        inv_shards.push(inv_shard_info);
    }

    eprintln!(
        "\nTotal across inverted shards: {} minimizers, {} bucket entries",
        total_minimizers, total_bucket_ids
    );
    eprintln!(
        "Ground truth: {} minimizers, {} bucket entries",
        inverted_truth.num_minimizers(),
        inverted_truth.num_bucket_entries()
    );

    if total_minimizers != inverted_truth.num_minimizers() {
        eprintln!("\n!!! MINIMIZER COUNT MISMATCH !!!");
        eprintln!(
            "Sum of shard minimizers ({}) != ground truth ({})",
            total_minimizers,
            inverted_truth.num_minimizers()
        );
        eprintln!("This indicates overlapping minimizers across shards.");
    }

    // Create and save manifest (bucket-partitioned since built from main shards)
    let manifest = ShardManifest {
        k: main_manifest.k,
        w: main_manifest.w,
        salt: main_manifest.salt,
        source_hash,
        total_minimizers,
        total_bucket_ids,
        has_overlapping_shards: true,
        shards: inv_shards,
    };

    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    manifest.save(&manifest_path)?;

    // Try to load and classify
    eprintln!("\nAttempting to load sharded inverted index...");
    match ShardedInvertedIndex::open(&inverted_path) {
        Ok(sharded) => {
            eprintln!("Successfully loaded!");

            let records = create_query_records(&seqs);
            let threshold = 0.1;

            let results_truth = classify_batch(&index, None, &records, threshold);
            let results_sharded =
                classify_batch_sharded_merge_join(&sharded, None, &records, threshold)?;

            eprintln!("\n=== Classification Results ===");
            eprintln!("Ground truth: {} results", results_truth.len());
            eprintln!("Sharded: {} results", results_sharded.len());

            if results_truth.len() != results_sharded.len() {
                eprintln!("\n!!! CLASSIFICATION RESULT MISMATCH !!!");
                eprintln!(
                    "Missing {} results ({:.2}%)",
                    results_truth.len() - results_sharded.len(),
                    100.0 * (results_truth.len() - results_sharded.len()) as f64
                        / results_truth.len() as f64
                );
            }

            compare_results("Ground-truth", &results_truth, "Sharded", &results_sharded);
        }
        Err(e) => {
            eprintln!("\n!!! LOAD FAILED !!!");
            eprintln!("Error: {}", e);
            eprintln!("\nThis demonstrates the bug: the manifest validation fails");
            eprintln!("because minimizer ranges are not contiguous when built from");
            eprintln!("bucket-partitioned main shards.");
            return Err(e);
        }
    }

    Ok(())
}
