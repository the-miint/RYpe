//! End-to-end classification benchmark: Legacy vs Parquet inverted index
//!
//! Compares classification performance with both index formats.

use anyhow::Result;
use rype::{classify_batch_sharded_merge_join, InvertedIndex, QueryRecord, ShardedInvertedIndex};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <main_index_path> <query_fastq>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  {} perf-data/n100-w200-fixed.ryidx perf-data/example-query-file/S1_1L_Sterivex_S000_L001_R1_001.trimmed.fastq.gz",
            args[0]
        );
        std::process::exit(1);
    }

    let main_index_path = Path::new(&args[1]);
    let query_path = Path::new(&args[2]);

    // Derive inverted index paths
    // For n100-w200-fixed.ryidx -> n100-w200-fixed.ryxdi.manifest
    let base_name = main_index_path.file_stem().unwrap().to_str().unwrap();
    let parent = main_index_path.parent().unwrap();
    let legacy_inverted_path = parent.join(format!("{}.ryxdi.manifest", base_name));
    let parquet_inverted_path = parent.join(format!("{}.parquet.ryxdi.manifest", base_name));

    println!("=== End-to-End Classification Benchmark ===\n");
    println!("Main index:     {}", main_index_path.display());
    println!("Query file:     {}", query_path.display());
    println!();

    // Read query sequences
    println!("1. Reading query sequences...");
    let t = Instant::now();
    let mut queries: Vec<(i64, Vec<u8>)> = Vec::new();
    let mut reader = needletail::parse_fastx_file(query_path)?;
    let mut read_id: i64 = 0;
    while let Some(record) = reader.next() {
        let record = record?;
        let seq = record.seq().to_vec();
        queries.push((read_id, seq));
        read_id += 1;
    }
    println!(
        "   Read {} sequences in {:.1} ms",
        queries.len(),
        t.elapsed().as_millis()
    );
    println!();

    // Parquet base path
    let parquet_base = parent.join(format!("{}.parquet.ryxdi", base_name));
    let parquet_inverted_dir = parquet_base.join("inverted");

    // Check for inverted indices
    let has_legacy = legacy_inverted_path.exists();
    let has_parquet = parquet_inverted_path.exists() && parquet_inverted_dir.exists();

    println!("2. Available inverted indices:");
    println!(
        "   Legacy:  {} ({})",
        if has_legacy { "✓" } else { "✗" },
        legacy_inverted_path.display()
    );
    println!(
        "   Parquet: {} ({})",
        if has_parquet { "✓" } else { "✗" },
        parquet_inverted_path.display()
    );
    println!();

    if !has_legacy {
        eprintln!("Error: No legacy inverted index found. Run:");
        eprintln!(
            "  cargo run --release --bin rype -- index invert -i {}",
            main_index_path.display()
        );
        std::process::exit(1);
    }

    // Create Parquet version if needed
    // Parquet shards go in {base}.parquet.ryxdi/inverted/shard.{id}.parquet
    if !has_parquet {
        println!("3. Creating Parquet inverted index...");
        let t = Instant::now();

        // Create directory structure
        println!("   Creating dir: {}", parquet_inverted_dir.display());
        std::fs::create_dir_all(&parquet_inverted_dir)?;

        // Load legacy inverted index (use base path without .manifest)
        let legacy_base = parent.join(format!("{}.ryxdi", base_name));
        println!("   Opening legacy from: {}", legacy_base.display());
        let legacy_sharded = ShardedInvertedIndex::open(&legacy_base)?;
        let manifest = legacy_sharded.manifest();

        // Create Parquet shards in the expected location
        for shard_info in &manifest.shards {
            let legacy_shard = legacy_sharded.load_shard(shard_info.shard_id)?;
            let parquet_shard_path =
                parquet_inverted_dir.join(format!("shard.{}.parquet", shard_info.shard_id));
            legacy_shard.save_shard_parquet(&parquet_shard_path, shard_info.shard_id, None)?;
        }

        // Create manifest for Parquet shards (copy from legacy)
        std::fs::copy(&legacy_inverted_path, &parquet_inverted_path)?;

        println!("   Created in {:.1} s", t.elapsed().as_secs_f64());
        println!();
    } else {
        println!("3. Parquet index already exists, skipping creation.\n");
    }

    // Benchmark parameters
    let threshold = 0.1;
    let batch_size = 10_000;
    let num_batches = (queries.len() + batch_size - 1) / batch_size;

    println!("4. Classification benchmark:");
    println!("   Threshold:   {}", threshold);
    println!("   Batch size:  {}", batch_size);
    println!("   Num batches: {}", num_batches);
    println!();

    // Benchmark Legacy format
    println!("--- Legacy Format ---");
    let legacy_base = parent.join(format!("{}.ryxdi", base_name));
    let legacy_sharded = ShardedInvertedIndex::open(&legacy_base)?;
    println!(
        "   Index loaded: {} shards",
        legacy_sharded.manifest().shards.len()
    );

    let t_total = Instant::now();
    let mut legacy_hits = 0usize;
    let mut legacy_classify_time_ms = 0u128;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(queries.len());

        // Create QueryRecord references
        let batch_refs: Vec<QueryRecord> = queries[start..end]
            .iter()
            .map(|(id, seq)| (*id, seq.as_slice(), None))
            .collect();

        let t_classify = Instant::now();
        let results =
            classify_batch_sharded_merge_join(&legacy_sharded, None, &batch_refs, threshold, None)?;
        legacy_classify_time_ms += t_classify.elapsed().as_millis();
        legacy_hits += results.len();
    }
    let legacy_total_ms = t_total.elapsed().as_millis();

    println!("   Total time:     {:.1} ms", legacy_total_ms);
    println!("   Classify time:  {:.1} ms", legacy_classify_time_ms);
    println!("   Hits:           {}", legacy_hits);
    println!(
        "   Reads/sec:      {:.0}",
        queries.len() as f64 / (legacy_total_ms as f64 / 1000.0)
    );
    println!();

    // Benchmark Parquet format
    println!("--- Parquet Format ---");
    // Open uses base path, not manifest path
    let parquet_sharded = ShardedInvertedIndex::open(&parquet_base)?;
    println!(
        "   Index loaded: {} shards",
        parquet_sharded.manifest().shards.len()
    );

    let t_total = Instant::now();
    let mut parquet_hits = 0usize;
    let mut parquet_classify_time_ms = 0u128;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(queries.len());

        let batch_refs: Vec<QueryRecord> = queries[start..end]
            .iter()
            .map(|(id, seq)| (*id, seq.as_slice(), None))
            .collect();

        let t_classify = Instant::now();
        let results = classify_batch_sharded_merge_join(
            &parquet_sharded,
            None,
            &batch_refs,
            threshold,
            None,
        )?;
        parquet_classify_time_ms += t_classify.elapsed().as_millis();
        parquet_hits += results.len();
    }
    let parquet_total_ms = t_total.elapsed().as_millis();

    println!("   Total time:     {:.1} ms", parquet_total_ms);
    println!("   Classify time:  {:.1} ms", parquet_classify_time_ms);
    println!("   Hits:           {}", parquet_hits);
    println!(
        "   Reads/sec:      {:.0}",
        queries.len() as f64 / (parquet_total_ms as f64 / 1000.0)
    );
    println!();

    // Summary
    println!("=== Summary ===");
    println!();
    println!("| Metric           | Legacy     | Parquet    | Speedup |");
    println!("|------------------|------------|------------|---------|");
    println!(
        "| Total time       | {:>8.0} ms | {:>8.0} ms | {:>6.2}x |",
        legacy_total_ms,
        parquet_total_ms,
        legacy_total_ms as f64 / parquet_total_ms.max(1) as f64
    );
    println!(
        "| Classify time    | {:>8.0} ms | {:>8.0} ms | {:>6.2}x |",
        legacy_classify_time_ms,
        parquet_classify_time_ms,
        legacy_classify_time_ms as f64 / parquet_classify_time_ms.max(1) as f64
    );
    println!(
        "| Reads/sec        | {:>10.0} | {:>10.0} | {:>6.2}x |",
        queries.len() as f64 / (legacy_total_ms as f64 / 1000.0),
        queries.len() as f64 / (parquet_total_ms as f64 / 1000.0),
        legacy_total_ms as f64 / parquet_total_ms.max(1) as f64
    );

    // Verify correctness
    if legacy_hits != parquet_hits {
        println!();
        println!(
            "WARNING: Hit counts differ! Legacy={}, Parquet={}",
            legacy_hits, parquet_hits
        );
    } else {
        println!();
        println!("✓ Results match ({} hits)", legacy_hits);
    }

    Ok(())
}
