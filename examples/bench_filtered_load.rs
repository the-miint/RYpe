//! Benchmark filtered Parquet loading using merge-scan approach
//!
//! Tests both uniform and clustered query patterns to show real-world performance.

use anyhow::Result;
use rype::InvertedIndex;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <shard_path>", args[0]);
        std::process::exit(1);
    }

    let shard_path = Path::new(&args[1]);
    let parquet_path = shard_path.with_extension("filter_test.parquet");

    println!("=== Benchmarking Filtered Parquet Loading (Merge-Scan) ===\n");

    // Load legacy and save as Parquet
    println!("1. Preparing test data...");
    let legacy = InvertedIndex::load_shard(shard_path)?;
    legacy.save_shard_parquet(&parquet_path, 0, None)?;

    let all_mins = legacy.minimizers();
    let num_mins = all_mins.len();

    // Get Parquet file size
    let parquet_size = std::fs::metadata(&parquet_path)?.len();
    println!("   Total minimizers: {}", num_mins);
    println!(
        "   Parquet file size: {:.1} MB",
        parquet_size as f64 / 1_000_000.0
    );
    println!("   Row groups: ~{}\n", (num_mins + 99_999) / 100_000);

    // Get full load baseline
    let mut full_times = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let _full = InvertedIndex::load_shard_parquet_with_params(
            &parquet_path,
            legacy.k,
            legacy.w,
            legacy.salt,
            0,
        )?;
        full_times.push(t.elapsed().as_micros());
    }
    full_times.sort();
    let full_baseline_us = full_times[1];
    println!(
        "Full load baseline: {:.1} ms\n",
        full_baseline_us as f64 / 1000.0
    );

    // Test CLUSTERED queries (contiguous minimizer ranges - simulates queries from specific regions)
    println!("=== Clustered Query Pattern (contiguous ranges) ===");
    println!("(Simulates queries from specific genomic regions)\n");
    println!("| Query Size | Filtered (ms) | Speedup | RGs Matched | Data Loaded |");
    println!("|------------|---------------|---------|-------------|-------------|");

    let clustered_cases = [
        ("100", 100),
        ("1K", 1_000),
        ("10K", 10_000),
        ("100K", 100_000),
        ("1M", 1_000_000),
    ];

    for (label, query_size) in clustered_cases {
        let query_size = query_size.min(num_mins);

        // Contiguous block from middle of index
        let start_idx = (num_mins - query_size) / 2;
        let query_mins: Vec<u64> = all_mins[start_idx..start_idx + query_size].to_vec();

        // Benchmark filtered load (3 runs, take median)
        let mut filtered_times = Vec::new();
        let mut filtered_result = None;
        for _ in 0..3 {
            let t = Instant::now();
            let filtered = InvertedIndex::load_shard_parquet_for_query(
                &parquet_path,
                legacy.k,
                legacy.w,
                legacy.salt,
                0,
                &query_mins,
            )?;
            filtered_times.push(t.elapsed().as_micros());
            filtered_result = Some(filtered);
        }
        filtered_times.sort();
        let filtered_us = filtered_times[1];

        let filtered = filtered_result.unwrap();
        let speedup = full_baseline_us as f64 / filtered_us.max(1) as f64;
        let data_fraction = filtered.minimizers().len() as f64 / num_mins as f64;

        // Estimate RGs matched
        let rgs_matched = query_size / 100_000 + 1;
        let total_rgs = (num_mins + 99_999) / 100_000;

        println!(
            "| {:>10} | {:>13.1} | {:>6.1}x | {:>5}/{:<5} | {:>10.1}% |",
            label,
            filtered_us as f64 / 1000.0,
            speedup,
            rgs_matched.min(total_rgs),
            total_rgs,
            data_fraction * 100.0
        );
    }

    // Test SPARSE queries (scattered across the index)
    println!("\n=== Sparse Query Pattern (scattered minimizers) ===");
    println!("(Simulates diverse queries touching many regions)\n");
    println!("| Query Size | Filtered (ms) | Speedup | Data Loaded |");
    println!("|------------|---------------|---------|-------------|");

    let sparse_cases = [("10", 10), ("100", 100), ("1K", 1_000)];

    for (label, query_size) in sparse_cases {
        let query_size = query_size.min(num_mins);

        // Sample uniformly
        let step = num_mins / query_size.max(1);
        let query_mins: Vec<u64> = (0..query_size)
            .map(|i| all_mins[(i * step).min(num_mins - 1)])
            .collect();

        // Benchmark
        let mut filtered_times = Vec::new();
        let mut filtered_result = None;
        for _ in 0..3 {
            let t = Instant::now();
            let filtered = InvertedIndex::load_shard_parquet_for_query(
                &parquet_path,
                legacy.k,
                legacy.w,
                legacy.salt,
                0,
                &query_mins,
            )?;
            filtered_times.push(t.elapsed().as_micros());
            filtered_result = Some(filtered);
        }
        filtered_times.sort();
        let filtered_us = filtered_times[1];

        let filtered = filtered_result.unwrap();
        let speedup = full_baseline_us as f64 / filtered_us.max(1) as f64;
        let data_fraction = filtered.minimizers().len() as f64 / num_mins as f64;

        println!(
            "| {:>10} | {:>13.1} | {:>6.1}x | {:>10.1}% |",
            label,
            filtered_us as f64 / 1000.0,
            speedup,
            data_fraction * 100.0
        );
    }

    // Test single minimizer (best case)
    println!("\n=== Best Case: Single Minimizer Query ===\n");

    let tiny_query: Vec<u64> = vec![all_mins[num_mins / 2]];
    let mut tiny_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        let _filtered = InvertedIndex::load_shard_parquet_for_query(
            &parquet_path,
            legacy.k,
            legacy.w,
            legacy.salt,
            0,
            &tiny_query,
        )?;
        tiny_times.push(t.elapsed().as_micros());
    }
    tiny_times.sort();
    let tiny_us = tiny_times[5];

    println!(
        "Single minimizer: {:.2} ms ({:.1}x speedup)",
        tiny_us as f64 / 1000.0,
        full_baseline_us as f64 / tiny_us as f64
    );

    // Also compare against legacy load
    println!("\n=== Legacy vs Parquet Full Load ===\n");

    let mut legacy_times = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let _l = InvertedIndex::load_shard(shard_path)?;
        legacy_times.push(t.elapsed().as_micros());
    }
    legacy_times.sort();
    let legacy_us = legacy_times[1];

    println!("Legacy load:    {:.1} ms", legacy_us as f64 / 1000.0);
    println!("Parquet full:   {:.1} ms", full_baseline_us as f64 / 1000.0);
    println!(
        "Ratio:          {:.2}x",
        legacy_us as f64 / full_baseline_us as f64
    );

    // Cleanup
    std::fs::remove_file(&parquet_path).ok();

    println!(
        "\nNote: Clustered queries benefit most from filtering (typical for real genomic data)."
    );
    println!("      Sparse queries touching >50% of row groups fall back to full load.");

    Ok(())
}
