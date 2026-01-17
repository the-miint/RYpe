//! Explore Parquet row group statistics for range filtering

use anyhow::Result;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use rype::InvertedIndex;
use std::fs::File;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <shard_path>", args[0]);
        std::process::exit(1);
    }

    let shard_path = Path::new(&args[1]);
    let parquet_path = shard_path.with_extension("stats_test.parquet");

    println!("Loading legacy shard: {}", shard_path.display());
    let index = InvertedIndex::load_shard(shard_path)?;

    println!("Saving as Parquet: {}", parquet_path.display());
    index.save_shard_parquet(&parquet_path, 0)?;

    // Explore the statistics
    println!("\n=== Row Group Statistics ===\n");

    let file = File::open(&parquet_path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();

    println!("Total row groups: {}", metadata.num_row_groups());
    println!("Total rows: {}\n", metadata.file_metadata().num_rows());

    // Sample first 5 and last 5 row groups
    let num_rg = metadata.num_row_groups();
    let sample_indices: Vec<usize> = if num_rg <= 10 {
        (0..num_rg).collect()
    } else {
        (0..5).chain((num_rg - 5)..num_rg).collect()
    };

    println!("| Row Group | Rows    | Min Minimizer        | Max Minimizer        |");
    println!("|-----------|---------|----------------------|----------------------|");

    for rg_idx in sample_indices {
        let rg = metadata.row_group(rg_idx);
        let col = rg.column(0); // minimizer column

        if let Some(stats) = col.statistics() {
            match stats {
                Statistics::Int64(s) => {
                    let min = s.min_opt().map(|v| *v as u64);
                    let max = s.max_opt().map(|v| *v as u64);
                    println!(
                        "| {:>9} | {:>7} | {:>20} | {:>20} |",
                        rg_idx,
                        rg.num_rows(),
                        min.map(|v| v.to_string()).unwrap_or("N/A".into()),
                        max.map(|v| v.to_string()).unwrap_or("N/A".into())
                    );
                }
                _ => println!(
                    "| {:>9} | {:>7} | unexpected type      |                      |",
                    rg_idx,
                    rg.num_rows()
                ),
            }
        } else {
            println!(
                "| {:>9} | {:>7} | no stats             | no stats             |",
                rg_idx,
                rg.num_rows()
            );
        }
    }

    // Check if stats are monotonically increasing (row groups are sorted)
    println!("\n=== Checking if row groups are sorted ===");
    let mut prev_max: Option<u64> = None;
    let mut sorted = true;
    let mut overlap_count = 0;

    for rg_idx in 0..num_rg {
        let rg = metadata.row_group(rg_idx);
        let col = rg.column(0);

        if let Some(Statistics::Int64(s)) = col.statistics() {
            let min = s.min_opt().map(|v| *v as u64);
            let max = s.max_opt().map(|v| *v as u64);

            if let (Some(prev), Some(curr_min)) = (prev_max, min) {
                if curr_min < prev {
                    if overlap_count < 5 {
                        println!(
                            "Row group {} min ({}) < previous max ({})",
                            rg_idx, curr_min, prev
                        );
                    }
                    sorted = false;
                    overlap_count += 1;
                }
            }
            prev_max = max;
        }
    }

    if sorted {
        println!("✓ All row groups are sorted! Range filtering will work perfectly.");
    } else {
        println!(
            "✗ {} row groups overlap with previous. Range filtering will still help but won't be perfect.",
            overlap_count
        );
    }

    // Simulate query filtering
    println!("\n=== Simulating Query Filtering ===");

    // Get actual min/max from the index
    let all_mins = index.minimizers();
    let global_min = all_mins.first().copied().unwrap_or(0);
    let global_max = all_mins.last().copied().unwrap_or(u64::MAX);

    println!("Global minimizer range: {} - {}", global_min, global_max);

    // Simulate a query that covers 10% of the range
    let range_size = global_max - global_min;
    let query_min = global_min + range_size / 4;
    let query_max = global_min + range_size / 4 + range_size / 10;

    println!(
        "Simulated query range (10% of data): {} - {}",
        query_min, query_max
    );

    // Count how many row groups would be loaded
    let mut matching_rg = 0;
    for rg_idx in 0..num_rg {
        let rg = metadata.row_group(rg_idx);
        let col = rg.column(0);

        if let Some(Statistics::Int64(s)) = col.statistics() {
            let rg_min = s.min_opt().map(|v| *v as u64).unwrap_or(0);
            let rg_max = s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX);

            // Check if row group overlaps with query range
            if rg_max >= query_min && rg_min <= query_max {
                matching_rg += 1;
            }
        }
    }

    println!(
        "Row groups matching query: {} / {} ({:.1}%)",
        matching_rg,
        num_rg,
        matching_rg as f64 / num_rg as f64 * 100.0
    );
    println!(
        "Potential I/O savings: {:.1}%",
        (1.0 - matching_rg as f64 / num_rg as f64) * 100.0
    );

    // Cleanup
    std::fs::remove_file(&parquet_path).ok();

    Ok(())
}
