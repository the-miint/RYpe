//! Benchmark harness for the Arrow single-bucket index build.
//!
//! Drives `build_index_from_arrow` directly from a directory of Parquet "chunk"
//! files (schema: feature_idx:int64, chunk_index:int32, chunk_data:utf8|binary),
//! mapping every feature into ONE bucket. This is the in-process equivalent of the
//! C-API `rype_index_build_from_arrow` path. Single-bucket builds enable the
//! cross-shard dedup filter by default (budget = memory budget / 4).
//!
//! Usage:
//!   cargo build --release --example arrow_build_bench
//!   target/release/examples/arrow_build_bench <chunk_dir> <output.ryxdi>
//!
//! Tuning via env (all optional):
//!   RYPE_BENCH_K=64  RYPE_BENCH_W=200  RYPE_BENCH_SALT=<u64>
//!   RYPE_BENCH_MAXMEM=<bytes>   (0/unset = auto-detect; also scales the dedup budget)
//!   RYPE_BENCH_ORIENT=0|1
//!   RYPE_BUILD_VERBOSE=1            (intermediate/final shard diagnostics)

use std::collections::BTreeSet;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;

const BUCKET_NAME: &str = "human";

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn list_parquet(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("parquet"))
        .collect();
    files.sort();
    files
}

/// Collect the distinct feature_idx values across all chunk files (projected read
/// of just that column) so the mapping covers exactly the features that appear.
fn distinct_features(files: &[PathBuf]) -> BTreeSet<i64> {
    let mut set = BTreeSet::new();
    for path in files {
        let f = File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)
            .unwrap_or_else(|e| panic!("parquet {path:?}: {e}"));
        let mask = ProjectionMask::columns(builder.parquet_schema(), ["feature_idx"]);
        let reader = builder
            .with_projection(mask)
            .build()
            .unwrap_or_else(|e| panic!("reader {path:?}: {e}"));
        for batch in reader {
            let batch = batch.unwrap_or_else(|e| panic!("batch {path:?}: {e}"));
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("feature_idx must be Int64");
            for i in 0..col.len() {
                set.insert(col.value(i));
            }
        }
    }
    set
}

fn mapping_batch(features: &BTreeSet<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("feature_idx", DataType::Int64, false),
        Field::new("bucket_name", DataType::Utf8, false),
    ]));
    let fids: Int64Array = features.iter().copied().collect();
    let names: StringArray = (0..features.len()).map(|_| Some(BUCKET_NAME)).collect();
    RecordBatch::try_new(
        schema,
        vec![Arc::new(fids) as ArrayRef, Arc::new(names) as ArrayRef],
    )
    .expect("mapping batch")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let chunk_dir = args
        .next()
        .expect("usage: arrow_build_bench <chunk_dir> <output.ryxdi>");
    let out_dir = args
        .next()
        .expect("usage: arrow_build_bench <chunk_dir> <output.ryxdi>");

    let k = env_usize("RYPE_BENCH_K", 64);
    let w = env_usize("RYPE_BENCH_W", 200);
    let salt: u64 = std::env::var("RYPE_BENCH_SALT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6148914691236517205);
    let max_mem = env_usize("RYPE_BENCH_MAXMEM", 0);
    let max_memory = if max_mem == 0 { None } else { Some(max_mem) };
    let orient = env_usize("RYPE_BENCH_ORIENT", 0) != 0;

    let files = list_parquet(&chunk_dir);
    assert!(!files.is_empty(), "no .parquet files in {chunk_dir}");
    eprintln!("[bench] {} chunk files in {chunk_dir}", files.len());

    let t_map = Instant::now();
    let features = distinct_features(&files);
    eprintln!(
        "[bench] {} distinct features (mapping scan {:.1}s)",
        features.len(),
        t_map.elapsed().as_secs_f64()
    );
    let mapping = std::iter::once(Ok(mapping_batch(&features)));

    // Lazy, streaming chunk iterator across all files (one open at a time).
    let chunk_files = files.clone();
    let chunks = chunk_files.into_iter().flat_map(|path| {
        let f = File::open(&path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
        ParquetRecordBatchReaderBuilder::try_new(f)
            .unwrap_or_else(|e| panic!("parquet {path:?}: {e}"))
            .build()
            .unwrap_or_else(|e| panic!("reader {path:?}: {e}"))
    });

    eprintln!(
        "[bench] build k={k} w={w} salt={salt} orient={orient} max_memory={} out={out_dir}",
        max_mem
    );
    let t0 = Instant::now();
    let stats = rype::parquet_index::build_index_from_arrow(
        std::path::Path::new(&out_dir),
        chunks,
        mapping,
        k,
        w,
        salt,
        orient,
        max_memory,
        None,
    )
    .expect("build_index_from_arrow failed");
    let secs = t0.elapsed().as_secs_f64();

    eprintln!("[bench] DONE in {secs:.1}s");
    println!(
        "buckets={} features={} total_minimizers={} num_shards={} build_secs={:.1}",
        stats.num_buckets, stats.num_features, stats.total_minimizers, stats.num_shards, secs
    );
}
