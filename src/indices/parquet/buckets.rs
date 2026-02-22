//! Bucket metadata Parquet I/O.
//!
//! This module handles reading and writing bucket metadata (names and sources)
//! to Parquet format within the index directory.

use crate::error::{Result, RypeError};
use crate::types::BucketFileStats;
use arrow::array::{
    Array, ArrayRef, Float64Array, LargeListArray, LargeListBuilder, LargeStringArray,
    LargeStringBuilder, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{WriterProperties, WriterVersion};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::files;

/// Write bucket metadata to Parquet.
///
/// Schema: `(bucket_id: u32, bucket_name: string, sources: list<string>,
///           file_stats_mean: f64, file_stats_median: f64, file_stats_stdev: f64,
///           file_stats_min: f64, file_stats_max: f64)`
///
/// When `bucket_file_stats` is `None` or a bucket has no stats, the stats columns
/// contain NaN for that row.
pub fn write_buckets_parquet(
    index_dir: &Path,
    bucket_names: &HashMap<u32, String>,
    bucket_sources: &HashMap<u32, Vec<String>>,
    bucket_file_stats: Option<&HashMap<u32, BucketFileStats>>,
) -> Result<()> {
    let path = index_dir.join(files::BUCKETS);

    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket_id", DataType::UInt32, false),
        Field::new("bucket_name", DataType::LargeUtf8, false),
        Field::new(
            "sources",
            DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
            false,
        ),
        Field::new("file_stats_mean", DataType::Float64, true),
        Field::new("file_stats_median", DataType::Float64, true),
        Field::new("file_stats_stdev", DataType::Float64, true),
        Field::new("file_stats_min", DataType::Float64, true),
        Field::new("file_stats_max", DataType::Float64, true),
    ]));

    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let file =
        File::create(&path).map_err(|e| RypeError::io(path.clone(), "create buckets file", e))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Collect and sort buckets by ID
    let empty_sources: Vec<String> = Vec::new();
    let mut buckets: Vec<(u32, &String, &Vec<String>)> = bucket_names
        .iter()
        .map(|(&id, name)| {
            let sources = bucket_sources.get(&id).unwrap_or(&empty_sources);
            (id, name, sources)
        })
        .collect();
    buckets.sort_by_key(|(id, _, _)| *id);

    // Build arrays
    let bucket_ids: Vec<u32> = buckets.iter().map(|(id, _, _)| *id).collect();
    let names: Vec<&str> = buckets.iter().map(|(_, name, _)| name.as_str()).collect();

    let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids));
    let bucket_name_array: ArrayRef = Arc::new(LargeStringArray::from(names));

    // Build list array for sources
    let mut list_builder = LargeListBuilder::new(LargeStringBuilder::new());
    for (_, _, sources) in &buckets {
        let values_builder = list_builder.values();
        for source in *sources {
            values_builder.append_value(source);
        }
        list_builder.append(true);
    }
    let sources_array: ArrayRef = Arc::new(list_builder.finish());

    // Build stats arrays
    let empty_stats: HashMap<u32, BucketFileStats> = HashMap::new();
    let stats_map = bucket_file_stats.unwrap_or(&empty_stats);

    let means: Vec<f64> = buckets
        .iter()
        .map(|(id, _, _)| stats_map.get(id).map(|s| s.mean).unwrap_or(f64::NAN))
        .collect();
    let medians: Vec<f64> = buckets
        .iter()
        .map(|(id, _, _)| stats_map.get(id).map(|s| s.median).unwrap_or(f64::NAN))
        .collect();
    let stdevs: Vec<f64> = buckets
        .iter()
        .map(|(id, _, _)| stats_map.get(id).map(|s| s.stdev).unwrap_or(f64::NAN))
        .collect();

    let mins: Vec<f64> = buckets
        .iter()
        .map(|(id, _, _)| stats_map.get(id).map(|s| s.min).unwrap_or(f64::NAN))
        .collect();
    let maxs: Vec<f64> = buckets
        .iter()
        .map(|(id, _, _)| stats_map.get(id).map(|s| s.max).unwrap_or(f64::NAN))
        .collect();

    let mean_array: ArrayRef = Arc::new(Float64Array::from(means));
    let median_array: ArrayRef = Arc::new(Float64Array::from(medians));
    let stdev_array: ArrayRef = Arc::new(Float64Array::from(stdevs));
    let min_array: ArrayRef = Arc::new(Float64Array::from(mins));
    let max_array: ArrayRef = Arc::new(Float64Array::from(maxs));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            bucket_id_array,
            bucket_name_array,
            sources_array,
            mean_array,
            median_array,
            stdev_array,
            min_array,
            max_array,
        ],
    )?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Read bucket metadata from Parquet.
///
/// Returns (bucket_names, bucket_sources, bucket_file_stats).
/// `bucket_file_stats` is `None` when the file was written by an older version
/// without stats columns. A deprecation warning is logged in that case.
#[allow(clippy::type_complexity)]
pub fn read_buckets_parquet(
    index_dir: &Path,
) -> Result<(
    HashMap<u32, String>,
    HashMap<u32, Vec<String>>,
    Option<HashMap<u32, BucketFileStats>>,
)> {
    let path = index_dir.join(files::BUCKETS);

    let file =
        File::open(&path).map_err(|e| RypeError::io(path.clone(), "open buckets file", e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    // Detect whether stats columns are present
    let has_stats = builder
        .schema()
        .column_with_name("file_stats_mean")
        .is_some();

    if !has_stats {
        log::info!(
            "Index at {} was built without file statistics. \
             Rebuild with a newer version to include per-bucket file stats.",
            index_dir.display()
        );
    }

    let reader = builder.build()?;

    let mut bucket_names = HashMap::new();
    let mut bucket_sources = HashMap::new();
    let mut bucket_file_stats: HashMap<u32, BucketFileStats> = HashMap::new();

    for batch in reader {
        let batch = batch?;
        let bucket_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                RypeError::format(
                    path.clone(),
                    "expected UInt32Array for bucket_id".to_string(),
                )
            })?;
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| {
                RypeError::format(
                    path.clone(),
                    "expected LargeStringArray for bucket_name".to_string(),
                )
            })?;
        let sources_list = batch
            .column(2)
            .as_any()
            .downcast_ref::<LargeListArray>()
            .ok_or_else(|| {
                RypeError::format(
                    path.clone(),
                    "expected LargeListArray for sources".to_string(),
                )
            })?;

        // Optionally read stats columns by name
        let mean_col = if has_stats {
            batch
                .column_by_name("file_stats_mean")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        } else {
            None
        };
        let median_col = if has_stats {
            batch
                .column_by_name("file_stats_median")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        } else {
            None
        };
        let stdev_col = if has_stats {
            batch
                .column_by_name("file_stats_stdev")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        } else {
            None
        };
        let min_col = if has_stats {
            batch
                .column_by_name("file_stats_min")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        } else {
            None
        };
        let max_col = if has_stats {
            batch
                .column_by_name("file_stats_max")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        } else {
            None
        };

        for i in 0..batch.num_rows() {
            let bucket_id = bucket_ids.value(i);
            let name = names.value(i).to_string();

            let sources_array = sources_list.value(i);
            let sources_str = sources_array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    RypeError::format(
                        path.clone(),
                        "expected LargeStringArray for sources items".to_string(),
                    )
                })?;
            let sources: Vec<String> = (0..sources_str.len())
                .map(|j| sources_str.value(j).to_string())
                .collect();

            bucket_names.insert(bucket_id, name);
            bucket_sources.insert(bucket_id, sources);

            // Read stats if columns are present and values are not NaN
            if let (Some(mean_arr), Some(median_arr), Some(stdev_arr)) =
                (mean_col, median_col, stdev_col)
            {
                let mean = mean_arr.value(i);
                let median = median_arr.value(i);
                let stdev = stdev_arr.value(i);
                if !mean.is_nan() && !median.is_nan() && !stdev.is_nan() {
                    // min/max columns may be absent in indices built between
                    // initial stats support and the min/max addition
                    let min = min_col.map(|c| c.value(i)).unwrap_or(f64::NAN);
                    let max = max_col.map(|c| c.value(i)).unwrap_or(f64::NAN);
                    bucket_file_stats.insert(
                        bucket_id,
                        BucketFileStats {
                            mean,
                            median,
                            stdev,
                            min,
                            max,
                        },
                    );
                }
            }
        }
    }

    let stats = if has_stats && !bucket_file_stats.is_empty() {
        Some(bucket_file_stats)
    } else {
        None
    };

    Ok((bucket_names, bucket_sources, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::parquet::manifest::create_index_directory;
    use tempfile::TempDir;

    #[test]
    fn test_buckets_parquet_round_trip() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("test.ryidx");
        create_index_directory(&index_dir).unwrap();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(2, "Archaea".to_string());
        bucket_names.insert(3, "Eukaryota".to_string());

        let mut bucket_sources = HashMap::new();
        bucket_sources.insert(
            1,
            vec!["ecoli.fna".to_string(), "bsubtilis.fna".to_string()],
        );
        bucket_sources.insert(2, vec!["haloferax.fna".to_string()]);
        bucket_sources.insert(3, vec![]); // Empty sources

        write_buckets_parquet(&index_dir, &bucket_names, &bucket_sources, None).unwrap();
        let (loaded_names, loaded_sources, loaded_stats) =
            read_buckets_parquet(&index_dir).unwrap();

        assert_eq!(loaded_names, bucket_names);
        assert_eq!(loaded_sources.get(&1), bucket_sources.get(&1));
        assert_eq!(loaded_sources.get(&2), bucket_sources.get(&2));
        // Empty sources should be preserved
        assert_eq!(loaded_sources.get(&3).map(|v| v.len()), Some(0));
        // No stats when None passed
        assert!(loaded_stats.is_none());
    }

    #[test]
    fn test_buckets_parquet_round_trip_with_stats() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("stats.ryidx");
        create_index_directory(&index_dir).unwrap();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "B1".to_string());
        bucket_names.insert(2, "B2".to_string());

        let mut bucket_sources = HashMap::new();
        bucket_sources.insert(1, vec!["f1.fa".to_string()]);
        bucket_sources.insert(2, vec!["f2.fa".to_string(), "f3.fa".to_string()]);

        let mut stats = HashMap::new();
        stats.insert(
            1,
            BucketFileStats {
                mean: 1000.0,
                median: 1000.0,
                stdev: 0.0,
                min: 1000.0,
                max: 1000.0,
            },
        );
        stats.insert(
            2,
            BucketFileStats {
                mean: 500.5,
                median: 400.0,
                stdev: 123.456,
                min: 200.0,
                max: 800.0,
            },
        );

        write_buckets_parquet(&index_dir, &bucket_names, &bucket_sources, Some(&stats)).unwrap();
        let (loaded_names, _loaded_sources, loaded_stats) =
            read_buckets_parquet(&index_dir).unwrap();

        assert_eq!(loaded_names, bucket_names);
        let loaded_stats = loaded_stats.expect("should have stats");
        assert_eq!(loaded_stats.len(), 2);

        let s1 = &loaded_stats[&1];
        assert_eq!(s1.mean, 1000.0);
        assert_eq!(s1.median, 1000.0);
        assert_eq!(s1.stdev, 0.0);
        assert_eq!(s1.min, 1000.0);
        assert_eq!(s1.max, 1000.0);

        let s2 = &loaded_stats[&2];
        assert!((s2.mean - 500.5).abs() < 1e-9);
        assert!((s2.median - 400.0).abs() < 1e-9);
        assert!((s2.stdev - 123.456).abs() < 1e-9);
        assert!((s2.min - 200.0).abs() < 1e-9);
        assert!((s2.max - 800.0).abs() < 1e-9);
    }

    #[test]
    fn test_buckets_parquet_backward_compat() {
        // Write a 3-column file manually (old format without stats)
        use parquet::basic::Compression;
        use parquet::file::properties::{WriterProperties, WriterVersion};

        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("old.ryidx");
        create_index_directory(&index_dir).unwrap();

        let path = index_dir.join(super::super::files::BUCKETS);
        let schema = Arc::new(Schema::new(vec![
            Field::new("bucket_id", DataType::UInt32, false),
            Field::new("bucket_name", DataType::LargeUtf8, false),
            Field::new(
                "sources",
                DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
                false,
            ),
        ]));

        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();

        let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(vec![1, 2]));
        let name_array: ArrayRef = Arc::new(LargeStringArray::from(vec!["Alpha", "Beta"]));
        let mut list_builder = LargeListBuilder::new(LargeStringBuilder::new());
        list_builder.values().append_value("src1");
        list_builder.append(true);
        list_builder.values().append_value("src2");
        list_builder.append(true);
        let sources_array: ArrayRef = Arc::new(list_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![bucket_id_array, name_array, sources_array],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Now read with new code — should get None for stats
        let (loaded_names, loaded_sources, loaded_stats) =
            read_buckets_parquet(&index_dir).unwrap();

        assert_eq!(loaded_names.len(), 2);
        assert_eq!(loaded_names[&1], "Alpha");
        assert_eq!(loaded_names[&2], "Beta");
        assert_eq!(loaded_sources[&1], vec!["src1".to_string()]);
        assert_eq!(loaded_sources[&2], vec!["src2".to_string()]);
        assert!(loaded_stats.is_none(), "Old format should have no stats");
    }
}
