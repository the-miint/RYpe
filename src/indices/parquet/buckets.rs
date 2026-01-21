//! Bucket metadata Parquet I/O.
//!
//! This module handles reading and writing bucket metadata (names and sources)
//! to Parquet format within the index directory.

use crate::error::{Result, RypeError};
use arrow::array::{
    Array, ArrayRef, LargeListArray, LargeListBuilder, LargeStringArray, LargeStringBuilder,
    UInt32Array,
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
/// Schema: `(bucket_id: u32, bucket_name: string, sources: list<string>)`
pub fn write_buckets_parquet(
    index_dir: &Path,
    bucket_names: &HashMap<u32, String>,
    bucket_sources: &HashMap<u32, Vec<String>>,
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

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![bucket_id_array, bucket_name_array, sources_array],
    )?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Read bucket metadata from Parquet.
///
/// Returns (bucket_names, bucket_sources).
#[allow(clippy::type_complexity)]
pub fn read_buckets_parquet(
    index_dir: &Path,
) -> Result<(HashMap<u32, String>, HashMap<u32, Vec<String>>)> {
    let path = index_dir.join(files::BUCKETS);

    let file =
        File::open(&path).map_err(|e| RypeError::io(path.clone(), "open buckets file", e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut bucket_names = HashMap::new();
    let mut bucket_sources = HashMap::new();

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
        }
    }

    Ok((bucket_names, bucket_sources))
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

        write_buckets_parquet(&index_dir, &bucket_names, &bucket_sources).unwrap();
        let (loaded_names, loaded_sources) = read_buckets_parquet(&index_dir).unwrap();

        assert_eq!(loaded_names, bucket_names);
        assert_eq!(loaded_sources.get(&1), bucket_sources.get(&1));
        assert_eq!(loaded_sources.get(&2), bucket_sources.get(&2));
        // Empty sources should be preserved
        assert_eq!(loaded_sources.get(&3).map(|v| v.len()), Some(0));
    }
}
