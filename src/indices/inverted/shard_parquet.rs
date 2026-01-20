//! Parquet format for shard serialization.

use anyhow::{anyhow, Context, Result};
use std::path::Path;

use super::InvertedIndex;
use crate::constants::{MAX_INVERTED_BUCKET_IDS, MAX_INVERTED_MINIMIZERS, PARQUET_BATCH_SIZE};
use crate::indices::sharded::ShardInfo;

impl InvertedIndex {
    /// Save this inverted index as a Parquet shard file.
    ///
    /// The Parquet schema is flattened: (minimizer: u64, bucket_id: u32)
    /// with one row per (minimizer, bucket_id) pair, sorted by minimizer then bucket_id.
    /// This enables row group filtering based on minimizer range statistics.
    ///
    /// # Memory Efficiency
    /// Streams directly from CSR format without materializing the full flattened
    /// dataset. Memory usage is O(BATCH_SIZE) instead of O(total_pairs).
    ///
    /// # Arguments
    /// * `path` - Output path (should end in .parquet)
    /// * `shard_id` - Shard identifier for manifest
    /// * `options` - Optional Parquet write options (compression, bloom filters, etc.)
    ///
    /// # Returns
    /// ShardInfo describing the written shard.
    pub fn save_shard_parquet(
        &self,
        path: &Path,
        shard_id: u32,
        options: Option<&super::super::parquet::ParquetWriteOptions>,
    ) -> Result<ShardInfo> {
        use arrow::array::{ArrayRef, UInt32Builder, UInt64Builder};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let opts = options.cloned().unwrap_or_default();

        if self.minimizers.is_empty() {
            // Empty shard - don't create file
            return Ok(ShardInfo {
                shard_id,
                min_start: 0,
                min_end: 0,
                is_last_shard: true,
                num_minimizers: 0,
                num_bucket_ids: 0,
            });
        }

        // Validate that minimizers are strictly increasing (required for correct loading)
        debug_assert!(
            self.minimizers.windows(2).all(|w| w[0] < w[1]),
            "InvertedIndex minimizers must be strictly increasing for Parquet serialization"
        );

        // Schema: (minimizer: u64, bucket_id: u32)
        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));

        // DRY: Use ParquetWriteOptions::to_writer_properties() as single source of truth
        let props = opts.to_writer_properties();

        let file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create Parquet shard: {}", path.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        // Stream from CSR format in batches without materializing all pairs
        let mut minimizer_builder = UInt64Builder::with_capacity(PARQUET_BATCH_SIZE);
        let mut bucket_id_builder = UInt32Builder::with_capacity(PARQUET_BATCH_SIZE);
        let mut pairs_in_batch = 0;

        for (i, &minimizer) in self.minimizers.iter().enumerate() {
            let start = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;

            for &bucket_id in &self.bucket_ids[start..end] {
                minimizer_builder.append_value(minimizer);
                bucket_id_builder.append_value(bucket_id);
                pairs_in_batch += 1;

                if pairs_in_batch >= PARQUET_BATCH_SIZE {
                    // Flush batch
                    let minimizer_array: ArrayRef = Arc::new(minimizer_builder.finish());
                    let bucket_id_array: ArrayRef = Arc::new(bucket_id_builder.finish());

                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![minimizer_array, bucket_id_array],
                    )?;
                    writer.write(&batch)?;

                    // Reset builders for next batch
                    minimizer_builder = UInt64Builder::with_capacity(PARQUET_BATCH_SIZE);
                    bucket_id_builder = UInt32Builder::with_capacity(PARQUET_BATCH_SIZE);
                    pairs_in_batch = 0;
                }
            }
        }

        // Flush remaining pairs
        if pairs_in_batch > 0 {
            let minimizer_array: ArrayRef = Arc::new(minimizer_builder.finish());
            let bucket_id_array: ArrayRef = Arc::new(bucket_id_builder.finish());

            let batch =
                RecordBatch::try_new(schema.clone(), vec![minimizer_array, bucket_id_array])?;
            writer.write(&batch)?;
        }

        writer.close()?;

        let min_start = self.minimizers[0];
        let min_end = 0; // Last shard marker

        Ok(ShardInfo {
            shard_id,
            min_start,
            min_end,
            is_last_shard: true,
            num_minimizers: self.minimizers.len(),
            num_bucket_ids: self.bucket_ids.len(),
        })
    }

    /// Load a Parquet shard with explicit parameters.
    ///
    /// This is the main entry point for loading Parquet shards. Parameters are
    /// provided by the manifest file that accompanies Parquet shards.
    ///
    /// # Performance
    /// Row groups are read in parallel using rayon, then concatenated and
    /// validated in a single pass to build the CSR structure.
    ///
    /// # Memory
    /// Uses a shared Bytes buffer to avoid opening N file descriptors.
    /// Peak memory usage is 2x the Parquet file size (buffer + decoded pairs).
    pub fn load_shard_parquet_with_params(
        path: &Path,
        k: usize,
        w: usize,
        salt: u64,
        source_hash: u64,
    ) -> Result<Self> {
        use arrow::array::{Array, UInt32Array, UInt64Array};
        use bytes::Bytes;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use rayon::prelude::*;
        use std::fs::File;
        use std::io::Read;

        // Validate k value (same as legacy format)
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value for Parquet shard: {} (must be 16, 32, or 64)",
                k
            ));
        }

        // Read entire file into memory once (avoids N file opens)
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open Parquet shard: {}", path.display()))?;
        let file_size = file.metadata()?.len() as usize;
        let mut buffer = Vec::with_capacity(file_size);
        file.read_to_end(&mut buffer)?;
        let bytes = Bytes::from(buffer);

        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
        let metadata = builder.metadata().clone();
        let num_row_groups = metadata.num_row_groups();

        if num_row_groups == 0 {
            return Ok(InvertedIndex {
                k,
                w,
                salt,
                source_hash,
                minimizers: Vec::new(),
                offsets: vec![0],
                bucket_ids: Vec::new(),
            });
        }

        // Parallel read of row groups using shared bytes
        let row_group_results: Vec<Result<Vec<(u64, u32)>>> = (0..num_row_groups)
            .into_par_iter()
            .map(|rg_idx| {
                let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
                let reader = builder.with_row_groups(vec![rg_idx]).build()?;

                let mut pairs = Vec::new();

                for batch in reader {
                    let batch = batch?;

                    let min_col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .context("Expected UInt64Array for minimizer column")?;

                    let bid_col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .context("Expected UInt32Array for bucket_id column")?;

                    for i in 0..batch.num_rows() {
                        pairs.push((min_col.value(i), bid_col.value(i)));
                    }
                }

                Ok(pairs)
            })
            .collect();

        // Concatenate row groups in order (they're already sorted globally since we write them that way)
        // This is O(n) vs O(n log k) for k-way merge, and avoids tuple overhead.
        let mut all_minimizers: Vec<u64> = Vec::new();
        let mut all_bucket_ids: Vec<u32> = Vec::new();

        for result in row_group_results {
            let pairs = result?;
            for (m, b) in pairs {
                all_minimizers.push(m);
                all_bucket_ids.push(b);
            }
        }

        if all_minimizers.is_empty() {
            return Ok(InvertedIndex {
                k,
                w,
                salt,
                source_hash,
                minimizers: Vec::new(),
                offsets: vec![0],
                bucket_ids: Vec::new(),
            });
        }

        // Validate global ordering and build CSR structure in one pass
        let mut minimizers: Vec<u64> = Vec::with_capacity(all_minimizers.len() / 2);
        let mut offsets: Vec<u32> = Vec::with_capacity(all_minimizers.len() / 2 + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(all_bucket_ids.len());

        offsets.push(0);
        let mut current_min = all_minimizers[0];
        let mut prev_min = all_minimizers[0];
        minimizers.push(current_min);

        for (i, (&m, &b)) in all_minimizers.iter().zip(all_bucket_ids.iter()).enumerate() {
            // Validation: minimizers must be non-decreasing
            if m < prev_min {
                return Err(anyhow!(
                    "Parquet shard has unsorted minimizers at row {}: {} < {} (corrupt file?)",
                    i,
                    m,
                    prev_min
                ));
            }
            prev_min = m;

            if m != current_min {
                // New minimizer - finalize previous
                offsets.push(bucket_ids_out.len() as u32);
                minimizers.push(m);
                current_min = m;
            }
            bucket_ids_out.push(b);
        }

        // Finalize last minimizer
        offsets.push(bucket_ids_out.len() as u32);

        // Validation: offsets must be monotonically increasing
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(anyhow!(
                "Parquet shard has invalid CSR offsets: first offset must be 0 and offsets must be monotonic"
            ));
        }

        // Validation: minimizers must be strictly increasing (after grouping)
        if minimizers.windows(2).any(|w| w[0] >= w[1]) {
            return Err(anyhow!(
                "Parquet shard has duplicate minimizers after merge (corrupt file?)"
            ));
        }

        // Validation: size limits (same as legacy format)
        if minimizers.len() > MAX_INVERTED_MINIMIZERS {
            return Err(anyhow!(
                "Parquet shard has too many minimizers: {} (limit: {})",
                minimizers.len(),
                MAX_INVERTED_MINIMIZERS
            ));
        }
        if bucket_ids_out.len() > MAX_INVERTED_BUCKET_IDS {
            return Err(anyhow!(
                "Parquet shard has too many bucket IDs: {} (limit: {})",
                bucket_ids_out.len(),
                MAX_INVERTED_BUCKET_IDS
            ));
        }

        minimizers.shrink_to_fit();
        offsets.shrink_to_fit();
        bucket_ids_out.shrink_to_fit();

        Ok(InvertedIndex {
            k,
            w,
            salt,
            source_hash,
            minimizers,
            offsets,
            bucket_ids: bucket_ids_out,
        })
    }

    /// Get row group statistics from a Parquet shard without loading data.
    ///
    /// Returns a list of (row_group_index, min_minimizer, max_minimizer) tuples.
    /// This can be used to plan which row groups to load for a query.
    pub fn get_parquet_row_group_stats(path: &Path) -> Result<Vec<(usize, u64, u64)>> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;
        use parquet::file::statistics::Statistics;
        use std::fs::File;

        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let metadata = reader.metadata();
        let num_row_groups = metadata.num_row_groups();

        let mut stats = Vec::with_capacity(num_row_groups);

        for rg_idx in 0..num_row_groups {
            let rg_meta = metadata.row_group(rg_idx);
            let col_meta = rg_meta.column(0); // minimizer column

            let (rg_min, rg_max) = if let Some(Statistics::Int64(s)) = col_meta.statistics() {
                let min = s.min_opt().map(|v| *v as u64).unwrap_or(0);
                let max = s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX);
                (min, max)
            } else {
                (0, u64::MAX) // No stats, assume full range
            };

            stats.push((rg_idx, rg_min, rg_max));
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::main::Index;
    use tempfile::TempDir;

    #[test]
    fn test_inverted_parquet_roundtrip() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.0.parquet");

        // Create an inverted index
        let mut index = Index::new(64, 50, 0xABCD).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.buckets.insert(3, vec![500, 600]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.bucket_names.insert(3, "C".into());

        let inverted = InvertedIndex::build_from_index(&index);
        let original_minimizers = inverted.minimizers().to_vec();
        let original_bucket_ids = inverted.bucket_ids().to_vec();

        // Save as Parquet
        let shard_info = inverted.save_shard_parquet(&path, 0, None)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, inverted.num_minimizers());
        assert_eq!(shard_info.num_bucket_ids, inverted.num_bucket_entries());

        // Load back
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;

        // Verify structure
        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xABCD);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());
        assert_eq!(loaded.num_bucket_entries(), inverted.num_bucket_entries());

        // Verify minimizers match
        assert_eq!(loaded.minimizers(), original_minimizers.as_slice());

        // Verify bucket_ids match
        assert_eq!(loaded.bucket_ids(), original_bucket_ids.as_slice());

        // Verify queries work
        let hits = loaded.get_bucket_hits(&[200, 300, 500]);
        assert_eq!(hits.get(&1), Some(&2)); // 200, 300
        assert_eq!(hits.get(&2), Some(&2)); // 200, 300
        assert_eq!(hits.get(&3), Some(&1)); // 500

        Ok(())
    }

    #[test]
    fn test_inverted_parquet_empty() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("empty.parquet");

        // Create an empty inverted index
        let index = Index::new(64, 50, 0).unwrap();
        let inverted = InvertedIndex::build_from_index(&index);

        // Save as Parquet
        let shard_info = inverted.save_shard_parquet(&path, 0, None)?;
        assert_eq!(shard_info.num_minimizers, 0);
        assert_eq!(shard_info.num_bucket_ids, 0);

        // Empty shard doesn't create file, so load should handle this
        // For now, just verify the shard_info is correct
        Ok(())
    }

    #[test]
    fn test_inverted_parquet_large_values() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("large.parquet");

        // Create minimizers with large values
        let minimizers: Vec<u64> = vec![
            1,
            1000,
            1_000_000,
            1_000_000_000,
            1_000_000_000_000,
            u64::MAX / 2,
            u64::MAX - 100,
            u64::MAX - 1,
        ];

        let mut index = Index::new(64, 50, 0x12345678).unwrap();
        index.buckets.insert(1, minimizers.clone());
        index.bucket_names.insert(1, "test".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Save and load
        inverted.save_shard_parquet(&path, 0, None)?;
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;

        // Verify all large values survived
        for &m in &minimizers {
            let found = loaded.minimizers().binary_search(&m).is_ok();
            assert!(found, "Large minimizer {} lost", m);
        }

        Ok(())
    }

    #[test]
    fn test_inverted_parquet_load_shard_requires_params() -> Result<()> {
        let tmp = TempDir::new()?;
        let parquet_path = tmp.path().join("shard.parquet");

        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as Parquet
        inverted.save_shard_parquet(&parquet_path, 0, None)?;

        // load_shard should error for Parquet files (they need manifest parameters)
        let result = InvertedIndex::load_shard(&parquet_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Parquet shards must be loaded via ShardedInvertedIndex"));

        // Direct loading with params should work
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &parquet_path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());

        Ok(())
    }

    #[test]
    fn test_inverted_parquet_many_buckets() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("many_buckets.parquet");

        // Create index with many buckets sharing some minimizers
        let mut index = Index::new(64, 50, 0xBEEF).unwrap();
        let shared = vec![100, 200, 300, 400, 500];
        for i in 0..50 {
            let mut mins = shared.clone();
            mins.push(1000 + i as u64); // Each bucket has one unique minimizer
            mins.sort();
            index.buckets.insert(i, mins);
            index.bucket_names.insert(i, format!("bucket_{}", i));
        }

        let inverted = InvertedIndex::build_from_index(&index);

        // Save and load
        inverted.save_shard_parquet(&path, 0, None)?;
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;

        // Shared minimizer 200 should map to all 50 buckets
        let hits = loaded.get_bucket_hits(&[200]);
        assert_eq!(hits.len(), 50);
        for i in 0..50 {
            assert_eq!(hits.get(&i), Some(&1));
        }

        Ok(())
    }
}
