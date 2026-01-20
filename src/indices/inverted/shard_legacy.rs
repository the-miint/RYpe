//! Legacy RYXS binary format for shard serialization.

use crate::error::{Result, RypeError};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::InvertedIndex;
use crate::constants::{
    MAX_INVERTED_BUCKET_IDS, MAX_INVERTED_MINIMIZERS, READ_BUF_SIZE, SHARD_MAGIC, SHARD_VERSION,
    WRITE_BUF_SIZE,
};
use crate::core::encoding::{decode_varint, encode_varint, VarIntError};
use crate::indices::sharded::ShardInfo;

impl InvertedIndex {
    /// Save a subset of this inverted index as a shard file (RYXS format v1).
    pub fn save_shard(
        &self,
        path: &Path,
        shard_id: u32,
        start_idx: usize,
        end_idx: usize,
        is_last_shard: bool,
    ) -> Result<ShardInfo> {
        let end_idx = end_idx.min(self.minimizers.len());
        if start_idx >= end_idx {
            return Err(RypeError::validation(format!(
                "Invalid shard range: start {} >= end {}",
                start_idx, end_idx
            )));
        }

        let shard_minimizers = &self.minimizers[start_idx..end_idx];
        let min_start = shard_minimizers[0];
        // min_end is 0 for the last shard, or when end_idx == len (no next minimizer).
        // The latter happens when building 1:1 inverted shards from main shards,
        // where each shard contains all its minimizers but isn't marked as last.
        let min_end = if is_last_shard || end_idx >= self.minimizers.len() {
            0
        } else {
            self.minimizers[end_idx]
        };

        let bucket_start = self.offsets[start_idx] as usize;
        let bucket_end = self.offsets[end_idx] as usize;
        let shard_bucket_ids = &self.bucket_ids[bucket_start..bucket_end];

        let base_offset = self.offsets[start_idx];
        let shard_offsets: Vec<u32> = self.offsets[start_idx..=end_idx]
            .iter()
            .map(|&o| o - base_offset)
            .collect();

        let mut writer = BufWriter::new(File::create(path)?);
        let max_bucket_id = shard_bucket_ids.iter().copied().max().unwrap_or(0);

        writer.write_all(SHARD_MAGIC)?;
        writer.write_all(&SHARD_VERSION.to_le_bytes())?;
        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.source_hash.to_le_bytes())?;
        writer.write_all(&shard_id.to_le_bytes())?;
        writer.write_all(&min_start.to_le_bytes())?;
        writer.write_all(&min_end.to_le_bytes())?;
        writer.write_all(&max_bucket_id.to_le_bytes())?;
        writer.write_all(&(shard_minimizers.len() as u64).to_le_bytes())?;
        writer.write_all(&(shard_bucket_ids.len() as u64).to_le_bytes())?;

        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;

        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);

        let flush_buf = |buf: &mut Vec<u8>,
                         encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>|
         -> Result<()> {
            if !buf.is_empty() {
                encoder.write_all(buf)?;
                buf.clear();
            }
            Ok(())
        };

        for &offset in &shard_offsets {
            write_buf.extend_from_slice(&offset.to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        let mut varint_buf = [0u8; 10];
        if !shard_minimizers.is_empty() {
            write_buf.extend_from_slice(&shard_minimizers[0].to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }

            let mut prev = shard_minimizers[0];
            for &min in &shard_minimizers[1..] {
                let delta = min - prev;
                let len = encode_varint(delta, &mut varint_buf);
                write_buf.extend_from_slice(&varint_buf[..len]);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
                prev = min;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        if max_bucket_id <= u8::MAX as u32 {
            for &bid in shard_bucket_ids {
                write_buf.push(bid as u8);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        } else if max_bucket_id <= u16::MAX as u32 {
            for &bid in shard_bucket_ids {
                write_buf.extend_from_slice(&(bid as u16).to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        } else {
            for &bid in shard_bucket_ids {
                write_buf.extend_from_slice(&bid.to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        encoder.finish()?;

        Ok(ShardInfo {
            shard_id,
            min_start,
            min_end,
            is_last_shard,
            num_minimizers: shard_minimizers.len(),
            num_bucket_ids: shard_bucket_ids.len(),
        })
    }

    /// Load a shard file into an InvertedIndex.
    ///
    /// Supports both legacy RYXS format and Parquet format (auto-detected by extension).
    ///
    /// # Parquet Shards
    /// Parquet shards should be loaded via `ShardedInvertedIndex::load_shard()` which
    /// provides the required parameters (k, w, salt, source_hash) from the manifest.
    /// Loading a Parquet shard directly requires these parameters and will fail.
    pub fn load_shard(path: &Path) -> Result<Self> {
        // Detect format based on extension
        if path.extension().map(|e| e == "parquet").unwrap_or(false) {
            return Err(RypeError::validation(format!(
                "Parquet shards must be loaded via ShardedInvertedIndex::load_shard() \
                 which provides parameters from the manifest. \
                 Use load_shard_parquet_with_params() directly if you have the parameters. \
                 Path: {}",
                path.display()
            )));
        }
        Self::load_shard_legacy(path)
    }

    /// Load a legacy RYXS format shard file.
    pub(super) fn load_shard_legacy(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != SHARD_MAGIC {
            return Err(RypeError::format(
                path,
                "Invalid shard format (expected RYXS)",
            ));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != SHARD_VERSION {
            return Err(RypeError::format(
                path,
                format!(
                    "Unsupported shard version: {} (expected {})",
                    version, SHARD_VERSION
                ),
            ));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(RypeError::validation(format!(
                "Invalid K value in shard: {} (must be 16, 32, or 64)",
                k
            )));
        }

        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let source_hash = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?; // shard_id
        reader.read_exact(&mut buf8)?; // min_start
        reader.read_exact(&mut buf8)?; // min_end

        reader.read_exact(&mut buf4)?;
        let max_bucket_id = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let num_minimizers = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let num_bucket_ids = u64::from_le_bytes(buf8) as usize;

        if num_minimizers > MAX_INVERTED_MINIMIZERS {
            return Err(RypeError::overflow(
                "shard minimizer count",
                MAX_INVERTED_MINIMIZERS,
                num_minimizers,
            ));
        }
        if num_bucket_ids > MAX_INVERTED_BUCKET_IDS {
            return Err(RypeError::overflow(
                "shard bucket IDs",
                MAX_INVERTED_BUCKET_IDS,
                num_bucket_ids,
            ));
        }

        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let mut read_buf = vec![0u8; READ_BUF_SIZE];
        let mut buf_pos = 0;
        let mut buf_len = 0;

        macro_rules! ensure_bytes {
            ($n:expr) => {{
                while buf_len - buf_pos < $n {
                    if buf_pos > 0 {
                        read_buf.copy_within(buf_pos..buf_len, 0);
                        buf_len -= buf_pos;
                        buf_pos = 0;
                    }
                    let space = read_buf.len() - buf_len;
                    if space == 0 {
                        read_buf.resize(read_buf.len() * 2, 0);
                    }
                    let n = decoder.read(&mut read_buf[buf_len..])?;
                    if n == 0 {
                        return Err(RypeError::encoding("Unexpected end of shard data"));
                    }
                    buf_len += n;
                }
            }};
        }

        let offsets_count = num_minimizers + 1;
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
            ensure_bytes!(4);
            let offset = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
            offsets.push(offset);
            buf_pos += 4;
        }

        if !offsets.is_empty() && offsets[0] != 0 {
            return Err(RypeError::format(
                path,
                "Invalid shard: first offset must be 0",
            ));
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(RypeError::format(
                path,
                "Invalid shard: offsets not monotonically increasing",
            ));
        }
        if !offsets.is_empty() && *offsets.last().unwrap() as usize != num_bucket_ids {
            return Err(RypeError::format(
                path,
                "Invalid shard: final offset doesn't match bucket_ids count",
            ));
        }

        let mut minimizers = Vec::with_capacity(num_minimizers);
        if num_minimizers > 0 {
            ensure_bytes!(8);
            let first = u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
            minimizers.push(first);
            buf_pos += 8;

            let mut prev = first;
            for i in 1..num_minimizers {
                // Ensure enough bytes for a complete varint.
                // Loop until we successfully decode or hit a real error.
                let (delta, consumed) = loop {
                    // Ensure we have at least 1 byte to attempt decoding.
                    // The loop handles Truncated errors by reading more data.
                    ensure_bytes!(1);

                    match decode_varint(&read_buf[buf_pos..buf_len]) {
                        Ok((delta, consumed)) => break (delta, consumed),
                        Err(VarIntError::Truncated(_)) => {
                            // Need more data - shift remaining bytes and read more
                            read_buf.copy_within(buf_pos..buf_len, 0);
                            buf_len -= buf_pos;
                            buf_pos = 0;
                            let n = decoder.read(&mut read_buf[buf_len..])?;
                            if n == 0 {
                                return Err(RypeError::encoding(format!(
                                    "Truncated varint at minimizer {} (EOF with continuation bit set, buf_len={})",
                                    i, buf_len
                                )));
                            }
                            buf_len += n;
                        }
                        Err(VarIntError::Overflow(bytes)) => {
                            return Err(RypeError::encoding(format!(
                                "Malformed varint at minimizer {}: exceeded 10 bytes (consumed {})",
                                i, bytes
                            )));
                        }
                    }
                };
                buf_pos += consumed;

                let val = prev.checked_add(delta).ok_or_else(|| {
                    RypeError::overflow(
                        format!("minimizer at index {} (prev={}, delta={})", i, prev, delta),
                        u64::MAX as usize,
                        (prev as u128 + delta as u128) as usize,
                    )
                })?;

                if val <= prev && i > 0 {
                    return Err(RypeError::format(
                        path,
                        format!(
                            "Minimizers not strictly increasing at index {} (prev={}, val={})",
                            i, prev, val
                        ),
                    ));
                }

                minimizers.push(val);
                prev = val;
            }
        }

        let mut bucket_ids = Vec::with_capacity(num_bucket_ids);
        if max_bucket_id <= u8::MAX as u32 {
            for _ in 0..num_bucket_ids {
                ensure_bytes!(1);
                bucket_ids.push(read_buf[buf_pos] as u32);
                buf_pos += 1;
            }
        } else if max_bucket_id <= u16::MAX as u32 {
            for _ in 0..num_bucket_ids {
                ensure_bytes!(2);
                let val = u16::from_le_bytes(read_buf[buf_pos..buf_pos + 2].try_into().unwrap());
                bucket_ids.push(val as u32);
                buf_pos += 2;
            }
        } else {
            for _ in 0..num_bucket_ids {
                ensure_bytes!(4);
                let val = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
                bucket_ids.push(val);
                buf_pos += 4;
            }
        }

        Ok(InvertedIndex {
            k,
            w,
            salt,
            source_hash,
            minimizers,
            offsets,
            bucket_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::main::Index;
    use anyhow::Result;
    use tempfile::NamedTempFile;

    #[test]
    fn test_shard_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0xABCD).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());

        let inverted = InvertedIndex::build_from_index(&index);

        let shard_info = inverted.save_shard(&path, 0, 0, inverted.num_minimizers(), true)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, inverted.num_minimizers());
        assert_eq!(shard_info.num_bucket_ids, inverted.num_bucket_entries());

        let loaded = InvertedIndex::load_shard(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xABCD);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());
        assert_eq!(loaded.num_bucket_entries(), inverted.num_bucket_entries());

        let hits = loaded.get_bucket_hits(&[200, 300]);
        assert_eq!(hits.get(&1), Some(&2));
        assert_eq!(hits.get(&2), Some(&2));

        Ok(())
    }

    #[test]
    fn test_shard_partial_range() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400, 500]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
        assert_eq!(inverted.num_minimizers(), 5);

        let shard_info = inverted.save_shard(&path, 0, 0, 3, false)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, 3);
        assert!(!shard_info.is_last_shard);
        assert_eq!(shard_info.min_start, 100);
        assert_eq!(shard_info.min_end, 400);

        let loaded = InvertedIndex::load_shard(&path)?;
        assert_eq!(loaded.num_minimizers(), 3);

        let hits = loaded.get_bucket_hits(&[100, 200, 300, 400, 500]);
        assert_eq!(hits.get(&1), Some(&3));

        Ok(())
    }

    /// Regression test for panic when saving a full-range shard with is_last_shard=false.
    /// This happens when building 1:1 inverted shards from a sharded main index,
    /// where each inverted shard contains all minimizers (end_idx = len) but only
    /// the final shard has is_last_shard=true.
    #[test]
    fn test_shard_full_range_not_last() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x9999).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400, 500]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
        assert_eq!(inverted.num_minimizers(), 5);

        // This is the scenario that caused the panic: saving full range with is_last_shard=false
        // This happens when building 1:1 inverted shards from sharded main index
        let shard_info = inverted.save_shard(&path, 0, 0, inverted.num_minimizers(), false)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, 5);
        assert!(!shard_info.is_last_shard);
        assert_eq!(shard_info.min_start, 100);
        // When end_idx == len, min_end should be 0 (no next minimizer)
        assert_eq!(shard_info.min_end, 0);

        // Verify the shard can be loaded and used
        let loaded = InvertedIndex::load_shard(&path)?;
        assert_eq!(loaded.num_minimizers(), 5);

        let hits = loaded.get_bucket_hits(&[100, 200, 300, 400, 500]);
        assert_eq!(hits.get(&1), Some(&5));

        Ok(())
    }

    #[test]
    fn test_shard_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTS").unwrap();

        let result = InvertedIndex::load_shard(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid shard format"));
    }

    /// Regression test for varint boundary bug in load_shard().
    ///
    /// The bug: load_shard() used `ensure_bytes!(1)` before decoding varints,
    /// but LEB128 varints can be up to 10 bytes. When a large delta value
    /// happened to span buffer boundaries during streaming decompression,
    /// only part of the varint was decoded, corrupting subsequent minimizers.
    ///
    /// This test creates minimizers with large gaps (requiring multi-byte varints)
    /// and verifies save/load round-trip preserves all values correctly.
    #[test]
    fn test_shard_varint_boundary_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        // Create minimizers with large gaps requiring multi-byte varints.
        // LEB128 encoding:
        // - 1 byte: values 0-127
        // - 2 bytes: values 128-16383
        // - 3 bytes: values 16384-2097151
        // - 10 bytes: max u64
        //
        // We intentionally create values with gaps of varying sizes to trigger
        // varints of different lengths, increasing the chance of hitting buffer boundaries.
        let mut minimizers: Vec<u64> = Vec::new();

        // Start with small values (small deltas, 1-byte varints)
        for i in 0..50 {
            minimizers.push(i * 10);
        }

        // Add medium gaps (2-3 byte varints)
        let mut val = 500;
        for _ in 0..100 {
            minimizers.push(val);
            val += 50000; // ~2-3 byte varint deltas
        }

        // Add large gaps (4+ byte varints)
        for _ in 0..50 {
            minimizers.push(val);
            val += 1_000_000_000; // ~5 byte varint deltas
        }

        // Add very large values near u64::MAX (requiring large absolute values and deltas)
        let high_base = 10_000_000_000_000_000_000u64;
        for i in 0..20 {
            minimizers.push(high_base + i * 100_000_000_000);
        }

        minimizers.sort();
        minimizers.dedup();

        // Create index with these minimizers
        let mut index = Index::new(64, 50, 0xDEADBEEF).unwrap();
        index.buckets.insert(1, minimizers.clone());
        index.bucket_names.insert(1, "test".into());

        let inverted = InvertedIndex::build_from_index(&index);
        let original_count = inverted.num_minimizers();

        // Save as shard
        let shard_info = inverted.save_shard(&path, 0, 0, original_count, true)?;
        assert_eq!(shard_info.num_minimizers, original_count);

        // Load shard back
        let loaded = InvertedIndex::load_shard(&path)?;

        // Verify all minimizers were preserved
        assert_eq!(
            loaded.num_minimizers(),
            original_count,
            "Minimizer count changed after save/load: {} -> {}",
            original_count,
            loaded.num_minimizers()
        );

        // Verify specific high-value minimizers weren't truncated
        for &m in &minimizers {
            let found = loaded.minimizers().binary_search(&m).is_ok();
            assert!(
                found,
                "Minimizer {} was lost after save/load round-trip (varint boundary bug?)",
                m
            );
        }

        // Verify the actual minimizer values match exactly
        assert_eq!(
            loaded.minimizers(),
            inverted.minimizers(),
            "Minimizer arrays differ after save/load"
        );

        // Verify queries still work correctly
        let hits = loaded.get_bucket_hits(&minimizers[..10]);
        assert_eq!(hits.get(&1), Some(&10));

        Ok(())
    }

    /// Regression test: ensure very large minimizer values near u64::MAX survive round-trip.
    /// This tests the upper range where varint deltas could overflow or be misinterpreted.
    #[test]
    fn test_shard_large_minimizers_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        // Create minimizers near the top of the u64 range
        let minimizers: Vec<u64> = vec![
            12297829382473034403,
            12297829382473034405,
            12297829382473034408,
            12297829382473034409,
            12297829382473034410,
            14168481312020516, // The actual problematic minimizer from the bug report
            u64::MAX - 1000,
            u64::MAX - 100,
            u64::MAX - 10,
            u64::MAX - 1,
        ];

        let mut sorted_mins = minimizers.clone();
        sorted_mins.sort();

        let mut index = Index::new(64, 200, 0x5555555555555555).unwrap();
        index.buckets.insert(30, sorted_mins.clone());
        index.bucket_names.insert(30, "bucket-125".into());

        let inverted = InvertedIndex::build_from_index(&index);
        let shard_info = inverted.save_shard(&path, 0, 0, inverted.num_minimizers(), true)?;

        let loaded = InvertedIndex::load_shard(&path)?;

        assert_eq!(loaded.num_minimizers(), shard_info.num_minimizers);

        // Check the specific minimizer that was missing in the bug
        let test_min = 14168481312020516u64;
        let found = loaded.minimizers().binary_search(&test_min).is_ok();
        assert!(
            found,
            "Minimizer {} (from bug report) not found after save/load",
            test_min
        );

        // Verify all minimizers survived
        for &m in &sorted_mins {
            let found = loaded.minimizers().binary_search(&m).is_ok();
            assert!(found, "Minimizer {} lost after round-trip", m);
        }

        Ok(())
    }
}
