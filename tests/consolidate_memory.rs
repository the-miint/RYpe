//! Memory-bounded-consolidation integration test.
//!
//! Verifies that [`rype::parquet_index::consolidate_shards_streaming`] honors
//! its `O(k · PARQUET_BATCH_SIZE · 12 B + max_shard_bytes)` memory bound:
//! peak allocation during consolidation must not scale with the total unique
//! minimizer count.
//!
//! Implementation: a crate-local `#[global_allocator]` wraps the system
//! allocator with atomic counters. The test snapshots `current()` before the
//! consolidation call and polls a separate `peak` tracker during it.
//!
//! The test is gated `#[ignore]` because it constructs ~30 MB of fixture data
//! on disk and is only meaningful on platforms where Rust's default allocator
//! is in use. Run with `cargo test --release -- --ignored
//! test_consolidate_streaming_peak_memory_is_bounded`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rype::parquet_index::{
    consolidate_shards_streaming, InvertedShardInfo, ParquetWriteOptions, MIN_SHARD_BYTES,
};

struct TrackingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            if new_size > layout.size() {
                let delta = new_size - layout.size();
                let now = CURRENT.fetch_add(delta, Ordering::SeqCst) + delta;
                PEAK.fetch_max(now, Ordering::SeqCst);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn snapshot_peak() -> usize {
    PEAK.load(Ordering::SeqCst)
}

fn reset_peak_to_current() {
    let c = CURRENT.load(Ordering::SeqCst);
    PEAK.store(c, Ordering::SeqCst);
}

fn current() -> usize {
    CURRENT.load(Ordering::SeqCst)
}

fn build_intermediate_shards(
    output_dir: &std::path::Path,
    shard_count: usize,
    unique_per_shard: usize,
    duplication: usize,
) -> Vec<InvertedShardInfo> {
    use rype::parquet_index::ShardAccumulator;
    use std::fs;

    fs::create_dir_all(output_dir.join("inverted")).unwrap();

    let mut infos = Vec::with_capacity(shard_count);
    for s in 0..shard_count {
        let base = (s as u64) * unique_per_shard as u64;
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(unique_per_shard * duplication);
        for i in 0..unique_per_shard {
            let m = base + i as u64;
            for _ in 0..duplication {
                pairs.push((m, 1));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();

        // Write shard.{s}.parquet directly via start_shard_id.
        let mut acc =
            ShardAccumulator::with_start_shard_id(output_dir, MIN_SHARD_BYTES, s as u32, None);
        acc.add_entries(&pairs);
        let info = acc.flush_shard().unwrap().unwrap();
        assert_eq!(info.shard_id, s as u32);
        infos.push(info);
    }
    infos
}

#[test]
#[ignore = "memory-bounded test; run with --ignored"]
fn test_consolidate_streaming_peak_memory_is_bounded() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output_dir = tmp.path().join("stream.ryxdi");

    // 4 shards × 1M unique entries each → 4M total unique (distinct per shard).
    // Legacy `merged: Vec<u64>` would grow to 32 MB + `merge_sorted_into`
    // allocates another 32–64 MB → peak ~100 MB. Streaming holds only
    // `k × batch × 12 B ≈ 4.8 MB` + `max_shard_bytes`.
    let infos = build_intermediate_shards(&output_dir, 4, 1_000_000, 1);

    let opts = ParquetWriteOptions::default();
    let max_shard_bytes = 16 * 1024 * 1024; // 16 MiB

    reset_peak_to_current();
    let baseline = current();

    let (shards, unique) =
        consolidate_shards_streaming(&output_dir, &infos, 1, max_shard_bytes, &opts).unwrap();

    let peak_delta = snapshot_peak().saturating_sub(baseline);

    assert_eq!(unique, 4_000_000, "total unique must equal input unique");
    assert!(!shards.is_empty());
    // Streaming bound: k × batch × 12 + max_shard_bytes ≈ 4.8 MB + 16 MB.
    // Allow 4× headroom for Arrow / Parquet writer buffers + allocator slack.
    let bound = 6 * max_shard_bytes; // 96 MiB
    assert!(
        peak_delta < bound,
        "streaming consolidation peak {} exceeded bound {} (baseline {})",
        peak_delta,
        bound,
        baseline
    );
    eprintln!(
        "streaming consolidation: peak_delta = {:.1} MiB (bound {} MiB, baseline {:.1} MiB)",
        peak_delta as f64 / (1024.0 * 1024.0),
        bound / (1024 * 1024),
        baseline as f64 / (1024.0 * 1024.0)
    );
}
