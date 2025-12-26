use pyo3::prelude::*;
use roaring::RoaringTreemap; 
use std::collections::{HashMap, VecDeque};
use std::cmp;

const K: usize = 64; 
const WINDOW_SIZE: usize = 20;
const SALT: u64 = 0x5555555555555555;

#[inline(always)]
fn base_to_bit(byte: u8) -> u64 {
    match byte {
        b'A' | b'a' | b'G' | b'g' => 1, 
        _ => 0,                         
    }
}

/// Helper to manage the Monotonic Queue logic efficiently
struct MinQueue {
    deque: VecDeque<(usize, u64)>,
    last_min: u64,
}

impl MinQueue {
    fn new() -> Self {
        Self {
            deque: VecDeque::with_capacity(WINDOW_SIZE),
            last_min: u64::MAX,
        }
    }

    #[inline(always)]
    fn push(&mut self, idx: usize, val: u64, out: &mut Vec<u64>) {
        // 1. Remove expired (older than window)
        if let Some(&(pos, _)) = self.deque.front() {
            if pos + WINDOW_SIZE <= idx {
                self.deque.pop_front();
            }
        }

        // 2. Maintain monotonicity
        while let Some(&(_, v)) = self.deque.back() {
            if v >= val {
                self.deque.pop_back();
            } else {
                break;
            }
        }

        // 3. Add new
        self.deque.push_back((idx, val));

        // 4. Extract Min (if window full)
        if idx >= WINDOW_SIZE - 1 {
            if let Some(&(_, min_h)) = self.deque.front() {
                if min_h != self.last_min {
                    out.push(min_h);
                    self.last_min = min_h;
                }
            }
        }
    }
}

/// Extracts minimizers for the Forward strand
/// (Used for Indexing - single strand only)
fn extract_fwd_only(seq: &str) -> Vec<u64> {
    let bytes = seq.as_bytes();
    let len = bytes.len();
    if len < K { return vec![]; }

    let mut mins = Vec::with_capacity(len / WINDOW_SIZE);
    let mut queue = MinQueue::new();
    let mut current_val: u64 = 0;

    // Pre-load K-1
    for i in 0..(K - 1) {
        current_val = (current_val << 1) | base_to_bit(bytes[i]);
    }

    // Slide
    for i in 0..=(len - K) {
        let next_bit = base_to_bit(bytes[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        
        // Hash and push
        queue.push(i, current_val ^ SALT, &mut mins);
    }
    mins
}

/// Extracts minimizers for BOTH Forward and RC strands in ONE PASS.
/// Returns (fwd_minimizers, rc_minimizers)
fn extract_dual_strand(seq: &str) -> (Vec<u64>, Vec<u64>) {
    let bytes = seq.as_bytes();
    let len = bytes.len();
    if len < K { return (vec![], vec![]); }

    let capacity = len / WINDOW_SIZE;
    let mut fwd_mins = Vec::with_capacity(capacity);
    let mut rc_mins = Vec::with_capacity(capacity);

    let mut fwd_queue = MinQueue::new();
    let mut rc_queue = MinQueue::new();

    let mut current_val: u64 = 0;

    // Pre-load K-1
    for i in 0..(K - 1) {
        current_val = (current_val << 1) | base_to_bit(bytes[i]);
    }

    // Single Slide Loop
    for i in 0..=(len - K) {
        let next_bit = base_to_bit(bytes[i + K - 1]);
        current_val = (current_val << 1) | next_bit;

        // 1. Forward Hash
        let h_fwd = current_val ^ SALT;
        fwd_queue.push(i, h_fwd, &mut fwd_mins);

        // 2. RC Hash (Bitwise Magic)
        // !current_val = Complement
        // .reverse_bits() = Reverse order
        // This generates the RC stream perfectly for K=64
        let h_rc = (!current_val).reverse_bits() ^ SALT;
        
        // Note: We feed h_rc into its own queue. 
        // Even though we generate these "backwards" relative to the RC physical string,
        // the grouping of K-mers into windows remains contiguous, so the Set of Minimizers 
        // generated is identical to scanning the RC string.
        rc_queue.push(i, h_rc, &mut rc_mins);
    }

    (fwd_mins, rc_mins)
}

#[pyclass]
struct RYEngine {
    buckets: HashMap<u32, RoaringTreemap>,
}

#[pymethods]
impl RYEngine {
    #[new]
    fn new() -> Self {
        RYEngine { buckets: HashMap::new() }
    }

    fn add_genome(&mut self, bucket_id: u32, sequence: &str) {
        // Index only the provided strand
        let mins = extract_fwd_only(sequence);
        let bitmap = self.buckets.entry(bucket_id).or_insert_with(RoaringTreemap::new);
        for m in mins { bitmap.insert(m); }
    }

    fn query(&self, sequence: &str, threshold: f64) -> Vec<(u32, f64)> {
        // Single pass extraction
        let (mins_fwd, mins_rc) = extract_dual_strand(sequence);
        
        if mins_fwd.is_empty() { return vec![]; }

        // Build Bitmaps
        let mut bitmap_fwd = RoaringTreemap::new();
        for m in &mins_fwd { bitmap_fwd.insert(*m); }
        let len_fwd = bitmap_fwd.len();

        let mut bitmap_rc = RoaringTreemap::new();
        for m in &mins_rc { bitmap_rc.insert(*m); }
        let len_rc = bitmap_rc.len();

        let mut results = Vec::new();

        for (id, bucket) in &self.buckets {
            let mut best_score = 0.0;
            
            // Score Forward
            if len_fwd > 0 {
                let intersect = bucket.intersection_len(&bitmap_fwd);
                let score = intersect as f64 / len_fwd as f64;
                if score > best_score { best_score = score; }
            }
            
            // Score RC
            if len_rc > 0 {
                let intersect = bucket.intersection_len(&bitmap_rc);
                let score = intersect as f64 / len_rc as f64;
                if score > best_score { best_score = score; }
            }

            if best_score >= threshold {
                results.push((*id, best_score));
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(cmp::Ordering::Equal));
        results
    }
    
    fn get_bucket_cardinality(&self, bucket_id: u32) -> u64 {
        self.buckets.get(&bucket_id).map(|b| b.len()).unwrap_or(0)
    }
}

#[pymodule]
fn ry_partitioner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RYEngine>()?;
    Ok(())
}

