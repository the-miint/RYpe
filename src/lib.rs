use pyo3::prelude::*;
use roaring::RoaringTreemap;
use std::collections::{HashMap, VecDeque};
use std::cmp;

// --- HARDCODED CONSTANTS ---
// Kept K=64 as requested.
const K: usize = 64; 
const WINDOW_SIZE: usize = 20;
const SALT: u64 = 0x5555555555555555;

#[inline(always)]
fn base_to_bit(byte: u8) -> u64 {
    match byte {
        b'A' | b'a' | b'G' | b'g' => 1, // Purine (R)
        _ => 0,                         // Pyrimidine (Y)
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
        // 1. Remove expired
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

/// Extracts minimizers for the Forward strand only
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
        queue.push(i, current_val ^ SALT, &mut mins);
    }
    mins
}

/// Extracts minimizers for BOTH Forward and RC strands in ONE PASS.
/// Optimized for K=64 using bitwise negation and reversal.
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

    // Slide
    for i in 0..=(len - K) {
        let next_bit = base_to_bit(bytes[i + K - 1]);
        
        // Update Forward
        current_val = (current_val << 1) | next_bit;

        // 1. Process Forward
        let h_fwd = current_val ^ SALT;
        fwd_queue.push(i, h_fwd, &mut fwd_mins);

        // 2. Process RC (Bitwise Magic for K=64)
        // RC(val) = reverse_bits(!val)
        let h_rc = (!current_val).reverse_bits() ^ SALT;
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
        let mins = extract_fwd_only(sequence);
        let bitmap = self.buckets.entry(bucket_id).or_insert_with(RoaringTreemap::new);
        for m in mins { 
            bitmap.insert(m); 
        }
    }

    /// Query a single long read (HiFi style)
    fn query(&self, sequence: &str, threshold: f64) -> Vec<(u32, f64)> {
        let (mins_fwd, mins_rc) = extract_dual_strand(sequence);
        if mins_fwd.is_empty() { return vec![]; }

        // Build bitmaps
        let mut b_fwd = RoaringTreemap::new();
        for m in &mins_fwd { b_fwd.insert(*m); }
        let l_fwd = b_fwd.len();

        let mut b_rc = RoaringTreemap::new();
        for m in &mins_rc { b_rc.insert(*m); }
        let l_rc = b_rc.len();

        let mut results = Vec::new();

        for (id, bucket) in &self.buckets {
            let mut best = 0.0;
            if l_fwd > 0 {
                let s = bucket.intersection_len(&b_fwd) as f64 / l_fwd as f64;
                if s > best { best = s; }
            }
            if l_rc > 0 {
                let s = bucket.intersection_len(&b_rc) as f64 / l_rc as f64;
                if s > best { best = s; }
            }
            if best >= threshold {
                results.push((*id, best));
            }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(cmp::Ordering::Equal));
        results
    }

    /// Query a Paired End Short Read (R1 + R2)
    /// Combines signals from both reads to overcome K=64 brittleness.
    /// Logic:
    ///   Scenario A: Fragment is FWD relative to Ref
    ///      -> R1 is Fwd, R2 is RC
    ///   Scenario B: Fragment is RC relative to Ref
    ///      -> R1 is RC, R2 is Fwd
    fn query_paired(&self, r1: &str, r2: &str, threshold: f64) -> Vec<(u32, f64)> {
        let (r1_f, r1_rc) = extract_dual_strand(r1);
        let (r2_f, r2_rc) = extract_dual_strand(r2);

        // If both reads fail length check, return empty
        if r1_f.is_empty() && r2_f.is_empty() { return vec![]; }

        // --- Build "Hypothesis A" Bitmap (Frag matches Ref FWD) ---
        // Combine R1_Fwd + R2_RC
        let mut b_hyp_a = RoaringTreemap::new();
        for m in &r1_f { b_hyp_a.insert(*m); }
        for m in &r2_rc { b_hyp_a.insert(*m); }
        let len_a = b_hyp_a.len();

        // --- Build "Hypothesis B" Bitmap (Frag matches Ref RC) ---
        // Combine R1_RC + R2_Fwd
        let mut b_hyp_b = RoaringTreemap::new();
        for m in &r1_rc { b_hyp_b.insert(*m); }
        for m in &r2_f { b_hyp_b.insert(*m); }
        let len_b = b_hyp_b.len();

        let mut results = Vec::new();

        for (id, bucket) in &self.buckets {
            let mut best = 0.0;

            // Check Hypothesis A
            if len_a > 0 {
                let score = bucket.intersection_len(&b_hyp_a) as f64 / len_a as f64;
                if score > best { best = score; }
            }

            // Check Hypothesis B
            if len_b > 0 {
                let score = bucket.intersection_len(&b_hyp_b) as f64 / len_b as f64;
                if score > best { best = score; }
            }

            if best >= threshold {
                results.push((*id, best));
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

