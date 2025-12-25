use pyo3::prelude::*;
use roaring::RoaringTreemap; 
use std::collections::HashMap;

// --- CONSTANTS ---
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

/// Extracts Minimizers from a sequence string.
fn extract_minimizers(seq: &str) -> Vec<u64> {
    let bytes = seq.as_bytes();
    let len = bytes.len();
    if len < K {
        return vec![];
    }

    let mut minimizers = Vec::with_capacity(len / WINDOW_SIZE);
    let mut window_hashes: Vec<u64> = Vec::with_capacity(WINDOW_SIZE);
    
    let mut current_val: u64 = 0;
    
    // 1. Initialize first K-mer
    for i in 0..K {
        current_val = (current_val << 1) | base_to_bit(bytes[i]);
    }

    // 2. Process sequence
    for i in 0..=(len - K) {
        let inverted = !current_val; 
        let rc_val = inverted.reverse_bits(); 

        let fwd_hash = current_val ^ SALT;
        let rev_hash = rc_val ^ SALT;
        let canonical = std::cmp::min(fwd_hash, rev_hash);

        window_hashes.push(canonical);

        if window_hashes.len() >= WINDOW_SIZE {
            let start_idx = window_hashes.len() - WINDOW_SIZE;
            let window = &window_hashes[start_idx..];
            if let Some(&min_h) = window.iter().min() {
                minimizers.push(min_h);
            }
        }

        if i < len - K {
            let next_bit = base_to_bit(bytes[i + K]);
            current_val = (current_val << 1) | next_bit;
        }
    }
    minimizers
}

#[pyclass]
struct RYEngine {
    buckets: HashMap<u32, RoaringTreemap>,
}

#[pymethods]
impl RYEngine {
    #[new]
    fn new() -> Self {
        RYEngine {
            buckets: HashMap::new(),
        }
    }

    fn add_genome(&mut self, bucket_id: u32, sequence: String) {
        let mins = extract_minimizers(&sequence);
        let bitmap = self.buckets.entry(bucket_id).or_insert_with(RoaringTreemap::new);
        for m in mins {
            bitmap.insert(m);
        }
    }

    fn query(&self, sequence: String, threshold: f64) -> Vec<(u32, f64)> {
        let mins = extract_minimizers(&sequence);
        if mins.is_empty() {
            return vec![];
        }
        
        let mut read_bitmap = RoaringTreemap::new();
        for m in &mins {
            read_bitmap.insert(*m);
        }
        let total_mins = read_bitmap.len();
        
        if total_mins == 0 {
            return vec![];
        }
        
        let mut results = Vec::new();

        for (id, bucket) in &self.buckets {
            let intersect_count = bucket.intersection_len(&read_bitmap);
            let score = intersect_count as f64 / total_mins as f64;
            
            if score >= threshold {
                results.push((*id, score));
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
    
    fn get_bucket_cardinality(&self, bucket_id: u32) -> u64 {
        match self.buckets.get(&bucket_id) {
            Some(b) => b.len(),
            None => 0,
        }
    }
}

// --- UPDATED MODULE DEFINITION ---
// PyO3 0.21+ requires `Bound<'_, PyModule>`
#[pymodule]
fn ry_partitioner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RYEngine>()?;
    Ok(())
}

// --- UNIT TESTS ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_to_bit() {
        assert_eq!(base_to_bit(b'A'), 1);
        assert_eq!(base_to_bit(b'G'), 1);
        assert_eq!(base_to_bit(b'C'), 0);
        assert_eq!(base_to_bit(b'T'), 0);
        assert_eq!(base_to_bit(b'N'), 0); 
    }

    #[test]
    fn test_extract_short_sequence() {
        let short_seq = "A".repeat(10);
        let mins = extract_minimizers(&short_seq);
        assert!(mins.is_empty());
    }

    #[test]
    fn test_canonicalization() {
        let len = K + WINDOW_SIZE + 10; 
        
        let seq = "A".repeat(len); 
        let mins_fwd = extract_minimizers(&seq);

        let rc_seq = "T".repeat(len);
        let mins_rc = extract_minimizers(&rc_seq);

        assert_eq!(mins_fwd, mins_rc);
        assert!(!mins_fwd.is_empty());
    }

    #[test]
    fn test_engine_exact_match() {
        let mut engine = RYEngine::new();
        let genome = "ACGT".repeat(100); 
        engine.add_genome(1, genome.clone());

        let results = engine.query(genome, 0.9);
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_engine_low_error_tolerance() {
        let mut engine = RYEngine::new();
        let base_seq = "ACGT".repeat(250); 
        engine.add_genome(10, base_seq.clone());

        let mut mutated_seq = String::from(&base_seq);
        unsafe {
            let bytes = mutated_seq.as_bytes_mut();
            for i in 0..8 {
                let idx = i * 100 + 50; 
                if idx < bytes.len() && bytes[idx] == b'A' { 
                    bytes[idx] = b'C'; 
                }
            }
        }

        let results = engine.query(mutated_seq, 0.3);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 10);
        println!("Score with <1% error: {}", results[0].1);
        assert!(results[0].1 > 0.3);
    }
}

