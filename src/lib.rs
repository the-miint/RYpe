use pyo3::prelude::*;
use roaring::RoaringTreemap;
use std::collections::{HashMap, VecDeque};
use rayon::prelude::*;
use needletail::parse_fastx_file;
use std::cmp;

const K: usize = 64; 
const WINDOW_SIZE: usize = 200;
const SALT: u64 = 0x5555555555555555;
const BATCH_SIZE: usize = 4096;

#[inline(always)]
fn base_to_bit(byte: u8) -> u64 {
    match byte {
        b'A' | b'a' | b'G' | b'g' => 1,
        _ => 0,
    }
}

// --- MINIMIZER LOGIC ---
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
        if let Some(&(pos, _)) = self.deque.front() {
            if pos + WINDOW_SIZE <= idx { self.deque.pop_front(); }
        }
        while let Some(&(_, v)) = self.deque.back() {
            if v >= val { self.deque.pop_back(); } else { break; }
        }
        self.deque.push_back((idx, val));
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

fn extract_fwd_only(seq: &str) -> Vec<u64> {
    let bytes = seq.as_bytes();
    let len = bytes.len();
    if len < K { return vec![]; }
    let mut mins = Vec::with_capacity(len / WINDOW_SIZE);
    let mut queue = MinQueue::new();
    let mut current_val: u64 = 0;
    for i in 0..(K - 1) { current_val = (current_val << 1) | base_to_bit(bytes[i]); }
    for i in 0..=(len - K) {
        let next_bit = base_to_bit(bytes[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        queue.push(i, current_val ^ SALT, &mut mins);
    }
    mins
}

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
    for i in 0..(K - 1) { current_val = (current_val << 1) | base_to_bit(bytes[i]); }
    for i in 0..=(len - K) {
        let next_bit = base_to_bit(bytes[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        let h_fwd = current_val ^ SALT;
        fwd_queue.push(i, h_fwd, &mut fwd_mins);
        let h_rc = (!current_val).reverse_bits() ^ SALT;
        rc_queue.push(i, h_rc, &mut rc_mins);
    }
    (fwd_mins, rc_mins)
}

// --- ENGINE CLASS ---

#[pyclass]
struct RYEngine {
    // We wrap buckets in a simpler thread-safe structure if needed, 
    // but here we only read buckets during query, so standard HashMap is fine.
    // However, to pass &self across thread boundaries in allow_threads, 
    // Rust needs to know it's safe. PyClass structs are generally Sync.
    buckets: HashMap<u32, RoaringTreemap>,
}

// We need to implement Sync to allow &self to be shared across threads
// RoaringTreemap is Send+Sync, HashMap is Send+Sync.
unsafe impl Sync for RYEngine {}

#[pymethods]
impl RYEngine {
    #[new]
    fn new() -> Self {
        RYEngine { buckets: HashMap::new() }
    }
    // ... inside #[pymethods] impl RYEngine ...

    /// Save the entire index to a binary file
    fn save(&self, path: &str) -> PyResult<()> {
        use std::io::Write;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        // 1. Write Number of Buckets (u32)
        let num_buckets = self.buckets.len() as u32;
        writer.write_all(&num_buckets.to_le_bytes()).unwrap();

        // 2. Write Each Bucket
        for (id, map) in &self.buckets {
            // Write Bucket ID
            writer.write_all(&id.to_le_bytes()).unwrap();

            // Serialize Roaring Bitmap to a buffer first to know its size
            let mut buf = Vec::new();
            map.serialize_into(&mut buf).unwrap();

            // Write Size (u64) then Data
            let size = buf.len() as u64;
            writer.write_all(&size.to_le_bytes()).unwrap();
            writer.write_all(&buf).unwrap();
        }
        Ok(())
    }

    /// Load the index from a binary file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        use std::io::Read;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 4];

        // 1. Read Num Buckets
        reader.read_exact(&mut buffer).unwrap();
        let num_buckets = u32::from_le_bytes(buffer);

        let mut buckets = HashMap::new();

        for _ in 0..num_buckets {
            // Read Bucket ID
            reader.read_exact(&mut buffer).unwrap();
            let id = u32::from_le_bytes(buffer);

            // Read Size
            let mut size_buf = [0u8; 8];
            reader.read_exact(&mut size_buf).unwrap();
            let _size = u64::from_le_bytes(size_buf);

            // Read Roaring Data
            // Roaring's deserialize consumes bytes from the reader.
            // In a production format, we'd limit the reader, but here we trust the stream.
            let map = RoaringTreemap::deserialize_from(&mut reader)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Roaring Error: {:?}", e)))?;

            buckets.insert(id, map);
        }

        Ok(RYEngine { buckets })
    }

    fn add_genome(&mut self, bucket_id: u32, sequence: &str) {
        let mins = extract_fwd_only(sequence);
        let bitmap = self.buckets.entry(bucket_id).or_insert_with(RoaringTreemap::new);
        for m in mins { bitmap.insert(m); }
    }
    
    fn get_bucket_cardinality(&self, bucket_id: u32) -> u64 {
        self.buckets.get(&bucket_id).map(|b| b.len()).unwrap_or(0)
    }

    // --- QUERY METHODS ---

    fn query(&self, sequence: &str, threshold: f64) -> Vec<(u32, f64)> {
        let (mins_fwd, mins_rc) = extract_dual_strand(sequence);
        if mins_fwd.is_empty() { return vec![]; }

        let mut b_fwd = RoaringTreemap::new();
        for m in &mins_fwd { b_fwd.insert(*m); }
        let l_fwd = b_fwd.len() as f64;

        let mut b_rc = RoaringTreemap::new();
        for m in &mins_rc { b_rc.insert(*m); }
        let l_rc = b_rc.len() as f64;

        let mut results = Vec::new();

        for (id, bucket) in &self.buckets {
            let mut best: f64 = 0.0;
            if l_fwd > 0.0 { best = best.max(bucket.intersection_len(&b_fwd) as f64 / l_fwd); }
            if l_rc > 0.0 { best = best.max(bucket.intersection_len(&b_rc) as f64 / l_rc); }
            if best >= threshold { results.push((*id, best)); }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(cmp::Ordering::Equal));
        results
    }

    fn query_paired(&self, r1: &str, r2: &str, threshold: f64) -> Vec<(u32, f64)> {
        let (r1_f, r1_rc) = extract_dual_strand(r1);
        let (r2_f, r2_rc) = extract_dual_strand(r2);
        if r1_f.is_empty() && r2_f.is_empty() { return vec![]; }

        let mut b_hyp_a = RoaringTreemap::new();
        for m in &r1_f { b_hyp_a.insert(*m); }
        for m in &r2_rc { b_hyp_a.insert(*m); }
        let len_a = b_hyp_a.len() as f64;

        let mut b_hyp_b = RoaringTreemap::new();
        for m in &r1_rc { b_hyp_b.insert(*m); }
        for m in &r2_f { b_hyp_b.insert(*m); }
        let len_b = b_hyp_b.len() as f64;

        let mut results = Vec::new();
        for (id, bucket) in &self.buckets {
            let mut best: f64 = 0.0;
            if len_a > 0.0 { best = best.max(bucket.intersection_len(&b_hyp_a) as f64 / len_a); }
            if len_b > 0.0 { best = best.max(bucket.intersection_len(&b_hyp_b) as f64 / len_b); }
            if best >= threshold { results.push((*id, best)); }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(cmp::Ordering::Equal));
        results
    }

    // --- PARALLEL BENCHMARKS (GIL RELEASED) ---

    fn benchmark_file(&self, py: Python, path: String) -> PyResult<HashMap<u32, HashMap<String, (u64, u64)>>> {
        // We release the GIL entirely for the duration of the file processing.
        // This allows Rayon to saturate all cores without fighting the Python interpreter.
        
        let path_clone = path.clone();
        
        let stats = py.allow_threads(move || {
            let mut reader = parse_fastx_file(&path_clone).expect("Invalid FASTQ path");
            let mut global_stats: HashMap<u32, HashMap<String, (u64, u64)>> = HashMap::new();
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            // Local helper to merge stats
            let merge_stats = |global: &mut HashMap<u32, HashMap<String, (u64, u64)>>, local: HashMap<u32, HashMap<String, (u64, u64)>>| {
                for (bid, t_map) in local {
                    let g_map = global.entry(bid).or_insert_with(HashMap::new);
                    for (thresh, (h, m)) in t_map {
                        let entry = g_map.entry(thresh).or_insert((0, 0));
                        entry.0 += h;
                        entry.1 += m;
                    }
                }
            };

            while let Some(record) = reader.next() {
                let seqrec = record.expect("Invalid FASTQ record");
                if let Ok(seq_str) = std::str::from_utf8(&seqrec.seq()) {
                    batch.push(seq_str.to_string());
                }

                if batch.len() >= BATCH_SIZE {
                    let batch_results: Vec<HashMap<u32, HashMap<String, (u64, u64)>>> = batch.par_iter()
                        .map(|seq| self.compute_read_stats_single(seq))
                        .collect();

                    for res in batch_results { merge_stats(&mut global_stats, res); }
                    batch.clear();
                }
            }

            // Flush remaining
            if !batch.is_empty() {
                let batch_results: Vec<HashMap<u32, HashMap<String, (u64, u64)>>> = batch.par_iter()
                    .map(|seq| self.compute_read_stats_single(seq))
                    .collect();
                for res in batch_results { merge_stats(&mut global_stats, res); }
            }

            global_stats
        });

        Ok(stats)
    }

    fn benchmark_paired(&self, py: Python, r1_path: String, r2_path: String) -> PyResult<HashMap<u32, HashMap<String, (u64, u64)>>> {
        let r1_c = r1_path.clone();
        let r2_c = r2_path.clone();

        let stats = py.allow_threads(move || {
            let mut r1_reader = parse_fastx_file(&r1_c).expect("Invalid R1 path");
            let mut r2_reader = parse_fastx_file(&r2_c).expect("Invalid R2 path");

            let mut global_stats: HashMap<u32, HashMap<String, (u64, u64)>> = HashMap::new();
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            while let (Some(rec1), Some(rec2)) = (r1_reader.next(), r2_reader.next()) {
                let r1 = rec1.expect("Error reading R1");
                let r2 = rec2.expect("Error reading R2");

                let s1 = std::str::from_utf8(&r1.seq()).unwrap_or("").to_string();
                let s2 = std::str::from_utf8(&r2.seq()).unwrap_or("").to_string();
                
                batch.push((s1, s2));

                if batch.len() >= BATCH_SIZE {
                    let batch_results: Vec<HashMap<u32, HashMap<String, (u64, u64)>>> = batch.par_iter()
                        .map(|(s1, s2)| self.compute_read_stats_paired(s1, s2))
                        .collect();
                    
                    // Merge
                    for res in batch_results {
                        for (bid, t_map) in res {
                            let g_map = global_stats.entry(bid).or_insert_with(HashMap::new);
                            for (thresh, (h, m)) in t_map {
                                let entry = g_map.entry(thresh).or_insert((0, 0));
                                entry.0 += h;
                                entry.1 += m;
                            }
                        }
                    }
                    batch.clear();
                }
            }

            // Flush
            if !batch.is_empty() {
                let batch_results: Vec<HashMap<u32, HashMap<String, (u64, u64)>>> = batch.par_iter()
                    .map(|(s1, s2)| self.compute_read_stats_paired(s1, s2))
                    .collect();
                for res in batch_results {
                    for (bid, t_map) in res {
                        let g_map = global_stats.entry(bid).or_insert_with(HashMap::new);
                        for (thresh, (h, m)) in t_map {
                            let entry = g_map.entry(thresh).or_insert((0, 0));
                            entry.0 += h;
                            entry.1 += m;
                        }
                    }
                }
            }
            global_stats
        });

        Ok(stats)
    }
}

// --- INTERNAL HELPERS ---
impl RYEngine {
    fn compute_read_stats_single(&self, seq: &str) -> HashMap<u32, HashMap<String, (u64, u64)>> {
        let (mins_fwd, mins_rc) = extract_dual_strand(seq);
        if mins_fwd.is_empty() { return HashMap::new(); }

        let mut b_fwd = RoaringTreemap::new();
        for m in &mins_fwd { b_fwd.insert(*m); }
        let l_fwd = b_fwd.len() as f64;

        let mut b_rc = RoaringTreemap::new();
        for m in &mins_rc { b_rc.insert(*m); }
        let l_rc = b_rc.len() as f64;

        let mut result = HashMap::new();

        for (id, bucket) in &self.buckets {
            let mut score: f64 = 0.0;
            if l_fwd > 0.0 { score = score.max(bucket.intersection_len(&b_fwd) as f64 / l_fwd); }
            if l_rc > 0.0 { score = score.max(bucket.intersection_len(&b_rc) as f64 / l_rc); }

            let stats = result.entry(*id).or_insert_with(HashMap::new);
            for t_int in 1..10 {
                let t_val = t_int as f64 / 10.0;
                let key = format!("{:.1}", t_val); 
                if score >= t_val { stats.insert(key, (1, 0)); } else { stats.insert(key, (0, 1)); }
            }
        }
        result
    }

    fn compute_read_stats_paired(&self, r1: &str, r2: &str) -> HashMap<u32, HashMap<String, (u64, u64)>> {
        let (r1_f, r1_rc) = extract_dual_strand(r1);
        let (r2_f, r2_rc) = extract_dual_strand(r2);

        if r1_f.is_empty() && r2_f.is_empty() { return HashMap::new(); }

        let mut b_hyp_a = RoaringTreemap::new();
        for m in &r1_f { b_hyp_a.insert(*m); }
        for m in &r2_rc { b_hyp_a.insert(*m); }
        let l_a = b_hyp_a.len() as f64;

        let mut b_hyp_b = RoaringTreemap::new();
        for m in &r1_rc { b_hyp_b.insert(*m); }
        for m in &r2_f { b_hyp_b.insert(*m); }
        let l_b = b_hyp_b.len() as f64;

        let mut result = HashMap::new();

        for (id, bucket) in &self.buckets {
            let mut score: f64 = 0.0;
            if l_a > 0.0 { score = score.max(bucket.intersection_len(&b_hyp_a) as f64 / l_a); }
            if l_b > 0.0 { score = score.max(bucket.intersection_len(&b_hyp_b) as f64 / l_b); }

            let stats = result.entry(*id).or_insert_with(HashMap::new);
            for t_int in 1..10 {
                let t_val = t_int as f64 / 10.0;
                let key = format!("{:.1}", t_val);
                if score >= t_val { stats.insert(key, (1, 0)); } else { stats.insert(key, (0, 1)); }
            }
        }
        result
    }
}

#[pymodule]
fn ry_partitioner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RYEngine>()?;
    Ok(())
}

