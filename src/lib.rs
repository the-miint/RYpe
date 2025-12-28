use pyo3::prelude::*;
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::collections::VecDeque;
use rayon::prelude::*;
use needletail::parse_fastx_file;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

const K: usize = 64; 
const WINDOW_SIZE: usize = 50; 
const SALT: u64 = 0x5555555555555555;
const BATCH_SIZE: usize = 10_000; 

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
    let mut mins = Vec::with_capacity(len / WINDOW_SIZE + 2);
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
    let capacity = len / WINDOW_SIZE + 2;
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
    buckets: HashMap<u32, RoaringTreemap>,
}

unsafe impl Sync for RYEngine {}

#[pymethods]
impl RYEngine {
    #[new]
    fn new() -> Self {
        RYEngine { buckets: HashMap::new() }
    }

    fn add_genome(&mut self, bucket_id: u32, sequence: &str) {
        let mins = extract_fwd_only(sequence);
        let bitmap = self.buckets.entry(bucket_id).or_insert_with(RoaringTreemap::new);
        for m in mins { bitmap.insert(m); }
    }
    
    fn get_bucket_cardinality(&self, bucket_id: u32) -> u64 {
        self.buckets.get(&bucket_id).map(|b| b.len()).unwrap_or(0)
    }

    // --- SAVE / LOAD ---

    fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        let num_buckets = self.buckets.len() as u32;
        writer.write_all(&num_buckets.to_le_bytes()).unwrap();

        for (id, map) in &self.buckets {
            writer.write_all(&id.to_le_bytes()).unwrap();
            let mut buf = Vec::new();
            map.serialize_into(&mut buf).unwrap();
            let size = buf.len() as u64;
            writer.write_all(&size.to_le_bytes()).unwrap();
            writer.write_all(&buf).unwrap();
        }
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 4];

        reader.read_exact(&mut buffer).unwrap();
        let num_buckets = u32::from_le_bytes(buffer);

        let mut buckets = HashMap::new();
        for _ in 0..num_buckets {
            reader.read_exact(&mut buffer).unwrap();
            let id = u32::from_le_bytes(buffer);

            let mut size_buf = [0u8; 8];
            reader.read_exact(&mut size_buf).unwrap();
            let _size = u64::from_le_bytes(size_buf);

            let map = RoaringTreemap::deserialize_from(&mut reader)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Roaring Error: {:?}", e)))?;
            buckets.insert(id, map);
        }
        Ok(RYEngine { buckets })
    }

    // --- BENCHMARK METHODS (Optimized with Fold/Reduce + Short Circuit) ---

    fn benchmark_file(&self, py: Python, path: String) -> PyResult<(u64, HashMap<u32, HashMap<String, u64>>)> {
        let path_clone = path.clone();
        
        let (total_reads, stats) = py.allow_threads(move || {
            let mut reader = parse_fastx_file(&path_clone).expect("Invalid FASTQ path");
            let mut global_stats: HashMap<u32, HashMap<String, u64>> = HashMap::new();
            let mut total_count: u64 = 0;
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            while let Some(record) = reader.next() {
                let seqrec = record.expect("Invalid FASTQ record");
                if let Ok(seq_str) = std::str::from_utf8(&seqrec.seq()) {
                    batch.push(seq_str.to_string());
                }

                if batch.len() >= BATCH_SIZE {
                    total_count += batch.len() as u64;
                    let batch_results = self.process_batch_single(&batch);
                    self.merge_stats_hits_only(&mut global_stats, batch_results);
                    batch.clear();
                }
            }

            if !batch.is_empty() {
                total_count += batch.len() as u64;
                let batch_results = self.process_batch_single(&batch);
                self.merge_stats_hits_only(&mut global_stats, batch_results);
            }

            (total_count, global_stats)
        });

        Ok((total_reads, stats))
    }

    fn benchmark_paired(&self, py: Python, r1_path: String, r2_path: String) -> PyResult<(u64, HashMap<u32, HashMap<String, u64>>)> {
        let r1_c = r1_path.clone();
        let r2_c = r2_path.clone();

        let (total_reads, stats) = py.allow_threads(move || {
            let mut r1_reader = parse_fastx_file(&r1_c).expect("Invalid R1 path");
            let mut r2_reader = parse_fastx_file(&r2_c).expect("Invalid R2 path");

            let mut global_stats: HashMap<u32, HashMap<String, u64>> = HashMap::new();
            let mut total_count: u64 = 0;
            let mut batch = Vec::with_capacity(BATCH_SIZE);

            while let (Some(rec1), Some(rec2)) = (r1_reader.next(), r2_reader.next()) {
                let r1 = rec1.expect("Error reading R1");
                let r2 = rec2.expect("Error reading R2");
                let s1 = std::str::from_utf8(&r1.seq()).unwrap_or("").to_string();
                let s2 = std::str::from_utf8(&r2.seq()).unwrap_or("").to_string();
                
                batch.push((s1, s2));

                if batch.len() >= BATCH_SIZE {
                    total_count += batch.len() as u64;
                    let batch_results = self.process_batch_paired(&batch);
                    self.merge_stats_hits_only(&mut global_stats, batch_results);
                    batch.clear();
                }
            }

            if !batch.is_empty() {
                total_count += batch.len() as u64;
                let batch_results = self.process_batch_paired(&batch);
                self.merge_stats_hits_only(&mut global_stats, batch_results);
            }
            (total_count, global_stats)
        });

        Ok((total_reads, stats))
    }
}

// --- INTERNAL HELPERS ---
impl RYEngine {
    
    fn merge_stats_hits_only(&self, global: &mut HashMap<u32, HashMap<String, u64>>, local: HashMap<u32, HashMap<String, u64>>) {
        for (bid, t_map) in local {
            let g_map = global.entry(bid).or_insert_with(HashMap::new);
            for (thresh, hits) in t_map {
                *g_map.entry(thresh).or_insert(0) += hits;
            }
        }
    }

    fn process_batch_single(&self, batch: &[String]) -> HashMap<u32, HashMap<String, u64>> {
        batch.par_iter()
            .fold(HashMap::new, |mut acc: HashMap<u32, HashMap<String, u64>>, seq| {
                let (mut mins_fwd, mut mins_rc) = extract_dual_strand(seq);
                mins_fwd.sort_unstable(); mins_fwd.dedup();
                mins_rc.sort_unstable(); mins_rc.dedup();
                
                let l_fwd = mins_fwd.len() as f64;
                let l_rc = mins_rc.len() as f64;
                if l_fwd == 0.0 { return acc; }

                for (id, bucket) in &self.buckets {
                    let mut score: f64 = 0.0;
                    if l_fwd > 0.0 {
                        let hits = mins_fwd.iter().filter(|m| bucket.contains(**m)).count();
                        score = score.max(hits as f64 / l_fwd);
                    }
                    if l_rc > 0.0 {
                        let hits = mins_rc.iter().filter(|m| bucket.contains(**m)).count();
                        score = score.max(hits as f64 / l_rc);
                    }
                    
                    if score < 0.1 { continue; }

                    let stats = acc.entry(*id).or_insert_with(HashMap::new);
                    for t_int in 1..10 {
                        let t_val = t_int as f64 / 10.0;
                        if score >= t_val {
                            let key = format!("{:.1}", t_val);
                            *stats.entry(key).or_insert(0) += 1;
                        }
                    }
                }
                acc
            })
            .reduce(HashMap::new, |mut map_a, map_b| {
                for (bid, t_map) in map_b {
                    let g_map = map_a.entry(bid).or_insert_with(HashMap::new);
                    for (thresh, hits) in t_map {
                        *g_map.entry(thresh).or_insert(0) += hits;
                    }
                }
                map_a
            })
    }

    fn process_batch_paired(&self, batch: &[(String, String)]) -> HashMap<u32, HashMap<String, u64>> {
        batch.par_iter()
            .fold(HashMap::new, |mut acc: HashMap<u32, HashMap<String, u64>>, (r1, r2)| {
                let (mut r1_f, mut r1_rc) = extract_dual_strand(r1);
                r1_f.sort_unstable(); r1_f.dedup();
                r1_rc.sort_unstable(); r1_rc.dedup();
                
                let (mut r2_f, mut r2_rc) = extract_dual_strand(r2);
                r2_f.sort_unstable(); r2_f.dedup();
                r2_rc.sort_unstable(); r2_rc.dedup();

                if r1_f.is_empty() && r2_f.is_empty() { return acc; }

                let mut hyp_a = r1_f.clone(); hyp_a.extend_from_slice(&r2_rc);
                hyp_a.sort_unstable(); hyp_a.dedup();
                let len_a = hyp_a.len() as f64;

                let mut hyp_b = r1_rc.clone(); hyp_b.extend_from_slice(&r2_f);
                hyp_b.sort_unstable(); hyp_b.dedup();
                let len_b = hyp_b.len() as f64;

                for (id, bucket) in &self.buckets {
                    let mut score: f64 = 0.0;
                    if len_a > 0.0 {
                        let hits = hyp_a.iter().filter(|m| bucket.contains(**m)).count();
                        score = score.max(hits as f64 / len_a);
                    }
                    if len_b > 0.0 {
                        let hits = hyp_b.iter().filter(|m| bucket.contains(**m)).count();
                        score = score.max(hits as f64 / len_b);
                    }

                    if score < 0.1 { continue; }

                    let stats = acc.entry(*id).or_insert_with(HashMap::new);
                    for t_int in 1..10 {
                        let t_val = t_int as f64 / 10.0;
                        if score >= t_val {
                            let key = format!("{:.1}", t_val);
                            *stats.entry(key).or_insert(0) += 1;
                        }
                    }
                }
                acc
            })
            .reduce(HashMap::new, |mut map_a, map_b| {
                for (bid, t_map) in map_b {
                    let g_map = map_a.entry(bid).or_insert_with(HashMap::new);
                    for (thresh, hits) in t_map {
                        *g_map.entry(thresh).or_insert(0) += hits;
                    }
                }
                map_a
            })
    }
}

#[pymodule]
fn ry_partitioner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RYEngine>()?;
    Ok(())
}

