//! Inspect command handlers for debugging minimizer matches.

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufRead;
use std::path::Path;

use rype::{extract_with_positions, Index, MinimizerWorkspace, Strand};

/// A match found in a reference sequence
#[derive(Debug, Clone)]
struct ReferenceMatch {
    file_path: String,
    seq_id: String,
    position: usize,
    strand: Strand,
    kmer: String,
}

/// Extract the k-mer nucleotide string at a given position.
/// For reverse complement strand, returns the RC of the k-mer.
fn extract_kmer_string(seq: &[u8], pos: usize, k: usize, strand: Strand) -> String {
    if pos + k > seq.len() {
        return "<out-of-bounds>".to_string();
    }
    let kmer_bytes = &seq[pos..pos + k];

    match strand {
        Strand::Forward => String::from_utf8_lossy(kmer_bytes).to_string(),
        Strand::ReverseComplement => kmer_bytes
            .iter()
            .rev()
            .map(|&b| match b {
                b'A' | b'a' => 'T',
                b'T' | b't' => 'A',
                b'G' | b'g' => 'C',
                b'C' | b'c' => 'G',
                other => other as char,
            })
            .collect(),
    }
}

/// Build a map of minimizer hash → all reference locations for a bucket
fn build_reference_minimizer_map(
    index: &Index,
    bucket_id: u32,
) -> Result<HashMap<u64, Vec<ReferenceMatch>>> {
    let mut map: HashMap<u64, Vec<ReferenceMatch>> = HashMap::new();
    let mut ws = MinimizerWorkspace::new();

    // Get source info for this bucket (format: "filepath::seqname")
    let sources = index
        .bucket_sources
        .get(&bucket_id)
        .ok_or_else(|| anyhow!("Bucket {} has no sources", bucket_id))?;

    // Group sources by file path
    let mut files_to_seqs: HashMap<String, HashSet<String>> = HashMap::new();
    for source in sources {
        let parts: Vec<&str> = source.split(Index::BUCKET_SOURCE_DELIM).collect();
        if parts.len() >= 2 {
            let file_path = parts[0].to_string();
            let seq_id = parts[1..].join(Index::BUCKET_SOURCE_DELIM);
            files_to_seqs.entry(file_path).or_default().insert(seq_id);
        }
    }

    // Scan each reference file
    for (file_path, target_seqs) in &files_to_seqs {
        let path = Path::new(file_path);
        if !path.exists() {
            log::warn!("Reference file not found: {}", file_path);
            continue;
        }

        let mut reader = match parse_fastx_file(path) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Failed to open reference file {}: {}", file_path, e);
                continue;
            }
        };

        while let Some(record) = reader.next() {
            let rec = match record {
                Ok(r) => r,
                Err(e) => {
                    log::warn!("Error reading record from {}: {}", file_path, e);
                    continue;
                }
            };
            let seq_id = String::from_utf8_lossy(rec.id()).to_string();

            // Only process sequences that are in this bucket's sources
            if !target_seqs.contains(&seq_id) {
                continue;
            }

            let seq = rec.seq();
            let minimizers = extract_with_positions(&seq, index.k, index.w, index.salt, &mut ws);

            for m in minimizers {
                let kmer = extract_kmer_string(&seq, m.position, index.k, m.strand);
                map.entry(m.hash).or_default().push(ReferenceMatch {
                    file_path: file_path.clone(),
                    seq_id: seq_id.clone(),
                    position: m.position,
                    strand: m.strand,
                    kmer,
                });
            }
        }
    }

    Ok(map)
}

/// Main inspect matches function
pub fn inspect_matches(
    index_path: &Path,
    queries_path: &Path,
    ids_file: &Path,
    bucket_filter: &[u32],
) -> Result<()> {
    // 1. Load the index
    log::info!("Loading index from {:?}", index_path);
    let index = Index::load(index_path)?;
    log::info!(
        "Index loaded: {} buckets, K={}, W={}",
        index.buckets.len(),
        index.k,
        index.w
    );

    // 2. Load sequence IDs to inspect
    log::info!("Loading sequence IDs from {:?}", ids_file);
    let target_ids: HashSet<String> = std::io::BufReader::new(File::open(ids_file)?)
        .lines()
        .map_while(Result::ok)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    log::info!("Loaded {} sequence IDs to inspect", target_ids.len());

    // 3. Validate bucket IDs exist
    for &bucket_id in bucket_filter {
        if !index.buckets.contains_key(&bucket_id) {
            return Err(anyhow!("Bucket {} does not exist in index", bucket_id));
        }
    }

    // 4. Build reference minimizer maps for each bucket
    log::info!(
        "Building reference minimizer maps for {} buckets...",
        bucket_filter.len()
    );
    let mut ref_maps: HashMap<u32, HashMap<u64, Vec<ReferenceMatch>>> = HashMap::new();
    for &bucket_id in bucket_filter {
        let bucket_name = index
            .bucket_names
            .get(&bucket_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        log::info!(
            "  Scanning references for bucket {} ({})...",
            bucket_id,
            bucket_name
        );
        ref_maps.insert(bucket_id, build_reference_minimizer_map(&index, bucket_id)?);
    }
    log::info!("Reference maps built.");

    // 5. Process query sequences
    log::info!("Processing query sequences from {:?}", queries_path);
    let mut reader = parse_fastx_file(queries_path).context("Failed to open query file")?;
    let mut ws = MinimizerWorkspace::new();
    let mut queries_processed = 0;
    let mut queries_with_matches = 0;

    while let Some(record) = reader.next() {
        let rec = record.context("Invalid query record")?;
        let id = String::from_utf8_lossy(rec.id()).to_string();

        if !target_ids.contains(&id) {
            continue;
        }

        queries_processed += 1;
        let seq = rec.seq();
        let minimizers = extract_with_positions(&seq, index.k, index.w, index.salt, &mut ws);

        // Find matches
        let mut has_output = false;
        for m in &minimizers {
            // Check which buckets contain this minimizer
            let mut bucket_matches: Vec<(u32, &str, &[ReferenceMatch])> = vec![];

            for &bucket_id in bucket_filter {
                if let Some(bucket) = index.buckets.get(&bucket_id) {
                    if bucket.binary_search(&m.hash).is_ok() {
                        let name = index
                            .bucket_names
                            .get(&bucket_id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        let ref_matches = ref_maps
                            .get(&bucket_id)
                            .and_then(|map| map.get(&m.hash))
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        bucket_matches.push((bucket_id, name, ref_matches));
                    }
                }
            }

            if !bucket_matches.is_empty() {
                if !has_output {
                    println!(">{}", id);
                    has_output = true;
                    queries_with_matches += 1;
                }

                let query_kmer = extract_kmer_string(&seq, m.position, index.k, m.strand);
                let strand_char = if m.strand == Strand::Forward {
                    '+'
                } else {
                    '-'
                };

                println!(
                    "  position: {}  strand: {}  kmer: {}  minimizer: 0x{:016X}",
                    m.position, strand_char, query_kmer, m.hash
                );

                for (bucket_id, bucket_name, ref_matches) in bucket_matches {
                    println!("    bucket: {} (id={})", bucket_name, bucket_id);

                    if ref_matches.is_empty() {
                        println!("      (no reference positions found - file may be missing)");
                        continue;
                    }

                    // Group reference matches by file path, then by seq_id
                    let mut by_file: HashMap<&str, HashMap<&str, Vec<&ReferenceMatch>>> =
                        HashMap::new();
                    for rm in ref_matches {
                        by_file
                            .entry(&rm.file_path)
                            .or_default()
                            .entry(&rm.seq_id)
                            .or_default()
                            .push(rm);
                    }

                    // Output grouped by file, then by sequence
                    let mut file_paths: Vec<_> = by_file.keys().collect();
                    file_paths.sort();
                    for file_path in file_paths {
                        println!("      file: {}", file_path);
                        let seqs = &by_file[file_path];
                        let mut seq_ids: Vec<_> = seqs.keys().collect();
                        seq_ids.sort();
                        for seq_id in seq_ids {
                            println!("        ref: {}", seq_id);
                            for rm in &seqs[seq_id] {
                                let ref_strand = if rm.strand == Strand::Forward {
                                    '+'
                                } else {
                                    '-'
                                };
                                println!(
                                    "          pos: {}  strand: {}  kmer: {}",
                                    rm.position, ref_strand, rm.kmer
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    log::info!(
        "Inspection complete: {} queries processed, {} had matches",
        queries_processed,
        queries_with_matches
    );
    Ok(())
}
