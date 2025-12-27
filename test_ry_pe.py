import argparse
import sys
import skbio
import ry_partitioner

def run_test(genome_paths, fastq1_path, fastq2_path=None, separate_buckets=False):
    print(f"--- RY Engine Test (Rust Parallel) ---")
    print(f"Genomes: {', '.join(genome_paths)}")
    print(f"Read 1:  {fastq1_path}")
    if fastq2_path:
        print(f"Read 2:  {fastq2_path}")
    print(f"Mode:    {'Separate Buckets' if separate_buckets else 'Single Merged Bucket'}")

    # 1. Initialize Engine
    print("\n[1/3] Initializing RY Engine...")
    engine = ry_partitioner.RYEngine()

    # 2. Index Genomes (This part stays in Python as it's one-time setup)
    print("[2/3] Indexing Genomes...")
    bucket_names = {}

    for i, g_path in enumerate(genome_paths):
        if separate_buckets:
            bucket_id = i + 1
            bucket_label = g_path.split('/')[-1]
        else:
            bucket_id = 1
            bucket_label = "Merged_Reference"

        bucket_names[bucket_id] = bucket_label
        print(f"      Loading {g_path} into Bucket {bucket_id}...")

        # We stick with skbio here for FASTA parsing as it's not the bottleneck
        for seq_record in skbio.io.read(g_path, format='fasta'):
            engine.add_genome(bucket_id, str(seq_record))

    print("\n      Bucket Cardinalities (Unique Minimizers):")
    for b_id, name in bucket_names.items():
        card = engine.get_bucket_cardinality(b_id)
        print(f"        Bucket {b_id} ({name}): {card:,}")

    # 3. Process Reads (The New Fast Part)
    print("\n[3/3] Processing Reads (Parallel Rust Execution)...")

    # We hand off the file paths to Rust. Rust handles parsing + threading.
    if fastq2_path:
        # Returns: {bucket_id: {"0.1": (hits, misses), "0.2": ...}}
        results = engine.benchmark_paired(fastq1_path, fastq2_path)
    else:
        results = engine.benchmark_file(fastq1_path)

    print("      Done.\n")

    # 4. Report Results
    # We iterate over the bucket IDs we know exist
    known_buckets = sorted(bucket_names.keys())

    for b_id in known_buckets:
        name = bucket_names[b_id]
        print(f"=== Results for Bucket {b_id}: {name} ===")
        print(f"{'Threshold':<10} | {'Hits':<10} | {'Misses':<10} | {'Hit Rate (%)':<12}")
        print("-" * 52)

        # Retrieve stats for this bucket.
        # If a bucket had 0 hits/misses total (empty file?), it might not be in results.
        bucket_stats = results.get(b_id, {})

        # We sort keys numerically: "0.1", "0.2" ... "0.9"
        sorted_thresholds = sorted(bucket_stats.keys(), key=lambda x: float(x))

        if not sorted_thresholds:
            print("      (No data returned for this bucket)")

        for t_str in sorted_thresholds:
            hits, misses = bucket_stats[t_str]
            total = hits + misses
            rate = (hits / total * 100.0) if total > 0 else 0.0

            print(f"{t_str:<10} | {hits:<10} | {misses:<10} | {rate:<12.2f}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RY Engine (Parallel Rust)")

    parser.add_argument("--genome", required=True,
                        help="Comma-separated paths to reference genomes (FASTA)")
    parser.add_argument("--reads1", required=True,
                        help="Path to Read 1 (FASTQ)")
    parser.add_argument("--reads2",
                        help="Path to Read 2 (FASTQ) - Optional for Paired End")
    parser.add_argument("--separate-buckets", action="store_true",
                        help="If set, each genome file gets a unique bucket ID.")

    args = parser.parse_args()
    genome_list = [x.strip() for x in args.genome.split(',')]

    run_test(genome_list, args.reads1, args.reads2, args.separate_buckets)

