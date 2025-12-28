import argparse
import os
import sys
import time
import skbio
import ry_partitioner

def main():
    parser = argparse.ArgumentParser(description="RY Partition Benchmark")
    parser.add_argument("--genome", required=True, help="Comma separated list of reference FASTA files")
    parser.add_argument("--reads1", required=True, help="Path to R1 FASTQ")
    parser.add_argument("--reads2", help="Path to R2 FASTQ (optional)", default=None)
    parser.add_argument("--separate-buckets", action="store_true", help="If set, each genome file gets its own bucket")
    parser.add_argument("--index", default="temp_benchmark.ryidx", help="Path to save/load index")

    args = parser.parse_args()

    # --- 1. Setup Buckets based on CLI ---
    genome_files = [x.strip() for x in args.genome.split(',')]
    buckets = {}
    bucket_names = {}

    if args.separate_buckets:
        # One bucket per file
        for i, fpath in enumerate(genome_files):
            b_id = i + 1
            buckets[b_id] = [fpath]
            # Name bucket after filename
            bucket_names[b_id] = os.path.basename(fpath)
    else:
        # All files in Bucket 1
        buckets[1] = genome_files
        bucket_names[1] = "Combined_Reference"

    # --- 2. Build or Load Index ---
    # Note: For benchmarking, we usually rebuild if the index name is generic,
    # but here we check if it exists to be safe.
    if os.path.exists(args.index):
        print(f"[1/3] Loading existing index from {args.index}...")
        try:
            engine = ry_partitioner.RYEngine.load(args.index)
        except Exception as e:
            print(f"      Error loading index: {e}. Rebuilding...")
            os.remove(args.index)
            engine = build_index(buckets, args.index)
    else:
        engine = build_index(buckets, args.index)

    # --- 3. Run Benchmark ---
    print("\n[3/3] Processing Reads (Parallel Rust Execution)...")
    t0 = time.time()

    if args.reads2:
        print(f"      Mode: Paired End")
        print(f"      R1: {args.reads1}")
        print(f"      R2: {args.reads2}")
        total_reads, results = engine.benchmark_paired(args.reads1, args.reads2)
    else:
        print(f"      Mode: Single End")
        print(f"      R1: {args.reads1}")
        total_reads, results = engine.benchmark_file(args.reads1)

    duration = time.time() - t0
    if duration == 0: duration = 0.001
    print(f"      Processed {total_reads:,} reads in {duration:.2f}s ({total_reads/duration:.0f} reads/sec)\n")

    # --- 4. Report Results ---
    known_buckets = sorted(bucket_names.keys())

    for b_id in known_buckets:
        name = bucket_names.get(b_id, f"Bucket {b_id}")
        print(f"=== Results for Bucket {b_id}: {name} ===")
        print(f"{'Threshold':<10} | {'Hits':<10} | {'Misses':<10} | {'Hit Rate (%)':<12}")
        print("-" * 55)

        bucket_stats = results.get(b_id, {})
        thresholds = [f"{x/10:.1f}" for x in range(1, 10)]

        for t_str in thresholds:
            hits = bucket_stats.get(t_str, 0)
            misses = total_reads - hits
            rate = (hits / total_reads * 100.0) if total_reads > 0 else 0.0
            print(f"{t_str:<10} | {hits:<10} | {misses:<10} | {rate:<12.2f}")
        print("\n")

def build_index(buckets, output_path):
    print("[1/3] Building new index...")
    engine = ry_partitioner.RYEngine()
    t0 = time.time()

    for b_id, files in buckets.items():
        print(f"      Building Bucket {b_id} from {len(files)} files...")
        for fpath in files:
            if not os.path.exists(fpath):
                print(f"      ERROR: Genome file not found: {fpath}")
                sys.exit(1)

            # Streaming reader to keep memory low during build
            for r in skbio.io.read(fpath, format='fasta'):
                engine.add_genome(b_id, str(r))

    print(f"      Saving index to {output_path}...")
    engine.save(output_path)
    print(f"      Build complete in {time.time() - t0:.2f}s")

    # Print stats
    print("\n[2/3] Index Statistics:")
    for b_id in buckets.keys():
        count = engine.get_bucket_cardinality(b_id)
        print(f"      Bucket {b_id}: {count:,} unique minimizers")

    return engine

if __name__ == "__main__":
    main()

