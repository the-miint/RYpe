import argparse
import sys
import skbio
import ry_partitioner
from collections import defaultdict

def run_test(genome_paths, fastq1_path, fastq2_path=None, separate_buckets=False):
    print(f"--- RY Engine Test ---")
    print(f"Genomes: {', '.join(genome_paths)}")
    print(f"Read 1:  {fastq1_path}")
    if fastq2_path:
        print(f"Read 2:  {fastq2_path}")
    print(f"Mode:    {'Separate Buckets' if separate_buckets else 'Single Merged Bucket'}")

    # 1. Initialize Engine
    print("\n[1/3] Initializing RY Engine...")
    engine = ry_partitioner.RYEngine()

    # 2. Index Genomes
    print("[2/3] Indexing Genomes...")

    # Map bucket_id -> Name for reporting
    bucket_names = {}

    # If merging, everything goes to Bucket 1
    current_bucket_id = 1

    for i, g_path in enumerate(genome_paths):
        if separate_buckets:
            bucket_id = i + 1
            bucket_label = g_path.split('/')[-1] # Simple filename
        else:
            bucket_id = 1
            bucket_label = "Merged_Reference"

        bucket_names[bucket_id] = bucket_label

        print(f"      Loading {g_path} into Bucket {bucket_id}...")

        contig_count = 0
        total_bases = 0

        # Load all contigs in this file
        for seq_record in skbio.io.read(g_path, format='fasta'):
            seq_str = str(seq_record)
            engine.add_genome(bucket_id, seq_str)
            contig_count += 1
            total_bases += len(seq_str)

    # Report Cardinalities
    print("\n      Bucket Cardinalities (Unique Minimizers):")
    for b_id, name in bucket_names.items():
        card = engine.get_bucket_cardinality(b_id)
        print(f"        Bucket {b_id} ({name}): {card:,}")

    # 3. Process Reads
    print("\n[3/3] Processing Reads...")

    # Thresholds: 0.1, 0.2 ... 0.9
    thresholds = [round(x * 0.1, 1) for x in range(1, 10)]

    # Stats Structure:
    # global_stats[bucket_id][threshold] = {'hits': 0, 'misses': 0}
    global_stats = defaultdict(lambda: {t: {'hits': 0, 'misses': 0} for t in thresholds})

    read_count = 0

    # Generator for reads
    if fastq2_path:
        # Paired End Iterator
        r1_iter = skbio.io.read(fastq1_path, format='fastq', phred_offset=33)
        r2_iter = skbio.io.read(fastq2_path, format='fastq', phred_offset=33)
        read_iterator = zip(r1_iter, r2_iter)
        is_paired = True
    else:
        # Single End Iterator
        read_iterator = skbio.io.read(fastq1_path, format='fastq', phred_offset=33)
        is_paired = False

    for item in read_iterator:
        read_count += 1

        # Query Engine
        if is_paired:
            r1, r2 = item
            # Paired Query: Returns best match score per bucket using combined signal
            results = engine.query_paired(str(r1), str(r2), 0.0)
        else:
            r1 = item
            # Single Query
            results = engine.query(str(r1), 0.0)

        # Convert results list [(id, score), ...] to dict for O(1) lookup
        # Default score is 0.0 if bucket not returned (no matches)
        scores_by_bucket = {b_id: score for b_id, score in results}

        # Update stats for EVERY bucket
        # (A read is tested against every bucket independently)
        for b_id in bucket_names.keys():
            score = scores_by_bucket.get(b_id, 0.0)

            for t in thresholds:
                if score >= t:
                    global_stats[b_id][t]['hits'] += 1
                else:
                    global_stats[b_id][t]['misses'] += 1

        if read_count % 1000 == 0:
            print(f"\r      Processed {read_count} reads...", end="")

    print(f"\r      Processed {read_count} reads. Done.    \n")

    # 4. Report Results
    for b_id, name in bucket_names.items():
        print(f"=== Results for Bucket {b_id}: {name} ===")
        print(f"{'Threshold':<10} | {'Hits':<10} | {'Misses':<10} | {'Hit Rate (%)':<12}")
        print("-" * 52)

        stats = global_stats[b_id]
        for t in thresholds:
            hits = stats[t]['hits']
            miss = stats[t]['misses']
            total = hits + miss
            rate = (hits / total * 100.0) if total > 0 else 0.0

            print(f"{t:<10} | {hits:<10} | {miss:<10} | {rate:<12.2f}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RY Engine (Single & Paired End)")

    parser.add_argument("--genome", required=True,
                        help="Comma-separated paths to reference genomes (FASTA)")

    parser.add_argument("--reads1", required=True,
                        help="Path to Read 1 (FASTQ)")

    parser.add_argument("--reads2",
                        help="Path to Read 2 (FASTQ) - Optional for Paired End")

    parser.add_argument("--separate-buckets", action="store_true",
                        help="If set, each genome file gets a unique bucket ID. Default: Merge all into Bucket 1.")

    args = parser.parse_args()

    # Parse comma-separated genomes
    genome_list = [x.strip() for x in args.genome.split(',')]

    run_test(genome_list, args.reads1, args.reads2, args.separate_buckets)

