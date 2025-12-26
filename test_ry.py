import argparse
import sys
import skbio
import ry_partitioner

def run_test(genome_path, fastq_path):
    print(f"--- RY Engine Test ---")
    print(f"Genome: {genome_path}")
    print(f"Reads:  {fastq_path}")

    # 1. Initialize Engine
    print("\n[1/3] Initializing RY Engine...")
    engine = ry_partitioner.RYEngine()

    # 2. Load Genome (Bucket ID = 1)
    # We load all chromosomes/contigs from the FASTA into the same bucket.
    print("[2/3] Indexing Genome...")
    contig_count = 0
    total_bases = 0

    # skbio.io.read yields Sequence objects. We convert them to string.
    # We use format='fasta'.
    for seq_record in skbio.io.read(genome_path, format='fasta'):
        # Convert skbio Sequence to string for the Rust engine
        seq_str = str(seq_record)
        engine.add_genome(1, seq_str)

        contig_count += 1
        total_bases += len(seq_str)

    bucket_cardinality = engine.get_bucket_cardinality(1)
    print(f"      Indexed {contig_count} contigs ({total_bases:,} bp).")
    print(f"      Bucket 1 Cardinality (Unique Minimizers): {bucket_cardinality:,}")

    # 3. Process Reads and Accumulate Stats
    print("\n[3/3] Processing Reads...")

    # Thresholds: 0.1, 0.2, ... 0.9
    thresholds = [round(x * 0.1, 1) for x in range(1, 10)]

    # Stats structure: {0.1: {'hits': 0, 'misses': 0}, ...}
    stats = {t: {'hits': 0, 'misses': 0} for t in thresholds}

    read_count = 0

    # Process FASTQ
    # variant='illumina1.8' corresponds to Phred+33, which is standard.
    # We assume phred_offset=33 as requested.
    for read in skbio.io.read(fastq_path, format='fastq', phred_offset=33):
        read_count += 1
        read_str = str(read)

        # Query the engine.
        # We set the engine threshold to 0.0 to get the raw score
        # so we can bin it ourselves.
        results = engine.query(read_str, 0.0)

        # Extract score for Bucket 1.
        # results is a list of (bucket_id, score), e.g., [(1, 0.85)]
        # If no minimizers matched at all, results might be empty or score is 0.
        score = 0.0
        for b_id, b_score in results:
            if b_id == 1:
                score = b_score
                break

        # Update stats for all thresholds
        for t in thresholds:
            if score >= t:
                stats[t]['hits'] += 1
            else:
                stats[t]['misses'] += 1

        if read_count % 1000 == 0:
            print(f"\r      Processed {read_count} reads...", end="")

    print(f"\r      Processed {read_count} reads. Done.    \n")

    # 4. Report Results
    print(f"{'Threshold':<10} | {'Hits':<10} | {'Misses':<10} | {'Hit Rate (%)':<12}")
    print("-" * 50)

    for t in thresholds:
        hits = stats[t]['hits']
        miss = stats[t]['misses']
        total = hits + miss
        rate = (hits / total * 100.0) if total > 0 else 0.0

        print(f"{t:<10} | {hits:<10} | {miss:<10} | {rate:<12.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RY Engine with Scikit-Bio")
    parser.add_argument("genome", help="Path to reference genome (FASTA)")
    parser.add_argument("fastq", help="Path to simulated reads (FASTQ)")

    args = parser.parse_args()

    run_test(args.genome, args.fastq)

