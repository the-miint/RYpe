#!/bin/bash

# ==============================================================================
# Script: generate_reads.sh
# Description: Simulates HiFi/Revio reads for all FASTA files in a directory using Badread.
#              Runs in parallel using xargs to maximize throughput.
# Usage: ./generate_reads.sh <input_genome_dir> <output_fastq_dir> <max_parallel_jobs>
# ==============================================================================

# 1. Input Validation
IN_DIR="$1"
OUT_DIR="$2"
JOBS="$3"

if [[ -z "$IN_DIR" || -z "$OUT_DIR" ]]; then
    echo "Usage: $0 <input_genome_dir> <output_fastq_dir> [parallel_jobs]"
    echo "Example: $0 ./ref_genomes ./simulated_reads 8"
    exit 1
fi

# Default to 4 jobs if not specified
if [[ -z "$JOBS" ]]; then
    JOBS=4
fi

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# 2. Export the function that processes a single file
#    We export this so xargs can call it in a subshell.
generate_one_genome() {
    local ref_path="$1"
    local out_dir="$2"
    
    # Extract filename without path
    local filename=$(basename "$ref_path")
    
    # Strip extension (handles .fasta, .fna, .fa)
    local base="${filename%.*}"
    
    local out_file="${out_dir}/${base}.fastq"

    echo "[START] Processing: $filename -> $out_file"

    # --- BADREAD COMMAND (Revio/HiFi Settings) ---
    # Identity: Mean 99.8%, Max 100%, Stdev 0.1 (Q30+ range)
    badread simulate \
        --reference "$ref_path" \
        --quantity 10x \
        --length 15000,10000 \
        --identity 99.8,100,0.1 \
        --glitches 0,0,0 \
        --junk_reads 0 \
        --random_reads 0 \
        --chimeras 0 \
        > "$out_file" 2>/dev/null

    # Check if badread succeeded
    if [[ $? -eq 0 ]]; then
        echo "[DONE]  Finished:   $filename"
    else
        echo "[ERR]   Failed:     $filename"
    fi
}

export -f generate_one_genome

# 3. Execution Phase
echo "--------------------------------------------------------"
echo "Starting Simulation"
echo "Input Dir:  $IN_DIR"
echo "Output Dir: $OUT_DIR"
echo "Parallelism: $JOBS concurrent processes"
echo "--------------------------------------------------------"

# Find all fasta/fna/fa files and pipe to xargs
# -print0 / -0 handles filenames with spaces correctly
# -P controls the number of parallel jobs
find "$IN_DIR" -type f \( -name "*.fasta" -o -name "*.fna" -o -name "*.fa" \) -print0 | \
    xargs -0 -n 1 -P "$JOBS" -I {} bash -c 'generate_one_genome "{}" "$@"' _ "$OUT_DIR"

echo "--------------------------------------------------------"
echo "All jobs completed."

