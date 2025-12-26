#!/bin/bash

# ==============================================================================
# Script: generate_short_reads.sh
# Description: Simulates NovaSeq-like paired-end short reads (150bp) using ART.
#              Output: Two files per genome (_1.fq, _2.fq).
#              Runs in parallel.
# Usage: ./generate_short_reads.sh <input_genome_dir> <output_fastq_dir> <max_parallel_jobs>
# ==============================================================================

# 1. Input Validation
IN_DIR="$1"
OUT_DIR="$2"
JOBS="$3"

if [[ -z "$IN_DIR" || -z "$OUT_DIR" ]]; then
    echo "Usage: $0 <input_genome_dir> <output_fastq_dir> [parallel_jobs]"
    echo "Example: $0 ./ref_genomes ./simulated_short_reads 8"
    exit 1
fi

if [[ -z "$JOBS" ]]; then JOBS=4; fi

# Verify art_illumina is installed
if ! command -v art_illumina &> /dev/null; then
    echo "[ERROR] art_illumina could not be found."
    echo "Please install it via bioconda: conda install art"
    exit 1
fi

mkdir -p "$OUT_DIR"

# 2. Define Processing Function
generate_one_genome_short() {
    local ref_path="$1"
    local out_dir="$2"
    
    local filename=$(basename "$ref_path")
    local base="${filename%.*}"
    local out_prefix="${out_dir}/${base}_"

    echo "[START] Processing: $filename"

    # --- ART COMMAND (NovaSeq Proxy) ---
    # -ss HS25 : HiSeq 2500 profile (Closest standard proxy for high-qual Illumina)
    # -i       : Input reference
    # -p       : Paired-end simulation
    # -l 150   : 150bp read length (NovaSeq X standard)
    # -f 30    : 30x Coverage
    # -m 450   : Mean fragment size (DNA insert size)
    # -s 50    : Fragment size standard deviation
    # -o       : Output prefix
    
    art_illumina \
        -ss HS25 \
        -i "$ref_path" \
        -p \
        -l 150 \
        -f 30 \
        -m 450 \
        -s 50 \
        -o "$out_prefix" \
        -na \
        >/dev/null 2>&1

    # Check result
    if [[ $? -eq 0 ]]; then
        # ART outputs .fq by default, usually we want .fastq for consistency
        # And ART appends '1.fq' and '2.fq' to the prefix.
        mv "${out_prefix}1.fq" "${out_prefix}R1.fastq"
        mv "${out_prefix}2.fq" "${out_prefix}R2.fastq"
        echo "[DONE]  Finished:   $filename"
    else
        echo "[ERR]   Failed:     $filename"
    fi
}

export -f generate_one_genome_short

# 3. Execution Phase
echo "--------------------------------------------------------"
echo "Starting NovaSeq Simulation (ART)"
echo "Input Dir:  $IN_DIR"
echo "Output Dir: $OUT_DIR"
echo "Parallelism: $JOBS concurrent processes"
echo "--------------------------------------------------------"

find "$IN_DIR" -type f \( -name "*.fasta" -o -name "*.fna" -o -name "*.fa" \) -print0 | \
    xargs -0 -n 1 -P "$JOBS" -I {} bash -c 'generate_one_genome_short "{}" "$@"' _ "$OUT_DIR"

echo "--------------------------------------------------------"
echo "All jobs completed."

