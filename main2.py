import ry_partitioner
import time
import random

def generate_random_dna(length):
    return "".join(random.choices("ACGT", k=length))

def mutate_sequence(seq, error_rate=0.05, mode="random"):
    chars = list(seq)
    num_mutations = int(len(seq) * error_rate)
    indices = random.sample(range(len(seq)), num_mutations)

    for i in indices:
        orig = chars[i]
        if mode == "transition":
            # RY Preserving Mutation (A<->G, C<->T)
            # This should be INVISIBLE to the engine
            if orig in "AG": chars[i] = "G" if orig == "A" else "A"
            else:            chars[i] = "T" if orig == "C" else "C"
        else:
            # Random Mutation (includes Transversions)
            # This breaks K-mers in RY space
            chars[i] = random.choice("ACGT")

    return "".join(chars)

def run_pipeline():
    engine = ry_partitioner.RYEngine()

    print("📂 Loading Target Genome (5Mbp)...")
    target_genome = generate_random_dna(5_000_000)
    engine.add_genome(42, target_genome)

    # 10kb Read
    original_read = target_genome[100000 : 110000]

    print("\n--- TEST 1: The 'Invisible' Mutation (Transitions) ---")
    print("Simulating 5% divergence using ONLY A<->G and C<->T.")
    # In standard alignment (BLAST), this is a 95% match.
    # In RY space, this should be a 100% match.
    transition_read = mutate_sequence(original_read, error_rate=0.05, mode="transition")

    matches = engine.query(transition_read, threshold=0.8)
    if matches:
        print(f"✅ SUCCESS: Found in Bucket {matches[0][0]} with Score {matches[0][1]:.1%}")
    else:
        print("❌ FAILED.")

    print("\n--- TEST 2: The 'Destructive' Mutation (Random/Transversions) ---")
    print("Simulating 5% random noise (includes A<->C, etc).")
    # With K=64, this destroys the signal.
    random_read = mutate_sequence(original_read, error_rate=0.05, mode="random")

    matches = engine.query(random_read, threshold=0.1) # Lower threshold to see if anything survives
    if matches:
        print(f"⚠️  Result: Score {matches[0][1]:.1%} (Low due to K=64 sensitivity)")
    else:
        print("❌ No matches (Expected with K=64 and high noise).")

if __name__ == "__main__":
    run_pipeline()

