# Compile (Linux/Mac)
# Ensure target/debug/ is in your library path or link explicitly
gcc -o c_example example.c -L./target/debug -lrype -lpthread -ldl -lm

# Run (Set LD_LIBRARY_PATH so it finds librype.so)
export LD_LIBRARY_PATH=./target/debug:$LD_LIBRARY_PATH
./c_example

