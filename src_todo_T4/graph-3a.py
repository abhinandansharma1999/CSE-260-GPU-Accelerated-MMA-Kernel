import matplotlib.pyplot as plt
import numpy as np

# Problem sizes
n = [256, 512, 1024, 2048]

# Performance in GFLOPS
naive = np.array([
    56.1467,
    75.3337,
    79.6381,
    52.7168
], dtype=float)

optimized = np.array([
    1619.533813,
    3040.507445,
    3640.449137,
    3671.855508
], dtype=float)

# --- Plot: Naive vs Optimized performance (GFLOPS) ---
plt.figure()
plt.plot(n, naive, marker="o", label="Naive")
plt.plot(n, optimized, marker="o", label="Optimized")
plt.xlabel("N")
plt.ylabel("Performance (GFLOPS)")
plt.title("Naive vs Optimized Performance")
plt.grid(True)
plt.legend()

# Save to file
plt.savefig("naive_vs_optimized_gflops.png", dpi=300)
plt.close()

# --- Plot: Speedup (Optimized / Naive) ---
speedup = optimized / naive

plt.figure()
plt.plot(n, speedup, marker="o")
plt.xlabel("N")
plt.ylabel("Speedup (Optimized / Naive)")
plt.title("Speedup in Performance")
plt.grid(True)

# Save to file
plt.savefig("speedup_gflops.png", dpi=300)
plt.close()
