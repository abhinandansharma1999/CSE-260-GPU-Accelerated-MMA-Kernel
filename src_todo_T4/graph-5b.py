import numpy as np
import matplotlib.pyplot as plt

P_peak = 7680.0          # GFLOPS
B_theoretical = 320.0    # GiB/s
B_actual = 220.0         # GiB/s

intensity = np.logspace(-2, 4, 200)

P_mem_theoretical = B_theoretical * intensity
P_mem_actual = B_actual * intensity
P_comp = np.full_like(intensity, P_peak)

N = 2048
I_point = N / 6.0              # FLOPs per byte
P_point = 3719.639176          # GFLOPS

plt.figure(figsize=(7,5))

plt.loglog(intensity, P_mem_theoretical, label="Memory roof (320 GiB/s)")
plt.loglog(intensity, P_mem_actual, label="Memory roof (220 GiB/s)")
plt.loglog(intensity, P_comp, label="Compute roof (7680 GFLOPS)")

plt.scatter(I_point, P_point, marker="o", color="red", zorder=5,
            label="Measured point (N=2048)")

plt.xlabel("Operational intensity (FLOPs / byte)")
plt.ylabel("Performance (GFLOPS)")
plt.title("Roofline Model for NVIDIA T4 (FP32)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("roofline_t4_320_vs_220.png", dpi=150)
plt.show()
