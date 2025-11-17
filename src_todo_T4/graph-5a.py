import numpy as np
import matplotlib.pyplot as plt

# Peak values
P_peak = 7680.0          # GFLOPS (compute peak)
B_peak = 320.0           # GB/s or GiB/s (bandwidth peak, used as-is)

# Intensity range for the roofline
intensity = np.logspace(-2, 4, 200)  # from 1e-2 to 1e4 FLOPs/byte

# Rooflines
P_mem = B_peak * intensity           # memory roof in GFLOPS
P_comp = np.full_like(intensity, P_peak)

# Your measured point for N = 2048
N = 2048
I_point = N / 6.0                    # FLOPs per byte (approx)
P_point = 3719.639176                # GFLOPS

plt.figure(figsize=(7,5))

plt.loglog(intensity, P_mem, label="Memory roof (320 GiB/s)")
plt.loglog(intensity, P_comp, label="Compute roof (7680 GFLOPS)")

plt.scatter(I_point, P_point, marker="o", color="red", zorder=5,
            label=f"Measured point (N=2048)")

plt.xlabel("Operational intensity (FLOPs / byte)")
plt.ylabel("Performance (GFLOPS)")
plt.title("Roofline Model for NVIDIA T4 (FP32)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("roofline_t4_n2048.png", dpi=150)
plt.show()
