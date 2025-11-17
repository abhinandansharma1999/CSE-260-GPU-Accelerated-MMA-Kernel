import matplotlib.pyplot as plt
import numpy as np

n = [
    255, 256, 400, 480, 511, 512, 650, 768, 800,
    1023, 1024, 1025, 1200, 1500, 1600, 1800,
    2000, 2047, 2048, 2049, 4095, 4096
]

blas = [
    5.93, 5.84, None, None, None, 17.4, None, 45.3, None,
    73.7, 73.6, 73.5, None, None, None, None,
    None, 171, 182, 175, 209.2, 213.4
]

cublas = [
    2456.2, 2515.3, None, None, None, 4573.6, None, 4333.1, None,
    4222.5, 4404.9, 3551.0, None, None, None, None,
    None, 4490.5, 4669.8, 4120.7, 4389.0, 4501.6
]

optimized = [
    1548.938251, 1613.962022, 1732.400490, 2571.979307, 2978.487796,
    3023.361726, 2546.392696, 3469.922551, 3052.137078,
    3570.483337, 3636.627606, 3061.100373, 3527.232486,
    3558.023903, 3923.016312, 3394.533889,
    3581.660903, 3574.264674, 3719.639176, 3320.856072,
    2944.886481, 3015.170254
]

blas = np.array(blas, dtype=float)
cublas = np.array(cublas, dtype=float)
optimized = np.array(optimized, dtype=float)

valid_blas = ~np.isnan(blas)
valid_cublas = ~np.isnan(cublas)
valid_opt = ~np.isnan(optimized)

plt.plot(np.array(n)[valid_blas], blas[valid_blas], marker="o", label="BLAS")
plt.plot(np.array(n)[valid_cublas], cublas[valid_cublas], marker="o", label="cuBLAS")
plt.plot(np.array(n)[valid_opt], optimized[valid_opt], marker="o", label="Optimized CUDA")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Performance (GFLOPS)")
plt.title("Performance Comparison (GFLOPS)")
plt.grid(True)
plt.legend()
plt.tight_layout()

filename = "performance_gflops_" + str(min(n)) + "_to_" + str(max(n)) + ".png"
plt.savefig(filename, dpi=150)
plt.show()
