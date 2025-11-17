import matplotlib.pyplot as plt
import numpy as np

n = [256, 512, 1024, 1025, 2047, 2048]

block64 = [
    1619.533813,
    3040.507445,
    3640.449137,
    3080.305968,
    3700.344693,
    3671.855508,
]

block32 = [
    1172.812403,
    1523.545205,
    1734.789248,
    1513.649790,
    1772.084763,
    1792.927926,
]

block8 = [
    341.678777,
    494.531518,
    572.951539,
    630.257954,
    683.449810,
    602.847130,
]

n = np.array(n, dtype=float)

plt.plot(n, block64, marker="o", label="Block size 64")
plt.plot(n, block32, marker="o", label="Block size 32")
plt.plot(n, block8,  marker="o", label="Block size 8")

plt.xlabel("Matrix size N")
plt.ylabel("Performance (GFLOPS)")
plt.title("Performance vs N for Different Thread Block Sizes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("blocksize_performance.png", dpi=150)
plt.show()
