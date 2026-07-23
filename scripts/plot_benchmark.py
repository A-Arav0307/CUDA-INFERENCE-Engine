import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

n = [100, 500, 1000, 2500, 5000, 10000, 20000, 50000, 100000]
cpu_ms = [9.92, 49.22, 98.88, 245.84, 490.73, 979.79, 1968.65, 4957.01, 9833.03]
naive_ms = [7.67, 37.00, 74.83, 180.24, 364.81, 737.09, 1453.97, 3477.91, 6957.89]
h2d_ms = [0.035, 0.114, 0.313, 0.584, 4.564, 39.15, 11.48, 33.37, 93.55]
compute_ms = [0.063, 0.097, 0.143, 0.225, 0.473, 1.075, 1.504, 3.707, 7.295]
d2h_ms = [0.012, 0.013, 0.020, 0.030, 0.199, 0.738, 1.448, 0.976, 1.814]
gpu_total_ms = [h + c + d for h, c, d in zip(h2d_ms, compute_ms, d2h_ms)]

surface = "#fcfcfb"
primary_ink = "#0b0b0b"
secondary_ink = "#52514e"
muted_ink = "#898781"
gridline = "#e1e0d9"
baseline = "#c3c2b7"

color_cpu = "#2a78d6"
color_naive = "#008300"
color_gpu = "#e87ba4"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
fig.patch.set_facecolor(surface)
ax.set_facecolor(surface)

ax.plot(n, cpu_ms, color=color_cpu, linewidth=2, marker="o", markersize=6, label="CPU")
ax.plot(n, naive_ms, color=color_naive, linewidth=2, marker="o", markersize=6, label="Naive CUDA (per-image)")
ax.plot(n, gpu_total_ms, color=color_gpu, linewidth=2, marker="o", markersize=6, label="Batched + tiled CUDA")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Batch size (number of images)", color=secondary_ink, fontsize=11)
ax.set_ylabel("Time (ms, log scale)", color=secondary_ink, fontsize=11)
ax.set_title("Inference time vs. batch size: CPU vs. naive CUDA vs. optimized CUDA", color=primary_ink, fontsize=13, pad=14)

ax.grid(True, which="major", color=gridline, linewidth=0.8)
ax.grid(True, which="minor", color=gridline, linewidth=0.4, alpha=0.6)
ax.tick_params(colors=muted_ink, labelsize=9)
for spine in ax.spines.values():
    spine.set_color(baseline)
    spine.set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# direct end-of-line labels (relief for the lower-contrast magenta series)
last_x = n[-1]
ax.annotate("CPU", (last_x, cpu_ms[-1]), xytext=(8, 0), textcoords="offset points",
            color=color_cpu, fontsize=10, fontweight="bold", va="center")
ax.annotate("Naive CUDA", (last_x, naive_ms[-1]), xytext=(8, 0), textcoords="offset points",
            color=color_naive, fontsize=10, fontweight="bold", va="center")
ax.annotate("Batched + tiled", (last_x, gpu_total_ms[-1]), xytext=(8, -2), textcoords="offset points",
            color=color_gpu, fontsize=10, fontweight="bold", va="center")

ax.set_xlim(80, last_x * 3.2)

legend = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=secondary_ink)

fig.tight_layout()

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/benchmark_scaling.png", facecolor=surface, bbox_inches="tight")
print("saved to docs/benchmark_scaling.png")
