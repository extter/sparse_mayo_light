import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

angles = [45, 60, 90, 180]

# --- Data: SSIM ---
ssim = {
    "FBP":            [0.65, 0.73, 0.84, 0.937],
    "TV":             [0.89, 0.91, 0.927, 0.952],
    "UNet (vs TV)":   [0.956, 0.956, 0.956, 0.980],
    "UNet (vs prep.)":[0.86, 0.912, 0.928, 0.955],
    "PnP":            [0.936, 0.948, 0.958, 0.966],
}
ssim_err = {
    "FBP":            [0.04, 0.04, 0.03, 0.012],
    "TV":             [0.02, 0.02, 0.018, 0.011],
    "UNet (vs TV)":   [0.005, 0.006, 0.006, 0.003],
    "UNet (vs prep.)":[0.03, 0.016, 0.011, 0.008],
    "PnP":            [0.012, 0.009, 0.007, 0.006],
}

# --- Data: PSNR ---
psnr = {
    "FBP":            [29.2, 31.0, 33.8, 36.6],
    "TV":             [34.8, 36.2, 37.7, 39.6],
    "UNet (vs TV)":   [39.3, 39.5, 39.6, 41.9],
    "UNet (vs prep.)":[32.8, 35.8, 37.1, 39.6],
    "PnP":            [37.3, 38.6, 39.9, 41.4],
}
psnr_err = {
    "FBP":            [1.5, 1.5, 1.5, 1.7],
    "TV":             [1.8, 1.8, 1.7, 1.5],
    "UNet (vs TV)":   [0.9, 0.9, 1.0, 1.4],
    "UNet (vs prep.)":[1.4, 1.5, 1.4, 1.4],
    "PnP":            [1.6, 1.4, 1.3, 1.2],
}

colors = {
    "FBP": "gray",
    "TV": "tab:blue",
    "UNet (vs TV)": "tab:red",
    "UNet (vs prep.)": "tab:orange",
    "PnP": "tab:green",
}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

alphas = {
    "FBP": 1.0,
    "TV": 1.0,
    "UNet (vs TV)": 0.25,   # quasi trasparente
    "UNet (vs prep.)": 1.0,
    "PnP": 1.0,
}

# SSIM subplot
for method in ssim:
    axes[0].errorbar(angles, ssim[method], yerr=ssim_err[method],
                      marker="o", capsize=3, label=method,
                      color=colors[method], alpha=alphas[method])
axes[0].set_xlabel("Number of angles")
axes[0].set_ylabel("SSIM")
axes[0].set_title("SSIM vs. number of angles")
axes[0].set_xticks(angles)
axes[0].grid(alpha=0.3)

# PSNR subplot
for method in psnr:
    axes[1].errorbar(angles, psnr[method], yerr=psnr_err[method],
                      marker="o", capsize=3, label=method,
                      color=colors[method], alpha=alphas[method])
axes[1].set_xlabel("Number of angles")
axes[1].set_ylabel("PSNR (dB)")
axes[1].set_title("PSNR vs. number of angles")
axes[1].set_xticks(angles)
axes[1].grid(alpha=0.3)


handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
for lh in leg.legend_handles:
    lh.set_alpha(1.0)

plt.tight_layout()
plt.savefig("psnr_ssim_vs_angles.png", dpi=300, bbox_inches="tight")
plt.show()