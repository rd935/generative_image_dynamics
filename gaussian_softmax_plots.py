import numpy as np
import matplotlib.pyplot as plt

def plot_weighting_curves(sigma=1.0, save_path="gaussian_vs_softmax.png"):
    # Metric values (importance metric m)
    m = np.linspace(-5, 5, 400)

    # "Softmax" weight: exp(m) (before normalization across pixels)
    w_softmax = np.exp(m)

    # Gaussian weight: exp(-m^2 / (2 * sigma^2))
    w_gaussian = np.exp(-(m**2) / (2 * sigma**2))

    # Normalize both to [0, 1] over this range so shapes are comparable
    w_softmax_norm = w_softmax / w_softmax.max()
    w_gaussian_norm = w_gaussian / w_gaussian.max()  # already max=1, but for symmetry

    plt.figure(figsize=(8, 5))
    plt.plot(m, w_softmax_norm, label="Softmax weight ~ exp(m)")
    plt.plot(m, w_gaussian_norm, label=r"Gaussian weight ~ exp(-m² / 2σ²)")
    plt.xlabel("Metric value m")
    plt.ylabel("Normalized weight")
    plt.title("Gaussian vs Softmax Weighting (Normalized Shapes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    plot_weighting_curves(sigma=1.0, save_path="gaussian_vs_softmax.png")
