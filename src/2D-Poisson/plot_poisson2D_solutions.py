import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def plot_solutions(evaluation_points, u_true, u_fdm, u_pinn, save_path=None):
    """
    Plot Analytical, FDM, and PINN solutions.

    Args:
        evaluation_points (array): Coordinates of evaluation points.
        u_true (array): Analytical solution.
        u_fdm (array): FDM solution.
        u_pinn (array): PINN solution.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot FDM solution
    c1 = axes[0].pcolormesh(
        evaluation_points[:, 0].reshape(1000, 1000),
        evaluation_points[:, 1].reshape(1000, 1000),
        u_fdm, shading='auto', cmap='viridis'
    )
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("FDM Solution")

    # Plot PINN solution
    c2 = axes[1].pcolormesh(
        evaluation_points[:, 0].reshape(1000, 1000),
        evaluation_points[:, 1].reshape(1000, 1000),
        u_pinn, shading='auto', cmap='viridis'
    )
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("PINN Solution")

    # Plot Analytical solution
    c3 = axes[2].pcolormesh(
        evaluation_points[:, 0].reshape(1000, 1000),
        evaluation_points[:, 1].reshape(1000, 1000),
        u_true, shading='auto', cmap='viridis'
    )
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title("Analytical Solution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()

def main():
    """Main function to load data and generate plots."""
    # File paths
    pinn_file = "results/pinn_2d_poisson.json"
    fdm_file = "results/fdm_2d_poisson.json"

    # Load data
    pinn_data = load_json(pinn_file)
    fdm_data = load_json(fdm_file)

    # Extract data
    evaluation_points = np.array(pinn_data["evaluation_points"])
    u_true = np.array(pinn_data["u_true"]).reshape(1000, 1000)
    u_pinn = np.array(pinn_data["u_pinn"]).reshape(1000, 1000)
    u_fdm = np.array(fdm_data["u_fdm"]).reshape(1000, 1000)

    # Output directory
    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "poisson2D_solutions.png")

    # Plot and save the results
    plot_solutions(evaluation_points, u_true, u_fdm, u_pinn, save_path)

if __name__ == "__main__":
    main()
    