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
        evaluation_points (array): X-coordinates of evaluation points.
        u_true (array): Analytical solution.
        u_fdm (array): FDM solution.
        u_pinn (array): PINN solution.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    axes[0].plot(evaluation_points, u_true, color='b', linestyle='-', linewidth=2)
    axes[0].set_title("Analytical Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True)

    axes[1].plot(evaluation_points, u_fdm, color='g', linestyle='-', linewidth=2)
    axes[1].set_title("FDM Solution")
    axes[1].set_xlabel("x")
    axes[1].grid(True)

    axes[2].plot(evaluation_points, u_pinn, color='r', linestyle='-', linewidth=2)
    axes[2].set_title("PINN Solution")
    axes[2].set_xlabel("x")
    axes[2].grid(True)

    fig.text(0.5, -0.05, "Plot of the Analytical, FDM, and PINN Solutions", ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    """Main function to load data and generate plots."""
    pinn_file = "results/pinn_1d_poisson.json"
    fdm_file = "results/fdm_1d_poisson.json"

    pinn_data = load_json(pinn_file)
    fdm_data = load_json(fdm_file)

    evaluation_points = np.array(pinn_data['evaluation_points']['0']).flatten()
    u_true = np.array(pinn_data['u_true'])
    u_pinn = np.array(pinn_data['u_pinn']['0']).flatten()
    u_fdm = np.array(fdm_data['u_fdm']) if 'u_fdm' in fdm_data else None

    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "poisson1D_solutions.png")    
    
    # Plot and save the results
    plot_solutions(
        evaluation_points,
        u_true,
        u_fdm,
        u_pinn,
        save_path=output_path
    )

if __name__ == "__main__":
    main()