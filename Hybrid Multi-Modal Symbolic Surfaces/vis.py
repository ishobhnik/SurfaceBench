import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_point_cloud(npz_path, output_path):
    data = np.load(npz_path)
    x, y, z = data["x"], data["y"], data["z"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.8)

    ax.set_title(os.path.basename(npz_path).replace("_samples.npz", ""))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "Sampled_Data")
    output_dir = os.path.join(base_dir, "Sampled_Data_Visualizations")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".npz"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".npz", ".png"))
            try:
                visualize_point_cloud(input_path, output_path)
            except Exception as e:
                print(f"[Error] {file}: {e}")

if __name__ == "__main__":
    main()
