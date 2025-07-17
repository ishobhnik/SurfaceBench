import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_and_save_surface(npz_path, output_path):
    data = np.load(npz_path)
    x, y, z = data['x'], data['y'], data['z']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_title(os.path.basename(npz_path))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    
    # Save to output_path
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "Grid_Sampled_Data")
    output_dir = os.path.join(base_dir, "Grid_Sampled_Visualizations")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".npz"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".npz", ".png"))
            try:
                visualize_and_save_surface(input_path, output_path)
            except Exception as e:
                print(f"[Error] {file}: {e}")

if __name__ == "__main__":
    main()
