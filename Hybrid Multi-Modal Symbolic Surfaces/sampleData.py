import numpy as np
import os

equations = [
    {
        "name": "Hybrid_Surface_Case_Switch",
        "expression": lambda x, y: np.where(x < 0, x**2, np.sin(y)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Dual_Domain",
        "expression": lambda x, y: np.where(y < 0, np.log(1 + np.abs(x)), np.exp(-y**2)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Region_Gated",
        "expression": lambda x, y: np.where(x * y > 0, x**2 + np.sin(y), -x**2 - np.cos(y)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Threshold_Logic",
        "expression": lambda x, y: np.where(x > y, np.tanh(x - y), 0),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Contact_Hybrid",
        "expression": lambda x, y: np.abs(x * y) + np.sin(x - y),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Bifurcation_Switch",
        "expression": lambda x, y: np.where(np.abs(x - y) < 1, np.cos(x), np.exp(-x**2)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Parabolic_Oscillatory",
        "expression": lambda x, y: np.where(y > 0, x**2, np.cos(y**2)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Circular_Domain",
        "expression": lambda x, y: np.where(x**2 + y**2 < 1, np.sin(x * y), np.log(1 + x**2)),
        "x_range": [-2, 2],
        "y_range": [-2, 2],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Threshold_Blend",
        "expression": lambda x, y: np.where(x * y < 0, np.tanh(x + y), np.sin(x - y)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    },
    {
        "name": "Hybrid_Surface_Linear_Nonlinear",
        "expression": lambda x, y: np.where(x > y, x, y**2 + np.sin(x)),
        "x_range": [-5, 5],
        "y_range": [-5, 5],
        "n_points": 5000
    }
]

# Output folder
output_dir = "Sampled_Data"
os.makedirs(output_dir, exist_ok=True)

# Sampling settings
for eq in equations:
    x_min, x_max = eq["x_range"]
    y_min, y_max = eq["y_range"]
    n_points = eq["n_points"]

    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)
    z = eq["expression"](x, y)

    path = os.path.join(output_dir, f"{eq['name']}_samples.npz")
    np.savez(path, x=x, y=y, z=z)
    print(f"[Saved] {path}")
