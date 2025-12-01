from src.image_renderer import ImageRenderer
from path_planner import PathPlanner
import open3d as o3d
import json
from pathlib import Path
import numpy as np

renderer = ImageRenderer()
wall_points_file = Path(__file__).parent / 'output' / 'wall_points.json'
if wall_points_file.exists():
    with open(wall_points_file, 'r') as f:
        wall_points_list = json.load(f)
    wall_points = np.array(wall_points_list)
    print(f"Loaded {len(wall_points)} wall points from: {wall_points_file}")

    # Visualize walls (optional - uncomment to see walls separately)
    renderer.plot_walls(wall_points, wall_color=(1.0, 0.0, 0.0))