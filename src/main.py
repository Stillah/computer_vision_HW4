import json
from pathlib import Path

import numpy as np
import open3d as o3d

from src.path_planner import PathPlanner, CameraWaypoint
from src.image_renderer import ImageRenderer
from typing import List


def save_path_json(path: List[CameraWaypoint], out_path: Path) -> None:
    data = [
        {
            "pos": wp.pos.tolist() if isinstance(wp.pos, np.ndarray) else list(wp.pos),
            "look_at": wp.look_at.tolist() if isinstance(wp.look_at, np.ndarray) else list(wp.look_at),
        }
        for wp in path
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    # Change one of these to pick your scene
    ply_map_path = 'input-data/Hall_exported.ply'

    ply_path = Path(ply_map_path)
    
    pcd = o3d.io.read_point_cloud(str(ply_path))

    # Adjust hyperparameters for proper path finding in different scenes
    planner = PathPlanner(vertical_axis=1, fixed_vertical_value=-1.4, target_num_points=37,
                          start_position=np.array([0.0, -1.4, 0.0]),
                          max_distance = 2.50,
                          step_size=0.05)
    waypoints = planner.get_path(pcd)

    # Save path for downstream renderer (JSON with pos and look_at per waypoint)
    out = Path(__file__).parent / 'output' / 'planned_path.json'
    save_path_json(waypoints, out)

    print(f"Planned {len(waypoints)} waypoints. Saved to: {out}")

    renderer = ImageRenderer()
    
    # Load and visualize wall points
    wall_points_file = Path(__file__).parent / 'output' / 'wall_points.json'
    if wall_points_file.exists():
        with open(wall_points_file, 'r') as f:
            wall_points_list = json.load(f)
        wall_points = np.array(wall_points_list)
        print(f"Loaded {len(wall_points)} wall points from: {wall_points_file}")
        
        # Create wall point cloud for visualization
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
        wall_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color for walls
        
        # Visualize the path with walls instead of full point cloud
        renderer.plot_path(wall_pcd, waypoints, show_axes=True, axis_length=10.0)
    else:
        print(f"Warning: Wall points file not found at {wall_points_file}")
        print("Falling back to full point cloud visualization")
        renderer.plot_path(pcd, waypoints, show_axes=True, axis_length=10.0)