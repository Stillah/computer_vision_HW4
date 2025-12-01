import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


@dataclass
class CameraWaypoint:
    """Represents a single camera waypoint.

    pos: (x,y,z) camera position in world coordinates
    look_at: (x,y,z) target point to look at
    """
    pos: np.ndarray
    look_at: np.ndarray


class PathPlanner:
    """Plans a camera path by ray casting from a starting position."""

    def __init__(self, vertical_axis: int = 1, fixed_vertical_value: float = -1.0,
                 target_num_points: int = 37,
                 start_position: np.ndarray = np.array([0.0, -1.0, 0.0]),
                 max_distance = 0.5,
                 step_size = 0.01,
                 default_look_at: Optional[np.ndarray] = None):
        """Initialize the path planner.
        
        Args:
            vertical_axis: Which axis is vertical (0=X, 1=Y, 2=Z). Default is 1 (Y).
            fixed_vertical_value: The fixed value for camera height on the vertical axis.
            default_look_at: If provided, all waypoints will use this as their look_at target instead of computed ones.
        """
        self.vertical_axis = vertical_axis
        self.fixed_vertical_value = fixed_vertical_value
        self.target_num_points = target_num_points
        self.start_pos = start_position
        self.max_distance = max_distance
        self.step_size = step_size
        self.default_look_at = default_look_at

    def get_path(self, 
                 pcd: o3d.geometry.PointCloud) -> List[CameraWaypoint]:
        """Compute a camera path by ray casting to find points of interest.

        Args:
            pcd: Open3D point cloud loaded from a PLY scene.

        Returns:
            List of CameraWaypoint entries describing camera motion and gaze.
        """
        # Downsample the point cloud for faster processing
        print(f'Original point cloud has {len(pcd.points)} points')
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
        print(f'Downsampled to {len(downsampled_pcd.points)} points')
        
        points = np.asarray(downsampled_pcd.points)
        if len(points) == 0:
            return []
        
        print('Separating walls from scene...')
        wall_points, objects = self._separate_walls_and_objects(points)
        self._save_wall_points(wall_points)
        # self._save_wall_points(objects) # save object points for visualization if you want
        
        # Find points of interest using ray casting from starting position
        print('Finding points of interest using ray casting...')
        interest_points = self._find_interest_points_by_raycasting(
            points, wall_points # change wall_points to objects if need to avoid objects or walls are not found
        )
        
        if len(interest_points) == 0:
            # Fallback: create a simple circular path around the scene center
            waypoints = self._create_fallback_path(points)
        else:
            # Convert interest points directly to waypoints
            print(f'Creating path from {len(interest_points)} interest points...')
            waypoints = self._convert_interest_points_to_waypoints(interest_points)
        
        return waypoints
    
    def _separate_walls_and_objects(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate wall points from object points using vertical slice analysis.
        
        Uses a vertical slice algorithm to identify walls as points that appear in
        the same horizontal position across multiple height levels.
        
        Args:
            points: All points in the scene (N, 3)
            
        Returns:
            wall_points: Points that are likely walls
            object_points: Points that are likely objects
        """
        # Get horizontal axes (not the vertical axis)
        h_axes = [i for i in range(3) if i != self.vertical_axis]
        
        # Get vertical and horizontal coordinates
        v_coords = points[:, self.vertical_axis]
        h_coords = points[:, h_axes]  # Shape: (N, 2)
        
        # Create vertical slices
        v_min, v_max = v_coords.min(), v_coords.max()
        v_range = v_max - v_min
        num_slices = 20
        slice_thickness = v_range / num_slices
        
        # Grid resolution for horizontal position
        h_grid_size = 0.2
        
        # Count how many slices each horizontal cell appears in
        horizontal_cell_counts = {}
        
        for slice_idx in range(num_slices):
            slice_min = v_min + slice_idx * slice_thickness
            slice_max = slice_min + slice_thickness
            
            # Get points in this slice
            in_slice = (v_coords >= slice_min) & (v_coords < slice_max)
            slice_points_h = h_coords[in_slice]
            
            if len(slice_points_h) == 0:
                continue
            
            # Discretize horizontal coordinates into grid cells
            h_mins = h_coords.min(axis=0)
            grid_cells = ((slice_points_h - h_mins) / h_grid_size).astype(int)
            
            # Get unique cells in this slice
            unique_cells = set(map(tuple, grid_cells))
            
            # Increment count for each cell
            for cell in unique_cells:
                horizontal_cell_counts[cell] = horizontal_cell_counts.get(cell, 0) + 1
        
        # Mark points as walls if their horizontal position appears in many slices
        wall_threshold = num_slices * 0.35
        wall_mask = np.zeros(len(points), dtype=bool)
        h_mins = h_coords.min(axis=0)
        
        for i, h_coord in enumerate(h_coords):
            cell = tuple(((h_coord - h_mins) / h_grid_size).astype(int))
            if cell in horizontal_cell_counts and horizontal_cell_counts[cell] >= wall_threshold:
                wall_mask[i] = True
        
        # Additional pass: mark points near existing walls as also being walls
        if np.any(wall_mask):
            try:
                wall_temp = points[wall_mask]
                tree = KDTree(wall_temp)
                
                non_wall_indices = np.where(~wall_mask)[0]
                if len(non_wall_indices) > 0:
                    non_wall_points = points[~wall_mask]
                    distances, _ = tree.query(non_wall_points, k=1)
                    
                    # Mark points within 0.15m of detected walls as also being walls
                    close_to_wall = distances < 0.15
                    wall_mask[non_wall_indices[close_to_wall]] = True
            except ImportError:
                pass
        
        wall_points = points[wall_mask]
        object_points = points[~wall_mask]
        
        print(f"Detected {len(wall_points)} wall points and {len(object_points)} object points")
        
        return wall_points, object_points
    
    def _save_wall_points(self, wall_points: np.ndarray) -> None:
        """Save wall points to a JSON file for visualization."""
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'wall_points.json'
        
        wall_points_list = wall_points.tolist()
        
        with open(output_file, 'w') as f:
            json.dump(wall_points_list, f, indent=2)
        
        print(f"Saved {len(wall_points)} wall points to: {output_file}")
    
    def _find_interest_points_by_raycasting(
        self, 
        all_points: np.ndarray, 
        wall_points: np.ndarray,
        num_rays: int = 32,
    ) -> List[Tuple[np.ndarray, float]]:
        """Find points of interest by iteratively casting rays and choosing furthest collision.
        
        Starting from (0, -1, 0), cast rays in all directions. Pick the furthest collision
        as the next point of interest, then repeat from that point until we have enough points.
        
        Args:
            all_points: All scene points
            wall_points: Points representing walls
            num_rays: Number of rays to cast in different directions
            
        Returns:
            List of tuples (look_at_point, distance) for interesting viewing directions
        """
        if len(wall_points) == 0:
            return []
        
        # First point is always (0, -1, 0)
        current_pos = self.start_pos.copy() if self.start_pos is not None else np.array([0.0, -1.0, 0.0])
        
        # Get horizontal axes
        h_axes = [i for i in range(3) if i != self.vertical_axis]
        
        # Build KDTree for wall points for faster ray casting
        try:
            wall_tree = KDTree(wall_points)
        except ImportError:
            wall_tree = None
        
        # Sequence of interest points
        interest_points = []
        target_num_points = self.target_num_points
        
        # Keep track of visited positions to avoid revisiting
        visited_positions = [current_pos.copy()]
        
        print(f"Starting ray casting sequence from {current_pos}...")
        
        while len(interest_points) < target_num_points - 1:
            print(f"\nIteration {len(interest_points) + 1}: Casting rays from {current_pos}")
            
            # Cast rays in all directions from current position
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
            ray_collisions = []
            
            for angle in angles:
                # Create ray direction (horizontal plane only)
                ray_dir = np.zeros(3)
                ray_dir[h_axes[0]] = np.cos(angle)
                ray_dir[h_axes[1]] = np.sin(angle)
                ray_dir[self.vertical_axis] = 0
                
                # Cast ray to find wall intersection
                max_distance = self.max_distance
                step_size = self.step_size
                
                hit_distance = None
                hit_point = None
                
                # March along the ray until we hit a wall
                for t in np.arange(0.5, max_distance, step_size):
                    ray_point = current_pos + t * ray_dir
                    
                    # Check if this point is close to a wall
                    if wall_tree is not None:
                        distance, _ = wall_tree.query(ray_point, k=1)
                        if distance < 0.3:
                            hit_distance = t
                            hit_point = ray_point
                            break
                    else:
                        distances = np.linalg.norm(wall_points - ray_point, axis=1)
                        if np.min(distances) < 0.3:
                            hit_distance = t
                            hit_point = ray_point
                            break
                
                # If no collision, use the furthest point of the ray
                if hit_point is None:
                    hit_distance = max_distance
                    hit_point = current_pos + max_distance * ray_dir
                
                ray_collisions.append((hit_point, hit_distance))
            
            # Choose the furthest collision point
            ray_collisions.sort(key=lambda x: x[1], reverse=True)
            
            # Find the furthest point that is not too close to already visited positions
            best_point = None
            best_distance = 0
            
            for collision_point, collision_dist in ray_collisions:
                # Check if this point is far enough from visited positions
                too_close = any(
                    np.linalg.norm(collision_point - visited_pos) < 1.0 
                    for visited_pos in visited_positions
                )
                
                if not too_close:
                    best_point = collision_point
                    best_distance = collision_dist
                    break
            
            # If all points are too close, just take the furthest one anyway
            if best_point is None and len(ray_collisions) > 0:
                best_point, best_distance = ray_collisions[0]
            
            # If we found a valid point, add it to our sequence
            if best_point is not None:
                interest_points.append((best_point, best_distance))
                visited_positions.append(best_point.copy())
                current_pos = best_point.copy()
                print(f"  Found point at distance {best_distance:.2f}m")
            else:
                print("  No more valid points found, stopping")
                break
        
        # Add the starting position as the first point to look at
        if len(interest_points) > 0:
            start_look_at = interest_points[0][0]
            start_distance = np.linalg.norm(start_look_at - visited_positions[0])
            interest_points.insert(0, (start_look_at, start_distance))
        
        print(f"\nFound {len(interest_points)} interest points in sequence")
        
        return interest_points
    
    def _convert_interest_points_to_waypoints(
        self, 
        interest_points: List[Tuple[np.ndarray, float]]
    ) -> List[CameraWaypoint]:
        """Convert the sequence of interest points into camera waypoints.
        
        Each waypoint is positioned at an interest point and looks at the next point in the sequence.
        
        Args:
            interest_points: List of (point, distance) tuples from ray casting
            
        Returns:
            List of camera waypoints
        """
        if len(interest_points) == 0:
            return []
        
        # Extract just the positions from interest_points
        positions = [point for point, _ in interest_points]
        
        # Create waypoints where each position looks at the next position
        waypoints = []
        for i in range(len(positions)):
            current_pos = positions[i]
            # Look at the next position in sequence (wrap around for the last point)
            next_idx = (i + 1) % len(positions)
            look_at = positions[next_idx]
            
            # Override with default_look_at if provided
            if self.default_look_at is not None:
                look_at = self.default_look_at
            
            waypoints.append(CameraWaypoint(pos=current_pos, look_at=look_at))
        
        print(f"Created {len(waypoints)} waypoints from interest points")
        
        return waypoints
    
    def _create_fallback_path(self, points: np.ndarray, num_waypoints: int = 8) -> List[CameraWaypoint]:
        """Create a simple circular path around the scene center as fallback.
        
        Args:
            points: All scene points
            num_waypoints: Number of waypoints to create
            
        Returns:
            List of waypoints forming a circular path
        """
        center = points.mean(axis=0)
        
        # Calculate radius based on scene size
        h_axes = [i for i in range(3) if i != self.vertical_axis]
        radius = np.std(points[:, h_axes]) * 1.5
        
        waypoints = []
        angles = np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)
        
        for angle in angles:
            pos = center.copy()
            pos[h_axes[0]] += radius * np.cos(angle)
            pos[h_axes[1]] += radius * np.sin(angle)
            pos[self.vertical_axis] = self.fixed_vertical_value
            
            # Override with default_look_at if provided
            look_at = self.default_look_at if self.default_look_at is not None else center
            
            waypoints.append(CameraWaypoint(pos=pos, look_at=look_at))
        
        return waypoints
