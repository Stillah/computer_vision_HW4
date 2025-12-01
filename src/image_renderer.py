import numpy as np
import open3d as o3d
from typing import Iterable, Tuple, Optional, List, Any
from pathlib import Path

class ImageRenderer:
    """Class for visualizing a scene and overlays like planned paths."""

    def plot_path(
        self,
        pcd: o3d.geometry.Geometry,
        path: Iterable[Any],
        show_look_vectors: bool = True,
        show_axes: bool = True,
        waypoint_size: float = 0.08,
        path_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        waypoint_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        look_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        axis_length: float = 1.0,
        window_name: str = "Planned Path Preview",
    ) -> None:
        """Plot the provided path on top of a scene.

        Args:
            pcd: Open3D geometry (typically a PointCloud) of the scene.
            path: Iterable of waypoints; each item can be:
                  - an object with attributes `.pos` (3,) and optional `.look_at` (3,)
                  - a dict with keys 'pos' and optional 'look_at'
                  - a tuple/list (pos,) or (pos, look_at)
                  pos and look_at are 3D coordinates in world space.
            show_look_vectors: If True, draw rays from positions toward look_at.
            waypoint_size: Sphere radius used to mark waypoints.
            path_color: RGB for the path polyline.
            waypoint_color: RGB for waypoint spheres.
            look_color: RGB for look direction rays.
            window_name: Viewer window title.
        """
        positions, look_ats = self._extract_waypoints(path)

        geoms: List[o3d.geometry.Geometry] = []
        geoms.append(pcd)

        # Path polyline
        if len(positions) >= 2:
            pts = o3d.utility.Vector3dVector(np.asarray(positions))
            lines = [[i, i + 1] for i in range(len(positions) - 1)]
            colors = [list(path_color) for _ in lines]
            line_set = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
            line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
            geoms.append(line_set)

        # Waypoint markers
        for p in positions:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=waypoint_size)
            sph.compute_vertex_normals()
            sph.paint_uniform_color(np.asarray(waypoint_color))
            sph.translate(np.asarray(p, dtype=float))
            geoms.append(sph)

        # Look direction vectors (short line segments)
        if show_look_vectors:
            look_pts = []
            look_lines = []
            look_cols = []
            idx = 0
            for pos, la in zip(positions, look_ats):
                if la is None:
                    continue
                pos = np.asarray(pos, dtype=float)
                la = np.asarray(la, dtype=float)
                v = la - pos
                n = np.linalg.norm(v)
                if n < 1e-6:
                    continue
                v = v / n
                end = pos + v * (waypoint_size * 6.0)
                look_pts.append(pos)
                look_pts.append(end)
                look_lines.append([idx, idx + 1])
                look_cols.append(list(look_color))
                idx += 2
            if look_pts:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.asarray(look_pts)),
                    lines=o3d.utility.Vector2iVector(np.asarray(look_lines, dtype=np.int32)),
                )
                ls.colors = o3d.utility.Vector3dVector(np.asarray(look_cols))
                geoms.append(ls)

        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        if show_axes:
            geoms.extend(self._axis_overlays(axis_length))
        # Call draw_geometries with best-effort compatibility across Open3D versions
        vis = getattr(o3d, "visualization", None)
        if vis is not None and hasattr(vis, "draw_geometries"):
            vis.draw_geometries(geoms, window_name=window_name)
        else:
            raise RuntimeError("Open3D visualization API not found. Please update Open3D.")

    def _extract_waypoints(self, path: Iterable[Any]) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        positions: List[np.ndarray] = []
        look_ats: List[Optional[np.ndarray]] = []

        for item in path:
            pos: Optional[np.ndarray] = None
            look: Optional[np.ndarray] = None

            if hasattr(item, "pos"):
                p = getattr(item, "pos")
                pos = np.asarray(p, dtype=float).reshape(3)
                if hasattr(item, "look_at"):
                    la = getattr(item, "look_at")
                    if la is not None:
                        look = np.asarray(la, dtype=float).reshape(3)
            elif isinstance(item, dict):
                if "pos" in item:
                    pos = np.asarray(item["pos"], dtype=float).reshape(3)
                if "look_at" in item and item["look_at"] is not None:
                    look = np.asarray(item["look_at"], dtype=float).reshape(3)
            elif isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    pos = np.asarray(item[0], dtype=float).reshape(3)
                if len(item) >= 2 and item[1] is not None:
                    look = np.asarray(item[1], dtype=float).reshape(3)

            if pos is not None:
                positions.append(pos)
                look_ats.append(look)

        return positions, look_ats

    def plot_walls(
        self,
        wall_points: np.ndarray,
        wall_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        point_size: float = 0.05,
        window_name: str = "Wall Points Visualization",
    ) -> None:
        """Visualize wall points only (without the full scene).
        
        Args:
            wall_points: Numpy array of wall point coordinates (N, 3).
            wall_color: RGB color for wall points.
            point_size: Size of spheres to visualize wall points.
            window_name: Viewer window title.
        """
        geoms: List[o3d.geometry.Geometry] = []
        
        # Create a point cloud for wall points with distinct color
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
        wall_pcd.paint_uniform_color(wall_color)
        geoms.append(wall_pcd)
        
        # Add coordinate frame
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        
        # Visualize
        vis = getattr(o3d, "visualization", None)
        if vis is not None and hasattr(vis, "draw_geometries"):
            vis.draw_geometries(geoms, window_name=window_name)
        else:
            raise RuntimeError("Open3D visualization API not found. Please update Open3D.")

    # ---------------- Axis helpers -----------------
    def _axis_overlays(self, length: float = 1.0) -> List[o3d.geometry.Geometry]:
        """Create arrow meshes and simple line labels for X,Y,Z axes.

        Args:
            length: Arrow length.
        Returns:
            List of Open3D geometry objects.
        """
        geoms: List[o3d.geometry.Geometry] = []
        shaft = length * 0.85
        head = length * 0.15
        r_shaft = length * 0.015
        r_head = length * 0.035

        # Factory to create a fresh arrow each time (avoid .copy())
        def make_arrow() -> o3d.geometry.TriangleMesh:
            arr = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=r_shaft,
                cone_radius=r_head,
                cylinder_height=shaft,
                cone_height=head,
            )
            arr.compute_vertex_normals()
            return arr

        # Rotation matrices
        def rot_x(theta: float) -> np.ndarray:
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def rot_y(theta: float) -> np.ndarray:
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # X arrow (rotate -90 deg about Y)
        x_arrow = make_arrow()
        x_arrow.rotate(rot_y(-np.pi / 2), center=(0, 0, 0))
        x_arrow.paint_uniform_color([1, 0, 0])
        geoms.append(x_arrow)
        geoms.append(self._axis_label('X', np.array([length, 0, 0]), length * 0.12, [1, 0, 0]))

        # Y arrow (rotate +90 deg about X)
        y_arrow = make_arrow()
        y_arrow.rotate(rot_x(np.pi / 2), center=(0, 0, 0))
        y_arrow.paint_uniform_color([0, 1, 0])
        geoms.append(y_arrow)
        geoms.append(self._axis_label('Y', np.array([0, length, 0]), length * 0.12, [0, 1, 0]))

        # Z arrow (original orientation)
        z_arrow = make_arrow()
        z_arrow.paint_uniform_color([0, 0, 1])
        geoms.append(z_arrow)
        geoms.append(self._axis_label('Z', np.array([0, 0, length]), length * 0.12, [0, 0, 1]))
        return geoms

    def _axis_label(self, char: str, position: np.ndarray, size: float, color: List[float]) -> o3d.geometry.LineSet:
        """Create a very simple line-based letter approximation at given position.

        Only supports X,Y,Z with crude strokes.
        """
        p = position.astype(float)
        s = size
        pts: List[List[float]] = []
        lines: List[List[int]] = []
        def add_point(x,y,z):
            pts.append([p[0] + x, p[1] + y, p[2] + z])
            return len(pts)-1
        if char == 'X':
            a = add_point(-s, -s, 0); b = add_point(s, s, 0)
            c = add_point(-s, s, 0); d = add_point(s, -s, 0)
            lines += [[a,b],[c,d]]
        elif char == 'Y':
            a = add_point(0, -s, 0); b = add_point(0, 0, 0)
            c = add_point(-s, s, 0); d = add_point(s, s, 0)
            lines += [[a,b],[c,b],[b,d]]
        elif char == 'Z':
            a = add_point(-s, -s, 0); b = add_point(s, -s, 0)
            c = add_point(-s, s, 0); d = add_point(s, s, 0)
            lines += [[a,b],[b,c],[c,d]]
        else:
            # Fallback: single point
            add_point(0,0,0)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(pts)),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(lines),1)))
        return ls
    


class VideoRenderer:

    def render_video(self, pcd, path: Path):
        pass