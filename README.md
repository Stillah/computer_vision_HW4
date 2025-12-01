# Computer Vision HW4 - Camera Path Planning and Gaussian Splatting Video Rendering

This project generates cinematic camera paths through 3D Gaussian Splatting scenes and renders smooth fly-through videos.

## Features

- **Automatic Path Planning**: Generates camera paths using ray casting to find points of interest
- **Wall Detection**: Separates walls from objects for intelligent path planning
- **Gaussian Splatting Rendering**: High-quality video rendering using GSplat
- **Look-At Targeting**: Camera orientations calculated from look-at targets for smooth viewing
- **Interpolation**: Smooth path interpolation for fluid camera movement
- **Configurable Parameters**: Adjustable FPS, camera roll, speed, and more

## Project Structure

```
HW4/
├── src/
│   ├── main.py              # Entry point for path generation
│   ├── path_planner.py      # Camera path planning logic
│   ├── detector.py          # Wall/object detection
│   ├── explorer.py          # Scene exploration utilities
│   ├── image_renderer.py    # Image rendering utilities
│   ├── video_render.ipynb   # Jupyter notebook for video rendering
│   └── output/
│       ├── planned_path.json    # Generated camera path
│       └── wall_points.json     # Detected wall points
|       + video examples
├── input-data/              # PLY scene files
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies
- numpy
- open3d
- scipy
- opencv-python
- plyfile
- torch
- gsplat (for rendering)

## Usage

### Step 1: Generate Camera Path

Run the path planner to generate a camera trajectory through your scene:

```bash
python -m src.main
```

This will:
- Load the PLY scene from `input-data/`
- Detect walls and objects
- Generate a camera path using ray casting
- Save the path to `src/output/planned_path.json`

Currently hyperparameters are optimized for the Conference Hall map.

### Step 2: Render Video (Kaggle)

For best performance, use Kaggle with GPU acceleration:

1. **Upload Required Files to Kaggle:**
   - `src/video_render.ipynb` - The rendering notebook
   - `src/output/planned_path.json` - Generated camera path
   - Your (exported) PLY scene file (e.g., Conference Hall)

2. **Configure Kaggle Dataset:**
   - Create a dataset with your PLY file
   - Create another dataset with `planned_path.json`
   - Link both datasets to your notebook

3. **Update Paths in Notebook:**
   ```python
   # In video_render.ipynb, update these paths:
   Config.INPUT_PLY_PATH = "/kaggle/input/your-dataset/Hall_exported.ply"
   Config.JSON_PATH = "/kaggle/input/your-dataset/planned_path.json"

   ```

4. **Run the Notebook:**
   - Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
   - Run all cells
   - Video will be saved to `output_results/cinematic_tour.mp4`

## Configuration

### Path Planning Parameters

Edit in `src/main.py` or when creating the `PathPlanner`:

```python
planner = PathPlanner(
    vertical_axis=1,              # Y-axis is up (0=X, 1=Y, 2=Z)
    fixed_vertical_value=-1.0,    # Camera height
    target_num_points=37,          # Number of waypoints
    start_position=np.array([0.0, -1.0, 0.0]),
    max_distance=0.5,              # Ray casting max distance
    step_size=0.01,                # Ray marching step
    default_look_at=None           # Optional: fixed look-at target
)
```

### Video Rendering Parameters

Edit in `video_render.ipynb` Config class:

```python
VIDEO_FPS = 60                    # Frame rate (30, 60, 120)
RENDER_WIDTH = 1280               # Video width
RENDER_HEIGHT = 720               # Video height
FOV_Y_DEGREES = 60.0              # Field of view
GIMBAL_SMOOTHING_SIGMA = 15.0     # Camera smoothing

# Camera roll (0=no rotation, 90=landscape left, -90=landscape right)
run_rendering_pipeline(camera_roll_degrees=90)
```

### Speed Control

**Adjust camera travel speed:**

```python
# In load_existing_path_pipeline():
interpolation_step=0.1  # Smaller = slower, more frames (0.1 = ~10 points/meter)
interpolation_step=0.2  # 2x faster
interpolation_step=0.5  # 5x faster
```

## Output Files

- `src/output/planned_path.json` - Camera waypoints with positions and look-at targets
- `src/output/wall_points.json` - Detected wall points for visualization
- `output_results/cinematic_tour.mp4` - Rendered video (from Kaggle)
- `output_results/path_vis.ply` - Path visualization point cloud

## Example Workflow

```bash
# 1. Generate path locally
python -m src.main

# 2. Upload to Kaggle:
#    - planned_path.json
#    - Your_Scene.ply
#    - video_render.ipynb

# 3. Run notebook on Kaggle with GPU

# 4. Download rendered video
```

## Troubleshooting

**Path generation fails:**
- Ensure PLY file exists in `input-data/`
- Check that scene has sufficient geometry
- Adjust `max_distance` or `target_num_points` parameters

**Video rendering is slow:**
- Use Kaggle GPU (not CPU)
- Reduce `RENDER_WIDTH` and `RENDER_HEIGHT`
- Increase `interpolation_step` for fewer frames

**Camera flipping/rolling incorrectly:**
- Adjust `camera_roll_degrees` in `run_rendering_pipeline()`
- Check `vertical_axis` matches your scene orientation
- Verify `GLOBAL_UP_VECTOR` is correct for your scene


## Technical Details

### Path Planning Algorithm
1. Downsample point cloud for efficiency
2. Separate walls from objects using vertical slice analysis
3. Cast rays from starting position in all horizontal directions
4. Select furthest collision points as waypoints
5. Sequence waypoints to create smooth path
6. Each waypoint looks at the next position in sequence

### Camera Orientation
- Uses Globally Aligned Frames (GAF) for level horizon
- Look-at targets interpolated alongside positions
- Orientation continuity checks prevent flipping
- Quaternion smoothing for stable motion
- Supports camera roll for landscape/portrait orientation

## Future improvements
- Use better hyperparameters for path planner
- Improve camera view angle
- Add slightly randomized movement to the sides
- Make object detection and adjust camera view based on it
