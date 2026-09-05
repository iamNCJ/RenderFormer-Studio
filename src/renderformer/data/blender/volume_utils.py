"""Volume utility functions for generating and reading VDB files."""
import json
import math
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
try:
    import openvdb as vdb
except ImportError:
    vdb = None
import trimesh


def _require_openvdb() -> None:
    if vdb is None:
        raise RuntimeError(
            "Python OpenVDB bindings are required for volume scenes. Use "
            "`conda env create -f environments/datagen.yml` and complete its "
            "documented pip steps, or use the dependency-only datagen image "
            "from "
            "docker/datagen.Dockerfile; the portable pip extras do not "
            "provide OpenVDB bindings."
        )


def _extract_dense_from_grid(
    density_grid: Any,
    min_coord: Tuple[int, int, int],
    max_coord: Tuple[int, int, int],
) -> np.ndarray:
    """Extract dense values from a grid inside [min_coord, max_coord] (inclusive)."""
    dims = [max_coord[i] - min_coord[i] + 1 for i in range(3)]
    dense = np.zeros(dims, dtype=np.float32)
    accessor = density_grid.getConstAccessor()
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                coord = (min_coord[0] + i, min_coord[1] + j, min_coord[2] + k)
                if accessor.isValueOn(coord):
                    dense[i, j, k] = accessor.getValue(coord)
    return dense


def _load_density_grid(input_vdb_path: str, grid_name: str) -> Tuple[Any, List[str]]:
    """Load VDB file and return the requested grid plus available grid names."""
    _require_openvdb()
    result = vdb.readAll(str(input_vdb_path))
    grids_list = result[0] if isinstance(result, tuple) else result

    density_grid = None
    available_names: List[str] = []
    for g in grids_list:
        if hasattr(g, "name"):
            available_names.append(g.name)
            if g.name == grid_name:
                density_grid = g
                break

    if density_grid is None:
        raise ValueError(
            f"Grid '{grid_name}' not found. Available grids: {available_names}"
        )
    return density_grid, available_names


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_lowfreq_density(res=32, seed=0, freq=2.0, tau0=0.35, tau1=0.65, alpha=1.0,
                         use_sphere_falloff=True, falloff_start=0.75, falloff_end=1.0):
    """
    Generates a smooth, low-frequency density field in [-0.5,0.5]^3 on cell centers.
    Returns float32 array of shape [res,res,res], values roughly in [0,1].
    """
    rng = np.random.default_rng(seed)

    lin = (np.arange(res) + 0.5) / res - 0.5
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

    # Low-frequency procedural field: sum of a few random-direction sinusoids
    # (avoids requiring Perlin/Simplex libs, yet stays smooth/band-limited)
    # Use random number of waves for more variation
    n_waves = rng.integers(3, 8)
    val = np.zeros_like(X, dtype=np.float32)
    for _ in range(n_waves):
        v = rng.normal(size=3)
        v /= (np.linalg.norm(v) + 1e-8)
        phase = rng.uniform(0, 2 * math.pi)
        # Wider amplitude range for more variation
        amp = rng.uniform(0.3, 1.2)

        dot = X * v[0] + Y * v[1] + Z * v[2]
        val += (amp * np.sin(2 * math.pi * freq * dot + phase)).astype(np.float32)

    # Normalize, but keep the distribution characteristics by using percentile-based normalization
    # This way different random seeds will have different density distributions
    p_min = rng.uniform(0.05, 0.15)  # Random lower percentile
    p_max = rng.uniform(0.85, 0.95)  # Random upper percentile
    v_min = np.percentile(val, p_min * 100)
    v_max = np.percentile(val, p_max * 100)
    val = np.clip((val - v_min) / (v_max - v_min + 1e-8), 0.0, 1.0)

    # soft threshold
    dens = smoothstep(tau0, tau1, val)

    # optional spherical falloff to keep volume compact
    if use_sphere_falloff:
        R = np.sqrt(X * X + Y * Y + Z * Z) / (math.sqrt(3) * 0.5)
        fall = smoothstep(falloff_start, falloff_end, R)  # 0 center -> 1 boundary
        dens = dens * (1.0 - fall)

    dens = (alpha * dens).astype(np.float32)
    return dens


def write_vdb_from_array(arr_xyz, out_path, grid_name="density", voxel_size=1.0, transpose=True, origin_offset=None):
    """
    Writes arr_xyz (numpy [X,Y,Z]) as a FloatGrid to .vdb.
    Centers the volume at world origin (0,0,0).
    Axis conventions can be confusing; by default we transpose to (Z,Y,X) which often
    aligns better with typical DCC expectations. If the volume looks rotated, flip transpose.
    
    Args:
        arr_xyz: Array of shape (X, Y, Z)
        out_path: Output VDB file path
        grid_name: Name of the grid
        voxel_size: Size of each voxel in world units
        transpose: Whether to transpose (X,Y,Z)->(Z,Y,X)
        origin_offset: DEPRECATED - kept for backward compatibility but ignored.
                      The volume is now always centered at origin using postTranslate.
    """
    _require_openvdb()
    a = arr_xyz
    if transpose:
        a = np.transpose(a, (2, 1, 0)).copy()  # (X,Y,Z)->(Z,Y,X)

    grid = vdb.FloatGrid()
    grid.name = grid_name

    # Copy array data
    grid.copyFromArray(a)

    # Create transform with voxel size
    transform = vdb.createLinearTransform(voxelSize=float(voxel_size))
    
    # Calculate translation to center the volume at origin
    # After copyFromArray, the grid spans from (0,0,0) to (res-1, res-1, res-1) in index space
    shape = np.array(a.shape)
    
    # Center in index space (use tuple/list instead of Vec3d)
    index_center = (
        (shape[0] - 1) / 2.0,
        (shape[1] - 1) / 2.0,
        (shape[2] - 1) / 2.0
    )
    
    # Convert to world space
    world_center = transform.indexToWorld(index_center)
    
    # Translate to move center to origin (negate each component)
    transform.postTranslate((-world_center[0], -world_center[1], -world_center[2]))
    
    # Apply the transform to the grid
    grid.transform = transform

    # Write to disk
    vdb.write(str(out_path), grids=[grid])


def read_vdb_density(vdb_path, grid_name="density") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read density values from VDB file using openvdb.
    Returns density array and metadata.
    """
    _require_openvdb()

    # Read VDB file
    try:
        grids = vdb.readAll(str(vdb_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read VDB file: {e}")
    
    # Handle different return types from readAll()
    # It might return a list of grids, or nested lists
    if not grids:
        raise ValueError("No grids found in VDB file")
    
    # Find the density grid
    density_grid = None
    for item in grids:
        # Handle case where item might be a list/tuple containing the grid
        if isinstance(item, (list, tuple)) and len(item) > 0:
            grid = item[0]  # Extract grid from list/tuple
        else:
            grid = item
        
        # Check if grid has a name attribute and matches
        if hasattr(grid, 'name') and getattr(grid, 'name', None) == grid_name:
            density_grid = grid
            break
    
    if density_grid is None:
        # Collect available grid names for error message
        available_grids = []
        for item in grids:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                g = item[0]
            else:
                g = item
            if hasattr(g, 'name'):
                grid_name_attr = getattr(g, 'name', None)
                if grid_name_attr is not None:
                    available_grids.append(str(grid_name_attr))
                else:
                    available_grids.append(f"<{type(g).__name__}>")
            else:
                available_grids.append(f"<{type(g).__name__}>")
        raise ValueError(f"Grid '{grid_name}' not found in VDB file. Available grids: {available_grids}")
    
    # Ensure density_grid is actually a grid object (not a list/tuple)
    if isinstance(density_grid, (list, tuple)):
        raise RuntimeError(f"Internal error: density_grid is a {type(density_grid).__name__}, expected a grid object")
    if not hasattr(density_grid, 'transform'):
        raise RuntimeError(f"Internal error: density_grid does not have expected grid attributes")
    
    # Get transform
    transform = density_grid.transform
    voxel_size = transform.voxelSize()[0]  # Assuming uniform voxel size
    
    # Get active voxel values
    # We need to iterate through active voxels and build a dense array
    # First, get the bounding box
    bbox = density_grid.evalActiveVoxelBoundingBox()
    if bbox is None:
        raise ValueError("No active voxels in density grid")
    
    # Handle different bbox return types (tuple or object with min/max methods)
    if isinstance(bbox, tuple):
        # bbox is a tuple, likely (min_coord, max_coord) or similar
        if len(bbox) >= 2:
            min_coord = bbox[0]
            max_coord = bbox[1]
        else:
            raise ValueError(f"Unexpected bbox tuple structure: {bbox}")
    elif hasattr(bbox, 'min') and hasattr(bbox, 'max'):
        # bbox is an object with min() and max() methods
        min_coord = bbox.min()
        max_coord = bbox.max()
    else:
        raise ValueError(f"Unexpected bbox type: {type(bbox)}")
    
    # Get dimensions
    dims = [max_coord[i] - min_coord[i] + 1 for i in range(3)]
    
    # Create dense array
    density_array = np.zeros(dims, dtype=np.float32)
    
    # Fill array with active voxel values
    accessor = density_grid.getConstAccessor()
    active_count = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                coord = (min_coord[0] + i, min_coord[1] + j, min_coord[2] + k)
                if accessor.isValueOn(coord):
                    density_array[i, j, k] = accessor.getValue(coord)
                    active_count += 1
    
    # Calculate world space bbox using transform
    # The VDB file is created with postTranslate to center the volume at origin
    # Convert index space coordinates to world space using the transform
    min_world = transform.indexToWorld(min_coord)
    max_world = transform.indexToWorld(max_coord)
    
    # Also calculate the center of the bbox in world space to verify it's at origin
    # The center should be at (0, 0, 0) if the volume was properly centered
    center_index = (
        (min_coord[0] + max_coord[0]) / 2.0,
        (min_coord[1] + max_coord[1]) / 2.0,
        (min_coord[2] + max_coord[2]) / 2.0
    )
    center_world = transform.indexToWorld(center_index)
    
    metadata = {
        'voxel_size': voxel_size,
        'min_coord': min_coord,
        'max_coord': max_coord,
        'dims': dims,
        'bbox_min': [min_world[i] for i in range(3)],
        'bbox_max': [max_world[i] for i in range(3)],
        'center_world': [center_world[i] for i in range(3)],  # Should be close to (0,0,0)
    }
    
    return density_array, metadata


def preprocess_external_vdb(
    input_vdb_path: str,
    output_vdb_path: str,
    target_res: int = 32,
    grid_name: str = "density",
    transpose: bool = True,
    normalize_values: bool = True,
    pad_ratio: float = 0.0,
    verbose: bool = True,
    fixed_bbox_min_coord: Optional[Tuple[int, int, int]] = None,
    fixed_bbox_max_coord: Optional[Tuple[int, int, int]] = None,
    global_value_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Preprocess an external VDB file (e.g. from Mantaflow/Houdini/Blender) to match
    our pipeline's canonical format:
      - Single 'density' grid
      - Resolution: target_res³ (default 32³)
      - Centered at world origin (0,0,0)
      - World space bbox: [-0.5, 0.5]³
      - voxel_size = 1.0 / target_res
      - Values normalized to [0, 1]
    
    Steps:
      1. Read the density grid from the input VDB
      2. Extract the active voxel bounding box as a dense array
      3. Pad to cubic (longest axis) if non-cubic
      4. Downsample/upsample to target_res³ via trilinear interpolation
      5. Optionally normalize values to [0, 1]
      6. Write out using write_vdb_from_array (centered at origin)
    
    Args:
        input_vdb_path: Path to the input VDB file
        output_vdb_path: Path to write the processed VDB file
        target_res: Target resolution per axis (default 32)
        grid_name: Name of the grid to extract (default "density")
        transpose: Whether to transpose when writing (should match pipeline, default True)
        normalize_values: Whether to normalize density values to [0, 1]
        pad_ratio: Extra padding around active region as a fraction of bbox size (0.0 = tight crop)
        verbose: Whether to print progress info
    
    Returns:
        Dict with preprocessing metadata
    """
    from scipy.ndimage import zoom as ndimage_zoom

    if verbose:
        print(f"Preprocessing: {input_vdb_path}")

    # --- Step 1: Read grids ---
    density_grid, _ = _load_density_grid(input_vdb_path, grid_name)

    # --- Step 2: Extract active region as dense array ---
    source_bbox = density_grid.evalActiveVoxelBoundingBox()
    source_min_coord, source_max_coord = source_bbox[0], source_bbox[1]

    if fixed_bbox_min_coord is not None and fixed_bbox_max_coord is not None:
        min_coord = fixed_bbox_min_coord
        max_coord = fixed_bbox_max_coord
        using_fixed_bbox = True
    else:
        min_coord = source_min_coord
        max_coord = source_max_coord
        using_fixed_bbox = False

    dims = [max_coord[i] - min_coord[i] + 1 for i in range(3)]

    if verbose:
        transform = density_grid.transform
        vs = transform.voxelSize()
        print(f"  Input grid: '{grid_name}'")
        print(f"  Voxel size: {vs}")
        print(f"  Active bbox (index): {source_min_coord} -> {source_max_coord}")
        if using_fixed_bbox:
            print(f"  Using fixed bbox (index): {min_coord} -> {max_coord}")
        print(f"  Active dims: {dims}")
        print(f"  Active voxel count: {density_grid.activeVoxelCount()}")

    # Build dense array from active region
    dense = _extract_dense_from_grid(density_grid, min_coord, max_coord)

    if verbose:
        print(f"  Dense array shape: {dense.shape}")
        print(f"  Value range: [{dense.min():.6f}, {dense.max():.6f}]")

    # --- Step 3: Pad to cubic ---
    max_dim = max(dims)
    if pad_ratio > 0:
        pad_voxels = int(max_dim * pad_ratio)
        max_dim += 2 * pad_voxels
    else:
        pad_voxels = 0

    # Symmetric padding to make cubic
    pad_widths = []
    for d in dims:
        total_pad = max_dim - d
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))

    if pad_ratio > 0:
        # Add extra padding on all sides
        pad_widths = [(pw[0] + pad_voxels, pw[1] + pad_voxels) for pw in pad_widths]

    cubic = np.pad(dense, pad_widths, mode='constant', constant_values=0.0)

    if verbose:
        print(f"  After cubic padding: {cubic.shape}")

    # --- Step 4: Resample to target_res³ ---
    current_res = cubic.shape[0]  # cubic, so all dims equal
    if current_res != target_res:
        scale_factor = target_res / current_res
        resampled = ndimage_zoom(cubic, scale_factor, order=1).astype(np.float32)
        # Ensure exact shape (zoom can be off by 1 sometimes)
        if resampled.shape != (target_res, target_res, target_res):
            final = np.zeros((target_res, target_res, target_res), dtype=np.float32)
            s = tuple(min(resampled.shape[i], target_res) for i in range(3))
            final[:s[0], :s[1], :s[2]] = resampled[:s[0], :s[1], :s[2]]
            resampled = final
    else:
        resampled = cubic.astype(np.float32)

    if verbose:
        print(f"  After resample: {resampled.shape}")
        print(f"  Value range: [{resampled.min():.6f}, {resampled.max():.6f}]")

    # --- Step 5: Normalize values to [0, 1] ---
    if normalize_values:
        v_max = global_value_max if global_value_max is not None else float(resampled.max())
        if v_max > 0:
            resampled = resampled / v_max
        # Clamp negatives from interpolation
        resampled = np.clip(resampled, 0.0, 1.0)
        if verbose:
            print(f"  After normalization (max={v_max:.6f}): [{resampled.min():.6f}, {resampled.max():.6f}]")

    # Ensure all voxels are marked active in OpenVDB by setting a tiny epsilon
    # for zero voxels. This guarantees the active bounding box spans the full
    # grid, which is required by read_vdb_to_micro_8_4.
    resampled = np.maximum(resampled, 1e-7).astype(np.float32)

    # --- Step 6: Write using pipeline format ---
    bbox_min_world = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    bbox_max_world = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    voxel_size = float((bbox_max_world[0] - bbox_min_world[0]) / target_res)

    write_vdb_from_array(
        resampled, output_vdb_path,
        grid_name="density",
        voxel_size=voxel_size,
        transpose=transpose,
    )

    if verbose:
        print(f"  Written to: {output_vdb_path}")
        print(f"  Target voxel_size: {voxel_size}")
        print(f"  World bbox: [-0.5, 0.5]³, center at origin")

    # --- Metadata ---
    meta = {
        "source_vdb": str(input_vdb_path),
        "source_grid": grid_name,
        "source_dims": dims,
        "source_voxel_size": float(density_grid.transform.voxelSize()[0]),
        "source_active_count": int(density_grid.activeVoxelCount()),
        "source_value_range": [float(dense.min()), float(dense.max())],
        "source_active_bbox_min": [int(source_min_coord[i]) for i in range(3)],
        "source_active_bbox_max": [int(source_max_coord[i]) for i in range(3)],
        "used_bbox_min": [int(min_coord[i]) for i in range(3)],
        "used_bbox_max": [int(max_coord[i]) for i in range(3)],
        "used_fixed_bbox": using_fixed_bbox,
        "target_res": target_res,
        "voxel_size": voxel_size,
        "bbox_min": bbox_min_world.tolist(),
        "bbox_max": bbox_max_world.tolist(),
        "transpose_written": transpose,
        "normalized": normalize_values,
        "global_value_max": float(global_value_max) if global_value_max is not None else None,
        "pad_ratio": pad_ratio,
    }
    return meta


def _preprocess_worker(args: Tuple[int, int, str, str, int, str, bool, bool, float, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]], Optional[float]]) -> Dict[str, Any]:
    """Top-level worker for multiprocessing (must be pickleable)."""
    (
        idx,
        total,
        in_path,
        out_path,
        target_res,
        grid_name,
        transpose,
        normalize_values,
        pad_ratio,
        global_bbox_min,
        global_bbox_max,
        global_value_max,
    ) = args
    try:
        meta = preprocess_external_vdb(
            in_path,
            out_path,
            target_res=target_res,
            grid_name=grid_name,
            transpose=transpose,
            normalize_values=normalize_values,
            pad_ratio=pad_ratio,
            verbose=False,
            fixed_bbox_min_coord=global_bbox_min,
            fixed_bbox_max_coord=global_bbox_max,
            global_value_max=global_value_max,
        )
        meta["output_vdb"] = out_path
        print(f"[{idx+1}/{total}] OK: {in_path} -> {out_path}", flush=True)
        return meta
    except Exception as e:
        print(f"[{idx+1}/{total}] FAIL: {in_path}: {e}", flush=True)
        return {"source_vdb": in_path, "error": str(e)}


def preprocess_external_vdb_batch(
    input_paths: List[str],
    output_dir: str,
    target_res: int = 32,
    grid_name: str = "density",
    transpose: bool = True,
    normalize_values: bool = True,
    pad_ratio: float = 0.0,
    num_workers: int = 0,
    use_global_bbox: bool = False,
    use_global_normalization: bool = False,
) -> List[Dict[str, Any]]:
    """
    Batch-preprocess multiple external VDB files in parallel.
    
    Args:
        input_paths: List of input VDB file paths
        output_dir: Directory to write processed VDB files
        target_res: Target resolution per axis (default 32)
        grid_name: Name of the grid to extract
        transpose: Whether to transpose when writing
        normalize_values: Whether to normalize density values to [0, 1]
        pad_ratio: Extra padding ratio
        num_workers: Number of parallel workers (0 = auto based on cpu_count)
    
    Returns:
        List of metadata dicts, one per input file
    """
    from multiprocessing import Pool, cpu_count

    if use_global_normalization and not normalize_values:
        raise ValueError("use_global_normalization requires normalize_values=True")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if num_workers <= 0:
        num_workers = min(cpu_count(), len(input_paths))

    global_bbox_min: Optional[Tuple[int, int, int]] = None
    global_bbox_max: Optional[Tuple[int, int, int]] = None
    global_value_max: Optional[float] = None
    if use_global_bbox or use_global_normalization:
        bbox_mins: List[Tuple[int, int, int]] = []
        bbox_maxs: List[Tuple[int, int, int]] = []
        value_max_candidates: List[float] = []

        for input_path in input_paths:
            density_grid, _ = _load_density_grid(input_path, grid_name)
            bbox = density_grid.evalActiveVoxelBoundingBox()
            min_coord, max_coord = bbox[0], bbox[1]
            bbox_mins.append(tuple(int(min_coord[i]) for i in range(3)))
            bbox_maxs.append(tuple(int(max_coord[i]) for i in range(3)))

            if use_global_normalization:
                dense_local = _extract_dense_from_grid(
                    density_grid,
                    tuple(int(min_coord[i]) for i in range(3)),
                    tuple(int(max_coord[i]) for i in range(3)),
                )
                value_max_candidates.append(float(dense_local.max()))

        if use_global_bbox:
            global_bbox_min = tuple(min(p[i] for p in bbox_mins) for i in range(3))
            global_bbox_max = tuple(max(p[i] for p in bbox_maxs) for i in range(3))

        if use_global_normalization:
            global_value_max = max(value_max_candidates) if value_max_candidates else 0.0

    work = []
    total = len(input_paths)
    for idx, in_path in enumerate(input_paths):
        stem = Path(in_path).stem
        out_path = str(out / f"{stem}_preprocessed.vdb")
        work.append(
            (
                idx,
                total,
                in_path,
                out_path,
                target_res,
                grid_name,
                transpose,
                normalize_values,
                pad_ratio,
                global_bbox_min,
                global_bbox_max,
                global_value_max,
            )
        )

    if num_workers == 1 or len(input_paths) == 1:
        results = [_preprocess_worker(w) for w in work]
    else:
        with Pool(num_workers) as pool:
            results = pool.map(_preprocess_worker, work)

    if use_global_bbox or use_global_normalization:
        batch_global_meta = {
            "use_global_bbox": use_global_bbox,
            "use_global_normalization": use_global_normalization,
            "global_bbox_min": list(global_bbox_min) if global_bbox_min is not None else None,
            "global_bbox_max": list(global_bbox_max) if global_bbox_max is not None else None,
            "global_value_max": global_value_max,
        }
        for r in results:
            if "error" not in r:
                r["batch_global"] = batch_global_meta

    return results


def generate_volume_vdb(
    output_dir: str,
    seed: int,
    freq: float = 2.0,
    tau0: float = 0.35,
    tau1: float = 0.65,
    alpha: float = 1.0,
    use_sphere_falloff: bool = True,
    transpose: bool = True,
    scattering_scale: Optional[List[float]] = None,
    absorption_scale: Optional[List[float]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a volume VDB file and return the path and metadata.
    
    Args:
        output_dir: Directory to save the VDB file
        seed: Random seed for volume generation
        freq: Frequency parameter for density generation
        tau0: Lower threshold for smoothstep
        tau1: Upper threshold for smoothstep
        alpha: Alpha multiplier for density
        use_sphere_falloff: Whether to use spherical falloff
        transpose: Whether to transpose when writing VDB
        scattering_scale: Scattering scale for volume shader (RGB, 3 values, optional, will be added to metadata if provided)
        absorption_scale: Absorption scale for volume shader (RGB, 3 values, optional, will be added to metadata if provided)
    
    Returns:
        Tuple of (vdb_path, metadata_dict)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    coarse_res = 8
    micro_per = 4
    micro_res = coarse_res * micro_per  # 32

    # Generate micro density as a normal dense grid 32^3
    micro32 = make_lowfreq_density(
        res=micro_res,
        seed=seed,
        freq=freq,
        tau0=tau0,
        tau1=tau1,
        alpha=alpha,
        use_sphere_falloff=use_sphere_falloff
    )

    # Canonical bbox and voxel size (world space)
    bbox_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    bbox_max = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    voxel_size = float((bbox_max[0] - bbox_min[0]) / micro_res)  # 1/32 world units

    # Write VDB using the 32^3 dense grid
    vdb_path = out / f"volume_{seed}.vdb"
    write_vdb_from_array(micro32, vdb_path, grid_name="density", voxel_size=voxel_size, transpose=transpose)

    # Metadata
    meta = {
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "coarse_res": coarse_res,
        "micro_per_coarse": micro_per,
        "micro_res": micro_res,
        "voxel_size": voxel_size,
        "grid_name": "density",
        "transpose_written": bool(transpose),
        "seed": seed,
    }
    
    # Add scattering and absorption scales to metadata if provided
    if scattering_scale is not None:
        meta["scattering_scale"] = [float(x) for x in scattering_scale]  # RGB values
    if absorption_scale is not None:
        meta["absorption_scale"] = [float(x) for x in absorption_scale]  # RGB values
    
    # Save metadata to JSON file (similar to export_openvdb.py)
    meta_json_path = out / f"volume_{seed}_meta.json"
    with open(meta_json_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return str(vdb_path), meta


def read_vdb_to_micro_8_4(vdb_path: str, transpose_written: bool = True) -> np.ndarray:
    """
    Read VDB file and convert to micro_8_4 format (8, 4, 8, 4, 8, 4).
    
    Args:
        vdb_path: Path to VDB file
        transpose_written: Whether the VDB was written with transpose (needs reverse transpose)
    
    Returns:
        Array of shape (8, 4, 8, 4, 8, 4)
    """
    density_array, metadata = read_vdb_density(vdb_path, grid_name="density")
    
    # VDB files are written as (Z, Y, X) if transpose=True, so we need to reverse transpose
    if transpose_written:
        # Reverse transpose: (Z, Y, X) -> (X, Y, Z)
        density_array = np.transpose(density_array, (2, 1, 0))
    
    # Ensure it's 32x32x32
    if density_array.shape != (32, 32, 32):
        raise ValueError(f"Expected density array shape (32, 32, 32), got {density_array.shape}")
    
    # Reshape to (8, 4, 8, 4, 8, 4)
    coarse_res = 8
    micro_per = 4
    micro_8_4 = density_array.reshape(coarse_res, micro_per, coarse_res, micro_per, coarse_res, micro_per).astype(np.float32)
    
    return micro_8_4


def compute_occupancy_from_density(density_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Compute occupancy mask from density array based on threshold.
    Returns boolean array where True indicates occupied voxels.
    """
    return density_array > threshold


def create_volume_occupancy_mesh(
    vdb_path: str,
    volume_transform: Dict[str, float],
    threshold: float = 0.5,
    transpose_written: bool = True
) -> Optional[trimesh.Trimesh]:
    """
    Create occupancy mesh from VDB file with volume transform applied.
    
    The density array is used directly as read from VDB (no transpose), and
    coordinates are calculated with the canonical origin-offset convention.
    
    Args:
        vdb_path: Path to VDB file
        volume_transform: Transform dict with keys: translation_x/y/z, rotation_x/y/z, scale_x/y/z
        threshold: Density threshold for occupancy
        transpose_written: DEPRECATED - kept for API compatibility but no longer used.
                          The density_array is used directly as read from VDB.
    
    Returns:
        Combined trimesh mesh of occupied voxels with transform applied
    """
    # Read VDB density
    # Note: We don't transpose density_array - we use it directly as read from VDB
    # The density_array from read_vdb_density is already in VDB index space order
    # arr_i, arr_j, arr_k correspond directly to VDB coordinates (min_coord + arr_index)
    # Keep the density in the VDB reader's native index-space order.
    density_array, vdb_metadata = read_vdb_density(vdb_path, grid_name="density")
    
    # Compute occupancy directly from density_array (no transpose)
    occupancy = compute_occupancy_from_density(density_array, threshold=threshold)
    occupied_indices = np.where(occupancy)
    num_occupied = len(occupied_indices[0])
    
    if num_occupied == 0:
        print("Warning: No occupied voxels found")
        return None
    
    print(f"Creating {num_occupied} cubes for occupied voxels...")
    
    # Get voxel size and metadata
    voxel_size = vdb_metadata['voxel_size']
    min_coord = vdb_metadata['min_coord']
    max_coord = vdb_metadata['max_coord']
    
    # Calculate origin_offset to center the mesh at the origin.
    # This makes the mesh center align with the volume center at (0,0,0)
    origin_offset = np.array([
        -((min_coord[i] + max_coord[i]) / 2.0 + 0.5) * voxel_size
        for i in range(3)
    ], dtype=np.float32)
    
    # Create base cube
    base_cube_size = voxel_size
    base_cube = trimesh.creation.box(extents=[base_cube_size, base_cube_size, base_cube_size])
    
    # Extract transform values
    translation = np.array([
        volume_transform['translation_x'],
        volume_transform['translation_y'],
        volume_transform['translation_z']
    ])
    rotation = np.array([
        np.deg2rad(volume_transform['rotation_x']),
        np.deg2rad(volume_transform['rotation_y']),
        np.deg2rad(volume_transform['rotation_z'])
    ])
    scale = np.array([
        volume_transform['scale_x'],
        volume_transform['scale_y'],
        volume_transform['scale_z']
    ])
    
    # Create list to store all cube meshes
    cube_meshes = []
    
    # Process occupied voxels
    batch_size = 1000
    for batch_start in range(0, num_occupied, batch_size):
        batch_end = min(batch_start + batch_size, num_occupied)
        
        for idx in range(batch_start, batch_end):
            # Get array indices
            arr_i, arr_j, arr_k = occupied_indices[0][idx], occupied_indices[1][idx], occupied_indices[2][idx]
            
            # Convert to VDB voxel coordinates (index space)
            vdb_i = min_coord[0] + arr_i
            vdb_j = min_coord[1] + arr_j
            vdb_k = min_coord[2] + arr_k
            
            # Calculate base position in canonical space (before volume transform)
            # Convert the voxel center from VDB index space to canonical space.
            # Voxel center = vdb_coord * voxel_size + origin_offset + voxel_size/2
            half_voxel = voxel_size / 2.0
            canonical_pos = np.array([
                vdb_i * voxel_size + origin_offset[0] + half_voxel,
                vdb_j * voxel_size + origin_offset[1] + half_voxel,
                vdb_k * voxel_size + origin_offset[2] + half_voxel
            ], dtype=np.float32)
            
            # Apply volume transform: scale -> rotation -> translation
            # Scale first
            scaled_pos = canonical_pos * scale
            # Then rotation
            import trimesh.transformations as tf
            # Create rotation matrix from Euler angles (XYZ order)
            # Use 'sxyz' (static XYZ) convention which matches Blender
            rotation_matrix_4x4 = tf.euler_matrix(rotation[0], rotation[1], rotation[2], 'sxyz')
            rotation_matrix = rotation_matrix_4x4[:3, :3]
            rotated_pos = rotation_matrix @ scaled_pos
            # Finally translation
            world_pos = rotated_pos + translation
            
            # Create cube copy
            cube = base_cube.copy()
            
            # Apply scale to cube size
            cube.apply_scale(scale)
            
            # Apply rotation
            cube.apply_transform(rotation_matrix_4x4)
            
            # Apply translation to world position
            cube.apply_translation(world_pos)
            
            cube_meshes.append(cube)
        
        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  Processed {batch_end} / {num_occupied} voxels...")
    
    # Combine all cubes into a single mesh
    if len(cube_meshes) > 0:
        print("Combining cubes into single mesh...")
        combined_mesh = trimesh.util.concatenate(cube_meshes)
        print(f"Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")
        return combined_mesh
    else:
        return None
