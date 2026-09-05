"""Generate convex and smooth environment-reflection shapes with PyPI bpy.

Shape types:
1. Ellipsoids - random axis ratios
2. Superellipsoids/Superquadrics - smooth convex shapes with varying roundness
3. Metaballs - multiple balls smoothly blended
4. Platonic solids (subdivided) - tetrahedron, octahedron, icosahedron, dodecahedron
5. Random convex hulls - from random points

Usage:
    # Run with the Python 3.11 environment that contains bpy==4.5.10:
    python create_convex_shapes.py --output_dir ./output --num_shapes 100 --seed 42
"""

import argparse
from importlib import metadata as package_metadata
import json
from math import cos, pi, sin
from pathlib import Path
import random
import uuid

import bpy
import bmesh
from mathutils import Vector
import numpy as np


REQUIRED_BPY_VERSION = "4.5.10"
REQUIRED_BPY_RUNTIME = (4, 5, 10)


def validate_bpy_runtime() -> None:
    """Require the pinned PyPI bpy distribution in the active interpreter."""
    try:
        distribution = package_metadata.distribution("bpy")
    except package_metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "shape generation requires bpy==4.5.10 from PyPI in the active "
            "Python environment"
        ) from error

    installed_version = distribution.version
    if installed_version != REQUIRED_BPY_VERSION:
        raise RuntimeError(
            "shape generation requires bpy==4.5.10 from PyPI exactly; "
            f"found bpy=={installed_version}"
        )

    runtime_version = tuple(bpy.app.version[:3])
    if runtime_version != REQUIRED_BPY_RUNTIME:
        raise RuntimeError(
            "shape generation requires the bpy 4.5.10 runtime exactly; "
            f"found {runtime_version}"
        )

    distribution_root = Path(distribution.locate_file("")).resolve()
    module_path = Path(bpy.__file__).resolve()
    if not module_path.is_relative_to(distribution_root):
        raise RuntimeError(
            "bpy resolved outside the active environment's PyPI distribution: "
            f"{module_path} (expected under {distribution_root})"
        )


def parse_args(argv: list[str] | None = None):
    """Parse direct-Python command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate convex shapes for env map reflection")
    parser.add_argument('--output_dir', type=str, default='./env_map_shapes_output', help='Output directory')
    parser.add_argument('--num_shapes', type=int, default=100, help='Number of shapes to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shape_types', type=str, default='all', 
                        help='Comma-separated list of shape types: ellipsoid,superellipsoid,metaball,platonic,convex_hull,all')
    parser.add_argument('--subdivisions', type=int, default=2, help='Subdivision level for smoother surfaces')
    parser.add_argument('--mesh_resolution', type=int, default=32, help='Resolution for parametric shapes')
    
    return parser.parse_args(argv)


def reset_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def apply_all_modifiers(obj):
    """Apply all modifiers to an object"""
    bpy.context.view_layer.objects.active = obj
    for modifier in obj.modifiers:
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def normalize_mesh(obj, target_radius=1.0):
    """Normalize mesh to fit in a sphere of given radius"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # Get bounding box
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8
    max_dist = max((v - center).length for v in bbox)
    
    if max_dist > 0:
        scale = target_radius / max_dist
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Center at origin
    obj.location = (0, 0, 0)


def add_subdivision(obj, levels=2):
    """Add subdivision surface modifier"""
    mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    mod.levels = levels
    mod.render_levels = levels
    apply_all_modifiers(obj)


def export_obj(obj, filepath):
    """Export object to OBJ file"""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False
    )


# =============================================================================
# Shape Generators
# =============================================================================

def create_ellipsoid(axis_ratios=(1.0, 1.0, 1.0), segments=32, rings=16):
    """Create an ellipsoid with given axis ratios"""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings)
    obj = bpy.context.active_object
    obj.scale = axis_ratios
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj, {'type': 'ellipsoid', 'axis_ratios': list(axis_ratios)}


def create_superellipsoid(e1=1.0, e2=1.0, resolution=32):
    """
    Create a superellipsoid (superquadric).
    e1, e2: exponents controlling roundness (1.0 = sphere, 0.1 = more box-like, 2.0 = pinched)
    Valid range: 0.1 to 2.0 for nice convex shapes
    """
    # Generate vertices using parametric equations
    vertices = []
    faces = []
    
    n_u = resolution
    n_v = resolution // 2
    
    def sgn(x):
        return 1 if x >= 0 else -1
    
    def superellipsoid_point(u, v, e1, e2):
        # u: -pi to pi (longitude)
        # v: -pi/2 to pi/2 (latitude)
        cos_v = cos(v)
        sin_v = sin(v)
        cos_u = cos(u)
        sin_u = sin(u)
        
        x = sgn(cos_v) * abs(cos_v) ** e1 * sgn(cos_u) * abs(cos_u) ** e2
        y = sgn(cos_v) * abs(cos_v) ** e1 * sgn(sin_u) * abs(sin_u) ** e2
        z = sgn(sin_v) * abs(sin_v) ** e1
        
        return (x, y, z)
    
    # Generate vertices
    for i in range(n_v + 1):
        v = -pi/2 + pi * i / n_v
        for j in range(n_u):
            u = -pi + 2 * pi * j / n_u
            vertices.append(superellipsoid_point(u, v, e1, e2))
    
    # Generate faces
    for i in range(n_v):
        for j in range(n_u):
            i0 = i * n_u + j
            i1 = i * n_u + (j + 1) % n_u
            i2 = (i + 1) * n_u + (j + 1) % n_u
            i3 = (i + 1) * n_u + j
            faces.append((i0, i1, i2, i3))
    
    # Create mesh
    mesh = bpy.data.meshes.new("Superellipsoid")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new("Superellipsoid", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Clean up mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj, {'type': 'superellipsoid', 'e1': e1, 'e2': e2}


def create_metaball(num_balls=3, radius_range=(0.3, 0.8), position_range=0.5):
    """
    Create a metaball shape with multiple balls blended together.
    Returns a mesh converted from the metaball.
    """
    # Create metaball
    mball = bpy.data.metaballs.new("MetaBall")
    mball.resolution = 0.05
    mball.render_resolution = 0.02
    
    ball_params = []
    for i in range(num_balls):
        element = mball.elements.new()
        element.type = 'BALL'
        element.radius = random.uniform(*radius_range)
        element.co = Vector([
            random.uniform(-position_range, position_range),
            random.uniform(-position_range, position_range),
            random.uniform(-position_range, position_range)
        ])
        ball_params.append({
            'radius': element.radius,
            'position': list(element.co)
        })
    
    # Create object from metaball
    obj = bpy.data.objects.new("MetaBall", mball)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Convert to mesh - this replaces the object, so get the new reference
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.active_object  # Get the new object reference after conversion
    
    return obj, {'type': 'metaball', 'num_balls': num_balls, 'balls': ball_params}


def create_platonic_solid(solid_type='icosahedron', subdivisions=2):
    """
    Create a platonic solid and subdivide for smoothness.
    solid_type: 'tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'
    """
    if solid_type == 'tetrahedron':
        # Tetrahedron vertices
        a = 1.0
        vertices = [
            (a, a, a),
            (a, -a, -a),
            (-a, a, -a),
            (-a, -a, a)
        ]
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
        
        mesh = bpy.data.meshes.new("Tetrahedron")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("Tetrahedron", mesh)
        bpy.context.collection.objects.link(obj)
        
    elif solid_type == 'cube':
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        
    elif solid_type == 'octahedron':
        # Octahedron vertices
        vertices = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        faces = [
            (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)
        ]
        
        mesh = bpy.data.meshes.new("Octahedron")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("Octahedron", mesh)
        bpy.context.collection.objects.link(obj)
        
    elif solid_type == 'dodecahedron':
        # Use icosphere as approximation, or construct manually
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=0)
        obj = bpy.context.active_object
        # Dodecahedron is dual of icosahedron - this is an approximation
        
    elif solid_type == 'icosahedron':
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=0)
        obj = bpy.context.active_object
        
    else:
        raise ValueError(f"Unknown solid type: {solid_type}")
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Add subdivision for smoothness
    if subdivisions > 0:
        add_subdivision(obj, levels=subdivisions)
    
    return obj, {'type': 'platonic', 'solid_type': solid_type, 'subdivisions': subdivisions}


def create_convex_hull(num_points=20, distribution='sphere'):
    """
    Create a convex hull from random points.
    distribution: 'sphere' (points on sphere surface), 'cube' (uniform in cube), 'gaussian'
    """
    # Generate random points
    points = []
    for _ in range(num_points):
        if distribution == 'sphere':
            # Random point on unit sphere
            theta = random.uniform(0, 2 * pi)
            phi = random.uniform(0, pi)
            r = random.uniform(0.5, 1.0)  # Some variation in radius
            x = r * sin(phi) * cos(theta)
            y = r * sin(phi) * sin(theta)
            z = r * cos(phi)
        elif distribution == 'cube':
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
        elif distribution == 'gaussian':
            x = random.gauss(0, 0.5)
            y = random.gauss(0, 0.5)
            z = random.gauss(0, 0.5)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        points.append((x, y, z))
    
    # Create mesh from points
    mesh = bpy.data.meshes.new("ConvexHull")
    obj = bpy.data.objects.new("ConvexHull", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Create BMesh and add vertices
    bm = bmesh.new()
    for p in points:
        bm.verts.new(p)
    
    bm.to_mesh(mesh)
    bm.free()
    
    # Enter edit mode and create convex hull
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.convex_hull()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj, {'type': 'convex_hull', 'num_points': num_points, 'distribution': distribution}


def create_torus(major_radius=1.0, minor_radius=0.3, major_segments=48, minor_segments=24):
    """Create a torus (donut shape) - good for varied normals"""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=major_segments,
        minor_segments=minor_segments
    )
    obj = bpy.context.active_object
    return obj, {'type': 'torus', 'major_radius': major_radius, 'minor_radius': minor_radius}


def create_blob(num_deformations=5, deform_strength=0.3):
    """
    Create a blob shape by deforming a sphere with multiple displacement modifiers.
    Creates organic-looking convex shapes.
    """
    # Start with a sphere
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    obj = bpy.context.active_object
    
    deform_params = []
    # Apply random vertex displacements using sculpt-like approach
    mesh = obj.data
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    
    for _ in range(num_deformations):
        # Pick a random direction
        direction = Vector([
            random.gauss(0, 1),
            random.gauss(0, 1),
            random.gauss(0, 1)
        ]).normalized()
        
        strength = random.uniform(-deform_strength, deform_strength)
        falloff = random.uniform(0.3, 0.8)
        
        # Displace vertices based on their alignment with direction
        for v in bm.verts:
            alignment = v.co.normalized().dot(direction)
            if alignment > 0:
                factor = (alignment ** (1/falloff)) * strength
                v.co += v.co.normalized() * factor
        
        deform_params.append({
            'direction': list(direction),
            'strength': strength,
            'falloff': falloff
        })
    
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Smooth the result
    bpy.ops.object.shade_smooth()
    
    return obj, {'type': 'blob', 'num_deformations': num_deformations, 'deformations': deform_params}


# =============================================================================
# Main Generation Logic
# =============================================================================

def generate_random_shape(shape_type, args):
    """Generate a random shape of the specified type"""
    
    if shape_type == 'ellipsoid':
        # Random axis ratios (keep convex by not making too extreme)
        axis_ratios = (
            random.uniform(0.5, 1.5),
            random.uniform(0.5, 1.5),
            random.uniform(0.5, 1.5)
        )
        obj, params = create_ellipsoid(axis_ratios, segments=args.mesh_resolution, rings=args.mesh_resolution//2)
        
    elif shape_type == 'superellipsoid':
        # Random exponents for varied shapes
        # Lower values = more boxy, higher = more pinched
        e1 = random.uniform(0.3, 1.5)
        e2 = random.uniform(0.3, 1.5)
        obj, params = create_superellipsoid(e1, e2, resolution=args.mesh_resolution)
        
    elif shape_type == 'metaball':
        num_balls = random.randint(2, 6)
        obj, params = create_metaball(num_balls)
        
    elif shape_type == 'platonic':
        solid_types = ['tetrahedron', 'cube', 'octahedron', 'icosahedron']
        solid_type = random.choice(solid_types)
        obj, params = create_platonic_solid(solid_type, subdivisions=args.subdivisions)
        
    elif shape_type == 'convex_hull':
        num_points = random.randint(10, 30)
        distribution = random.choice(['sphere', 'cube', 'gaussian'])
        obj, params = create_convex_hull(num_points, distribution)
        add_subdivision(obj, levels=args.subdivisions)
        params['subdivisions'] = args.subdivisions
        
    elif shape_type == 'torus':
        major_radius = random.uniform(0.8, 1.2)
        minor_radius = random.uniform(0.2, 0.5)
        obj, params = create_torus(major_radius, minor_radius)
        
    elif shape_type == 'blob':
        num_deformations = random.randint(3, 8)
        deform_strength = random.uniform(0.2, 0.4)
        obj, params = create_blob(num_deformations, deform_strength)
        add_subdivision(obj, levels=args.subdivisions)
        params['subdivisions'] = args.subdivisions
        
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    return obj, params


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_bpy_runtime()
    if args.num_shapes < 1:
        raise ValueError("num_shapes must be at least 1")
    if args.subdivisions < 0:
        raise ValueError("subdivisions must be nonnegative")
    if args.mesh_resolution < 4:
        raise ValueError("mesh_resolution must be at least 4")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse shape types
    if args.shape_types == 'all':
        shape_types = ['ellipsoid', 'superellipsoid', 'metaball', 'platonic', 'convex_hull', 'torus', 'blob']
    else:
        shape_types = [s.strip() for s in args.shape_types.split(',')]
    supported_shape_types = {
        'ellipsoid', 'superellipsoid', 'metaball', 'platonic',
        'convex_hull', 'torus', 'blob',
    }
    unknown_shape_types = sorted(set(shape_types) - supported_shape_types)
    if not shape_types or unknown_shape_types:
        raise ValueError(f"unsupported shape types: {unknown_shape_types or shape_types}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_metadata_path = output_dir / "all_shapes.json"
    if global_metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {global_metadata_path}")
    
    # Generate shapes
    all_metadata = []
    
    # Create a shuffled queue for balanced distribution
    # Each cycle through shape_types is shuffled to ensure variety
    shape_queue = []
    
    for i in range(args.num_shapes):
        reset_scene()
        
        # Refill and shuffle queue when empty (ensures balanced distribution)
        if not shape_queue:
            shape_queue = shape_types.copy()
            random.shuffle(shape_queue)
        
        # Pick shape type from queue
        shape_type = shape_queue.pop()
        
        # Generate shape. A failed item aborts the explicit batch instead of
        # silently changing its size or shape-type distribution.
        obj, params = generate_random_shape(shape_type, args)
        
        # Normalize to unit sphere
        normalize_mesh(obj, target_radius=0.45)
        
        # Generate unique ID
        shape_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"renderformer-env-shape:{args.seed}:{i}:{shape_type}",
            )
        )
        
        # Create shape directory
        shape_dir = output_dir / shape_id
        shape_dir.mkdir(parents=True, exist_ok=False)
        
        # Export OBJ
        obj_path = shape_dir / "original.obj"
        export_obj(obj, str(obj_path))
        
        # Save metadata
        metadata = {
            'id': shape_id,
            'index': i,
            'shape_params': params,
            'seed': args.seed,
            'obj_path': str(obj_path)
        }
        
        metadata_path = shape_dir / "metadata.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        all_metadata.append(metadata)
        
        print(
            f"[W0] {i + 1}/{args.num_shapes}: generated {shape_type} {shape_id}",
            flush=True,
        )
    
    # Save global metadata
    with global_metadata_path.open('w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"generated {len(all_metadata)} shapes to {output_dir}", flush=True)
    print(f"metadata: {global_metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
