"""RenderFormer V2 Blender extension: portable prepared-frame export."""

bl_info = {
    "name": "RenderFormer V2 Tools",
    "author": "RenderFormer maintainers",
    "version": (2, 0, 0),
    "blender": (4, 5, 10),
    "location": "View3D > Sidebar > RenderFormer V2",
    "description": "Export Blender scenes as portable RenderFormer V2 prepared frames",
    "category": "Import-Export",
}

import importlib
import json
import math
import os

import bmesh
import bpy
import numpy as np

from .prepared_frame import (
    HOMO_DIFFUSE_SPECULAR,
    HOMO_METALLIC_ROUGHNESS,
    HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
    SVBRDF_DIFFUSE_SPECULAR,
    SVBRDF_METALLIC_ROUGHNESS,
    build_object_config,
    first_not_none,
    image_or_constant,
    prepare_frame_directory,
    required_texture_maps,
    sanitize_object_key,
)


def _require_runtime_dependency(module_name, distribution_name=None):
    distribution_name = distribution_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        missing = getattr(error, "name", None) or module_name
        raise RuntimeError(
            f"RenderFormer V2 export requires {distribution_name!r} "
            f"(failed to import {missing!r}). Install the wheel-bearing V2 release "
            "ZIP, or install the pinned V2 dependencies into Blender's Python "
            f"environment. Original import error: {error}"
        ) from error


def _require_export_dependencies():
    return {
        "imageio": _require_runtime_dependency("imageio.v3", "imageio"),
        "pillow": _require_runtime_dependency("PIL.Image", "Pillow"),
    }


def _principled_bsdf(material):
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    group = material.node_tree.nodes.get("Group")
    if group is not None and group.node_tree is not None:
        for child in group.node_tree.nodes:
            if child.type == "BSDF_PRINCIPLED":
                return child
    return None


def _specular_roughness_group(material):
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if (
            node.type == "GROUP"
            and node.node_tree is not None
            and node.node_tree.name.startswith("SpecularRoughnessBSDF")
        ):
            return node
    return None


def _uses_vertex_colors(material):
    if material is None or not material.use_nodes:
        return False
    return any(
        node.type in {"VERTEX_COLOR", "ATTRIBUTE"}
        and any(output.links for output in node.outputs)
        for node in material.node_tree.nodes
    )


def _find_image_node(node, visited=None):
    if node is None:
        return None
    visited = set() if visited is None else visited
    node_id = id(node)
    if node_id in visited:
        return None
    visited.add(node_id)
    if node.type in {"TEX_IMAGE", "TEX_ENVIRONMENT"} and node.image is not None:
        return node
    for socket in node.inputs:
        for link in socket.links:
            found = _find_image_node(link.from_node, visited)
            if found is not None:
                return found
    return None


def _input_image_node(shader_node, input_name):
    if shader_node is None:
        return None
    socket = shader_node.inputs.get(input_name)
    if socket is None:
        return None
    for link in socket.links:
        found = _find_image_node(link.from_node)
        if found is not None and found.type == "TEX_IMAGE":
            return found
    return None


def _extract_image(node):
    if node is None or node.image is None:
        return None
    image = node.image
    width, height = image.size
    if width == 0 or height == 0 or len(image.pixels) == 0:
        image.reload()
        width, height = image.size
    if width == 0 or height == 0 or len(image.pixels) == 0:
        raise ValueError(f"image {image.name!r} has no readable pixels")
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(
        height, width, image.channels
    )
    return pixels[::-1].copy()


def _input_image(shader_node, input_name):
    return _extract_image(_input_image_node(shader_node, input_name))


def _input_float(shader_node, *names, default=0.0):
    if shader_node is None:
        return float(default)
    for name in names:
        socket = shader_node.inputs.get(name)
        if socket is not None:
            return float(socket.default_value)
    return float(default)


def _input_rgb(shader_node, name, default):
    if shader_node is None:
        return list(default)
    socket = shader_node.inputs.get(name)
    return list(socket.default_value[:3]) if socket is not None else list(default)


def _socket_has_links(shader_node, *names):
    if shader_node is None:
        return False
    return any(
        socket is not None and bool(socket.links)
        for socket in (shader_node.inputs.get(name) for name in names)
    )


def _emission(bsdf):
    if bsdf is None:
        return False, [0.0, 0.0, 0.0]
    if _socket_has_links(bsdf, "Emission Color", "Emission", "Emission Strength"):
        raise ValueError("V2 prepared frames support constant, not textured, emission")
    strength = _input_float(bsdf, "Emission Strength", default=0.0)
    color_socket = first_not_none(
        bsdf.inputs.get("Emission Color"), bsdf.inputs.get("Emission")
    )
    color = (
        list(color_socket.default_value[:3])
        if color_socket is not None
        else [1.0, 1.0, 1.0]
    )
    return strength > 0.0, [float(channel) * strength for channel in color]


def _detect_displacement(material):
    if material is None or not material.use_nodes:
        return None
    output = next(
        (node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL"),
        None,
    )
    if output is None:
        return None
    socket = output.inputs.get("Displacement")
    if socket is None or not socket.links:
        return None
    displacement = socket.links[0].from_node
    if displacement.type != "DISPLACEMENT":
        raise ValueError("V2 export supports displacement only through a Displacement node")
    height_socket = displacement.inputs.get("Height")
    height_node = (
        _find_image_node(height_socket.links[0].from_node)
        if height_socket is not None and height_socket.links
        else None
    )
    if height_node is None or height_node.type != "TEX_IMAGE":
        raise ValueError("V2 displacement requires an image texture connected to Height")
    return {
        "image": height_node,
        "scale": _input_float(displacement, "Scale", default=0.05),
        "midlevel": _input_float(displacement, "Midlevel", default=0.5),
    }


def _surface_has_image(bsdf, sr_node):
    inputs = (
        (bsdf, "Base Color"),
        (bsdf, "Metallic"),
        (bsdf, "Specular IOR Level"),
        (bsdf, "Specular"),
        (bsdf, "Roughness"),
        (bsdf, "Normal"),
        (sr_node, "Diffuse"),
        (sr_node, "Specular"),
        (sr_node, "Roughness"),
        (sr_node, "Normal"),
    )
    return any(_input_image_node(node, name) is not None for node, name in inputs)


def _resize_rgb(data, size):
    data = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
    if data.shape[:2] == (size, size):
        return data
    encoded = (data * 255.0).round().astype(np.uint8)
    image_module = _require_runtime_dependency("PIL.Image", "Pillow")
    resized = image_module.fromarray(encoded, mode="RGB").resize(
        (size, size), image_module.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _write_rgb(path, image, fallback, size):
    iio = _require_runtime_dependency("imageio.v3", "imageio")
    data = image_or_constant(image, fallback, size)
    data = _resize_rgb(data, size)
    iio.imwrite(path, (np.clip(data, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def _write_texture_set(bsdf, sr_node, texture_dir, size):
    os.makedirs(texture_dir, exist_ok=True)
    if sr_node is not None and bsdf is None:
        material_type = SVBRDF_DIFFUSE_SPECULAR
        maps = {
            "diffuse.png": (
                _input_image(sr_node, "Diffuse"),
                _input_rgb(sr_node, "Diffuse", [0.4, 0.4, 0.4]),
            ),
            "specular.png": (
                _input_image(sr_node, "Specular"),
                _input_rgb(sr_node, "Specular", [0.04, 0.04, 0.04]),
            ),
            "roughness.png": (
                _input_image(sr_node, "Roughness"),
                _input_float(sr_node, "Roughness", default=0.5),
            ),
            "normal.png": (
                _input_image(sr_node, "Normal"),
                [0.5, 0.5, 1.0],
            ),
        }
    elif bsdf is not None:
        transmission = _input_float(
            bsdf, "Transmission Weight", "Transmission", default=0.0
        )
        if transmission > 1e-6 or _socket_has_links(
            bsdf, "Transmission Weight", "Transmission"
        ):
            raise ValueError(
                "V2 prepared-frame export supports transmission only for homogeneous materials"
            )
        metallic_image = _input_image(bsdf, "Metallic")
        metallic = _input_float(bsdf, "Metallic", default=0.0)
        if metallic_image is not None or metallic > 1e-6:
            material_type = SVBRDF_METALLIC_ROUGHNESS
            maps = {
                "basecolor.png": (
                    _input_image(bsdf, "Base Color"),
                    _input_rgb(bsdf, "Base Color", [0.5, 0.5, 0.5]),
                ),
                "metallic.png": (metallic_image, metallic),
                "roughness.png": (
                    _input_image(bsdf, "Roughness"),
                    _input_float(bsdf, "Roughness", default=0.5),
                ),
                "normal.png": (
                    _input_image(bsdf, "Normal"),
                    [0.5, 0.5, 1.0],
                ),
            }
        else:
            material_type = SVBRDF_DIFFUSE_SPECULAR
            specular_image = first_not_none(
                _input_image(bsdf, "Specular IOR Level"),
                _input_image(bsdf, "Specular"),
            )
            specular = _input_float(
                bsdf, "Specular IOR Level", "Specular", default=0.5
            )
            maps = {
                "diffuse.png": (
                    _input_image(bsdf, "Base Color"),
                    _input_rgb(bsdf, "Base Color", [0.5, 0.5, 0.5]),
                ),
                "specular.png": (specular_image, specular),
                "roughness.png": (
                    _input_image(bsdf, "Roughness"),
                    _input_float(bsdf, "Roughness", default=0.5),
                ),
                "normal.png": (
                    _input_image(bsdf, "Normal"),
                    [0.5, 0.5, 1.0],
                ),
            }
    else:
        raise ValueError("textured V2 materials require a supported shader node")

    for filename, (image, fallback) in maps.items():
        _write_rgb(os.path.join(texture_dir, filename), image, fallback, size)
    missing = [
        filename
        for filename in required_texture_maps(material_type)
        if not os.path.isfile(os.path.join(texture_dir, filename))
    ]
    if missing:
        raise RuntimeError("failed to create required material maps: " + ", ".join(missing))
    return material_type


def _write_heightmap(displacement, texture_dir, size):
    image = _extract_image(displacement["image"])
    if image is None:
        raise ValueError("height image could not be read")
    height = image[..., 0]
    encoded = (np.clip(height, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if encoded.shape != (size, size):
        image_module = _require_runtime_dependency("PIL.Image", "Pillow")
        encoded = np.asarray(
            image_module.fromarray(encoded, mode="L").resize(
                (size, size), image_module.Resampling.LANCZOS
            )
        )
    os.makedirs(texture_dir, exist_ok=True)
    iio = _require_runtime_dependency("imageio.v3", "imageio")
    iio.imwrite(os.path.join(texture_dir, "height.png"), encoded)


def _homogeneous_material(bsdf, sr_node):
    if sr_node is not None:
        return HOMO_DIFFUSE_SPECULAR, {
            "diffuse_specular_params": {
                "diffuse": _input_rgb(sr_node, "Diffuse", [0.4, 0.4, 0.4]),
                "specular": _input_rgb(sr_node, "Specular", [0.04, 0.04, 0.04]),
                "roughness": _input_float(sr_node, "Roughness", default=0.5),
            }
        }
    if bsdf is not None:
        transmission = _input_float(
            bsdf, "Transmission Weight", "Transmission", default=0.0
        )
        common = {
            "base_color": _input_rgb(bsdf, "Base Color", [0.5, 0.5, 0.5]),
            "metallic": _input_float(bsdf, "Metallic", default=0.0),
            "roughness": _input_float(bsdf, "Roughness", default=0.5),
        }
        if transmission > 1e-6:
            return HOMO_METALLIC_ROUGHNESS_TRANSMISSION, {
                "transmission_params": {
                    **common,
                    "transmission_weight": transmission,
                    "ior": _input_float(bsdf, "IOR", default=1.5),
                }
            }
        return HOMO_METALLIC_ROUGHNESS, {"metallic_roughness_params": common}
    return HOMO_DIFFUSE_SPECULAR, {
        "diffuse_specular_params": {
            "diffuse": [0.4, 0.4, 0.4],
            "specular": [0.0, 0.0, 0.0],
            "roughness": 1.0,
        }
    }


def _export_obj(bm, output_path):
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    count = len(bm.faces)
    triangles = np.zeros((count, 3, 3), dtype=np.float32)
    normals = np.zeros_like(triangles)
    uv_layer = bm.loops.layers.uv.active
    uvs = np.zeros((count, 3, 2), dtype=np.float32) if uv_layer else None
    for face_index, face in enumerate(bm.faces):
        for vertex_index, loop in enumerate(face.loops):
            triangles[face_index, vertex_index] = loop.vert.co[:]
            normal = loop.vert.normal if face.smooth else face.normal
            normals[face_index, vertex_index] = normal[:]
            if uvs is not None:
                uvs[face_index, vertex_index] = loop[uv_layer].uv[:]
    vertices = triangles.reshape(-1, 3)
    vertex_normals = normals.reshape(-1, 3)
    texture_coordinates = uvs.reshape(-1, 2) if uvs is not None else None
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# RenderFormer V2 world-space triangle mesh\n")
        handle.writelines(
            f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertices
        )
        if texture_coordinates is not None:
            handle.writelines(
                f"vt {u:.9g} {v:.9g}\n" for u, v in texture_coordinates
            )
        handle.writelines(
            f"vn {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertex_normals
        )
        for face_index in range(count):
            indices = [face_index * 3 + offset + 1 for offset in range(3)]
            if texture_coordinates is None:
                corners = " ".join(f"{index}//{index}" for index in indices)
            else:
                corners = " ".join(
                    f"{index}/{index}/{index}" for index in indices
                )
            handle.write(f"f {corners}\n")


def _unique_key(name, used):
    base = sanitize_object_key(name)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _material_for_object(material, obj_key, split_dir, texture_size):
    bsdf = _principled_bsdf(material)
    sr_node = _specular_roughness_group(material)
    if material is not None and material.use_nodes and bsdf is None and sr_node is None:
        raise ValueError(
            f"V2 export supports Principled BSDF or SpecularRoughnessBSDF, got {material.name!r}"
        )
    if _uses_vertex_colors(material):
        raise ValueError(
            "the maintained V2 raw profile does not preserve per-triangle vertex colors"
        )

    is_lighting, emissive = _emission(bsdf)
    displacement = _detect_displacement(material)
    has_texture = _surface_has_image(bsdf, sr_node) or displacement is not None
    texture_path = None
    use_heightmap = displacement is not None
    params = {}
    if has_texture:
        texture_dir = os.path.join(split_dir, obj_key)
        material_type = _write_texture_set(bsdf, sr_node, texture_dir, texture_size)
        texture_path = f"split/{obj_key}"
        if displacement is not None:
            _write_heightmap(displacement, texture_dir, texture_size)
    else:
        material_type, params = _homogeneous_material(bsdf, sr_node)

    return build_object_config(
        obj_key=obj_key,
        material_type=material_type,
        is_lighting=is_lighting,
        emissive=emissive,
        texture_map_path=texture_path,
        use_heightmap=use_heightmap,
        **params,
    )


def _export_environment(scene, frame_dir):
    world = scene.world
    if world is None or not world.use_nodes:
        return None
    background = world.node_tree.nodes.get("Background")
    if background is None:
        return None
    color = background.inputs.get("Color")
    environment_node = None
    if color is not None:
        for link in color.links:
            found = _find_image_node(link.from_node)
            if found is not None and found.type == "TEX_ENVIRONMENT":
                environment_node = found
                break
    if environment_node is None:
        return None

    image = environment_node.image
    output_name = "env_map.exr"
    output_path = os.path.join(frame_dir, output_name)
    original_path = image.filepath_raw
    original_format = image.file_format
    try:
        image.filepath_raw = output_path
        image.file_format = "OPEN_EXR"
        image.save()
    finally:
        image.filepath_raw = original_path
        image.file_format = original_format

    rotation = [0.0, 0.0, 0.0]
    mapping = next(
        (node for node in world.node_tree.nodes if node.type == "MAPPING"), None
    )
    if mapping is not None:
        rotation = [
            math.degrees(float(value))
            for value in mapping.inputs["Rotation"].default_value[:3]
        ]
    strength = _input_float(background, "Strength", default=1.0)
    return {
        "env_map_path": output_name,
        "rotation_x": rotation[0],
        "rotation_y": rotation[1],
        "rotation_z": rotation[2],
        "strength": strength,
    }


def _next_frame_directory(folder, frame_override, current_frame):
    frame = frame_override if frame_override >= 0 else current_frame
    return os.path.join(folder, f"frame_{frame:06d}"), f"{frame:06d}"


def _export_prepared_frame(context, frame_override=-1):
    _require_export_dependencies()
    scene = context.scene
    unsupported = [
        obj.name
        for obj in scene.objects
        if not obj.hide_render and obj.type in {"LIGHT", "VOLUME"}
    ]
    if unsupported:
        raise ValueError(
            "V2 prepared-frame export requires emissive mesh lighting and does not "
            "export Blender LIGHT/VOLUME objects: " + ", ".join(unsupported)
        )

    folder = bpy.path.abspath(scene.rf2_output_folder or "//renderformer_v2")
    frame_dir, frame_name = _next_frame_directory(
        folder, frame_override, scene.frame_current
    )
    prepare_frame_directory(frame_dir)
    split_dir = os.path.join(frame_dir, "split")
    os.makedirs(split_dir)

    object_configs = {}
    used_keys = set()
    for obj in [item for item in scene.objects if item.type == "MESH" and not item.hide_render]:
        nonempty_slots = [slot for slot in obj.material_slots if slot.material is not None]
        if len(nonempty_slots) > 1:
            raise ValueError(
                f"V2 prepared-frame export supports one material per mesh; {obj.name!r} has "
                f"{len(nonempty_slots)}"
            )
        obj_key = _unique_key(obj.name, used_keys)
        disabled_modifiers = []
        source_material = nonempty_slots[0].material if nonempty_slots else None
        displacement = _detect_displacement(source_material)
        if displacement is not None:
            for modifier in obj.modifiers:
                if modifier.type == "SUBSURF":
                    disabled_modifiers.append(
                        (modifier, modifier.show_render, modifier.show_viewport)
                    )
                    modifier.show_render = False
                    modifier.show_viewport = False

        bm = bmesh.new()
        eval_obj = None
        try:
            depsgraph = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            temp_mesh = eval_obj.to_mesh()
            try:
                bm.from_mesh(temp_mesh)
            finally:
                eval_obj.to_mesh_clear()
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.transform(eval_obj.matrix_world)
            obj_path = os.path.join(split_dir, f"{obj_key}.obj")
            _export_obj(bm, obj_path)
            material = (
                eval_obj.material_slots[0].material
                if eval_obj.material_slots and eval_obj.material_slots[0].material
                else source_material
            )
            config = _material_for_object(
                material, obj_key, split_dir, scene.rf2_texture_resolution
            )
            config["material"]["smooth_shading"] = (
                sum(1 for face in bm.faces if face.smooth) > len(bm.faces) / 2
            )
            object_configs[obj_key] = config
        finally:
            bm.free()
            for modifier, show_render, show_viewport in disabled_modifiers:
                modifier.show_render = show_render
                modifier.show_viewport = show_viewport

    if not object_configs:
        raise ValueError("V2 prepared-frame export found no visible mesh geometry")

    camera = scene.camera or next(
        (item for item in scene.objects if item.type == "CAMERA"), None
    )
    if camera is None:
        raise ValueError("V2 prepared-frame export requires an active camera")
    c2w = np.asarray(camera.matrix_world, dtype=np.float32)
    position = c2w[:3, 3]
    look_at = position - c2w[:3, 2]
    up = c2w[:3, 1]

    config = {
        "scene_name": f"blender_frame_{frame_name}",
        "version": "2.0.0",
        "objects": object_configs,
        "cameras": [
            {
                "position": position.tolist(),
                "look_at": look_at.tolist(),
                "up": up.tolist(),
                "fov": math.degrees(camera.data.angle),
            }
        ],
        "lighting": [],
        "env_map": _export_environment(scene, frame_dir),
        "volumes": None,
        "wall_holes": None,
    }
    config_path = os.path.join(frame_dir, "scene_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return frame_dir, len(object_configs)


class RENDERFORMER_V2_PT_export(bpy.types.Panel):
    bl_label = "RenderFormer V2 Export"
    bl_idname = "RENDERFORMER_V2_PT_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RenderFormer V2"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "rf2_output_folder")
        layout.prop(context.scene, "rf2_texture_resolution")
        layout.operator("renderformer_v2.export_prepared_frame", icon="FILE_FOLDER")


class RENDERFORMER_V2_PT_animation(bpy.types.Panel):
    bl_label = "RenderFormer V2 Animation"
    bl_idname = "RENDERFORMER_V2_PT_animation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RenderFormer V2"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "rf2_animation_folder")
        layout.prop(context.scene, "rf2_start_frame")
        layout.prop(context.scene, "rf2_end_frame")
        layout.operator("renderformer_v2.export_animation", icon="RENDER_ANIMATION")
        if context.window_manager.rf2_export_progress > 0:
            layout.progress(
                factor=context.window_manager.rf2_export_progress,
                type="BAR",
                text=f"{int(context.window_manager.rf2_export_progress * 100)}%",
            )


class RENDERFORMER_V2_OT_export_prepared_frame(bpy.types.Operator):
    bl_idname = "renderformer_v2.export_prepared_frame"
    bl_label = "Export V2 Prepared Frame"
    bl_description = "Export OBJ, material maps, environment, and scene_config.json"
    bl_options = {"REGISTER"}

    frame_override: bpy.props.IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        try:
            frame_dir, object_count = _export_prepared_frame(
                context, self.frame_override
            )
        except (ValueError, RuntimeError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {object_count} objects to {frame_dir}")
        return {"FINISHED"}


class RENDERFORMER_V2_OT_export_animation(bpy.types.Operator):
    bl_idname = "renderformer_v2.export_animation"
    bl_label = "Export V2 Animation"
    bl_description = "Export each selected frame as a V2 prepared-frame directory"
    bl_options = {"REGISTER"}

    _timer = None

    def execute(self, context):
        scene = context.scene
        if scene.rf2_start_frame > scene.rf2_end_frame:
            self.report({"ERROR"}, "Start frame must not exceed end frame")
            return {"CANCELLED"}
        self._original_folder = scene.rf2_output_folder
        self._original_frame = scene.frame_current
        scene.rf2_output_folder = scene.rf2_animation_folder
        self._current_frame = scene.rf2_start_frame
        self._total_frames = scene.rf2_end_frame - scene.rf2_start_frame + 1
        context.window_manager.rf2_export_progress = 0.0
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.scene.rf2_output_folder = self._original_folder
        context.scene.frame_set(self._original_frame)
        context.window_manager.rf2_export_progress = 0.0

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._current_frame > context.scene.rf2_end_frame:
            self._finish(context)
            self.report({"INFO"}, "V2 animation export complete")
            return {"FINISHED"}
        context.scene.frame_set(self._current_frame)
        result = bpy.ops.renderformer_v2.export_prepared_frame(
            frame_override=self._current_frame
        )
        if "FINISHED" not in result:
            self._finish(context)
            self.report({"ERROR"}, f"V2 export failed at frame {self._current_frame}")
            return {"CANCELLED"}
        completed = self._current_frame - context.scene.rf2_start_frame + 1
        context.window_manager.rf2_export_progress = completed / self._total_frames
        self._current_frame += 1
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._finish(context)


_CLASSES = (
    RENDERFORMER_V2_PT_export,
    RENDERFORMER_V2_PT_animation,
    RENDERFORMER_V2_OT_export_prepared_frame,
    RENDERFORMER_V2_OT_export_animation,
)


def register():
    bpy.types.Scene.rf2_output_folder = bpy.props.StringProperty(
        name="V2 Output Folder", subtype="DIR_PATH", default="//renderformer_v2"
    )
    bpy.types.Scene.rf2_animation_folder = bpy.props.StringProperty(
        name="V2 Animation Folder",
        subtype="DIR_PATH",
        default="//renderformer_v2_animation",
    )
    bpy.types.Scene.rf2_texture_resolution = bpy.props.IntProperty(
        name="Material-map Resolution", default=256, min=32, max=4096
    )
    bpy.types.Scene.rf2_start_frame = bpy.props.IntProperty(
        name="Start Frame", default=1, min=1
    )
    bpy.types.Scene.rf2_end_frame = bpy.props.IntProperty(
        name="End Frame", default=250, min=1
    )
    bpy.types.WindowManager.rf2_export_progress = bpy.props.FloatProperty(default=0.0)
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.rf2_output_folder
    del bpy.types.Scene.rf2_animation_folder
    del bpy.types.Scene.rf2_texture_resolution
    del bpy.types.Scene.rf2_start_frame
    del bpy.types.Scene.rf2_end_frame
    del bpy.types.WindowManager.rf2_export_progress


if __name__ == "__main__":
    register()
