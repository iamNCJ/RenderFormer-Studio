"""RenderFormer V1 Blender extension: strict, direct RF1 HDF5 export."""

bl_info = {
    "name": "RenderFormer V1 Tools",
    "author": "RenderFormer maintainers",
    "version": (1, 1, 0),
    "blender": (4, 5, 10),
    "location": "View3D > Sidebar > RenderFormer V1",
    "description": "Export Blender scenes and animations to strict RF1 HDF5",
    "category": "Import-Export",
}

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Color
import numpy as np

from .schema import (
    angle_weighted_face_vertex_normals,
    lower_triangular_mask,
    make_texture_tile,
    require_v1_features,
    validate_rf1_arrays,
)


def _require_runtime_dependency(module_name, distribution_name=None):
    distribution_name = distribution_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        missing = getattr(error, "name", None) or module_name
        raise RuntimeError(
            f"RenderFormer V1 export requires {distribution_name!r} "
            f"(failed to import {missing!r}). Install the wheel-bearing V1 release "
            "ZIP, or install the pinned V1 dependencies into Blender's Python "
            f"environment. Original import error: {error}"
        ) from error


@contextmanager
def _quiet_stdout():
    old_stdout = sys.stdout
    stream = open(os.devnull, "w")
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
        stream.close()


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


def _has_linked_node_type(node_tree, node_type):
    return any(
        node.type == node_type and any(output.links for output in node.outputs)
        for node in node_tree.nodes
    )


def _has_image_texture(material):
    return bool(
        material
        and material.use_nodes
        and _has_linked_node_type(material.node_tree, "TEX_IMAGE")
    )


def _has_displacement(material):
    if material is None or not material.use_nodes:
        return False
    for node in material.node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            socket = node.inputs.get("Displacement")
            return bool(socket and socket.links)
    return False


def _uses_vertex_colors(material):
    if material is None or not material.use_nodes:
        return False
    for node in material.node_tree.nodes:
        if node.type not in {"VERTEX_COLOR", "ATTRIBUTE"}:
            continue
        for output in node.outputs:
            if any(
                link.to_socket.name in {"Base Color", "Diffuse"}
                for link in output.links
            ):
                return True
    return False


def _socket_float(node, *names, default=0.0):
    if node is None:
        return float(default)
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return float(socket.default_value)
    return float(default)


def _socket_is_active(node, *names):
    if node is None:
        return False
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None and (socket.links or float(socket.default_value) > 1e-6):
            return True
    return False


def _has_environment_texture(scene):
    world = scene.world
    return bool(
        world
        and world.use_nodes
        and _has_linked_node_type(world.node_tree, "TEX_ENVIRONMENT")
    )


def _scene_level_capabilities(scene):
    visible = [obj for obj in scene.objects if not obj.hide_render]
    require_v1_features(
        has_environment_texture=_has_environment_texture(scene),
        has_light_object=any(obj.type == "LIGHT" for obj in visible),
        has_volume_object=any(obj.type == "VOLUME" for obj in visible),
    )


def _object_material(obj, eval_obj, temp_mesh):
    material_slots = [slot for slot in eval_obj.material_slots if slot.material is not None]
    material = material_slots[0].material if material_slots else None
    bsdf = _principled_bsdf(material)
    sr_node = _specular_roughness_group(material)
    if material is not None and material.use_nodes and bsdf is None and sr_node is None:
        raise ValueError(
            f"RF1 export cannot represent shader graph on {obj.name!r}; use a "
            "Principled BSDF or SpecularRoughnessBSDF material"
        )

    uses_vertex_colors = _uses_vertex_colors(material)
    require_v1_features(
        has_uv_texture=_has_image_texture(material),
        has_metallic=_socket_is_active(bsdf, "Metallic"),
        has_transmission=_socket_is_active(bsdf, "Transmission Weight", "Transmission"),
        has_displacement=_has_displacement(material),
        has_multiple_materials=len(material_slots) > 1,
        has_untriangulated_vertex_colors=(
            uses_vertex_colors
            and any(len(polygon.vertices) != 3 for polygon in temp_mesh.polygons)
        ),
        vertex_colors_with_topology_processing=(
            uses_vertex_colors
            and (obj.rf1_fix_object_normal or obj.rf1_recompute_vn)
        ),
    )
    return material, bsdf, sr_node, uses_vertex_colors


def _default_material_vector():
    return np.asarray(
        [0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )


def _principled_material_vector(bsdf):
    diffuse = list(bsdf.inputs["Base Color"].default_value[:3])
    specular = _socket_float(bsdf, "Specular IOR Level", "Specular", default=0.5)
    roughness = _socket_float(bsdf, "Roughness", default=0.5)
    return np.asarray(
        diffuse
        + [specular] * 3
        + [roughness]
        + [0.5, 0.5, 1.0]
        + [0.0, 0.0, 0.0],
        dtype=np.float32,
    )


def _specular_roughness_material_vector(node):
    diffuse = list(node.inputs["Diffuse"].default_value[:3])
    specular = list(node.inputs["Specular"].default_value[:3])
    roughness = float(node.inputs["Roughness"].default_value)
    return np.asarray(
        diffuse + specular + [roughness] + [0.5, 0.5, 1.0] + [0.0, 0.0, 0.0],
        dtype=np.float32,
    )


def _emission(bsdf):
    if bsdf is None:
        return False, np.zeros(3, dtype=np.float32)
    strength = _socket_float(bsdf, "Emission Strength", default=0.0)
    color_socket = bsdf.inputs.get("Emission Color")
    if color_socket is None:
        color_socket = bsdf.inputs.get("Emission")
    color = (
        np.asarray(color_socket.default_value[:3], dtype=np.float32)
        if color_socket is not None
        else np.ones(3, dtype=np.float32)
    )
    return strength > 0.0, color * strength


def _light_material_vector(emissive):
    vector = np.zeros(13, dtype=np.float32)
    vector[:3] = 1.0
    vector[6] = 1.0
    vector[7:10] = [0.5, 0.5, 1.0]
    vector[10:13] = emissive
    return vector


def _face_diffuse_from_vertex_colors(mesh):
    color_layer = mesh.color_attributes.get("Color")
    if color_layer is None:
        raise ValueError("vertex-color material is linked but color attribute 'Color' is missing")
    face_colors = np.zeros((len(mesh.polygons), 3), dtype=np.float32)
    for face_index, polygon in enumerate(mesh.polygons):
        samples = []
        if color_layer.domain == "CORNER":
            samples = [color_layer.data[index].color[:3] for index in polygon.loop_indices]
        elif color_layer.domain == "POINT":
            samples = [color_layer.data[index].color[:3] for index in polygon.vertices]
        else:
            raise ValueError(
                f"RF1 vertex colors require POINT or CORNER domain, got {color_layer.domain}"
            )
        srgb = [np.asarray(Color(value).from_scene_linear_to_srgb()) for value in samples]
        face_colors[face_index] = np.mean(srgb, axis=0)
    return face_colors


def _geometry_from_bmesh(bm):
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    triangles = np.zeros((len(bm.faces), 3, 3), dtype=np.float32)
    normals = np.zeros_like(triangles)
    for face_index, face in enumerate(bm.faces):
        for vertex_index, loop in enumerate(face.loops):
            triangles[face_index, vertex_index] = loop.vert.co[:]
            source_normal = loop.vert.normal if face.smooth else face.normal
            normals[face_index, vertex_index] = source_normal[:]
    return triangles, normals


def _process_normals(obj, triangles, normals, *, is_light):
    if not obj.rf1_fix_object_normal and not obj.rf1_recompute_vn:
        return triangles, normals
    _require_runtime_dependency("networkx")
    trimesh = _require_runtime_dependency("trimesh")
    if obj.rf1_fix_object_normal and not is_light:
        mesh = trimesh.Trimesh(
            vertices=triangles.reshape(-1, 3),
            faces=np.arange(triangles.shape[0] * 3).reshape(-1, 3),
            process=True,
        )
        mesh.fix_normals()
        triangles = np.asarray(mesh.triangles, dtype=np.float32)
        normals = angle_weighted_face_vertex_normals(
            mesh.faces,
            mesh.face_normals,
            mesh.face_angles,
            len(mesh.vertices),
        )

    if obj.rf1_recompute_vn:
        mesh = trimesh.Trimesh(
            vertices=triangles.reshape(-1, 3),
            faces=np.arange(triangles.shape[0] * 3).reshape(-1, 3),
            process=bool(obj.rf1_use_smooth_shading),
        )
        if obj.rf1_use_smooth_shading:
            mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(30.0))
        triangles = np.asarray(mesh.vertices[mesh.faces], dtype=np.float32)
        normals = angle_weighted_face_vertex_normals(
            mesh.faces,
            mesh.face_normals,
            mesh.face_angles,
            len(mesh.vertices),
        )
    return triangles, normals


def _texture_for_object(
    *,
    count,
    bsdf,
    sr_node,
    is_light,
    emissive,
    face_diffuse,
):
    if is_light:
        vector = _light_material_vector(emissive)
    elif sr_node is not None:
        vector = _specular_roughness_material_vector(sr_node)
    elif bsdf is not None:
        vector = _principled_material_vector(bsdf)
    else:
        vector = _default_material_vector()

    if face_diffuse is None:
        return np.repeat(make_texture_tile(vector)[None], count, axis=0)
    if face_diffuse.shape != (count, 3):
        raise ValueError(
            f"vertex-color triangle count {face_diffuse.shape[0]} does not match geometry {count}"
        )
    tiles = np.repeat(make_texture_tile(vector)[None], count, axis=0)
    mask = lower_triangular_mask()
    for channel in range(3):
        tiles[:, channel, mask] = face_diffuse[:, channel, None]
    return tiles


def _render_reference(scene, output_path, resolution):
    simple_exr = _require_runtime_dependency("simple_exr", "simple-exr")

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4096
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "OPEN_EXR"

    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.get_devices()
    scene.cycles.device = "CPU"
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue
        scene.cycles.device = "GPU"
        break

    background = scene.world.node_tree.nodes.get("Background") if scene.world else None
    original_strength = None
    if background is not None:
        original_strength = background.inputs["Strength"].default_value
        background.inputs["Strength"].default_value = 0.0
    try:
        scene.render.filepath = os.path.abspath(output_path)
        with _quiet_stdout():
            bpy.ops.render.render(animation=False, write_still=True)
        return simple_exr.read_exr(output_path).copy()
    finally:
        if background is not None:
            background.inputs["Strength"].default_value = original_strength


def _next_output_path(folder, frame_override):
    if frame_override >= 0:
        stem = f"{frame_override:06d}"
    else:
        numeric = [
            int(os.path.splitext(name)[0])
            for name in os.listdir(folder)
            if os.path.splitext(name)[0].isdigit()
        ]
        stem = str(max(numeric, default=-1) + 1)
    return os.path.join(folder, f"{stem}.h5"), stem


def _export_rf1(context, frame_override=-1):
    h5py = _require_runtime_dependency("h5py")
    scene = context.scene
    _scene_level_capabilities(scene)
    folder = bpy.path.abspath(scene.rf1_output_folder or "//renderformer_v1")
    os.makedirs(folder, exist_ok=True)

    triangle_chunks = []
    normal_chunks = []
    texture_chunks = []
    for obj in [item for item in scene.objects if item.type == "MESH" and not item.hide_render]:
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        try:
            material, bsdf, sr_node, uses_vertex_colors = _object_material(
                obj, eval_obj, temp_mesh
            )
            face_diffuse = (
                _face_diffuse_from_vertex_colors(temp_mesh) if uses_vertex_colors else None
            )
            bm.from_mesh(temp_mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.transform(eval_obj.matrix_world)
            triangles, normals = _geometry_from_bmesh(bm)
            is_light, emissive = _emission(bsdf)
            triangles, normals = _process_normals(
                obj, triangles, normals, is_light=is_light
            )
            texture = _texture_for_object(
                count=triangles.shape[0],
                bsdf=bsdf,
                sr_node=sr_node,
                is_light=is_light,
                emissive=emissive,
                face_diffuse=face_diffuse,
            )
            triangle_chunks.append(triangles)
            normal_chunks.append(normals)
            texture_chunks.append(texture)
        finally:
            bm.free()
            eval_obj.to_mesh_clear()

    if not triangle_chunks:
        raise ValueError("RF1 export found no visible mesh geometry")

    camera = scene.camera or next(
        (item for item in scene.objects if item.type == "CAMERA"), None
    )
    if camera is None:
        raise ValueError("RF1 export requires an active camera")

    triangles = np.concatenate(triangle_chunks).astype(np.float32, copy=False)
    vn = np.concatenate(normal_chunks).astype(np.float32, copy=False)
    texture = np.concatenate(texture_chunks).astype(np.float32, copy=False)
    c2w = np.asarray(camera.matrix_world, dtype=np.float32)[None]
    fov = np.asarray([math.degrees(camera.data.angle)], dtype=np.float32)
    validate_rf1_arrays(triangles, vn, texture, c2w, fov)

    output_path, stem = _next_output_path(folder, frame_override)
    reference = None
    temporary_reference = os.path.join(folder, f".rf1-reference-{stem}.exr")
    if scene.rf1_include_reference_render:
        try:
            reference = _render_reference(
                scene, temporary_reference, scene.rf1_resolution
            )
        finally:
            if os.path.exists(temporary_reference):
                os.remove(temporary_reference)

    metadata = {
        "format": "v1",
        "schema_version": 1,
        "scene_name": scene.name,
        "texture_encoding": "rf_v1_13ch",
        "exporter": "renderformer_v1_tools",
    }
    temporary_h5 = output_path + ".tmp"
    if os.path.exists(temporary_h5):
        os.remove(temporary_h5)
    try:
        with h5py.File(temporary_h5, "w") as handle:
            handle.create_dataset(
                "triangles", data=triangles, compression="gzip", compression_opts=9
            )
            handle.create_dataset("vn", data=vn, compression="gzip", compression_opts=9)
            handle.create_dataset(
                "texture", data=texture, compression="gzip", compression_opts=9
            )
            handle.create_dataset("c2w", data=c2w, compression="gzip", compression_opts=9)
            handle.create_dataset("fov", data=fov, compression="gzip", compression_opts=9)
            if reference is not None:
                handle.create_dataset(
                    "img",
                    data=np.asarray(reference, dtype=np.float16),
                    compression="gzip",
                    compression_opts=9,
                )
            handle.attrs["created"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["renderformer"] = "1.1.0"
            handle.attrs["export_metadata"] = json.dumps(metadata, sort_keys=True)
        os.replace(temporary_h5, output_path)
    finally:
        if os.path.exists(temporary_h5):
            os.remove(temporary_h5)
    return output_path, triangles.shape[0]


class RENDERFORMER_V1_PT_export(bpy.types.Panel):
    bl_label = "RenderFormer V1 Export"
    bl_idname = "RENDERFORMER_V1_PT_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RenderFormer V1"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "rf1_output_folder")
        layout.prop(context.scene, "rf1_resolution")
        layout.prop(context.scene, "rf1_include_reference_render")
        layout.operator("renderformer_v1.export_h5", icon="EXPORT")


class RENDERFORMER_V1_PT_animation(bpy.types.Panel):
    bl_label = "RenderFormer V1 Animation"
    bl_idname = "RENDERFORMER_V1_PT_animation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RenderFormer V1"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "rf1_animation_folder")
        layout.prop(context.scene, "rf1_start_frame")
        layout.prop(context.scene, "rf1_end_frame")
        layout.operator("renderformer_v1.export_animation", icon="RENDER_ANIMATION")
        if context.window_manager.rf1_export_progress > 0:
            layout.progress(
                factor=context.window_manager.rf1_export_progress,
                type="BAR",
                text=f"{int(context.window_manager.rf1_export_progress * 100)}%",
            )


class RENDERFORMER_V1_PT_object(bpy.types.Panel):
    bl_label = "RF1 Object Geometry"
    bl_idname = "RENDERFORMER_V1_PT_object"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RenderFormer V1"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.object, "rf1_recompute_vn")
        layout.prop(context.object, "rf1_fix_object_normal")
        layout.prop(context.object, "rf1_use_smooth_shading")


class RENDERFORMER_V1_OT_export_h5(bpy.types.Operator):
    bl_idname = "renderformer_v1.export_h5"
    bl_label = "Export RF1 HDF5"
    bl_description = "Export the active scene to the strict RenderFormer V1 HDF5 schema"
    bl_options = {"REGISTER"}

    frame_override: bpy.props.IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        try:
            output_path, triangle_count = _export_rf1(context, self.frame_override)
        except (ValueError, RuntimeError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {triangle_count} triangles to {output_path}")
        return {"FINISHED"}


class RENDERFORMER_V1_OT_export_animation(bpy.types.Operator):
    bl_idname = "renderformer_v1.export_animation"
    bl_label = "Export RF1 Animation"
    bl_description = "Export each selected animation frame as strict RF1 HDF5"
    bl_options = {"REGISTER"}

    _timer = None

    def execute(self, context):
        scene = context.scene
        if scene.rf1_start_frame > scene.rf1_end_frame:
            self.report({"ERROR"}, "Start frame must not exceed end frame")
            return {"CANCELLED"}
        self._original_folder = scene.rf1_output_folder
        self._original_frame = scene.frame_current
        scene.rf1_output_folder = scene.rf1_animation_folder
        self._current_frame = scene.rf1_start_frame
        self._total_frames = scene.rf1_end_frame - scene.rf1_start_frame + 1
        context.window_manager.rf1_export_progress = 0.0
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.scene.rf1_output_folder = self._original_folder
        context.scene.frame_set(self._original_frame)
        context.window_manager.rf1_export_progress = 0.0

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._current_frame > context.scene.rf1_end_frame:
            self._finish(context)
            self.report({"INFO"}, "RF1 animation export complete")
            return {"FINISHED"}
        context.scene.frame_set(self._current_frame)
        result = bpy.ops.renderformer_v1.export_h5(frame_override=self._current_frame)
        if "FINISHED" not in result:
            self._finish(context)
            self.report({"ERROR"}, f"RF1 export failed at frame {self._current_frame}")
            return {"CANCELLED"}
        completed = self._current_frame - context.scene.rf1_start_frame + 1
        context.window_manager.rf1_export_progress = completed / self._total_frames
        self._current_frame += 1
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._finish(context)


_CLASSES = (
    RENDERFORMER_V1_PT_export,
    RENDERFORMER_V1_PT_animation,
    RENDERFORMER_V1_PT_object,
    RENDERFORMER_V1_OT_export_h5,
    RENDERFORMER_V1_OT_export_animation,
)


def register():
    bpy.types.Scene.rf1_resolution = bpy.props.IntProperty(
        name="Reference Resolution", default=512, min=64, max=8192
    )
    bpy.types.Scene.rf1_output_folder = bpy.props.StringProperty(
        name="RF1 Output Folder", subtype="DIR_PATH", default="//renderformer_v1"
    )
    bpy.types.Scene.rf1_animation_folder = bpy.props.StringProperty(
        name="RF1 Animation Folder",
        subtype="DIR_PATH",
        default="//renderformer_v1_animation",
    )
    bpy.types.Scene.rf1_start_frame = bpy.props.IntProperty(
        name="Start Frame", default=1, min=1
    )
    bpy.types.Scene.rf1_end_frame = bpy.props.IntProperty(
        name="End Frame", default=250, min=1
    )
    bpy.types.Scene.rf1_include_reference_render = bpy.props.BoolProperty(
        name="Include Cycles Reference",
        description="Store a 4096-sample reference render in the optional img dataset",
        default=False,
    )
    bpy.types.WindowManager.rf1_export_progress = bpy.props.FloatProperty(default=0.0)
    bpy.types.Object.rf1_recompute_vn = bpy.props.BoolProperty(
        name="Recompute Vertex Normals", default=True
    )
    bpy.types.Object.rf1_fix_object_normal = bpy.props.BoolProperty(
        name="Fix Winding/Normals",
        description="Use trimesh processing; this can change topology",
        default=False,
    )
    bpy.types.Object.rf1_use_smooth_shading = bpy.props.BoolProperty(
        name="Smooth Recomputed Normals", default=True
    )
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.rf1_resolution
    del bpy.types.Scene.rf1_output_folder
    del bpy.types.Scene.rf1_animation_folder
    del bpy.types.Scene.rf1_start_frame
    del bpy.types.Scene.rf1_end_frame
    del bpy.types.Scene.rf1_include_reference_render
    del bpy.types.WindowManager.rf1_export_progress
    del bpy.types.Object.rf1_recompute_vn
    del bpy.types.Object.rf1_fix_object_normal
    del bpy.types.Object.rf1_use_smooth_shading


if __name__ == "__main__":
    register()
