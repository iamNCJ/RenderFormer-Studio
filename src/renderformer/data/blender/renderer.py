import os
BLENDER_BACKEND = os.getenv('BLENDER_BACKEND', 'CUDA')
import json
import tempfile
import imageio
import simple_exr
import h5py
import numpy as np
import trimesh
import time
from tqdm import tqdm

from renderformer.data.blender.generated_config import CameraConfig, ObjectConfig, GeneratedConfig, VolumeConfig
from renderformer.data.blender.texture_sample_fn import texture_sample
from dacite import from_dict, Config
from renderformer.data.blender.volume_utils import read_vdb_to_micro_8_4

from contextlib import nullcontext


def hdr_to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Convert linear HDR RGB/RGBA image data to an sRGB PNG preview."""
    arr = np.asarray(image, dtype=np.float32)
    clipped = np.clip(arr, 0.0, 1.0)

    rgb = clipped[..., :3]
    srgb = np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )

    display = clipped.copy()
    display[..., :3] = srgb
    return np.rint(display * 255.0).clip(0, 255).astype(np.uint8)


def _maybe_save_blend(
    dump_blend_file: bool,
    output_image_path: str,
    pack_all,
    save_blend_file,
) -> None:
    if dump_blend_file:
        pack_all()
        save_blend_file(output_image_path + ".blend")


def scene_to_img(
        scene_config: GeneratedConfig,
        mesh_path: str,
        output_obj_path: str,
        output_image_path: str,
        save_img: bool = False,
        resolution: int = 256,
        save_scene_obj: bool = False,
        spp: int = 4096,
        local_model: bool = False,
        dump_blend_file: bool = False,
        skip_rendering: bool = False,
        include_background: bool = False,
        render_repeats: int = 1,
        suppress_blender_stdout: bool = True,
        return_render_timing: bool = False,
        rgb_only_render: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], list[float]]:
    # Import the environment's PyPI helper before bpy. Importing bpy can add a
    # user Blender extension directory to sys.path, which must not shadow the
    # release-pinned helper package.
    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.material import create_specular_roughness_material, create_emissive_material, create_full_principled_bsdf_material
    from bpy_helper.scene import reset_scene, import_3d_model, scene_meshes
    from bpy_helper.light import set_env_light
    from bpy_helper.io import save_blend_file, create_compositing_nodes, render_with_compositing_nodes
    import bpy

    def setup_blender_scene(scene_config: GeneratedConfig, mesh_path: str) -> None:
        reset_scene()
        
        # Set Cycles engine and experimental feature set early (required for adaptive subdivision)
        # This must be done before setting up materials and objects
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
        
        setup_stdout_ctx = nullcontext()
        if suppress_blender_stdout:
            from bpy_helper.utils import stdout_redirected
            setup_stdout_ctx = stdout_redirected()

        with setup_stdout_ctx:
        # with nullcontext():
            # import_3d_model(mesh_path)
            split_mesh_path = os.path.dirname(mesh_path) + '/split'
            
            for obj_key, obj_config in scene_config.objects.items():
                import_3d_model(f'{split_mesh_path}/{obj_key}.obj')
                material_config = obj_config.material
                # Check if object has emissive (either is_lighting=True or emissive value > 0)
                has_emissive = obj_config.is_lighting or max(material_config.emissive) > 0.0
                if has_emissive:
                    strength = max(material_config.emissive)
                    normalized_color = [c / strength for c in material_config.emissive] if strength > 0.0 else [1.0, 1.0, 1.0]
                    # print(f"Creating emissive material for {obj_key} with strength {strength} and color {normalized_color}")
                    material = create_emissive_material(
                        strength=strength,
                        color=normalized_color,
                        material_name=f"{obj_key}"
                    )
                    # print(f"Emissive material created for {obj_key}", material)
                else:
                    # Create material based on material_type
                    from renderformer.data.blender.material_types import (
                        MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
                        MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
                        MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
                        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
                        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
                        supports_svbrdf,
                    )
                    
                    material_type = material_config.material_type
                    
                    if material_type in [MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR, MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR]:
                        # Use diffuse-specular material
                        params = material_config.diffuse_specular_params
                        if params is None:
                            # Fallback for SVBRDF (will use textures)
                            params = {'diffuse': [0.5, 0.5, 0.5], 'specular': [0.5, 0.5, 0.5], 'roughness': 0.5}
                        material = create_specular_roughness_material(
                            diffuse_color=tuple(params['diffuse']),
                            specular_color=tuple(params['specular']),
                            roughness=params['roughness'],
                            material_name=f"{obj_key}"
                        )
                    else:
                        # Use Principled BSDF for metallic-roughness materials
                        if material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS:
                            params = material_config.metallic_roughness_params
                            material = create_full_principled_bsdf_material(
                                base_color=tuple(params['base_color']),
                                metallic=params['metallic'],
                                roughness=params['roughness'],
                                material_name=f"{obj_key}"
                            )
                        elif material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION:
                            params = material_config.metallic_roughness_transmission_params
                            material = create_full_principled_bsdf_material(
                                base_color=tuple(params['base_color']),
                                metallic=params['metallic'],
                                roughness=params['roughness'],
                                transmission_weight=params['transmission_weight'],
                                ior=params['ior'],
                                material_name=f"{obj_key}"
                            )
                        else:
                            # SVBRDF metallic-roughness - will set up textures below
                            # Create base material with default values
                            material = create_full_principled_bsdf_material(
                                base_color=(0.5, 0.5, 0.5),
                                metallic=0.0,
                                roughness=0.5,
                                material_name=f"{obj_key}"
                            )
                    
                    # Handle texture maps for SVBRDF materials
                    if supports_svbrdf(material_type) and material_config.texture_map_path:
                        if material_type == MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR:
                            # Create texture nodes for diffuse-specular
                            diffuse_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                            normal_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                            roughness_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                            specular_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')

                            normal_map_node = material.node_tree.nodes.new(type='ShaderNodeNormalMap')
                            normal_map_node.space = 'TANGENT'
                            normal_map_node.inputs['Strength'].default_value = 1.0

                            # Load PBR textures
                            diffuse_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/diffuse.png')
                            diffuse_texture_node.image.colorspace_settings.name = 'Non-Color'
                            specular_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/specular.png')
                            specular_texture_node.image.colorspace_settings.name = 'Non-Color'
                            normal_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/normal.png')
                            normal_texture_node.image.colorspace_settings.name = 'Non-Color'
                            roughness_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/roughness.png')
                            roughness_texture_node.image.colorspace_settings.name = 'Non-Color'

                            # Connect texture nodes to Principled BSDF
                            bsdf = material.node_tree.nodes["Group"]
                            material.node_tree.links.new(bsdf.inputs["Diffuse"], diffuse_texture_node.outputs["Color"])
                            material.node_tree.links.new(normal_map_node.inputs['Color'], normal_texture_node.outputs["Color"])
                            material.node_tree.links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])
                            material.node_tree.links.new(bsdf.inputs["Roughness"], roughness_texture_node.outputs["Color"])
                            material.node_tree.links.new(bsdf.inputs["Specular"], specular_texture_node.outputs["Color"])
                            
                            # Add displacement mapping if heightmap is enabled
                            if material_config.use_heightmap:
                                height_path = f'{split_mesh_path}/{obj_key}/height.png'
                                if os.path.exists(height_path):
                                    # Create heightmap texture node
                                    height_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                                    height_texture_node.image = bpy.data.images.load(height_path)
                                    height_texture_node.image.colorspace_settings.name = 'Non-Color'
                                    
                                    # Create scale node (Value node) for displacement strength
                                    scale_node = material.node_tree.nodes.new(type='ShaderNodeValue')
                                    scale_node.name = 'Scale'
                                    scale_node.outputs[0].default_value = 0.050
                                    
                                    # Create displacement node
                                    displacement_node = material.node_tree.nodes.new(type='ShaderNodeDisplacement')
                                    displacement_node.inputs['Midlevel'].default_value = 0.500
                                    displacement_node.space = 'OBJECT'
                                    
                                    # Connect nodes
                                    material.node_tree.links.new(displacement_node.inputs['Height'], height_texture_node.outputs['Color'])
                                    material.node_tree.links.new(displacement_node.inputs['Scale'], scale_node.outputs[0])
                                    
                                    # Connect to Material Output
                                    material_output = material.node_tree.nodes.get("Material Output")
                                    if material_output is None:
                                        material_output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
                                    material.node_tree.links.new(displacement_node.outputs['Displacement'], material_output.inputs['Displacement'])
                                    
                                    # Set displacement method to 'BOTH' for proper displacement rendering
                                    material.displacement_method = 'BOTH'
                        
                        elif material_type == MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS:
                            # Create texture nodes for metallic-roughness
                            base_color_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                            normal_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                            roughness_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                            metallic_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')

                            normal_map_node = material.node_tree.nodes.new(type='ShaderNodeNormalMap')
                            normal_map_node.space = 'TANGENT'
                            normal_map_node.inputs['Strength'].default_value = 1.0

                            # Load PBR textures (try basecolor.png first, fallback to base_color.png)
                            basecolor_path = f'{split_mesh_path}/{obj_key}/basecolor.png'
                            base_color_path = f'{split_mesh_path}/{obj_key}/base_color.png'
                            if os.path.exists(basecolor_path):
                                base_color_texture_node.image = bpy.data.images.load(basecolor_path)
                            elif os.path.exists(base_color_path):
                                base_color_texture_node.image = bpy.data.images.load(base_color_path)
                            else:
                                raise FileNotFoundError(f"Neither basecolor.png nor base_color.png found in {split_mesh_path}/{obj_key}/")
                            base_color_texture_node.image.colorspace_settings.name = 'Non-Color'
                            metallic_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/metallic.png')
                            metallic_texture_node.image.colorspace_settings.name = 'Non-Color'
                            normal_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/normal.png')
                            normal_texture_node.image.colorspace_settings.name = 'Non-Color'
                            roughness_texture_node.image = bpy.data.images.load(f'{split_mesh_path}/{obj_key}/roughness.png')
                            roughness_texture_node.image.colorspace_settings.name = 'Non-Color'

                            # Connect texture nodes to Principled BSDF
                            bsdf = material.node_tree.nodes["Principled BSDF"]
                            material.node_tree.links.new(bsdf.inputs["Base Color"], base_color_texture_node.outputs["Color"])
                            material.node_tree.links.new(normal_map_node.inputs['Color'], normal_texture_node.outputs["Color"])
                            material.node_tree.links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])
                            material.node_tree.links.new(bsdf.inputs["Roughness"], roughness_texture_node.outputs["Color"])
                            material.node_tree.links.new(bsdf.inputs["Metallic"], metallic_texture_node.outputs["Color"])
                            
                            # Add displacement mapping if heightmap is enabled
                            if material_config.use_heightmap:
                                height_path = f'{split_mesh_path}/{obj_key}/height.png'
                                if os.path.exists(height_path):
                                    # Create heightmap texture node
                                    height_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
                                    height_texture_node.image = bpy.data.images.load(height_path)
                                    height_texture_node.image.colorspace_settings.name = 'Non-Color'
                                    
                                    # Create scale node (Value node) for displacement strength
                                    scale_node = material.node_tree.nodes.new(type='ShaderNodeValue')
                                    scale_node.name = 'Scale'
                                    scale_node.outputs[0].default_value = 0.050
                                    
                                    # Create displacement node
                                    displacement_node = material.node_tree.nodes.new(type='ShaderNodeDisplacement')
                                    displacement_node.inputs['Midlevel'].default_value = 0.500
                                    displacement_node.space = 'OBJECT'
                                    
                                    # Connect nodes
                                    material.node_tree.links.new(displacement_node.inputs['Height'], height_texture_node.outputs['Color'])
                                    material.node_tree.links.new(displacement_node.inputs['Scale'], scale_node.outputs[0])
                                    
                                    # Connect to Material Output
                                    material_output = material.node_tree.nodes.get("Material Output")
                                    if material_output is None:
                                        material_output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
                                    material.node_tree.links.new(displacement_node.outputs['Displacement'], material_output.inputs['Displacement'])
                                    
                                    # Set displacement method to 'BOTH' for proper displacement rendering
                                    material.displacement_method = 'BOTH'
                    
                    elif material_config.rand_tri_diffuse_seed:  # Use vertex color as diffuse color when have per-triangle diffuse color
                        if material_type in [MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR, MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR]:
                            bsdf = material.node_tree.nodes["Group"]
                        else:
                            bsdf = material.node_tree.nodes["Principled BSDF"]
                        vcol = material.node_tree.nodes.new(type="ShaderNodeVertexColor")
                        if material_type in [MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR, MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR]:
                            material.node_tree.links.new(vcol.outputs['Color'], bsdf.inputs['Diffuse'])
                        else:
                            material.node_tree.links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])
                
                for obj in scene_meshes():
                    if obj.name == obj_key:
                        obj.data.materials.clear()  # Clear all materials
                        obj.data.materials.append(material)
                        if has_emissive:
                            obj.visible_camera = False  # hide light source from camera
                        obj.rotation_mode = 'XYZ'
                        obj.rotation_euler = (0.0, 0.0, 0.0)  # Set rotation to (0, 0, 0)
                        
                        # Enable adaptive subdivision for objects with heightmap
                        if material_config.use_heightmap:
                            # Remove existing Subdivision modifier if any
                            # if "Subdivision" in obj.modifiers:
                            #     bpy.ops.object.modifier_remove(modifier="Subdivision")
                            
                            # Add Subdivision Surface modifier with SIMPLE type
                            bpy.context.view_layer.objects.active = obj
                            bpy.ops.object.modifier_add(type='SUBSURF')
                            obj.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
                            obj.modifiers["Subdivision"].levels = 6
                            obj.modifiers["Subdivision"].render_levels = 6

                            
                            # Enable adaptive subdivision
                            obj.cycles.use_adaptive_subdivision = True
            
            # Import and setup volumes
            if scene_config.volumes is not None:
                for volume_idx, volume_config in enumerate(scene_config.volumes):
                    # Import VDB file
                    vdb_path = volume_config.vdb_path
                    bpy.ops.object.volume_import(filepath=str(vdb_path))
                    
                    # Find the imported volume object
                    volume_obj = None
                    for obj in bpy.context.scene.objects:
                        if obj.type == 'VOLUME' and obj not in [v for v in bpy.context.scene.objects if v.type == 'VOLUME' and v != obj]:
                            # Check if this is the most recently imported volume
                            volume_obj = obj
                            break
                    
                    # Get the most recently imported volume (should be the one we just imported)
                    volume_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'VOLUME']
                    if volume_objects:
                        volume_obj = volume_objects[-1]  # Get the last one (most recently imported)
                    
                    if volume_obj is None:
                        print(f"Warning: Failed to import VDB file: {vdb_path}")
                        continue
                    
                    # Apply transform
                    transform = volume_config.transform
                    volume_obj.location = (transform.translation_x, transform.translation_y, transform.translation_z)
                    volume_obj.rotation_euler = (
                        np.deg2rad(transform.rotation_x),
                        np.deg2rad(transform.rotation_y),
                        np.deg2rad(transform.rotation_z)
                    )
                    volume_obj.scale = (transform.scale_x, transform.scale_y, transform.scale_z)
                    
                    # Setup volume shader with scattering and absorption scales
                    # Get or create material
                    if volume_obj.data.materials and volume_obj.data.materials[0]:
                        material = volume_obj.data.materials[0]
                    else:
                        material = bpy.data.materials.new(name=f"VolumeMaterial_{volume_idx}")
                        volume_obj.data.materials.append(material)
                    
                    # Enable nodes
                    material.use_nodes = True
                    nodes = material.node_tree.nodes
                    links = material.node_tree.links
                    
                    # Clear existing nodes
                    nodes.clear()
                    
                    # Create Volume Info node
                    volume_info = nodes.new(type='ShaderNodeVolumeInfo')
                    volume_info.location = (-600, 0)
                    
                    # Create first Multiply node (for Absorption: Density * absorption_scale)
                    multiply_abs = nodes.new(type='ShaderNodeVectorMath')
                    multiply_abs.operation = 'MULTIPLY'
                    multiply_abs.location = (-400, 100)
                    # Set the multiplier vector for absorption (RGB values)
                    multiply_abs.inputs[1].default_value = tuple(volume_config.absorption_scale)
                    
                    # Create second Multiply node (for Scatter: Density * scattering_scale)
                    multiply_scatter = nodes.new(type='ShaderNodeVectorMath')
                    multiply_scatter.operation = 'MULTIPLY'
                    multiply_scatter.location = (-400, -100)
                    # Set the multiplier vector for scattering (RGB values)
                    multiply_scatter.inputs[1].default_value = tuple(volume_config.scattering_scale)
                    
                    # Create Volume Coefficients node
                    volume_coeff = nodes.new(type='ShaderNodeVolumeCoefficients')
                    volume_coeff.location = (-200, 0)
                    # Set phase function to Henyey-Greenstein
                    volume_coeff.phase = 'HENYEY_GREENSTEIN'
                    # Set anisotropy to 0
                    volume_coeff.inputs['Anisotropy'].default_value = 0.0
                    # Set emission to (0, 0, 0)
                    volume_coeff.inputs['Emission Coefficients'].default_value = (0.0, 0.0, 0.0)
                    
                    # Create Material Output node
                    material_output = nodes.new(type='ShaderNodeOutputMaterial')
                    material_output.location = (0, 0)
                    
                    # Connect nodes
                    # Volume Info Density -> Multiply (Absorption)
                    links.new(volume_info.outputs['Density'], multiply_abs.inputs[0])
                    # Volume Info Density -> Multiply (Scatter)
                    links.new(volume_info.outputs['Density'], multiply_scatter.inputs[0])
                    # Multiply -> Volume Coefficients
                    links.new(multiply_abs.outputs['Vector'], volume_coeff.inputs['Absorption Coefficients'])
                    links.new(multiply_scatter.outputs['Vector'], volume_coeff.inputs['Scatter Coefficients'])
                    # Volume Coefficients -> Material Output Volume
                    links.new(volume_coeff.outputs['Volume'], material_output.inputs['Volume'])
                    
                    print(f"Volume {volume_idx} imported and shader setup completed with scattering={volume_config.scattering_scale} (RGB), absorption={volume_config.absorption_scale} (RGB)")

    def render_scene(camera_config: CameraConfig, output_image_path: str, output_obj_path: str, save_img_this: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        camera_pos = np.array(camera_config.position)
        look_at = np.array(camera_config.look_at)
        up = np.array(camera_config.up)
        fov = camera_config.fov
        
        c2w = look_at_to_c2w(camera_pos, look_at, up)
        camera = create_camera(c2w, fov)
        camera.data.clip_start = 1e-6  # set near plane to 1e-6 to avoid z-fighting
        bpy.context.scene.camera = camera

        # save_blend_file("output_dev/scene.blend")
        if skip_rendering:
            empty_img = np.zeros((resolution, resolution, 4), dtype=np.float32)
            empty_rgb = np.zeros((resolution, resolution, 3), dtype=np.float32)
            empty_depth = np.zeros((resolution, resolution, 1), dtype=np.float32)
            return empty_img, c2w, empty_depth, empty_rgb, empty_rgb, empty_rgb, empty_rgb
        render_stdout_ctx = nullcontext()
        if suppress_blender_stdout:
            from bpy_helper.utils import stdout_redirected
            render_stdout_ctx = stdout_redirected()

        with render_stdout_ctx, tempfile.TemporaryDirectory() as temp_dir:
        # with tempfile.TemporaryDirectory() as temp_dir:
        # with tempfile.NamedTemporaryFile(suffix=".exr") as img_f:
            # temp_img_path = img_f.name if not local_model else output_image_path.replace(".png", ".exr")
            bpy.context.scene.render.filepath = os.path.abspath(temp_dir + '/out.exr')
            # bpy.ops.render.render(animation=False, write_still=True)
            if rgb_only_render:
                bpy.ops.render.render(animation=False, write_still=True)
                img = simple_exr.read_exr(temp_dir + '/out.exr').copy()
                rgb = img[..., :3].copy()
                depth = np.zeros((resolution, resolution, 1), dtype=np.float32)
                diffuse = np.zeros((resolution, resolution, 3), dtype=np.float32)
                glossy = np.zeros((resolution, resolution, 3), dtype=np.float32)
                normal = np.zeros((resolution, resolution, 3), dtype=np.float32)
                albedo = np.zeros((resolution, resolution, 3), dtype=np.float32)
            else:
                render_with_compositing_nodes(output_folder_path=temp_dir, verbose=False)
                # if not local_model:
                # img = imageio.v3.imread(temp_dir + '/out.exr').copy()
                img = simple_exr.read_exr(temp_dir + '/out.exr').copy()
                # rgb = imageio.v3.imread(temp_dir + '/rgb0001.exr').copy()
                rgb = simple_exr.read_exr(temp_dir + '/rgb0001.exr').copy()
                # depth = imageio.v3.imread(temp_dir + '/depth0001.exr').copy()
                depth = simple_exr.read_exr(temp_dir + '/depth0001.exr').copy()
                # diffuse = imageio.v3.imread(temp_dir + '/diffuse0001.exr').copy()
                diffuse = simple_exr.read_exr(temp_dir + '/diffuse0001.exr').copy()
                # glossy = imageio.v3.imread(temp_dir + '/glossy0001.exr').copy()
                glossy = simple_exr.read_exr(temp_dir + '/glossy0001.exr').copy()
                # normal = imageio.v3.imread(temp_dir + '/normal0001.exr').copy()
                normal = simple_exr.read_exr(temp_dir + '/normal0001.exr').copy()
                albedo = simple_exr.read_exr(temp_dir + '/albedo0001.exr').copy()
            # else:
                # raise NotImplementedError("Local model is not supported")
                # import cv2
                # img = cv2.imread(temp_dir + '/out.exr', cv2.IMREAD_UNCHANGED)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if save_img_this:
                # Save PNG previews in display space. EXR/H5 outputs remain linear HDR.
                imageio.v3.imwrite(output_image_path, hdr_to_display_uint8(img))
                imageio.v3.imwrite(output_image_path + '_rgb.png', hdr_to_display_uint8(rgb))
                imageio.v3.imwrite(output_image_path + '_diffuse.png', hdr_to_display_uint8(diffuse))
                imageio.v3.imwrite(output_image_path + '_glossy.png', hdr_to_display_uint8(glossy))
                imageio.v3.imwrite(output_image_path + '_normal.png', (normal * 255).clip(0, 255).astype(np.uint8))
                imageio.v3.imwrite(output_image_path + '_albedo.png', hdr_to_display_uint8(albedo))

                # Save EXR (HDR, full dynamic range)
                simple_exr.write_exr(output_image_path.replace('.png', '.exr'), img)
                simple_exr.write_exr(output_image_path + '_rgb.exr', rgb)
                simple_exr.write_exr(output_image_path + '_diffuse.exr', diffuse)
                simple_exr.write_exr(output_image_path + '_glossy.exr', glossy)
                simple_exr.write_exr(output_image_path + '_depth.exr', depth)
                simple_exr.write_exr(output_image_path + '_normal.exr', normal)
                simple_exr.write_exr(output_image_path + '_albedo.exr', albedo)

        if save_scene_obj:
            bpy.ops.wm.obj_export(
                filepath=output_obj_path,
                export_uv=False,
                export_normals=True,
                export_colors=True,
                export_materials=True,
                export_triangulated_mesh=True,
                up_axis='Z',
                forward_axis='Y'
            )

        return img, c2w, depth, diffuse, glossy, normal, albedo

    setup_blender_scene(scene_config, mesh_path)

    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = spp
    bpy.context.scene.render.film_transparent = True if not include_background else False
    bpy.context.scene.render.image_settings.color_mode = 'RGBA' if not include_background else 'RGB'
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    if BLENDER_BACKEND in ('CPU', 'NONE', ''):
        bpy.context.scene.cycles.device = 'CPU'
    else:
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = BLENDER_BACKEND
    bpy.context.scene.render.threads = int(os.getenv('BLENDER_THREADS', '8'))
    bpy.context.scene.render.threads_mode = 'FIXED'

    if scene_config.env_map is not None:
        set_env_light(
            scene_config.env_map.env_map_path,
            strength=scene_config.env_map.strength,
            rotation_euler=(
                scene_config.env_map.rotation_x / 180.0 * np.pi,
                scene_config.env_map.rotation_y / 180.0 * np.pi,
                scene_config.env_map.rotation_z / 180.0 * np.pi,
            ),
            keep_other_lights=True
        )
    else:
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.  # remove all ambient

    if not rgb_only_render:
        create_compositing_nodes(
            enable_depth=True,
            enable_diffuse=True,
            enable_glossy=True,
            enable_normal=True,
            enable_albedo=True,
        )

    if render_repeats < 1:
        raise ValueError(f"render_repeats must be >= 1, got {render_repeats}")

    results = []
    repeat_times_sec: list[float] = []
    for repeat_idx in range(render_repeats):
        repeat_start = time.perf_counter()
        repeat_results = []
        for i, camera_config in tqdm(list(enumerate(scene_config.cameras)), desc=f"render repeat {repeat_idx + 1}/{render_repeats}"):
            save_img_this = save_img and (repeat_idx == render_repeats - 1)
            img, c2w, depth, diffuse, glossy, normal, albedo = render_scene(
                camera_config,
                f"{output_image_path}_{i}.png",
                f"{output_obj_path}_{i}.obj",
                save_img_this,
            )
            repeat_results.append((img, c2w, depth, diffuse, glossy, normal, albedo))
        repeat_time = time.perf_counter() - repeat_start
        repeat_times_sec.append(repeat_time)
        print(f"[RenderTiming] repeat={repeat_idx + 1}/{render_repeats}, elapsed_sec={repeat_time:.6f}")
        if repeat_idx == render_repeats - 1:
            results = repeat_results

    if repeat_times_sec:
        mean_all = float(np.mean(repeat_times_sec))
        print(f"[RenderTiming] avg_repeat_sec(all_{len(repeat_times_sec)})={mean_all:.6f}")
        if len(repeat_times_sec) > 1:
            mean_wo_first = float(np.mean(repeat_times_sec[1:]))
            print(f"[RenderTiming] avg_repeat_sec(exclude_first_{len(repeat_times_sec) - 1})={mean_wo_first:.6f}")

    # bpy.ops.file.pack_all()
    # save_blend_file('debug.blend')
    # exit(0)

    _maybe_save_blend(dump_blend_file, output_image_path, bpy.ops.file.pack_all, save_blend_file)

    if return_render_timing:
        return results, repeat_times_sec
    return results


def load_scene_config(scene_config_path: str) -> GeneratedConfig:
    with open(scene_config_path, 'r') as f:
        scene_config_dict = json.load(f)
    return from_dict(data_class=GeneratedConfig, data=scene_config_dict, config=Config(strict=True))


def get_projection_matrix(fovy, aspect_wh, near, far):
    proj_mtx = np.zeros((1, 4, 4), dtype=np.float32)
    proj_mtx[:, 0, 0] = 1.0 / (np.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / np.tan(fovy / 2.0)
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(c2w, proj_mtx):
    w2c = np.zeros(c2w.shape, dtype=c2w.dtype)
    w2c[:, :3, :3] = c2w[:, :3, :3].transpose(0, 2, 1)
    w2c[:, :3, 3:] = -np.matmul(c2w[:, :3, :3].transpose(0, 2, 1), c2w[:, :3, 3:])
    w2c[:, 3, 3] = 1.0
    mvp_mtx = np.matmul(proj_mtx, w2c)
    return mvp_mtx[0]


def is_black_image(img: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if an image is mostly black (indicating camera is inside scene or no visible content)
    
    Args:
        img: Image array with shape (H, W, C) in range [0, 1]
        threshold: Threshold for considering image as black (default: 0.01)
    
    Returns:
        bool: True if image is mostly black, False otherwise
    """
    # Check if the image is mostly black
    mean_intensity = np.mean(img[..., :3])
    return mean_intensity < threshold

def is_z_fighting(depth: np.ndarray, threshold: float = 1e-2) -> bool:
    """
    Check if the depth image is z-fighting
    """
    return np.min(depth[..., :3]) < threshold
