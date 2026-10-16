import bpy
import numpy as np
import argparse
import sys
from pathlib import Path


def water_material():
    watmat = bpy.data.materials.new(name="Water")
    watmat.use_nodes = True
    nodes = watmat.node_tree.nodes
    links = watmat.node_tree.links
    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (300, 0)

    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_node.location = (0, 0)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    principled_node.inputs['Base Color'].default_value = (0.464, 0.820, 1.0, 1.0)
    principled_node.inputs['Metallic'].default_value = 0.0
    principled_node.inputs['Roughness'].default_value = 0.01
    principled_node.inputs['IOR'].default_value = 1.333
    principled_node.inputs['Transmission Weight'].default_value = 1.0
    principled_node.inputs['Alpha'].default_value = 1.0
    return watmat


def geometry_nodes(volume: bpy.types.Object, material, threshold: float):
    geo_modifier = volume.modifiers.new(name="WaterSurface", type='NODES')
    tree = bpy.data.node_groups.new(name="WaterSurfaceRender", type='GeometryNodeTree')
    geo_modifier.node_group = tree

    tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = tree.nodes
    links = tree.links

    input_node = nodes.new(type='NodeGroupInput')
    input_node.location = (-600, 0)

    volume_to_mesh = nodes.new(type='GeometryNodeVolumeToMesh')
    volume_to_mesh.location = (-350, 0)
    volume_to_mesh.resolution_mode = 'GRID'
    volume_to_mesh.inputs['Threshold'].default_value = threshold
    volume_to_mesh.inputs['Adaptivity'].default_value = 0.0

    smoothing_node = nodes.new(type='GeometryNodeSetShadeSmooth')
    smoothing_node.location = (-100, 0)
    smoothing_node.domain = 'FACE'

    set_material_node = nodes.new(type='GeometryNodeSetMaterial')
    set_material_node.location = (150, 0)
    set_material_node.inputs['Material'].default_value = material

    output_node = nodes.new(type='NodeGroupOutput')
    output_node.location = (400, 0)

    links.new(input_node.outputs['Geometry'], volume_to_mesh.inputs['Volume'])
    links.new(volume_to_mesh.outputs['Mesh'], smoothing_node.inputs['Geometry'])
    links.new(smoothing_node.outputs['Geometry'], set_material_node.inputs['Geometry'])
    links.new(set_material_node.outputs['Geometry'], output_node.inputs['Geometry'])

    return geo_modifier


def import_vdb_sequence(vdb_dir: Path):
    files = sorted(vdb_dir.glob("*.vdb"))
    if not files:
        raise ValueError(f"No VDB files found in {vdb_dir}")

    bpy.ops.object.volume_import(filepath=str(files[0].resolve()))
    volume = bpy.context.active_object
    volume.name = "Water"

    volume.data.filepath = str(files[0].resolve())
    volume.data.is_sequence = True
    volume.data.frame_start = 1
    volume.data.frame_duration = len(files)
    volume.data.frame_offset = 0
    volume.data.sequence_mode = 'CLIP'
    # rotate by 90 around x-axis since by default volume is misoriented
    volume.rotation_euler[0] = np.pi / 2

    return volume, len(files)


def setup_floor():
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -0.1))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (10, 10, 10)

    material = bpy.data.materials.new(name="Floor")
    material.diffuse_color = (0.5, 0.5, 0.5, 1)
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs['Roughness'].default_value = 0.5
    floor.data.materials.append(material)


def setup_camera():
    bpy.ops.object.camera_add(location=(8, -12, 7))
    camera = bpy.context.active_object
    camera.name = "Camera"

    direction = -camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 50

    bpy.context.scene.camera = camera


def setup_lighting():
    for name, location, power, size in [
        ("Key", (5, -5, 8), 1500, 5),
        ("Fill", (-5, -3, 5), 800, 5),
        ("Rim", (0, 5, 7), 2000, 4),
    ]:
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.active_object
        light.name = name
        light.data.energy = power
        light.data.shape = 'DISK'
        light.data.size = size
        light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()


def setup_render(nframes: int, output_dir: Path):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = 24

    scene.frame_start = 1
    scene.frame_end = nframes
    scene.frame_set(1)

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output_dir.resolve()) + "/"


def setup_scene(vdb_dir: Path, output_dir: Path, threshold: float = 0.1):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    volume, nframes = import_vdb_sequence(vdb_dir)
    material = water_material()
    volume.data.materials.append(material)
    geometry_nodes(volume, material, threshold)

    setup_floor()
    setup_camera()
    setup_lighting()
    setup_render(nframes, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str((output_dir / "water.blend").resolve()))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("vdb_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.1)
    parsed = parser.parse_args(args)

    setup_scene(parsed.vdb_dir, parsed.output_dir, parsed.threshold)
