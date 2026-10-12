import typer
import bpy
from pathlib import Path

def water_material():
    watmat = bpy.data.materials.new(name="Water")
    watmat.use_nodes = True
    nodes = watmat.node_tree.nodes
    links = watmat.node_tree.links

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (300, 0)

    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_node.location = (0, 0)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    principled_node.inputs['Base Color'].default_value = (0.0, 0.4, 0.8, 1.0)
    principled_node.inputs['Metallic'].default_value = 0.0
    principled_node.inputs['Roughness'].default_value = 0.0
    principled_node.inputs['IOR'].default_value = 1.33
    principled_node.inputs['Transmission Weight'].default_value = 1.0
    principled_node.inputs['Alpha'].default_value = 1.0
    return watmat

def geometry_nodes(volume: bpy.types.Object):
    geo_modifier = volume.modifiers.new(name="WaterSurface", type='NODES')
    geometry_node_tree = bpy.data.node_groups.new(name="WaterSurfaceRender", type='GeometryNodeTree')
    geo_modifier.node_group = geometry_node_tree

    nodes = geometry_node_tree.nodes
    links = geometry_node_tree.links
    nodes.clear()

    geometry_node_tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    geometry_node_tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    input_node = nodes.new(type='NodeGroupInput')
    input_node.location = (-300, 0)
    volume_to_mesh = nodes.new(type='GeometryNodeVolumeToMesh')
    volume_to_mesh.location = (0, 0)
    smoothing_node = nodes.new(type='GeometryNodeSetShadeSmooth')
    smoothing_node.domain = 'FACE'
    set_material_node = nodes.new(type='GeometryNodeSetMaterial')
    set_material_node.inputs['Material'].default_value = volume.data.materials[0]
    output_node = nodes.new(type='NodeGroupOutput')
    output_node.location = (300, 0)
    links.new(input_node.outputs['Geometry'], volume_to_mesh.inputs['Volume'])
    links.new(volume_to_mesh.outputs['Mesh'], smoothing_node.inputs['Geometry'])
    links.new(smoothing_node.outputs['Geometry'], set_material_node.inputs['Geometry'])
    links.new(set_material_node.outputs['Geometry'], output_node.inputs['Geometry'])
    return geo_modifier

def setup_scene(vdb_dir: Path):
    sample = vdb_dir.glob("*.vdb").__next__()
    bpy.ops.object.volume_import(filepath=sample.as_posix())
    volume = bpy.context.active_object
    nframes = len(list(vdb_dir.glob("*.vdb")))
    volume.data.frame_duration = nframes
    watermat = water_material()
    volume.data.materials.append(watermat)
    geometry_modifier = geometry_nodes(volume)

if __name__ == "__main__":
    typer.run(setup_scene)

