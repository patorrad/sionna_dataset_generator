# HELPER FUNCTIONS FOR SCENE GENERATION FROM HEIGHTMAP
# Pydelatin is doing the heavy lifting for taking the heightmap to a mesh,
# https://github.com/kylebarron/pydelatin


import drjit as dr
import mitsuba as mi
from matplotlib import pyplot as plt
from pydelatin import Delatin
from pydelatin.util import rescale_positions
import numpy as np
mi.set_variant("llvm_ad_rgb")


def render_hm(heightmap, title='lunar mesh', rescale=False, detail =.1, base = True, display=True):
    # RENDER GENERATED TERRAIN
    heightmap = np.copy(heightmap) - np.min(heightmap)
    if rescale:
        scale_val, _ = heightmap.shape
        # print(scale_val)
        heightmap/=scale_val
        
    base_height=0
    
    if base:
        base_height = 20
    
    if rescale:
        tin = Delatin(heightmap, max_error=detail/scale_val, base_height=base_height/scale_val)
    else:
        tin = Delatin(heightmap, max_error=detail, base_height=base_height)
        
    vertices = tin.vertices
    triangles = tin.triangles
    bounds1, bounds2 = heightmap.shape
    
    if rescale: 
        bounds = [-0.5, -0.5, 0.5, 0.5]
        rescaled_vertices = rescale_positions(vertices, bounds)
    else:
        bounds = [-bounds1//2, -bounds2//2, bounds1//2, bounds2//2]
        rescaled_vertices = rescale_positions(vertices, bounds)

    vertex_positions = np.ravel(rescaled_vertices)
    face_indices = np.ravel(triangles)

    # CHANGE IF USING CUDA
    vertex_positions_dr = dr.llvm.Float(*vertex_positions.astype(np.float32))
    face_indices_dr = dr.llvm.UInt(*face_indices)
    # RENDER 
    mesh = mi.Mesh(
        "terrain_mesh",
        vertex_count=vertices.shape[0],
        face_count=triangles.shape[0],
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = vertex_positions_dr
    mesh_params["faces"] = face_indices_dr
    mesh_params.update()
    
    if display:
        if rescale:
            origin1 = [0,-2,1]
        else:
            origin1 = [0, -2048, 1024]
        above = [0, -1, 2]
        scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path"},
                "light": {
                    "type": "directional",
                    "direction": [1, .2, -.4],  # Adjust light direction
                    "irradiance": {"type": "rgb", "value": [3.0, 3.0, 3.0]},  # Increase irradiance for brighter light
                },
                "sensor": {
                    "type": "perspective",
                    "to_world": mi.ScalarTransform4f.look_at(
                        mi.ScalarPoint3f(origin1),
                        mi.ScalarPoint3f([0, 0, 0]),
                        mi.ScalarPoint3f([0, 0, 1])
                    ),
                },
                "terrain_mesh": mesh,
            }
        )
        
        img = mi.render(scene, spp=32)
        plt.figure(figsize=(12, 8))  
        plt.axis("off")
        plt.title(title)
        plt.imshow(mi.util.convert_to_bitmap(img))
        plt.show()
            
    return mesh


def save_scene_xml(filename="lunar_scene.xml", plyname="terrain_mesh.ply"):
    scene_xml = f"""<?xml version="1.0" ?>
<scene version="2.1.0">
    <default name="spp" value="4096"/>
    <default name="resx" value="1024"/>
    <default name="resy" value="768"/>
    <integrator type="path">
        <integer name="max_depth" value="12"/>
    </integrator>
    <bsdf type="twosided" id="mat-itu_concrete">
        <bsdf type="diffuse">
            <rgb value="0.539479 0.539479 0.53948" name="reflectance"/>
        </bsdf>
    </bsdf>
    <bsdf type="twosided" id="mat-itu_marble">
        <bsdf type="diffuse">
            <rgb value="0.701101 0.644479 0.48515" name="reflectance"/>
        </bsdf>
    </bsdf>
    <bsdf type="twosided" id="mat-itu_metal">
        <bsdf type="diffuse">
            <rgb value="0.219526 0.219526 0.254152" name="reflectance"/>
        </bsdf>
    </bsdf>
    <bsdf type="twosided" id="mat-itu_wood">
        <bsdf type="diffuse">
            <rgb value="0.043 0.58 0.184" name="reflectance"/>
        </bsdf>
    </bsdf>
    <bsdf type="twosided" id="mat-itu_wet_ground">
        <bsdf type="diffuse">
            <rgb value="0.91 0.569 0.055" name="reflectance"/>
        </bsdf>
    </bsdf>
    <emitter type="directional">
        <vector name="direction" x="1" y="0.2" z="-0.4"/>
        <rgb name="irradiance" value="3.0, 3.0, 3.0"/>
    </emitter>
    <sensor type="perspective">
        <transform name="to_world">
            <lookat origin="0,-2,1" target="0, 0, 0" up="0, 0, 1"/>
        </transform>
    </sensor>
    <shape type="ply" id="mesh-ground">
        <string name="filename" value="{plyname}"/>
        <ref id="mat-itu_concrete" name="bsdf"/>
        <boolean name="face_normals" value="true"/>
    </shape>
</scene>"""

    with open(filename, "w") as f:
        f.write(scene_xml)
    print(f"Scene saved to {filename}")
    
    
    
def save_scene(heightmap, filename, rescale=False, detail =.1, base = True, display=True):
    mesh = render_hm(heightmap=heightmap, title=filename, rescale=rescale, detail=detail, base=base, display=display)
    mesh.write_ply(f'{filename}_mesh.ply')
    plyname = filename.split("/")[-1]
    save_scene_xml(f'{filename}_scene.xml', f'{plyname}_mesh.ply')
    
    
    
if __name__=='__main__':
    print('loading dtms')
    ex = np.load("dtms_heightmap_samples/generated_numpy/test_samples1024 (1).npy")

    
    print('saving scene')
    save_scene(ex[2], 'scenes/test_scenes/test1', display=False)