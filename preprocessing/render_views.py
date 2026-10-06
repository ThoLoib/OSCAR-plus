import os
import json
import shutil
import bpy
import bmesh
import sys
import mathutils
from mathutils import Vector, Matrix, Euler
import numpy as np
import math
import time
import pickle
from PIL import Image
import random
import argparse

## solve the division problem
#from decimal import Decimal, getcontext
#getcontext().prec = 28  # Set the precision for the decimal calculations.
 
# Both locations are passed in by the caller (preprocess_gallery.py or the
# Stage-4 scripts), which resolve them from config/paths.yaml. This script runs
# inside Blender's own Python, so it reads them from the environment instead of
# the YAML — and refuses to guess if they are missing.
object_folder = os.environ.get('OBJECT_FOLDER', '')
object_images = os.environ.get('OBJECT_IMAGES', '')
if not object_folder or not object_images:
    raise SystemExit(
        "render_views.py: set OBJECT_FOLDER (CAD directory) and OBJECT_IMAGES "
        "(output directory). Normally you do not call this script directly — "
        "use: python3 preprocess_gallery.py --dataset <name> --step render")
allowed_exts = ['.glb', '.obj', '.ply']  # PLY added for BOP datasets (T-LESS, LM-O, ITODD)
overwrite_existing = os.environ.get('OVERWRITE_EXISTING', '0') == '1'
# Optional: Wandzeit je Objekt (Import + alle View-Renders) als JSON ablegen —
# reine Messung, Render-Ergebnisse unveraendert. Die Differenz zwischen
# aeusserer Wandzeit und script_total_s ist der Blender-Binaerstart.
render_timing_json = os.environ.get('RENDER_TIMING_JSON', '').strip()
_timing_t0 = time.perf_counter()
_timing_per_object = {}
render_only = os.environ.get('RENDER_ONLY', '').strip()
num_views = int(os.environ.get('NUM_VIEWS', '42'))  # Ablation O4; default 42 (full icosphere)
# Parallel sharding (opt-in). When SHARD_TOTAL > 1, this process renders only
# the models whose position in the sorted model list satisfies
#   index % SHARD_TOTAL == SHARD_INDEX
# so N processes with SHARD_INDEX=0..N-1 split the dataset into N disjoint,
# balanced parts and can run concurrently.
# Default SHARD_TOTAL=1 => renders everything: behaviour is unchanged.
shard_index = int(os.environ.get('SHARD_INDEX', '0'))
shard_total = int(os.environ.get('SHARD_TOTAL', '1'))


# ---------------------------------------------------------------------------
# Icosphere viewpoint generation (following CNOS, Nguyen et al. ICCV 2023)
# ---------------------------------------------------------------------------
# CNOS uses 42 viewpoints from a Blender icosphere at subdivision level 1
# (upper hemisphere only).  Subdivision level 2 gives 162 viewpoints but
# showed no improvement over 42 in their experiments (CNOS Table 3).
#
# Ref: Nguyen et al., "CNOS: A Strong Baseline for CAD-based Novel Object
#      Segmentation", ICCV 2023.
#      Code: https://github.com/nv-nguyen/cnos
#      Viewpoint generation: https://github.com/nv-nguyen/template-pose
#
# We generate all 42 icosphere vertices and order them by farthest-point
# sampling (FPS) so that any prefix (the first N views) gives the best
# possible angular coverage for that N.  This allows the thesis ablation
# O4 to test any V in {8, 16, 42} — or any other count up to 42 — without
# needing separate rendering runs.
# ---------------------------------------------------------------------------

def _generate_icosphere_vertices(subdivisions=1):
    """Generate vertices of an icosphere via recursive subdivision.

    Starts from a regular icosahedron (12 vertices, 20 faces) and
    subdivides each triangle into 4 smaller triangles, projecting new
    vertices onto the unit sphere.

    Subdivision 0 → 12 vertices
    Subdivision 1 → 42 vertices  (CNOS default)
    Subdivision 2 → 162 vertices

    Returns:
        np.ndarray of shape (N, 3) — unit-sphere vertex positions.
    """
    # --- Icosahedron base vertices ---
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    verts = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    # Normalize to unit sphere
    verts = [np.array(v, dtype=np.float64) for v in verts]
    verts = [v / np.linalg.norm(v) for v in verts]

    # --- Icosahedron faces (20 triangles) ---
    faces = [
        (0,11,5), (0,5,1), (0,1,7), (0,7,10), (0,10,11),
        (1,5,9), (5,11,4), (11,10,2), (10,7,6), (7,1,8),
        (3,9,4), (3,4,2), (3,2,6), (3,6,8), (3,8,9),
        (4,9,5), (2,4,11), (6,2,10), (8,6,7), (9,8,1),
    ]

    # --- Subdivide ---
    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        for tri in faces:
            mids = []
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                if edge not in edge_midpoints:
                    mid = (verts[edge[0]] + verts[edge[1]]) / 2.0
                    mid = mid / np.linalg.norm(mid)  # project onto sphere
                    edge_midpoints[edge] = len(verts)
                    verts.append(mid)
                mids.append(edge_midpoints[edge])
            a, b, c = tri
            m0, m1, m2 = mids
            new_faces.extend([
                (a, m0, m2), (b, m1, m0), (c, m2, m1), (m0, m1, m2)
            ])
        faces = new_faces

    return np.array(verts, dtype=np.float64)


def _fps_ordering(points):
    """Order points by farthest-point sampling for maximum spread.

    Starts from the point with highest elevation (most "top-down" view),
    then iteratively picks the point farthest from all already-selected
    points.  Any prefix of the returned ordering gives the best angular
    coverage for that count.

    Args:
        points: (N, 3) array of unit-sphere positions.

    Returns:
        List of indices into points, length N.
    """
    N = len(points)
    # Start from highest-elevation point
    start = int(np.argmax(points[:, 2]))
    selected = [start]
    min_dists = np.full(N, np.inf)

    for _ in range(N - 1):
        last = points[selected[-1]]
        # Geodesic distance ≈ Euclidean distance on unit sphere is monotonic
        dists = np.linalg.norm(points - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        # Exclude already selected
        min_dists_copy = min_dists.copy()
        for idx in selected:
            min_dists_copy[idx] = -1.0
        next_idx = int(np.argmax(min_dists_copy))
        selected.append(next_idx)

    return selected


def generate_icosphere_positions(num_views, distance, ratio, subdivisions=1):
    """Generate camera positions from an icosphere, ordered by FPS.

    Following CNOS (Nguyen et al., ICCV 2023): icosphere subdivision 1
    gives 42 upper-hemisphere viewpoints.  FPS ordering ensures that any
    prefix (first N views) gives approximately optimal angular coverage.

    Args:
        num_views: How many views to generate (max 42 for subdiv=1).
        distance: Object bounding-box max dimension.
        ratio: Camera distance multiplier (default 1.15).
        subdivisions: Icosphere subdivision level (1→42, 2→162).

    Returns:
        List of Vector positions, length = min(num_views, available).
    """
    all_verts = _generate_icosphere_vertices(subdivisions)

    # CNOS uses all 42 icosphere vertices (full sphere), not just the upper
    # hemisphere.  This provides view coverage for objects that can be seen
    # from any angle (not just tabletop).
    print(f"Icosphere subdiv={subdivisions}: {len(all_verts)} vertices")

    # FPS ordering for optimal subsets
    ordering = _fps_ordering(all_verts)

    # Scale to camera distance
    r = distance * ratio
    n = min(num_views, len(all_verts))
    positions = []
    for i in range(n):
        v = all_verts[ordering[i]]
        positions.append(Vector((v[0] * r, v[1] * r, v[2] * r)))

    return positions


# Generic model filenames that indicate "one model per directory" layout.
# When a mesh file has one of these names, the parent directory is the object ID.
# When the filename is NOT generic (e.g. airplane_test_0001.obj), the filename
# stem IS the object ID — this handles MI3DOR, HouseCat6D, SHREC'18 where
# multiple objects live in the same directory.
_GENERIC_MODEL_NAMES = {
    'textured_simple.obj', 'model.obj', 'textured.obj',
    'model.glb', 'model.ply', 'mesh.obj', 'mesh.ply',
}


def infer_model_id(file_path):
    norm = os.path.normpath(file_path)
    parts = norm.split(os.sep)
    fname = parts[-1].lower()

    # ycbv: .../{object_id}/meshes/textured_simple.obj
    if len(parts) >= 3 and parts[-2] == 'meshes':
        return parts[-3]

    # Generic filename (model.ply, textured_simple.obj, etc.)
    # → parent directory is the object ID (BOP, ycbv direct)
    if fname in _GENERIC_MODEL_NAMES and len(parts) >= 2:
        return parts[-2]

    # Specific filename (airplane_test_0001.obj, 02691156_xxx.obj, etc.)
    # → filename stem is the object ID (MI3DOR, SHREC'18, HouseCat6D)
    return os.path.splitext(os.path.basename(file_path))[0]


def file_priority(fname):
    """Priority for choosing between multiple mesh files for the same object.

    Lower = preferred.  Only matters when multiple files map to the same
    model_id (e.g. textured_simple.obj and model.obj in the same dir).
    For datasets where each file IS a separate object (MI3DOR, SHREC'18),
    there's only one file per model_id so priority is irrelevant.
    """
    base = fname.lower()
    if base == 'textured_simple.obj':
        return 0
    if base == 'model.obj':
        return 1
    if base == 'textured.obj':
        return 2
    if base.endswith('.glb'):
        return 3
    if base.endswith('.ply'):
        return 4
    return 9


model_choice = {}
for dirpath, _, files in os.walk(object_folder):
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in allowed_exts:
            continue
        file_path = os.path.join(dirpath, fname)
        model_id = infer_model_id(file_path)
        prio = file_priority(fname)
        current = model_choice.get(model_id)
        if current is None or prio < current[0]:
            model_choice[model_id] = (prio, file_path)

model_files = sorted((mid, fp) for mid, (_, fp) in model_choice.items())

pending_models = []
for _shard_idx, (model_id, filename) in enumerate(model_files):
    if shard_total > 1 and _shard_idx % shard_total != shard_index:
        continue
    if render_only and model_id != render_only:
        continue
    folder_path = os.path.join(object_images, model_id)
    if overwrite_existing:
        pending_models.append((model_id, filename))
    elif not os.path.isdir(folder_path):
        pending_models.append((model_id, filename))
    else:
        # Check if all views are rendered (resumability after interruption)
        all_done = all(
            os.path.exists(os.path.join(folder_path, f'{model_id}_{v}.png'))
            for v in range(num_views)
        )
        if not all_done:
            pending_models.append((model_id, filename))

print(f"Total models found: {len(model_files)}")
print(f"Models to render now: {len(pending_models)} (overwrite_existing={overwrite_existing})")
for mid, fname in pending_models[:20]:
    print(f" - {fname} (output folder: {mid})")

# Final cleanup (optional)
#bpy.ops.wm.quit_blender()

bpy.context.scene.render.engine = 'CYCLES'
# small samples for fast rendering
bpy.context.scene.cycles.samples = 16
# bpy.context.scene.cycles.samples = 128
# Color management: use Standard, not Blender's default Filmic. Filmic compresses
# the white world background to grey (~206) and crushes the shading of untextured
# CAD models into a narrow mid-grey band, producing the washed-out white-on-white
# look. Standard maps the white world to 255 and preserves lamp shading, matching
# the dataset-provided reference renders (white bg, shaded grey object).
bpy.context.scene.view_settings.view_transform = 'Standard'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    if 'NVIDIA' in d['name']:
        d["use"] = 1 # Using all devices, include GPU and CPU
    else:
        d["use"] = 0 # Using all devices, include GPU and CPU)

render_prefs = bpy.context.preferences.addons['cycles'].preferences
render_device_type = render_prefs.compute_device_type
compute_device_type = render_prefs.devices[0].type if len(render_prefs.devices) > 0 else None
# Check if the compute device type is GPU
if render_device_type == 'CUDA' and compute_device_type == 'CUDA':
    # GPU is being used for rendering
    print("Using GPU for rendering")
else:
    # GPU is not being used for rendering
    print("Not using GPU for rendering")


# if the object is too far away from the origin, pull it closer
def check_object_location(mesh_objects, max_distance):
    # Compute the maximum distance of any object from the origin
    max_obj_distance = max(obj.location.length for obj in mesh_objects)

    # If any object is too far from the origin, move all mesh_objects closer to the origin
    if max_obj_distance > max_distance:
        bbox_center, _ = compute_bounding_box(mesh_objects)
        for obj in mesh_objects:
            obj.location -= bbox_center
        bpy.context.view_layer.update()

    # Compute the maximum distance again and check if it's within range
    max_obj_distance = max(obj.location.length for obj in mesh_objects)
    if max_obj_distance > max_distance:
        print("Objects are still too far from the origin. Please adjust the object locations and try again.")
        return False
    else:
        return True

# compute the bounding box of the mesh objects
def compute_bounding_box(mesh_objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in mesh_objects:
        matrix_world = obj.matrix_world
        mesh = obj.data

        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co

            min_coords = Vector((min(min_coords[i], global_coord[i]) for i in range(3)))
            max_coords = Vector((max(max_coords[i], global_coord[i]) for i in range(3)))

    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    return bbox_center, bbox_size

# normalize objects 
def normalize_and_center_objects(mesh_objects, normalization_range):

    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    # Check the location of the objects and move them closer to the origin if necessary
    check_object_location(mesh_objects, 1000)

    # Compute the bounding box of the objects again after making adjustments
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    # Normalize the objects within a certain range
    max_dimension = max(bbox_size.x, bbox_size.y, bbox_size.z)
    scaling_factor = normalization_range / max_dimension

    for obj in mesh_objects:
        mesh = obj.data
        matrix_world = obj.matrix_world
        inv_matrix_world = matrix_world.inverted()
        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co
            global_coord -= bbox_center
            global_coord *= scaling_factor
            vert.co = inv_matrix_world @ global_coord
        mesh.update()
        obj.data.update()

    bpy.context.view_layer.update()
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    return bbox_center, bbox_size

# check if rendered object will cross the boundary of the image
def project_points_to_camera_space(obj, camera):
    bpy.context.view_layer.update()
    # Get the 8 corners of the bounding box in local space
    bbox_local = [Vector(corner) for corner in obj.bound_box]

    # Transform bounding box corners to world space
    bbox_world = [obj.matrix_world @ corner for corner in bbox_local]
    bbox_world = [np.array(corner) for corner in bbox_world]  # convert to numpy

    # Get the 4x4 transformation matrix of the camera
    RT = np.array(camera.matrix_world.inverted())
    RT = RT[:3, :4]  # Remove the last row to make it a 3x4 matrix

    # Get the intrinsic matrix K from the camera properties
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    f_x = width / 2.0 / np.tan(camera.data.angle / 2.0)
    f_y = height / 2.0 / np.tan(camera.data.angle / 2.0)
    c_x = width / 2.0
    c_y = height / 2.0

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    bbox_camera = []
    bbox_image = []

    for vertex in bbox_world:
        # Transform from world to camera space
        XYZ_camera = np.dot(RT, np.append(vertex, 1))  # Append 1 to make it a 4-element vector for multiplication with RT

        # Project from camera space to image space
        XYZ_image = np.dot(K, XYZ_camera)

        # Homogenize to get pixel coordinates
        XYZ_image /= XYZ_image[2]

        bbox_camera.append(XYZ_camera)
        bbox_image.append(XYZ_image[:2])  # Keep only x and y

    # Check if the coordinates are within the normalized device coordinates [-1, 1]
    is_within_ndc = all(np.all(np.abs(vertex[:2]) <= 1) for vertex in bbox_image)

    # print(is_within_ndc)
    return bbox_image

# prepare the scene
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Create lights
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()

def create_light(name, light_type, energy, location, rotation):
    bpy.ops.object.light_add(type=light_type, align='WORLD', location=location, scale=(1, 1, 1))
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.rotation_euler = rotation
    return light

def three_point_lighting():
    
    # Key light
    key_light = create_light(
        name="KeyLight",
        light_type='AREA',
        energy=1000,
        location=(4, -4, 4),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    key_light.data.size = 2

    # Fill light
    fill_light = create_light(
        name="FillLight",
        light_type='AREA',
        energy=300,
        location=(-4, -4, 2),
        rotation=(math.radians(45), 0, math.radians(135))
    )
    fill_light.data.size = 2

    # Rim/Back light
    rim_light = create_light(
        name="RimLight",
        light_type='AREA',
        energy=600,
        location=(0, 4, 0),
        rotation=(math.radians(45), 0, math.radians(225))
    )
    rim_light.data.size = 2

def get_3x4_RT_matrix_from_blender(cam):
            # Use matrix_world instead to account for all constraints
            location, rotation = cam.matrix_world.decompose()[0:2]
            R_world2bcam = rotation.to_matrix().transposed()

            # Use location from matrix_world to account for constraints:     
            T_world2bcam = -1*R_world2bcam @ location

            # put into 3x4 matrix
            RT = Matrix((
                R_world2bcam[0][:] + (T_world2bcam[0],),
                R_world2bcam[1][:] + (T_world2bcam[1],),
                R_world2bcam[2][:] + (T_world2bcam[2],)
                ))
            return RT

def setup_camera_lighting(cam, key=1.8, fill=0.5):
    """Camera-relative sun rig, parented to the camera so it orbits with it.

    OSCAR's fixed world-space three-point rig, combined with the orbiting
    camera, lit each of the 42 icosphere views differently (front views bright,
    back views crushed to black). Parenting SUN lamps to the camera keeps a
    constant key/fill direction relative to the viewer, so every view is shaded
    the same even grey as the dataset-provided renders. Suns (no distance
    falloff) plus the grey world ambient floor keep shadows off pure black.
    Directions/energies tuned to match the reference (obj mean ~190, min ~100).
    """
    def sun(name, energy, rot_deg):
        bpy.ops.object.light_add(type='SUN')
        o = bpy.context.active_object
        o.name = name
        o.data.energy = energy
        o.parent = cam
        o.matrix_parent_inverse = cam.matrix_world.inverted()
        o.rotation_euler = Euler([math.radians(a) for a in rot_deg], 'XYZ')
        return o
    sun("KeySun",  key,  (-35,  30, 0))   # upper-left of the viewer
    sun("FillSun", fill, ( 30, -25, 0))   # lower-right fill


def setup_render_world():
    """White background to the camera, zero ambient for object lighting.

    The camera sees a pure-white world (background -> 255 under Standard color
    management), while shading/indirect rays see black, so the object is lit
    ONLY by the three-point lamp rig. This reproduces the dataset-provided
    renders (white bg + shaded grey object) instead of letting the white world
    flood the untextured surface into a flat mid-grey wash. Uses a Light Path
    "Is Camera Ray" mask, so it stays a single opaque RGB pass (no compositing).
    """
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputWorld')
    bg = nt.nodes.new('ShaderNodeBackground')
    lp = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixRGB')
    mix.inputs['Color1'].default_value = (0.3, 0.3, 0.3, 1.0)  # non-camera: grey ambient floor
    mix.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)      # camera rays: white background
    nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(mix.outputs['Color'], bg.inputs['Color'])
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove orphan data (optional but helps prevent memory bloat)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)

#def create_camera():
#    bpy.ops.object.camera_add(location=(0, -3, 1), rotation=(math.radians(90), 0, 0))
#    cam = bpy.context.active_object
#    bpy.context.scene.camera = cam
#    return cam

### Main Code ###

# SHREC'18 (SketchUp/3D-Warehouse export) needs two source-specific fixups that
# must NOT touch other datasets:
#   1. Inverted "d" convention (d = transparency, 0 = opaque) — corrected by
#      alpha = 1 - d. Other datasets use standard d (opaque) or none, so
#      inverting their alpha would wrongly make them transparent.
#   2. Double-walled geometry: nearly every surface is two coincident faces,
#      whose overlapping twins often carry different materials (textured vs an
#      untextured black one). Once opaque they z-fight into a checkerboard;
#      resolved by collapsing each coincident set to one face, keeping the
#      textured one.
IS_SHREC18 = 'shrec18' in object_folder

def _material_has_texture(mat):
    if mat is None or not mat.use_nodes:
        return False
    return any(node.type == 'TEX_IMAGE' for node in mat.node_tree.nodes)

def recalculate_normals_outward(obj):
    """Weld the mesh, then make face normals consistent and outward-pointing.

    Many MI3DOR .obj meshes are unwelded "triangle soup" (coincident duplicate
    vertices, no `vn` lines) with inconsistent face winding, so Blender derives
    inward-pointing normals for whole panels (e.g. the flat bed headboard). Any
    normal-dependent shading (diffuse, AO) then leaves those faces pure black.
    Recalculating alone can't help because "outside" is undefined on a non-
    manifold soup — so first merge vertices by distance to make the surface
    manifold, THEN recompute outward normals. Skipped for meshes with authored
    split normals (textured datasets like GSO/YCB-V) to preserve their shading.
    """
    me = obj.data
    if getattr(me, "has_custom_normals", False):
        return
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()


def resolve_coincident_faces(obj):
    me = obj.data
    materials = me.materials
    textured_flags = [_material_has_texture(m) for m in materials]

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()

    # Group faces by the rounded positions of their vertices — NOT by vertex
    # index, and WITHOUT welding vertices. An earlier version welded via
    # remove_doubles to make coincident faces share indices, but remove_doubles
    # drops coincident duplicate faces itself, arbitrarily, before the
    # keep-the-textured-face choice below runs — which deleted textured faces
    # (e.g. a TV screen image) whenever the untextured twin happened to win, and
    # also collapsed nearby distinct geometry. Position grouping leaves the kept
    # face — its vertices, UVs and material — completely untouched.
    groups = {}
    for face in bm.faces:
        key = tuple(sorted(
            (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
            for v in face.verts))
        groups.setdefault(key, []).append(face)

    to_delete = []
    for faces in groups.values():
        if len(faces) < 2:
            continue
        # Keep a textured face if any twin has one; else keep the first.
        keep = next((f for f in faces
                     if f.material_index < len(textured_flags)
                     and textured_flags[f.material_index]), faces[0])
        to_delete.extend(f for f in faces if f is not keep)

    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context='FACES')
        bm.to_mesh(me)
    bm.free()
    me.update()

# Create new folder structure
os.makedirs(object_images, exist_ok=True)

# Set up the camera-relative light rig ONCE, parented to the scene camera so it
# orbits with every icosphere view. Every model is normalized to a unit bbox at
# the origin, so one rig lights all of them the same. Deselect afterwards so the
# loop's object.delete() cannot remove a light.
setup_camera_lighting(bpy.data.objects['Camera'])
bpy.ops.object.select_all(action='DESELECT')

for _obj_idx, (model_id, filename) in enumerate(pending_models):
    _timing_obj_start = time.perf_counter()
    # Create a folder for each model
    model_dir = os.path.join(object_images, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Copy the model file into it
    src = filename
    dst = os.path.join(model_dir, os.path.basename(filename))


    print(f"📁 Created folder for '{model_id}'")

    model_path = filename
 #   clear_scene()
 #   three_point_lighting()
    print(f"📥 Importing model: {model_path}")
    bpy.ops.object.delete()

    # Import model into Blender
    _, ext = os.path.splitext(filename)
    if ext.lower() == ".glb" or ext.lower() == ".gltf":
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif ext.lower() == ".obj":
        bpy.ops.import_scene.obj(filepath=model_path)
    elif ext.lower() == ".ply":
        bpy.ops.import_mesh.ply(filepath=model_path)
        # PLY models (BOP datasets) often have vertex colors but no material.
        # Create a material that uses vertex colors so they render correctly.
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            if not mesh.vertex_colors and not mesh.color_attributes:
                continue
            mat = bpy.data.materials.new(name="VertexColorMat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            # Vertex Color node → Principled BSDF → Output
            vc_node = nodes.new('ShaderNodeVertexColor')
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            output = nodes.new('ShaderNodeOutputMaterial')
            links.new(vc_node.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    print('begin*************')
    # Assuming objects are mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    print(mesh_objects)

    # Fix inward/inconsistent normals (untextured CAD with no `vn`) so surfaces
    # like the bed headboard don't render as solid black. No-op for meshes with
    # authored normals.
    for obj in mesh_objects:
        recalculate_normals_outward(obj)

    # SHREC'18 (SketchUp/3D-Warehouse export) writes the MTL "d" field with the
    # inverse of its official meaning: it uses d as TRANSPARENCY (0 = opaque,
    # 1 = fully transparent), not dissolve (1 = opaque). Evidence: 14465 of the
    # 15362 materials are d 0 and only 18 are d 1 — the dataset is furniture, so
    # "almost everything opaque, a few glass/window parts transparent" is the
    # only sensible reading. Blender's importer takes d literally as dissolve
    # and sets Principled Alpha = d, so d 0 becomes fully transparent (the whole
    # object vanishes). Correct it by inverting: alpha = 1 - d. This is safe
    # because every material in this dataset carries an explicit d line, so the
    # imported Alpha always equals the file's d.
    if IS_SHREC18:
        for obj in mesh_objects:
            for mat in obj.data.materials:
                if mat is None or not mat.use_nodes:
                    continue
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is None:
                    continue
                alpha_input = bsdf.inputs.get("Alpha")
                if alpha_input is not None and not alpha_input.is_linked:
                    alpha_input.default_value = 1.0 - alpha_input.default_value

        for obj in mesh_objects:
            resolve_coincident_faces(obj)

    # Compute the bounding box for the objects
    normalization_range = 1.0
    bbox_center, bbox_size = normalize_and_center_objects(mesh_objects, normalization_range)

    distance = max(bbox_size.x, bbox_size.y, bbox_size.z)
    ratio = 1.15

    camera = bpy.context.scene.camera
    name = model_id

    # --- Generate icosphere viewpoints (CNOS-style, FPS-ordered) ---
    ico_positions = generate_icosphere_positions(num_views, distance, ratio)
    actual_num_views = len(ico_positions)
    print(f"  Rendering {actual_num_views} icosphere views for '{model_id}'")

    # --- Step -1: transparent background render to calibrate camera distance ---
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    camera.location = ico_positions[0]  # use first view for calibration

    direction = (bbox_center - camera.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = quat.to_euler()
    camera.data.clip_start = 0.1
    camera.data.clip_end = max(1000, distance * 2)
    bpy.context.scene.camera = bpy.data.objects['Camera']
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    bg_path = os.path.join(model_dir, f'{name}_bg.png')
    bpy.context.scene.render.filepath = bg_path
    if not os.path.exists(bg_path) or overwrite_existing:
        bpy.ops.render.render(write_still=True)

    # Check if object fits in frame; increase ratio if needed
    img = Image.open(bg_path)
    img_array = np.array(img)
    if np.sum(img_array < 10) > 1020000:
        print(name, 'WARNING: rendered image may contain too much white space')

    while True:
        flag_list = []
        for obj in mesh_objects:
            bbox_image = project_points_to_camera_space(obj, camera)
            if np.max(np.array(bbox_image) > 512) or np.min(np.array(bbox_image) < 0):
                flag_list.append(0)
                ratio += 0.1
                # Regenerate positions with new ratio
                ico_positions = generate_icosphere_positions(num_views, distance, ratio)
                camera.location = ico_positions[0]
                direction = (bbox_center - camera.location).normalized()
                quat = direction.to_track_quat('-Z', 'Y')
                camera.rotation_euler = quat.to_euler()
        if len(flag_list) == 0:
            break

    # --- White background to the camera, lamp-only lighting for the object ---
    setup_render_world()
    bpy.context.scene.render.film_transparent = False

    # --- Render all views ---
    for view_idx, cam_pos in enumerate(ico_positions):
        camera.location = cam_pos

        # Point camera at bounding box center
        direction = (bbox_center - camera.location).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()

        camera.data.clip_start = 0.1
        camera.data.clip_end = max(1000, distance * 2)
        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        file_path = os.path.join(model_dir, f'{name}_{view_idx}.png')
        bpy.context.scene.render.filepath = file_path
        if os.path.exists(file_path) and not overwrite_existing:
            continue

        bpy.ops.render.render(write_still=True)

        # Save camera matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(model_dir, f"{model_id}_view{view_idx}_CamMatrix.npy")
        np.save(RT_path, RT)

    _timing_per_object[model_id] = time.perf_counter() - _timing_obj_start

if render_timing_json:
    with open(render_timing_json, 'w') as _f:
        json.dump({'per_object_s': _timing_per_object,
                   'script_total_s': time.perf_counter() - _timing_t0}, _f, indent=1)
    print(f"[timing] {len(_timing_per_object)} objects -> {render_timing_json}")

bpy.ops.wm.quit_blender()
