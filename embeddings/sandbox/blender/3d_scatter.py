import bpy
import pandas as pd
import os
import mathutils
from sklearn.svm import LinearSVC
import pickle


FACTOR = 0.219938
COLORS = [(0.069, 0.35, 1, 1), (0.475, 0, 1, 1), (0.069, 0.35, 1, 0.75)]
SHEETS = [
    "real",
    "es-digital-line",
    "es-digital-paragraph",
    "es-digital-seq",
    "es-digital-rotation",
    "es-digital-zoom",
    "es-render-seq",
]

IMAGE_ROOT = bpy.path.abspath("//plots")
RENDER_RESULT = False
ANIMATION_PAUSE = 40
ANIMATION_STEP = 50
INCLUDE_TRAIL = False
GRADIENT_GREEN = (0.002, 0.077, 0.021, 1)
GRADIENT_RED = (0.183, 0, 0.008, 1)
GRADIENT_YELLOW = (0.897, 0.779, 0.250, 1)
PLANE_NAMES = ["PC1_PC2", "PC1_PC3", "PC2_PC3"]
FEATURES = [
    "v_info_blocks",
    "v_density",
    "layout",
    "columns",
    "grades",
    "source",
    "shadows",
    "header_badge",
    "signed",
    "stamped",
    "table_pos",
]


##############################
# Block 1: Synthetic Animation
##############################


def get_samples_materials():
    blue_mat = None  # bpy.data.materials.get("BlueMaterial")
    green_mat = None  # bpy.data.materials.get("GreenMaterial")
    material_synth_prox = None
    materials = [blue_mat, green_mat, material_synth_prox]

    for i, material in enumerate(materials):
        if material is None:
            mat_name = "BlueMaterial" if i == 0 else "GreenMaterial"
            material = bpy.data.materials.new(name=mat_name)
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = COLORS[i]
                bsdf.inputs["Roughness"].default_value = 1.0
            materials[i] = material

    return materials


def load_data(data_name: str = None):

    data = {}

    if data_name == "position":
        path = bpy.path.abspath("//plots/df_all_pca_donut.xlsx")
        sheets = SHEETS
    elif data_name == "distance":
        path = bpy.path.abspath("//plots/df_real_to_synth_distance_pca_donut.xlsx")
        sheets = SHEETS[1:]
    elif data_name == "f1":
        path = bpy.path.abspath("//plots/df_f1_donut.xlsx")
        sheets = [data_name]
    elif data_name == "planes":
        path = bpy.path.abspath("//plots/hiperplanes.xlsx")
        sheets = FEATURES
    else:
        path = bpy.path.abspath("//plots/hiperplanes.xlsx")
        sheets = SHEETS

    for sheet_name in sheets:
        df = pd.read_excel(path, sheet_name=sheet_name)
        data[sheet_name] = df

    return data


def scatter_plot(collection_name: str, df: pd.DataFrame, mat, keyframes: bool = False, r: float = 0.015):
    scatter_coll = bpy.data.collections.get(collection_name)
    if scatter_coll is None:
        scatter_coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(scatter_coll)

    for index, row in df.iterrows():
        x = row["PC1"] * FACTOR
        y = row["PC2"] * FACTOR
        z = row["PC3"] * FACTOR

        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, z))
        sphere = bpy.context.active_object

        if collection_name == "real_samples":
            sphere.name = row["name"]
        else:
            sphere.name = get_blender_name(row["name"])

        if sphere.data.materials:
            sphere.data.materials[0] = mat
        else:
            sphere.data.materials.append(mat)

        if keyframes:
            sphere.keyframe_insert(data_path="location", frame=0)

        scatter_coll.objects.link(sphere)
        if sphere.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(sphere)


def real_samples_scatter_plot(df, mat):
    scatter_plot("real_samples", df, mat)


def synth_samples_scatter_plot(df, mat, collection_name_detail: str = ""):
    scatter_plot(f"synthetic_samples_{collection_name_detail}", df, mat, keyframes=True, r=0.01)


def get_blender_name(excel_name: str) -> str:
    parts = excel_name.split("_", 1)
    return parts[1] if len(parts) > 1 else excel_name


def animate_synth_samples(data):
    synthetic_sheet_names = SHEETS[1:]
    current_frame = 0
    planes = ["PC1_PC2", "PC1_PC3", "PC2_PC3"]
    sheets_sequence = [
        "es-digital-line",
        "es-digital-paragraph",
        "es-digital-seq",
        "es-digital-rotation",
        "es-digital-zoom",
        "es-render-seq",
    ]

    for i, sheet_name in enumerate(synthetic_sheet_names):
        df = data[sheet_name]

        for _, row in df.iterrows():
            blender_name = get_blender_name(row["name"])
            sphere = bpy.data.objects.get(blender_name)

            if sphere is not None:
                x = row["PC1"] * FACTOR
                y = row["PC2"] * FACTOR
                z = row["PC3"] * FACTOR
                sphere.location = (x, y, z)

                if i == 0:
                    sphere.keyframe_insert(data_path="location", frame=current_frame)
                else:
                    sphere.keyframe_insert(data_path="location", frame=current_frame)
                    sphere.keyframe_insert(data_path="location", frame=current_frame + ANIMATION_PAUSE)

            else:
                print(f"Object {row['name']} not found for sheet {sheet_name}")

        if i == 0:
            current_frame += ANIMATION_STEP
        else:
            current_frame += ANIMATION_STEP + ANIMATION_PAUSE

    for plane in planes:
        mat_name = "Material__" + plane
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            print(f"Material {mat_name} not found")
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
        create_texture_sequence_node(mat, plane, sheets_sequence)


def create_texture_sequence_node(material, plane, sheets):
    nt = material.node_tree
    nodes = nt.nodes
    links = nt.links

    output_node = None
    for node in nodes:
        if node.type == "OUTPUT_MATERIAL":
            output_node = node
            break
    if output_node is None:
        output_node = nodes.new("ShaderNodeOutputMaterial")
        output_node.location = (300, 0)

    bsdf = None
    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf = node
            break
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    bsdf.inputs["Metallic"].default_value = 0.75
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.0
    bsdf.inputs["Alpha"].default_value = 1.0

    image_nodes = []
    for i, sheet in enumerate(sheets):
        # Img path: <IMAGE_ROOT>/<sheet>/<plane>_<sheet>.png
        image_file = os.path.join(IMAGE_ROOT, sheet, f"{plane}_{sheet}.png")
        if not os.path.isfile(image_file):
            print(f"Image not found: {image_file}")
            continue
        try:
            img = bpy.data.images.load(image_file, check_existing=True)
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
            continue

        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.label = f"{plane}_{sheet}"
        tex_node.location = (-600, 300 - i * 200)
        image_nodes.append(tex_node)

    if not image_nodes:
        print(f"No loaded imgs for {plane}")
        return

    if len(image_nodes) == 1:
        links.new(image_nodes[0].outputs["Color"], bsdf.inputs["Base Color"])
        return

    mix_nodes = []
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.location = (-300, 300)
    mix.inputs["Fac"].default_value = 0.0  # Show first img initially
    links.new(image_nodes[0].outputs["Color"], mix.inputs[1])
    links.new(image_nodes[1].outputs["Color"], mix.inputs[2])
    mix_nodes.append(mix)

    previous_node = mix
    for i in range(2, len(image_nodes)):
        mix_next = nodes.new("ShaderNodeMixRGB")
        mix_next.blend_type = "MIX"
        mix_next.location = (-100, 300 - (i - 1) * 200)
        mix_next.inputs["Fac"].default_value = 0.0  # Show previous img initially
        links.new(previous_node.outputs["Color"], mix_next.inputs[1])
        links.new(image_nodes[i].outputs["Color"], mix_next.inputs[2])
        mix_nodes.append(mix_next)
        previous_node = mix_next

    links.new(previous_node.outputs["Color"], bsdf.inputs["Base Color"])

    current_frame = 0
    if mix_nodes:
        first_mix = mix_nodes[0]
        first_mix.inputs["Fac"].default_value = 0.0
        first_mix.inputs["Fac"].keyframe_insert(data_path="default_value", frame=current_frame)
        first_mix.inputs["Fac"].default_value = 1.0
        first_mix.inputs["Fac"].keyframe_insert(data_path="default_value", frame=current_frame + ANIMATION_STEP)
        current_frame += ANIMATION_STEP

        for mix_node in mix_nodes[1:]:
            mix_node.inputs["Fac"].default_value = 0.0
            mix_node.inputs["Fac"].keyframe_insert(data_path="default_value", frame=current_frame)
            mix_node.inputs["Fac"].default_value = 1.0
            mix_node.inputs["Fac"].keyframe_insert(
                data_path="default_value", frame=current_frame + ANIMATION_PAUSE + ANIMATION_STEP
            )
            current_frame += ANIMATION_PAUSE + ANIMATION_STEP


def render_animation_from_cameras(hide_target: bool = False):
    if hide_target:
        bpy.data.collections["real_samples"].hide_render = True

    camera_names = ["Camera-ISO", "Camera-PC1vsPC2", "Camera-PC1vsPC3", "Camera-PC2vsPC3"]

    blend_filepath = bpy.data.filepath
    base_dir = os.path.dirname(blend_filepath)
    output_dir = os.path.join(base_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.use_file_extension = True

    for cam_name in camera_names:
        cam = bpy.data.objects.get(cam_name)
        if cam is None:
            print(f"Camera '{cam_name}' not found!")
            continue

        scene.camera = cam

        output_file = os.path.join(output_dir, f"results_{cam_name}.mp4")
        scene.render.filepath = output_file

        print(f"Rendering animation from camera: {cam_name} into file {output_file}")
        bpy.ops.render.render(animation=True)


def create_segment_trail_curves(obj, trail_mat):

    if not obj.animation_data or not obj.animation_data.action:
        return []

    fcurves = [fc for fc in obj.animation_data.action.fcurves if fc.data_path == "location" and fc.array_index == 0]
    if not fcurves:
        return []
    keyframes = sorted({int(point.co.x) for point in fcurves[0].keyframe_points})
    segments = []
    original_frame = bpy.context.scene.frame_current
    num_segments = len(keyframes) - 1

    for i in range(num_segments):
        frame_start = keyframes[i]
        frame_end = keyframes[i + 1]
        bpy.context.scene.frame_set(frame_start)
        pos_start = obj.matrix_world.translation.copy()
        bpy.context.scene.frame_set(frame_end)
        pos_end = obj.matrix_world.translation.copy()

        curve_data = bpy.data.curves.new(name=f"{obj.name}_Segment_{frame_start}_{frame_end}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_depth = 0.001  # Line width
        spline = curve_data.splines.new(type="POLY")
        spline.points.add(1)
        spline.points[0].co = (pos_start.x, pos_start.y, pos_start.z, 1)
        spline.points[1].co = (pos_end.x, pos_end.y, pos_end.z, 1)

        curve_obj = bpy.data.objects.new(curve_data.name, curve_data)

        traj_coll = bpy.data.collections.get("trajectories")
        if traj_coll is None:
            traj_coll = bpy.data.collections.new("trajectories")
            bpy.context.scene.collection.children.link(traj_coll)
        traj_coll.objects.link(curve_obj)

        if len(curve_obj.data.materials) == 0:
            curve_obj.data.materials.append(trail_mat)
        else:
            curve_obj.data.materials[0] = trail_mat

        curve_obj.hide_render = True
        curve_obj.keyframe_insert(data_path="hide_render", frame=frame_end - 1)
        curve_obj.hide_render = False
        curve_obj.keyframe_insert(data_path="hide_render", frame=frame_end)
        curve_obj.hide_render = False
        curve_obj.keyframe_insert(data_path="hide_render", frame=frame_end + ANIMATION_PAUSE - 1)
        curve_obj.hide_render = True
        curve_obj.keyframe_insert(data_path="hide_render", frame=frame_end + ANIMATION_PAUSE)

        segments.append(curve_obj)

    bpy.context.scene.frame_set(original_frame)

    return segments


def animate_synthetic_transformations():
    """
    Create an animation where synthetic samples travel along the 3D space
    generated by PCA. Include synthetic samples trails if needed.
    """

    materials = get_samples_materials()
    data = load_data("position")

    real_samples_scatter_plot(data["real"], materials[0])
    synth_samples_scatter_plot(data["es-digital-line"], materials[1])

    animate_synth_samples(data)

    if INCLUDE_TRAIL:
        synthetic_coll = bpy.data.collections.get("synthetic_samples")
        if synthetic_coll is not None:
            for obj in synthetic_coll.objects:
                create_segment_trail_curves(obj, materials[1])

    if RENDER_RESULT:
        render_animation_from_cameras(hide_target=True)


###############################
# Block 2: Static Visualization
###############################


def show_real_samples_by_proximity(sheet: str, include_synth: bool = False):
    """
    Create a static 3D plot where those real samples that are closest to synthetic samples
    are green (while the opposite are red).
    """

    data_pos = load_data("position")
    data_dist = load_data("distance")

    real_samples_as_gradient(
        collection_name="real_samples_proximity_gradient",
        data_to_plot=data_pos["real"],
        colored_by=data_dist[sheet],
        grad_key="mean_distance",
    )

    if include_synth:
        materials = get_samples_materials()
        synth_samples_scatter_plot(data_pos[sheet], materials[1])

    add_texture_to_planes()


def show_real_samples_by_f1(synth_version: str, include_synth: bool = False):
    """
    Create a static 3D plot where those real with better f1 score
    are green (while the opposite are red).
    """

    data_pos = load_data("position")
    data_f1 = load_data("f1")

    real_samples_as_gradient(
        collection_name=f"real_samples_f1_gradient_{synth_version}",
        data_to_plot=data_pos["real"],
        colored_by=data_f1["f1"],
        grad_key=synth_version,
        re_arrange_colors=True,
    )

    if include_synth:
        materials = get_samples_materials()
        synth_samples_scatter_plot(data_pos[synth_version], materials[1])

    add_texture_to_planes()


def show_real_samples_by_proximity_vs_f1_mismatch():
    pass


def real_samples_as_gradient(
    collection_name: str,
    data_to_plot: pd.DataFrame,
    colored_by: pd.DataFrame,
    grad_key: str,
    range: tuple = None,
    r: float = 0.015,
    re_arrange_colors: bool = False,
):

    if not range:
        min_grad = colored_by[grad_key].min()
        max_grad = colored_by[grad_key].max()
        mid_grad = (min_grad + max_grad) / 2
    else:
        min_grad = 0.0
        max_grad = 1.0
        mid_grad = 0.5

    scatter_coll = bpy.data.collections.get(collection_name)
    if scatter_coll is None:
        scatter_coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(scatter_coll)

    for (index, row), (_, row_to_color) in zip(data_to_plot.iterrows(), colored_by.iterrows()):
        x = row["PC1"] * FACTOR
        y = row["PC2"] * FACTOR
        z = row["PC3"] * FACTOR

        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, z))
        sphere = bpy.context.active_object

        sphere.name = row["name"]

        grad_value = row_to_color[grad_key]
        color = compute_gradient_color(
            grad_value,
            min_grad,
            mid_grad,
            max_grad,
            GRADIENT_GREEN,
            GRADIENT_YELLOW,
            GRADIENT_RED,
        )

        if re_arrange_colors:
            color = compute_gradient_color(
                grad_value,
                min_grad,
                mid_grad,
                max_grad,
                GRADIENT_RED,
                GRADIENT_YELLOW,
                GRADIENT_GREEN,
            )

        mat_name = f"GradientMat_{collection_name}_{index}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = color

        if sphere.data.materials:
            sphere.data.materials[0] = mat
        else:
            sphere.data.materials.append(mat)

        scatter_coll.objects.link(sphere)
        if sphere.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(sphere)


def compute_gradient_color(value, min_val, mid_val, max_val, color_min, color_mid, color_max):
    if value <= mid_val:

        factor = (value - min_val) / (mid_val - min_val) if mid_val != min_val else 0.0
        r = color_min[0] + factor * (color_mid[0] - color_min[0])
        g = color_min[1] + factor * (color_mid[1] - color_min[1])
        b = color_min[2] + factor * (color_mid[2] - color_min[2])
        a = color_min[3] + factor * (color_mid[3] - color_min[3])
    else:

        factor = (value - mid_val) / (max_val - mid_val) if max_val != mid_val else 0.0
        r = color_mid[0] + factor * (color_max[0] - color_mid[0])
        g = color_mid[1] + factor * (color_max[1] - color_mid[1])
        b = color_mid[2] + factor * (color_max[2] - color_mid[2])
        a = color_mid[3] + factor * (color_max[3] - color_mid[3])

    return (r, g, b, a)


def add_texture_to_planes():

    img_path = os.path.join(IMAGE_ROOT, "blank_plot", "blank_plot.png")

    try:
        img = bpy.data.images.load(img_path)
    except Exception as e:
        print(f"Imposible to load {img_path}: {e}")
        return

    for plane_name in PLANE_NAMES:
        plane_obj = bpy.data.objects.get(plane_name)
        if plane_obj is None:
            print(f"Object {plane_name} not found in scene")
            continue

        if plane_obj.data.materials:
            mat = plane_obj.data.materials[0]
            if mat is None:
                mat = bpy.data.materials.new(name=f"Mat_{plane_name}")
                plane_obj.data.materials[0] = mat
        else:
            mat = bpy.data.materials.new(name=f"Mat_{plane_name}")
            plane_obj.data.materials.append(mat)

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = nodes.get("Principled BSDF")
        if not bsdf:
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)

        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = img
        tex_image.location = (-300, 300)

        if not bsdf.inputs["Base Color"].is_linked:
            links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])

        material_output = nodes.get("Material Output")
        if not material_output:
            material_output = nodes.new("ShaderNodeOutputMaterial")
            material_output.location = (200, 0)
        if not material_output.inputs["Surface"].is_linked:
            links.new(bsdf.outputs["BSDF"], material_output.inputs["Surface"])


def create_plane_from_hyperplane(w, b, size=10):

    normal = mathutils.Vector(w).normalized()

    if abs(normal.z) > 1e-6:
        # For the z axis, we assume the current behavior is correct.
        point = mathutils.Vector((0, 0, -b / w[2]))
    elif abs(normal.y) > 1e-6:
        # For the y axis, invert the sign to obtain the expected intercept.
        point = mathutils.Vector((0, -b / w[1], 0))
    else:
        # For the x axis, invert the sign as well.
        point = mathutils.Vector((-b / w[0], 0, 0))

    point = mathutils.Vector([p_i * f_i for p_i, f_i in zip(point, (FACTOR, FACTOR, FACTOR))])

    # Create the plane in Blender with the desired size at the calculated point
    bpy.ops.mesh.primitive_plane_add(size=size, location=point)
    plane = bpy.context.active_object

    # The default normal of the created plane is (0, 0, 1)
    default_normal = mathutils.Vector((0, 0, 1))
    # Calculate the rotation needed to align the default normal to the hyperplane normal
    rotation_quat = default_normal.rotation_difference(normal)

    # Apply the rotation to the plane
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = rotation_quat

    mat_name = "HyperplaneMaterial"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = COLORS[2]
            bsdf.inputs["Alpha"].default_value = 0.2
            bsdf.inputs["Roughness"].default_value = 1
            bsdf.inputs["Metallic"].default_value = 1

    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)

    return plane


def show_planes():

    data = load_data("planes")

    for key, values in data.items():

        collection_name = f"cluster_division_{key}"
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)

        model_path = bpy.path.abspath(f"//plots/model_{key}.npz")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        clf_ = LinearSVC()
        clf_.classes_ = model_data["classes"]
        clf_.coef_ = model_data["coef"]
        clf_.intercept_ = model_data["intercept"]

        # print(clf_.predict([[-0.8, -1, 3]]))

        for _, row in data[key].iterrows():
            w = [row["w1"], row["w2"], row["w3"]]
            b = row["b"]
            plane_obj = create_plane_from_hyperplane(w, b)
            new_collection.objects.link(plane_obj)
            bpy.context.scene.collection.objects.unlink(plane_obj)


def new_samples_target_features(target_name):

    target_obj = bpy.data.objects.get(target_name)
    target_features = {}

    if target_obj:
        coords = target_obj.location
        location = [coords[0] / FACTOR, coords[1] / FACTOR, coords[2] / FACTOR]

    data = load_data("planes")

    for key in data.keys():
        model_path = bpy.path.abspath(f"//plots/model_{key}.npz")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        clf_ = LinearSVC()
        clf_.classes_ = model_data["classes"]
        clf_.coef_ = model_data["coef"]
        clf_.intercept_ = model_data["intercept"]

        pred = clf_.predict([location])[0]

        if hasattr(pred, "item"):
            target_features[key] = pred.item()
        else:
            target_features[key] = pred

    print(target_features.keys())
    print(target_features.values())

    return target_features


def get_trajectory_intersections(epsilon=1e-6):
    """
    Count how many plane meshes are intersected by the line segment
    between two target objects in the scene.
    """

    # Gather all collections whose names start with "cluster_division_"
    plane_collections = [coll for coll in bpy.data.collections if coll.name.startswith("cluster_division_")]

    # Retrieve the two endpoint objects for the trajectory
    obj_a = bpy.data.objects["new_samples_target"]
    obj_b = bpy.data.objects["new_samples_target_"]

    new_samples_target_features("new_samples_target")
    new_samples_target_features("new_samples_target_")

    # Collect all mesh objects (planes) from those collections
    planes = []
    for coll in plane_collections:
        planes.extend([obj for obj in coll.objects if obj.type == "MESH"])

    # Define the segment endpoints in world space
    P1 = obj_a.location if hasattr(obj_a, "location") else obj_a
    P2 = obj_b.location if hasattr(obj_b, "location") else obj_b
    d = P2 - P1  # direction vector from P1 to P2

    count = 0

    for plane in planes:

        # Compute the plane normal in world space (local Z axis → world)
        normal = plane.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
        normal.normalize()
        P0 = plane.matrix_world.translation  # a point on the plane

        # Check for near-parallelism: if denom ≈ 0, no intersection
        denom = normal.dot(d)
        if abs(denom) < epsilon:
            continue

        # Compute the intersection parameter t along the segment
        t = normal.dot(P0 - P1) / denom
        if not (0.0 <= t <= 1.0):
            # Intersection lies outside the segment
            continue

        # Calculate the intersection point in world coordinates
        intersec_world = P1 + t * d

        # Transform the intersection point into the plane’s local space
        intersec_local = plane.matrix_world.inverted() @ intersec_world

        # Get the plane’s local bounding box corners to find min/max X and Y
        local_corners = [mathutils.Vector(corner) for corner in plane.bound_box]
        xs = [v.x for v in local_corners]
        ys = [v.y for v in local_corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # If the local intersection point falls inside the bounding box, count it
        if (
            min_x - epsilon <= intersec_local.x <= max_x + epsilon
            and min_y - epsilon <= intersec_local.y <= max_y + epsilon
        ):
            count += 1

    return count


if __name__ == "__main__":

    # animate_synthetic_transformations()

    # show_real_samples_by_proximity("es-render-seq", include_synth=True)
    # show_real_samples_by_f1("es-digital-seq", include_synth=False)

    # TODO
    # show_real_samples_by_feature()

    # show_planes()

    """Plot a specific synth subset"""
    # materials = get_samples_materials()
    # data = load_data("position")
    # subset = "es-render-seq"
    # synth_samples_scatter_plot(data[subset], materials[1], collection_name_detail=subset)

    """ Get the features from the position of the target object """
    # new_samples_target_features("new_samples_target")

    """ Get the number of intersections between the trajectory AB and the planes it trespasses"""
    num_planes = get_trajectory_intersections()
    print(f"Num of planes trespassed: {num_planes}")

    # """ Get wormhole maps"""
    # get_wormholes()
