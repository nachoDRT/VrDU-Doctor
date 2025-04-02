import bpy
import pandas as pd
import os

FACTOR = 0.219938
COLORS = [(0.069, 0.35, 1, 1), (0.475, 0, 1, 1)]
SHEETS = [
    "real",
    "es-digital-line",
    "es-digital-paragraph",
    "es-digital-seq",
    "es-digital-rotation",
    "es-digital-zoom",
    "es-render-seq",
]

RENDER_RESULT = True
RENDER_PAUSE = 50


def get_samples_materials():
    blue_mat = None  # bpy.data.materials.get("BlueMaterial")
    green_mat = None  # bpy.data.materials.get("GreenMaterial")
    materials = [blue_mat, green_mat]

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


def load_data():
    path = bpy.path.abspath("//plots/df_all_pca_donut.xlsx")
    data = {}
    for sheet_name in SHEETS:
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
            sphere.name = get_blender_name(row["name"])  # row["name"]

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


def synth_samples_scatter_plot(df, mat):
    scatter_plot("synthetic_samples", df, mat, keyframes=True, r=0.01)


def get_blender_name(excel_name: str) -> str:
    parts = excel_name.split("_", 1)
    return parts[1] if len(parts) > 1 else excel_name


def animate_synth_samples(data, frame_step=20):
    synthetic_sheet_names = SHEETS[1:]
    current_frame = frame_step

    for sheet_name in synthetic_sheet_names:
        df = data[sheet_name]

        for index, row in df.iterrows():
            blender_name = get_blender_name(row["name"])
            sphere = bpy.data.objects.get(blender_name)

            if sphere is not None:
                x = row["PC1"] * FACTOR
                y = row["PC2"] * FACTOR
                z = row["PC3"] * FACTOR
                sphere.location = (x, y, z)

                sphere.keyframe_insert(data_path="location", frame=current_frame)
                sphere.keyframe_insert(data_path="location", frame=current_frame + RENDER_PAUSE)

            else:
                print(f"Objeto {row['name']} no encontrado para la sheet {sheet_name}")

        current_frame += frame_step + RENDER_PAUSE


def render_animation_from_cameras(hide_target: bool = False):
    if hide_target:
        bpy.data.collections["real_samples"].hide_render = True

    camera_names = ["Camera-PC1vsPC2", "Camera-PC1vsPC3", "Camera-PC2vsPC3"]

    blend_filepath = bpy.data.filepath
    base_dir = os.path.dirname(blend_filepath)
    output_dir = os.path.join(base_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene

    # Configurar salida de render a formato MP4
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
        curve_obj.keyframe_insert(data_path="hide_render", frame=frame_end + RENDER_PAUSE - 1)
        curve_obj.hide_render = True
        curve_obj.keyframe_insert(data_path="hide_render", frame=frame_end + RENDER_PAUSE)

        segments.append(curve_obj)

    bpy.context.scene.frame_set(original_frame)

    return segments


if __name__ == "__main__":

    materials = get_samples_materials()
    data = load_data()

    real_samples_scatter_plot(data["real"], materials[0])
    synth_samples_scatter_plot(data["es-digital-line"], materials[1])

    animate_synth_samples(data, frame_step=50)

    synthetic_coll = bpy.data.collections.get("synthetic_samples")
    if synthetic_coll is not None:
        for obj in synthetic_coll.objects:
            create_segment_trail_curves(obj, materials[1])

    if RENDER_RESULT:
        render_animation_from_cameras(hide_target=True)
