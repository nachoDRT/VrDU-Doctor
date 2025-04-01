import bpy
import pandas as pd

FACTOR = 0.219938
COLORS = [(0.069, 0.35, 1, 1), (0.475, 0, 1, 1)]
SHEETS = [
    "real",
    "synthetic_es-digital-line-degradation-seq",
    "synthetic_es-digital-paragraph-degradation-seq",
    "synthetic_es-digital-seq",
    "synthetic_es-digital-rotation-degradation-seq",
    "synthetic_es-digital-zoom-degradation-seq",
    "synthetic_es-render-seq",
]


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


def scatter_plot(collection_name: str, df: pd.DataFrame, mat, keyframes: bool = False):
    scatter_coll = bpy.data.collections.get(collection_name)
    if scatter_coll is None:
        scatter_coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(scatter_coll)

    for index, row in df.iterrows():
        x = row["PC1"] * FACTOR
        y = row["PC2"] * FACTOR
        z = row["PC3"] * FACTOR

        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.015, location=(x, y, z))
        sphere = bpy.context.active_object

        sphere.name = row["name"]

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
    scatter_plot("synthetic_samples", df, mat, keyframes=True)


def get_blender_name(excel_name: str) -> str:
    parts = excel_name.split("_", 1)
    return parts[1] if len(parts) > 1 else excel_name


def animate_synth_samples(data, frame_step=50):

    synthetic_sheet_names = SHEETS[1:]

    for i, sheet_name in enumerate(synthetic_sheet_names[1:], start=1):
        print(sheet_name)
        frame = i * frame_step
        df = data[sheet_name]

        for index, row in df.iterrows():
            blender_name = get_blender_name(row["name"])
            sphere = bpy.data.objects.get(blender_name)
            if sphere is not None:
                x = row["PC1"] * FACTOR
                y = row["PC2"] * FACTOR
                z = row["PC3"] * FACTOR

                sphere.location = (x, y, z)
                sphere.keyframe_insert(data_path="location", frame=frame)
                sphere.keyframe_insert(data_path="location", frame=frame + 20)

            else:
                print(f"Objeto {row['name']} no encontrado para la sheet {sheet_name}")


if __name__ == "__main__":

    materials = get_samples_materials()
    data = load_data()

    real_samples_scatter_plot(data["real"], materials[0])
    synth_samples_scatter_plot(data["synthetic_es-digital-line-degradation-seq"], materials[1])

    animate_synth_samples(data, frame_step=50)
