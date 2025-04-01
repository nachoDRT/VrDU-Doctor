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


def get_real_samples_material():

    blue_mat = None  # bpy.data.materials.get("BlueMaterial")
    green_mat = None  # bpy.data.materials.get("GreenMaterial")

    materials = [blue_mat, green_mat]

    for i, material in enumerate(materials):

        if material is None:
            material = bpy.data.materials.new(name="BlueMaterial")
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get("Principled BSDF")

            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = COLORS[i]

            materials[i] = material

    return materials


def load_data():
    path = bpy.path.abspath("//plots/df_all_pca_donut.xlsx")

    data = {}

    for sheet_name in SHEETS:
        df = pd.read_excel(path, sheet_name=sheet_name)
        data[sheet_name] = df

    return data


def real_samples_scatter_plot(df, mat):

    collection_name = "real_samples"
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

        scatter_coll.objects.link(sphere)
        if sphere.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(sphere)


def synth_samples_scatter_plot(df, mat):
    pass


if __name__ == "__main__":

    materials = get_real_samples_material()

    data = load_data()
    real_samples_scatter_plot(data["real"], materials[0])
    real_samples_scatter_plot(data["synthetic_es-digital-line-degradation-seq"], materials[1])
