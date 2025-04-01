import bpy
import pandas as pd


FACTOR = 0.219938
REAL_SAMPLES_BLUE = (0.069, 0.35, 1, 1)


def get_real_samples_material():

    blue_mat = bpy.data.materials.get("BlueMaterial")
    if blue_mat is None:
        blue_mat = bpy.data.materials.new(name="BlueMaterial")
        blue_mat.use_nodes = True
        bsdf = blue_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = REAL_SAMPLES_BLUE

    return blue_mat


def load_data():
    path = bpy.path.abspath("//plots/df_all_pca_donut.xlsx")
    df = pd.read_excel(path)

    return df


def generate_scatter_plot(df, mat):

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

        if sphere.data.materials:
            sphere.data.materials[0] = mat
        else:
            sphere.data.materials.append(mat)

        scatter_coll.objects.link(sphere)
        if sphere.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(sphere)


if __name__ == "__main__":

    real_samples_mat = get_real_samples_material()
    df_data = load_data()
    generate_scatter_plot(df_data, real_samples_mat)
