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


if __name__ == "__main__":

    real_samples_mat = get_real_samples_material()
    df_data = load_data()
    generate_scatter_plot(df_data, real_samples_mat)
