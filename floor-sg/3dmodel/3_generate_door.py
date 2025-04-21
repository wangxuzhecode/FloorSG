import numpy as np
import pyvista as pv


def save_obj(filepath, filename, mesh):
    vertices = mesh.points  # 获取顶点坐标 (N x 3 numpy array)
    faces = mesh.faces  # 获取面片数据 (包含面片顶点索引的 numpy array)
    with open(filepath, 'w') as f:
        f.write("mtllib " + filename +".mtl\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        idx = 0
        f.write("vt 0.0 0.0\nvt 1.0 0.0\nvt 0.0 1.0\nvt 1.0 1.0\n")

        f.write("usemtl " + filename + "\n")
        while idx < len(faces):
            num_vertices = faces[idx]  # 面片顶点数量
            idx += 1
            face = faces[idx:idx + num_vertices]
            idx += num_vertices
            # `.obj` 格式的面片是 1-based 索引，因此需要加 1
            f.write(f"f {' '.join([str(i + 1)+'/'+str(i+1) for i in face])}\n")

def save_mtl(filepath, file_name):
    mtl_content = """
    newmtl red_material
    Ka 1.000 1.000 1.000
    Kd 1.000 1.000 1.000
    Ks 0.000 0.000 0.000
    map_Kd door.jpg
    """
    # 保存 mtl 文件
    with open(filepath, "w") as mtl_file:
        mtl_file.write("newmtl " + file_name + '\n')
        mtl_file.write("Ka 1.000 1.000 1.000\n")
        mtl_file.write("Kd 1.000 1.000 1.000\n")
        mtl_file.write("Ks 0.000 0.000 0.000\n")
        mtl_file.write("map_Kd door.jpg\n")

path_roomBound = 'area4_project_door.txt'
path_save_mesh = 'E:\\FloorSG\\code\\S3DIS_total\\data\\area4\\door\\'
door_mesh = []
with open(path_roomBound, 'r') as f:
    data = f.readlines()
    data = [(x.strip('\n').split(' ')) for x in data]
    new_data = []
    for i in data:
        ps = []
        for j in i:
            if j != '':
                ps.append(float(j))
        new_data.append(np.array(ps))

for i, room in enumerate(new_data):
    print(room.shape)
    A = np.array([room[0], room[1], 0])  # 线段A的坐标 (x1, y1, z1)
    B = np.array([room[3], room[4], 0])  # 线段B的坐标 (x2, y2, z2)
    height = 2  # 给定的高度
    top_A = A + np.array([0, 0, height])  # 顶部A
    top_B = B + np.array([0, 0, height])  # 顶部B
    vertices = np.array([A, B, top_A, top_B])
    faces = np.array([
        [3, 0, 1, 2],  # 第一个三角形 ABC
        [3, 1, 2, 3]   # 第二个三角形 ABD
    ])
    mesh = pv.PolyData(vertices, faces)
    # door_mesh.append(mesh)
    # mesh.save(path_save_mesh +str(i)+ ".obj")
    save_obj(path_save_mesh +str(i)+ ".obj", str(i), mesh)
    save_mtl(path_save_mesh +str(i)+ ".mtl", str(i))
# combined_mesh = door_mesh[0]
# for mesh in door_mesh[1:]:
#     combined_mesh = combined_mesh.merge(mesh)

# 保存合并后的 mesh 为 obj 文件
# combined_mesh.save(path_save_mesh + '.obj')