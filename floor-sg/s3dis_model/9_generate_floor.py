"""
读取txt文件并将其拉伸为三维ply文件
"""
import pyvista as pv
import numpy as np
import scipy

# 设置绘图模式为文档模式，即白色底
pv.set_plot_theme("document")


# 将房间边界转换为mesh文件
def polygon23D(points, isExtrude, show_cap):
    remove_last_points = points
    x = remove_last_points[::3].reshape(-1, 1)
    y = remove_last_points[1::3].reshape(-1, 1)
    z = np.zeros((len(x), 1))
    print(x.shape)
    height = remove_last_points[2]
    height1 = 2
    poly = np.hstack((x, y, z))
    poly_connectivity = [x.shape[0]] + [i for i in range(x.shape[0])]
    print(poly_connectivity)
    # Create a PolyData object from the points array and triangle
    polygon = pv.PolyData(poly, poly_connectivity).triangulate()

    if isExtrude is True:
        # Extrude the polygon along the z-axis to create a 3D model
        mesh = polygon.extrude([0, 0, height], capping=show_cap)
    else:
        mesh = polygon

    return mesh

def save_mtl(filepath, file_name, rgb):
    mtl_content = """
    newmtl red_material
    Ka 1.000 0.000 0.000
    Kd 1.000 0.000 0.000
    Ks 0.000 0.000 0.000
    d 1.0
    illum 2
    """
    # 保存 mtl 文件
    with open(filepath, "w") as mtl_file:
        mtl_file.write("newmtl " + file_name + '\n')
        mtl_file.write("Ka " + str(rgb[0]) + " " + str(rgb[1]) + " " + str(rgb[2]) + "\n")
        mtl_file.write("Kd " + str(rgb[0]) + " " + str(rgb[1]) + " " + str(rgb[2])+ "\n")
        mtl_file.write("Ks 0.000 0.000 0.000\n")
        mtl_file.write("d 1.0\n")
        mtl_file.write("illum 2\n")

def save_obj(filepath, filename, mesh):
    vertices = mesh.points  # 获取顶点坐标 (N x 3 numpy array)
    faces = mesh.faces  # 获取面片数据 (包含面片顶点索引的 numpy array)
    with open(filepath, 'w') as f:
        f.write("mtllib " + filename +".mtl\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        idx = 0
        f.write("usemtl " + filename + "\n")
        while idx < len(faces):
            num_vertices = faces[idx]  # 面片顶点数量
            idx += 1
            face = faces[idx:idx + num_vertices]
            idx += num_vertices
            # `.obj` 格式的面片是 1-based 索引，因此需要加 1
            f.write(f"f {' '.join([str(i + 1) for i in face])}\n")


def genModel(path_roomBound, color_lst = None, path_ori_point_cloud=None, path_save=None, isExtrude=True, show_cap=True, show_edge=False):
    """
    :param path_roomBound: 房间边界文件txt
    :param path_ori_point_cloud: 房间原型的点云文件ply，如果有则生成两个画面，否则仅有mesh画面
    :param path_save: mesh文件存储路径
    :param show_cap: 是否为mesh生成上下底面
    :param show_edge: 是否显示线框模型
    :return:
    """
    # 读取文件
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

    # mesh文件列表
    meshes = []

    if path_ori_point_cloud is None:
        # 创建画布
        p = pv.Plotter()
        # p.renderer.set_color_cycler(list(color_map.values()))
        for i, room in enumerate(new_data):
            print(i)
            mesh = polygon23D(room, isExtrude, show_cap)
            if path_save is not None:

                cur_color = color_lst[i]
                save_mtl(path_save + str(i) + '.mtl', str(i), cur_color/255)
                save_obj(path_save + str(i) + '_.obj', str(i), mesh)
            #     mesh.save(path_save + str(i) + '_.obj', binary=False)
            # meshes.append(mesh)

def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def get_color_lst(mat_path, floor_path):
    color_lst = []
    mat_data = scipy.io.loadmat(mat_path)
    color = mat_data['Lrgb']
    img_work = read_txt_to_matrix(floor_path)
    floorplan = np.array(img_work, np.uint32)
    for i in range(1, np.max(floorplan)+1):
        idx = np.argwhere(floorplan==i)
        if idx.shape[0]==0:
            continue
        cur_color = color[idx[0,0], idx[0,1]]
        color_lst.append(cur_color)
    return color_lst

if __name__ == '__main__':
    # roomBound = './data/exp1/area5full_with_height.txt'
    # path_point_cloud = './data/exp1/area5_full.ply'
    # path_save_mesh = './data/exp1/res/'

    # roomBound = './data/area4/area4full_with_height.txt'
    # path_point_cloud = './data/area4/Area_4_sub.ply'
    # path_save_mesh = './data/area4/res/'

    # roomBound = './data/area6/Area_6_with_height.txt'

    roomBound = 'area4_wall.txt'
    path_save_mesh = 'E:\\FloorSG\\code\\S3DIS_total\\data\\area4\\floor\\'
    color_lst = get_color_lst(r'C:\2024\Code\FloorSG\Seg\area4_color.mat', r'E:\FloorSG\Expriment_output\S3DIS\Boundary_optim\area4.txt')
    genModel(roomBound, isExtrude=False, color_lst = color_lst, show_cap=False, path_save=path_save_mesh)