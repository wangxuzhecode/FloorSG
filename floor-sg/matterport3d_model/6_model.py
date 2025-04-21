import glob

import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt


def get_data(in_path):
    with open(in_path, 'r') as f:
        data = f.readlines()
        data = [(x.strip('\n').split(' ')) for x in data]
        new_data = []
        for i in data:
            ps = []
            for j in i:
                if j != '':
                    ps.append(float(j))
            new_data.append(np.array(ps))
    return new_data

def is_point_on_line_segment(P1, P2, P):
    """
    判断点 P 是否在由 P1 和 P2 定义的线段上
    P1, P2, P 是包含坐标元组的点 (x, y)
    """
    x1, y1 = P1
    x2, y2 = P2
    x, y = P

    # 计算叉积，判断共线
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # 如果叉积为 0，表示共线
    if cross_product != 0:
        return False

    # 检查点是否在范围内
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        return True
    return False



def split_long_line(long_line, short_line):

    (x1, y1), (x2, y2) = long_line
    (a1, b1), (a2, b2) = short_line

    # 创建一个分割点列表，初始化为长线段的端点
    points = [(x1, y1), (x2, y2)]

    # 判断短线段的端点是否在长线段上

    points.append((a1, b1))

    points.append((a2, b2))

    # 去重并排序分割点
    points = sorted(set(points))

    # 将分割点组成新的线段
    segments = []
    for i in range(len(points) - 1):
        segments.append([points[i], points[i + 1]])

    return segments


def is_point_on_line(long_line, point, epsilon=1e-5):
    """
    判断一个点是否在线段上。
    long_line: 长线段的端点 [(x1, y1), (x2, y2)]
    point: 需要判断的点 (px, py)
    """
    (x1, y1), (x2, y2) = long_line
    (px, py) = point
    if (px==x1 and py==y1) or (px==x2 and py==y2):
        return True

    # 计算点到线段的距离，判断点是否在线段上
    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])
    line_len = np.linalg.norm(line_vec)

    if line_len < epsilon:
        return False  # 长线段太短，无法判断

    # 投影计算
    proj_len = np.dot(point_vec, line_vec) / line_len
    if proj_len < 0 or proj_len > line_len:
        return False  # 点不在该线段的延长线上

    # 计算点到线段的距离
    dist = np.abs(np.cross(line_vec, point_vec)) / line_len
    return dist < epsilon  # 如果距离小于容差，说明点在直线上


def split_wall(rt_walls, element, ele_type):
    p1 = (element[0], element[1])
    p2 = (element[3], element[4])

    for i, walls in enumerate(rt_walls):
        for j, wall in enumerate(walls):
            st_p = wall[0]
            ed_p = wall[1]
            long_line = [st_p, ed_p]
            if is_point_on_line(long_line, p1) == True and is_point_on_line(long_line, p2) == True:
                short_line = [p1, p2]
                split_segments = split_long_line(long_line, short_line)
                rt_walls[i].pop(j)
                for seg in split_segments:
                    if (seg[0] == p1 and seg[1] == p2) or (seg[0] == p2 and seg[1] == p1):
                        cur_wall = (seg[0], seg[1], ele_type)
                        rt_walls[i].append(cur_wall)
                    else:
                        cur_wall = (seg[0], seg[1], 0)
                        rt_walls[i].append(cur_wall)
    return rt_walls

roomBound = 'scene3_wall.txt'
door_path = 'scene3_project_door.txt'
window_path = 'scene3_project_window.txt'
walls = get_data(roomBound)
doors = get_data(door_path)
windows = get_data(window_path)
res_sp_walls = [[] for _ in walls]

# for i, room in enumerate(walls):
#     # if i!=9:
#     #     continue
#     x = room[::3].reshape(-1, 1)  # 提取 x 坐标
#     y = room[1::3].reshape(-1, 1)  # 提取 y 坐标
#     z = room[2::3].reshape(-1, 1)
#     poly = np.hstack((x, y, z))  # 连接成 (x, y, z) 坐标
#     hash_wall = [0 for _ in range(poly.shape[0])]
#     # print(poly.shape[0])
#     tmp = -1
#     for j in range(poly.shape[0]):
#         st_p = (poly[tmp, 0], poly[tmp, 1])
#         ed_p = (poly[j, 0], poly[j, 1])
#         tmp+=1
#         res_sp_walls[i].append((st_p, ed_p, 0))

path_dir = r"E:\FloorSG\Expriment_output\S3DIS\wall_scene3\*.txt"
txt_files = glob.glob(path_dir)
wall_line = []
for i, txt_file in enumerate(txt_files):
    # if i!=1:continue
    cur_array = []
    with open(txt_file, 'r') as file:
        for line in file:
            # 去掉行尾的换行符，并将每行的数字按空格分割成列表
            row = list(map(float, line.strip().split()))
            cur_array.append(row)
    for data in cur_array:
        st_p = (data[0], data[1])
        ed_p = (data[2], data[3])
        res_sp_walls[i].append((st_p, ed_p, 0))
print(len(res_sp_walls[0]), len(res_sp_walls[1]))


for door in doors:
    res_sp_walls = split_wall(res_sp_walls, door, 1)
for door in windows:
    res_sp_walls = split_wall(res_sp_walls, door, 2)


print(len(res_sp_walls[0]), len(res_sp_walls[1]))
fig, ax = plt.subplots()
p = pv.Plotter()
res_mesh = []
for room_walls in res_sp_walls:
    for wall in room_walls:
        if wall[2]==0:
            A = np.array([wall[0][0], wall[0][1], 0])  # 线段A的坐标 (x1, y1, z1)
            B = np.array([wall[1][0], wall[1][1], 0])  # 线段B的坐标 (x2, y2, z2)
            height = 2.4  # 给定的高度
            top_A = A + np.array([0, 0, height])  # 顶部A
            top_B = B + np.array([0, 0, height])  # 顶部B
            vertices = np.array([A, B, top_A, top_B])
            faces = np.array([
                [3, 0, 1, 2],  # 第一个三角形 ABC
                [3, 1, 2, 3]  # 第二个三角形 ABD
            ])
            mesh = pv.PolyData(vertices, faces)
            p.add_mesh(mesh)
            res_mesh.append(mesh)
        elif wall[2]==1:
            height = 2  # 给定的高度
            A = np.array([wall[0][0], wall[0][1], height])  # 线段A的坐标 (x1, y1, z1)
            B = np.array([wall[1][0], wall[1][1], height])  # 线段B的坐标 (x2, y2, z2)

            top_A = A + np.array([0, 0, 0.4])  # 顶部A
            top_B = B + np.array([0, 0, 0.4])  # 顶部B
            vertices = np.array([A, B, top_A, top_B])
            faces = np.array([
                [3, 0, 1, 2],  # 第一个三角形 ABC
                [3, 1, 2, 3]  # 第二个三角形 ABD
            ])
            mesh = pv.PolyData(vertices, faces)
            p.add_mesh(mesh)
            res_mesh.append(mesh)
        else:
            window_top = 2
            window_bottom = 0.7
            A = np.array([wall[0][0], wall[0][1], 0])  # 线段A的坐标 (x1, y1, z1)
            B = np.array([wall[1][0], wall[1][1], 0])  # 线段B的坐标 (x2, y2, z2)
            top_A = A + np.array([0, 0, window_bottom])  # 顶部A
            top_B = B + np.array([0, 0, window_bottom])  # 顶部B
            vertices = np.array([A, B, top_A, top_B])
            faces = np.array([
                [3, 0, 1, 2],  # 第一个三角形 ABC
                [3, 1, 2, 3]  # 第二个三角形 ABD
            ])
            mesh1 = pv.PolyData(vertices, faces)
            p.add_mesh(mesh1)
            A = np.array([wall[0][0], wall[0][1], window_top])  # 线段A的坐标 (x1, y1, z1)
            B = np.array([wall[1][0], wall[1][1], window_top])  # 线段B的坐标 (x2, y2, z2)
            top_A = A + np.array([0, 0, 0.4])  # 顶部A
            top_B = B + np.array([0, 0, 0.4])  # 顶部B
            vertices = np.array([A, B, top_A, top_B])
            faces = np.array([
                [3, 0, 1, 2],  # 第一个三角形 ABC
                [3, 1, 2, 3]  # 第二个三角形 ABD
            ])
            mesh2 = pv.PolyData(vertices, faces)
            p.add_mesh(mesh2)
            res_mesh.append(mesh1)
            res_mesh.append(mesh2)

combined_mesh = res_mesh[0]
for mesh in res_mesh[1:]:
    combined_mesh = combined_mesh.merge(mesh)

# 保存合并后的 mesh 为 obj 文件
combined_mesh.save('scene3_wall.obj')

p.show()
#         if wall[2]==0:
#             plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'b-', label='Extended Line')
#         else:
#             plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'r-', label='Extended Line')
#
# ax.set_aspect('equal')  # 设置坐标轴比例相等
# # ax.set_axis_off()  # 关闭坐标轴显示
# plt.show()



# for door in windows:
#     p1 = (door[0], door[1])
#     p2 = (door[3], door[4])
#     np_p1 = np.array(p1)
#     np_p2 = np.array(p2)
#     for
#     long_line = [st_p, ed_p]



# for i, room in enumerate(walls):
#
#     x = room[::3].reshape(-1, 1)  # 提取 x 坐标
#     y = room[1::3].reshape(-1, 1)  # 提取 y 坐标
#     z = room[2::3].reshape(-1, 1)
#     poly = np.hstack((x, y, z))  # 连接成 (x, y, z) 坐标
#     hash_wall = [0 for _ in range(poly.shape[0])]
#     res_wall = []
#     print(poly.shape[0])
#     for j in range(poly.shape[0]-1):
#         st_p = (poly[j, 0], poly[j, 1])
#         ed_p = (poly[j+1, 0], poly[j+1, 1])
#         # res_wall.append(st_p)
#         flag = False
        # for door in doors:
        #     p1 = (door[0], door[1])
        #     p2 = (door[3], door[4])
        #     np_p1 = np.array(p1)
        #     np_p2 = np.array(p2)
        #     long_line = [st_p, ed_p]
        #     if is_point_on_line(long_line,  p1)==True and is_point_on_line(long_line, p2)==True:
        #         # print(p1,p2)
        #         # print(st_p, ed_p)
        #         flag = True
        #         short_line = [p1, p2]
        #         split_segments = split_long_line(long_line, short_line)
        #         print(len(split_segments))
        #         for seg in split_segments:
        #             if (seg[0]== p1 and seg[1]==p2) or (seg[0]==p2 and seg[1]== p1):
        #                 cur_wall = (seg[0], seg[1], 1)
        #                 res_wall.append(cur_wall)
        #             else:
        #                 cur_wall = (seg[0], seg[1], 0)
        #                 res_wall.append(cur_wall)
    #     for door in windows:
    #         p1 = (door[0], door[1])
    #         p2 = (door[3], door[4])
    #         np_p1 = np.array(p1)
    #         np_p2 = np.array(p2)
    #         long_line = [st_p, ed_p]
    #         if is_point_on_line(long_line,  p1)==True and is_point_on_line(long_line, p2)==True:
    #             # print(p1,p2)
    #             # print(st_p, ed_p)
    #             flag = True
    #             short_line = [p1, p2]
    #             split_segments = split_long_line(long_line, short_line)
    #             print(len(split_segments))
    #             for seg in split_segments:
    #                 if (seg[0]== p1 and seg[1]==p2) or (seg[0]==p2 and seg[1]== p1):
    #                     cur_wall = (seg[0], seg[1], 2)
    #                     res_wall.append(cur_wall)
    #                 else:
    #                     cur_wall = (seg[0], seg[1], 0)
    #                     res_wall.append(cur_wall)
    #     if flag == False:
    #         cur_wall = (st_p, ed_p, 0)
    #         res_wall.append(cur_wall)
    # print(len(res_wall))
    #             print(door)
    #             flag = True
    #             long_line = LineSegment(st_p[0], st_p[1], ed_p[0], ed_p[1])  # 长线段
    #             short_line = LineSegment(p1[0], p1[1], p2[0], p2[1])  # 短线段
    #             split_segments = split_line_by_short_segment(long_line, short_line)
    #             print(len(split_segments))
    #
    #                 # print(seg.p1, seg.p2)
    #                 if (np.array_equal(seg.p1, np_p1) and np.array_equal(seg.p2, np_p2)) or (np.array_equal(seg.p1, np_p2) and np.array_equal(seg.p2, np_p1)):
    #                     print('door')
    #                     cur_wall = (seg.p1, seg.p2, 1)
    #                     res_wall.append(cur_wall)
    #                 else:
    #                     cur_wall = (seg.p1, seg.p2, 0)
    #                     res_wall.append(cur_wall)
    #     if flag == False:
    #         cur_wall = (st_p, ed_p, 0)
    #         res_wall.append(cur_wall)
    # print(len(res_wall))