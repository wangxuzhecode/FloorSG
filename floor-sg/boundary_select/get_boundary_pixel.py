import math
import os
import numpy as np
import sys

def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def convert_format(points):
    converted_points = []
    for point in points:
        x, y = point[0]  # point 是形如 [[x y]] 的列表
        converted_points.append((x, y))
    return converted_points

def find_nearest_point(x, y, points):
    min_distance = float('inf')  # 初始设为无穷大
    nearest_point = None
    for point in points:
        px, py = point
        distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
    return nearest_point

def change_points(st, ed):
    if st[0]==ed[0]:
        if st[1]<ed[1]:
            res_st = st
            res_ed = ed
        else:
            res_st = ed
            res_ed = st
    else:
        if st[0]<ed[0]:
            res_st = st
            res_ed = ed
        else:
            res_st = ed
            res_ed = st
    return res_st, res_ed

def get_boundary_map(resized_image2):
    boundary_map = np.zeros_like(resized_image2)
    for i in range(resized_image2.shape[0]):
        for j in range(resized_image2.shape[1]):
            if resized_image2[i][j] != 0:
                for a, b in zip([1, 0, -1, 0, 1, -1, 1, -1], [0, 1, 0, -1, 1, -1, -1, 1]):
                    if (i + a < resized_image2.shape[0] and i + a >= 0) and (
                            j + b < resized_image2.shape[1] and j + b >= 0):
                        if resized_image2[i][j] != resized_image2[i + a][j + b]:
                            boundary_map[i][j] = 1
                            break
                    else:
                        boundary_map[i][j] = 1
    return boundary_map

def are_collinear(p1, p2, p3):
    """检查点 p1, p2, p3 是否共线"""
    return np.isclose(np.cross(p2 - p1, p3 - p1), 0)


def merge_lines(segments):
    """合并共线且重叠的线段"""
    merged_segments = []

    def is_overlap_or_connected(seg1, seg2):
        """检查两个线段是否重叠或连接"""

        def project_to_1d(p, seg):
            return min(seg[0], seg[1]) <= p <= max(seg[0], seg[1])

        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        if are_collinear(np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])) and \
                are_collinear(np.array([x1, y1]), np.array([x2, y2]), np.array([x4, y4])):
            return (project_to_1d(x3, (x1, x2)) and project_to_1d(y3, (y1, y2))) or \
                   (project_to_1d(x4, (x1, x2)) and project_to_1d(y4, (y1, y2)))
        return False

    def merge_two_segments(seg1, seg2):
        """合并两个重叠或连接的线段"""
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        x_min = min(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        x_max = max(x1, x2, x3, x4)
        y_max = max(y1, y2, y3, y4)
        return (x_min, y_min, x_max, y_max)

    while segments:
        seg = segments.pop(0)
        merged = False
        for i in range(len(merged_segments)):
            if is_overlap_or_connected(seg, merged_segments[i]):
                merged_segments[i] = merge_two_segments(seg, merged_segments[i])
                merged = True
                break
        if not merged:
            merged_segments.append(seg)

    return merged_segments

def get_selected_points_by_range(point_cloud, x_min, x_max, y_min, y_max):
    filtered_points = point_cloud[
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max)
        ]
    return filtered_points

def get_selected_points_by_enlarge_range(points, x_range_min, x_range_max, y_range_min, y_range_max, gridsize, theta, mode):
    candidate_points = None
    if mode == 1: ## x equals
        for i in range(1, 4):
            selected_points1 = get_selected_points_by_range(points, x_range_min - gridsize * i, x_range_min - (i - 1) * gridsize, y_range_min, y_range_max)
            selected_points2 = get_selected_points_by_range(points, x_range_max + (i - 1) * gridsize, x_range_max + i * gridsize, y_range_min, y_range_max)
            if candidate_points is None:
                candidate_points = selected_points1 if selected_points1.shape[0] > selected_points2.shape[0] else selected_points2
            else:
                if candidate_points.shape[0]<selected_points1.shape[0]: candidate_points = selected_points1
                if candidate_points.shape[0]<selected_points2.shape[0]: candidate_points = selected_points2
            if selected_points1.shape[0]>theta or selected_points2.shape[0]>theta:
                return selected_points1 if selected_points1.shape[0]>selected_points2.shape[0] else selected_points2
    else:
        for i in range(1, 4):
            selected_points1 = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_min - gridsize * i, y_range_min - (i - 1) * gridsize)
            selected_points2 = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_max + (i - 1) * gridsize, y_range_max + gridsize * i)
            if candidate_points is None:
                candidate_points = selected_points1 if selected_points1.shape[0] > selected_points2.shape[0] else selected_points2
            else:
                if candidate_points.shape[0]<selected_points1.shape[0]: candidate_points = selected_points1
                if candidate_points.shape[0]<selected_points2.shape[0]: candidate_points = selected_points2
            if selected_points1.shape[0]>theta or selected_points2.shape[0]>theta:
                return selected_points1 if selected_points1.shape[0]>selected_points2.shape[0] else selected_points2
    return candidate_points

def get_points_by_roomline(roomline, points, theta = 50):
    gridsize = 0.12
    min_data = [-9.393884660000000, -24.515441890000000]
    height = 126
    width = 356
    st_x, st_y, ed_x, ed_y = roomline

    ## calculate the region


    if st_x == ed_x:
        x_range_min = min_data[0] + st_x * gridsize
        x_range_max = min_data[0] + (st_x + 1) * gridsize

        y_range_min = min_data[1] + st_y * gridsize
        y_range_max = min_data[1] + (ed_y + 1) * gridsize

        selected_points = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_min, y_range_max)
        if selected_points.shape[0] < theta:
            selected_points1 = get_selected_points_by_enlarge_range(points, x_range_min, x_range_max, y_range_min, y_range_max, gridsize, theta, 1)
            print('larger region',selected_points1.shape)
            if selected_points1.shape[0]>selected_points.shape[0]: selected_points = selected_points1
    else:
        x_range_min = min_data[0] + st_x * gridsize
        x_range_max = min_data[0] + (ed_x + 1) * gridsize

        y_range_min = min_data[1] + st_y * gridsize
        y_range_max = min_data[1] + (st_y + 1) * gridsize
        selected_points = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_min, y_range_max)
        if selected_points.shape[0] < theta:
            selected_points1 = get_selected_points_by_enlarge_range(points, x_range_min, x_range_max, y_range_min, y_range_max, gridsize, theta, 2)
            print('larger region',selected_points1.shape)
            if selected_points1.shape[0]>selected_points.shape[0]: selected_points = selected_points1

    print(selected_points.shape)
    return selected_points

if __name__ == '__main__':

    area_name = r'C:\2024\FloorSG\ISPRS_result\MRF_filter_refine_filter_res_pad.txt'
    img_orig = read_txt_to_matrix(area_name)
    image = np.array(img_orig,dtype=np.uint32)
    print(image.shape)
    dir_path = r'C:\2024\FloorSG\ISPRS_result\room_boundary'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    point_path = 'C:\\2024\\data\\ISPRS\\TUB1\\vertical.txt'
    matrix = []
    with open(point_path, 'r') as file:
        for line in file:
            # 将每行的数字按空格或制表符分割，并转换为float类型，存储为一行矩阵
            row = list(map(float, line.split()))
            matrix.append(row)
    pointcloud = np.array(matrix)[:, 0:3]

    # np.savetxt('I:\\FloorSG\\Expriment_output\\S3DIS\\Boundary_optim\\area2_boundary.txt', boundary_map, fmt='%d', delimiter=' ')
    print(np.max(image))
    for i in range(1,np.max(image)+1):
        print('第',i,'个房间')
        # if i!=17:
        #     continue
        # if i!=1 and i!=17 and i!=18 and i!=19 and i!=28 and i!=31 and i!=34 and i!=37 and i!=41 and i!=42 and i!=43 and i!=44:
        #     continue
        new_image = np.zeros(shape=image.shape, dtype=np.uint32)
        new_image[image == i] = 255
        cur_boundary_map = get_boundary_map(new_image)
        boundary_lst = []
        for x in range(cur_boundary_map.shape[0]):
            for y in range(cur_boundary_map.shape[1]):
                if cur_boundary_map[x][y]==1:
                    boundary_lst.append((x,y))
        roomline_path = os.path.join(dir_path, str(i) + '.txt')
        num_edge = 0
        cur_pos = boundary_lst[0]
        hash_map = np.zeros_like(cur_boundary_map)
        segments = []
        while 1:
            if num_edge == len(boundary_lst):
                break
            # print(cur_pos)
            for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
                if (cur_pos[0] + a < image.shape[0] and cur_pos[0] + a >= 0) and (cur_pos[1] + b < image.shape[1] and cur_pos[1] + b >= 0):
                    if cur_boundary_map[cur_pos[0] + a][cur_pos[1] + b]==1 and hash_map[cur_pos[0] + a][cur_pos[1] + b]==0:
                        res_st, res_ed = change_points(cur_pos, (cur_pos[0] + a, cur_pos[1] + b))
                        # with open(roomline_path, "a") as f:
                        #     f.write(f"{res_st[0]}\t{res_st[1]}\t{res_ed[0]}\t{res_ed[1]}\t{11}\t{11}\n")
                        segments.append((res_st[0],res_st[1],res_ed[0], res_ed[1]))
                        hash_map[cur_pos[0] + a][cur_pos[1] + b]=1
                        cur_pos=(cur_pos[0] + a,cur_pos[1] + b)
                        num_edge+=1
                        break

        merged_segments = merge_lines(segments)
        # print("合并后的线段:")

        for num, seg in enumerate(merged_segments):
            # print(seg)
            selected_points = get_points_by_roomline(seg, pointcloud)
            cur_path = os.path.join(dir_path, str(i) + '_' + str(num) + '.txt')
            np.savetxt(cur_path, selected_points)
            # with open(roomline_path, "a") as f:
            #     res_st, res_ed = change_points((seg[0], seg[1]), (seg[2],seg[3]))
            #     f.write(f"{res_st[1]}\t{res_st[0]}\t{res_ed[1]}\t{res_ed[0]}\t{11}\t{11}\n")