
import maxflow as mf
import numpy as np
import math
import sys

def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def get_area_by_label(img_work, label):
    area_list = []
    for y in range(len(img_work)):
        for x in range(len(img_work[0])):
            cur_label = img_work[y][x]
            if cur_label == label:
                area_list.append((y, x))
    return area_list


def get_area_range(area_list):
    max_y = -1
    max_x = -1
    min_y = 1e5
    min_x = 1e5
    for pos in area_list:
        cur_y = pos[0]
        cur_x = pos[1]
        max_y = max(max_y, cur_y)
        max_x = max(max_x, cur_x)
        min_y = min(min_y, cur_y)
        min_x = min(min_x, cur_x)
    return max_y, min_y, max_x, min_x

def get_neighboring_point(img_work, y, x):
    neighboring_pos = []
    for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
        if (x + a < len(img_work[0]) and x + a >= 0) and (y + b < len(img_work) and y + b >= 0):
            neighboring_pos.append((y + b, x + a))
    return neighboring_pos

def get_boundary_point_4(area_list, img_work, area_range, label):
    boundary_list = []
    for pos in area_list:
        cur_y = pos[0]
        cur_x = pos[1]
        neighboring_pos = get_neighboring_point(img_work, cur_y, cur_x)
        for (n_y, n_x) in neighboring_pos:
            if img_work[n_y][n_x] != label and pos not in boundary_list:
                boundary_list.append(pos)
                break
        if (cur_y == area_range['max_y'] or cur_y == area_range['min_y'] or cur_x == area_range['max_x'] or cur_x ==
            area_range['min_x']) and pos not in boundary_list:
            boundary_list.append(pos)
    return boundary_list

def process(img_work, img_wall, boundary_list, cur_label):
    # img_name = 'D:\\G2\\floorplan\\room_segmentation_refine\\data\\watershed_result\\mrf_result4_3.txt'
    # img_rt = read_txt_to_matrix(img_name)
    change_lst = []
    for pos in boundary_list:
        y = pos[0]
        x = pos[1]
        # if pos == (1,3):
        #     continue
        for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            n_x = x+a
            n_y = y+b
            if (n_x < len(img_work[0]) and n_x >= 0) and (n_y < len(img_work) and n_y >= 0):
                if img_work[n_y][n_x] == '0':
                    num = 1
                    while 1:
                        n_x += a
                        n_y += b
                        if num>3:
                            break
                        if (n_x < len(img_work[0]) and n_x >= 0) and (n_y < len(img_work) and n_y >= 0):
                            if img_wall[n_y][n_x] == '-100' or img_work[n_y][n_x] == '0':
                                num+=1
                            if img_wall[n_y][n_x] == '-200':
                                print(pos,a,b)
                                while 1:
                                    n_x -= a
                                    n_y -= b
                                    if img_work[n_y][n_x] == cur_label:
                                        break
                                    # img_rt[n_y][n_x] = cur_label
                                    change_lst.append((n_y,n_x))
                                    print(n_y, n_x)
                                    print(1)

                                break
                            if img_work[n_y][n_x] != cur_label and img_work[n_y][n_x]!='0':
                                print(3)
                                if num == 1:
                                    break
                                n_x -= a * math.ceil(num / 2)
                                n_y -= b * math.ceil(num / 2)
                                while 1:
                                    n_x -= a
                                    n_y -= b
                                    if img_work[n_y][n_x] == cur_label:
                                        break
                                    # img_rt[n_y][n_x] = cur_label
                                    change_lst.append((n_y,n_x))

                                break
                        else:
                            print(4)
                            while 1:
                                n_x -= a
                                n_y -= b
                                if img_work[n_y][n_x] == cur_label:
                                    break
                                change_lst.append((n_y, n_x))
                                # img_rt[n_y][n_x] = cur_label
                            break
        # break
    for pos in change_lst:
        y = pos[0]
        x = pos[1]
        img_work[y][x] = cur_label
    return img_work

def save_matrix_to_txt(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            row_str = ' '.join(map(str, row))  # 将每行转换为字符串，并用制表符分隔
            file.write(row_str + '\n')  # 写入每行数据到文件

# def get_neighboring_point(img_work, y,x):
#     neighboring_pos = []
#     for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
#         if (x + a < len(img_work[0]) and x + a >= 0) and (y + b < len(img_work) and y + b >= 0):
#             neighboring_pos.append((y+b, x+a))
#     return neighboring_pos

def padding(img_work):
    for y in range(len(img_work)):
        for x in range(len(img_work[0])):
            if img_work[y][x] == '0':
                neighbor_lst = get_neighboring_point(img_work, y, x)
                num = 0
                label = '0'
                flag = False
                for n_pos in neighbor_lst:
                    if flag == False and img_work[n_pos[0]][n_pos[1]] != img_work[y][x]:
                        flag = True
                        num += 1
                        label = img_work[n_pos[0]][n_pos[1]]
                    elif img_work[n_pos[0]][n_pos[1]] == label and flag == True:
                        num = num + 1
                if num == 4:
                    img_work[y][x] = label
                    print(y+1, x+1)
    return img_work

def filter_abnormal_point(img_work):
    for y in range(len(img_work)):
        for x in range(len(img_work[0])):
            if img_work[y][x] != '0':
                neighbor_lst = get_neighboring_point(img_work, y, x)
                num = 0
                label = '0'
                num2=0
                for n_pos in neighbor_lst:
                    if img_work[n_pos[0]][n_pos[1]] != img_work[y][x]:
                        num = num+1
                    # if img_work[n_pos[0]][n_pos[1]]
                if num >=3 or (num==2 and len(neighbor_lst)==3):
                    img_work[y][x] = 0
    return img_work

def padding2(img_work):
    for y in range(len(img_work)):
        for x in range(len(img_work[0])):
            if img_work[y][x] == '0':
                neighbor_lst = get_neighboring_point(img_work, y, x)
                num = 0
                num_0 = 0
                label = '0'
                flag = False
                for n_pos in neighbor_lst:
                    if flag == False and img_work[n_pos[0]][n_pos[1]] != img_work[y][x]:
                        flag = True
                        num += 1
                        label = img_work[n_pos[0]][n_pos[1]]
                    elif img_work[n_pos[0]][n_pos[1]] == label and flag== True:
                        num = num+1
                    if img_work[n_pos[0]][n_pos[1]] == '0':
                        num_0 = num_0+1
                if (num ==3 and num_0==1) or num ==4:
                    img_work[y][x] = label
                    print(3)
    return img_work


if __name__ == '__main__':

    img_name = sys.argv[1]
    img_work = read_txt_to_matrix(img_name)

    for i in range(20):
        img_work = filter_abnormal_point(img_work)

    out_path = sys.argv[2]

    save_matrix_to_txt(img_work, out_path)