from os.path import join
import numpy as np
# from ply import read_ply
import glob
import os
from collections import deque
from collections import Counter
import sys
import tqdm


def most_frequent_matrix_value(positions, matrix):
    if not positions or not isinstance(matrix, np.ndarray):
        return None

    # Initialize a counter to count occurrences of matrix values
    value_counter = Counter()

    # Iterate through positions and count matrix values
    for pos in positions:
        row, col = pos
        if 0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1]:
            value = matrix[row, col]
            value_counter[value] += 1

    # Find the most common value
    most_common = value_counter.most_common()

    # Handle ties by returning 0 if multiple values have the same frequency
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return 0
    else:
        return most_common[0][0]


def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def save_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            row_str = ' '.join(map(str, row))  # 将每行转换为字符串，用空格分隔元素
            f.write(row_str + '\n')  # 写入文件，每行后加上换行符

def nearest_nonzero_point(matrix, start_x, start_y):
    rows = len(matrix)
    cols = len(matrix[0])
    distances = [[float('inf')] * (cols) for _ in range(rows)]
    nearest_point = None
    min_distance = float('inf')
    queue = deque([(start_x, start_y)])
    distances[start_x][start_y] = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if distances[nx][ny] == float('inf'):  # 如果当前位置还未访问过
                    distances[nx][ny] = distances[x][y] + 1
                    if matrix[nx][ny] != 0:
                        if distances[nx][ny] < min_distance:
                            min_distance = distances[nx][ny]
                            nearest_point = (nx, ny)
                    queue.append((nx, ny))

    return nearest_point

def nearest_nonzero(matrix, r, c):
    if matrix.size == 0 or r < 0 or c < 0 or r >= matrix.shape[0] or c >= matrix.shape[1]:
        return []

    rows, cols = matrix.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向

    queue = deque([(r, c)])
    visited = np.zeros_like(matrix, dtype=bool)
    visited[r, c] = True
    min_distance = float('inf')
    result = []

    while queue:
        curr_r, curr_c = queue.popleft()

        for dr, dc in directions:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                if matrix[nr, nc] != 0:
                    distance = abs(nr - r) + abs(nc - c)
                    if distance < min_distance:
                        min_distance = distance
                        result = [(nr, nc)]
                    elif distance == min_distance:
                        result.append((nr, nc))
                queue.append((nr, nc))

    return result

def bfs_nearest_non_zero(matrix, start_r, start_c, max_depth=10):
    rows, cols = matrix.shape
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上、右、下、左
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque([(start_r, start_c, 0)])  # (行, 列, 当前深度)
    visited[start_r, start_c] = True
    distance_map = {}  # 用于记录每个距离的点位置和对应值

    while queue:
        r, c, depth = queue.popleft()

        if depth > max_depth:
            continue

        if matrix[r, c] != 0:
            if depth not in distance_map:
                distance_map[depth] = []
            distance_map[depth].append((r, c, matrix[r, c]))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(matrix, nr, nc) and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc, depth + 1))

    if not distance_map:
        return None

    min_distance = min(distance_map.keys())
    values = [val for r, c, val in distance_map[min_distance]]

    value_counts = Counter(values)
    most_common = value_counts.most_common()

    if not most_common:
        return None

    max_count = most_common[0][1]
    most_common_values = [value for value, count in most_common if count == max_count]

    if len(most_common_values) > 1:
        return None

    # 返回最近点的位置（行，列）
    for r, c, val in distance_map[min_distance]:
        if val == most_common_values[0]:
            return (r, c)

def bfs_nearest_non_zero2(matrix, start_r, start_c, max_depth=10):
    rows, cols = matrix.shape
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上、右、下、左
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque([(start_r, start_c, 0)])  # (行, 列, 当前深度)
    visited[start_r, start_c] = True
    distance_map = {}  # 用于记录每个距离的点位置和对应值

    while queue:
        r, c, depth = queue.popleft()

        if depth > max_depth:
            continue

        if matrix[r, c] != 0:
            if depth not in distance_map:
                distance_map[depth] = []
            distance_map[depth].append((r, c, matrix[r, c]))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(matrix, nr, nc) and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc, depth + 1))

    if not distance_map:
        return None

    min_distance = min(distance_map.keys())
    values = [val for r, c, val in distance_map[min_distance]]

    value_counts = Counter(values)
    most_common = value_counts.most_common()

    if not most_common:
        return None

    max_count = most_common[0][1]
    most_common_values = [value for value, count in most_common if count == max_count]

    if len(most_common_values) > 1:
        return None

    # 返回最近点的位置（行，列）
    for r, c, val in distance_map[min_distance]:
        if val == most_common_values[0]:
            return (r, c)



def find_non_zero_points(matrix):
    # 返回所有非零点的坐标
    return np.array(np.nonzero(matrix))


def is_valid(matrix, r, c):
    # 检查坐标 (r, c) 是否在矩阵范围内
    return 0 <= r < matrix.shape[0] and 0 <= c < matrix.shape[1]
if __name__ == '__main__':

    label_name = sys.argv[1]
    label_orig = read_txt_to_matrix(label_name)
    label_orig = np.array(label_orig,dtype=np.uint32)

    seg_name = sys.argv[2]
    seg_orig = read_txt_to_matrix(seg_name)
    seg_orig = np.array(seg_orig,dtype=np.uint32)

    res = np.zeros_like(seg_orig, seg_orig.dtype)
    for i in tqdm.tqdm(range(seg_orig.shape[0])):
        for j in range(label_orig.shape[1]):
            if label_orig[i][j]==1 and seg_orig[i][j]==0:
                # result = nearest_nonzero(seg_orig, i, j)
                result = bfs_nearest_non_zero2(seg_orig,i,j,3)
                if result!=None:
                    res[i][j] = seg_orig[result[0]][result[1]]

                # res[i][j] = l
                # flag = False
                # pos_0 = result[0]
                # for idx in range(1,len(result)):
                #     cur_pos = result[idx]
                #     if seg_orig[pos_0[0]][pos_0[1]]!=seg_orig[cur_pos[0]][cur_pos[1]]:
                #         flag=True
                # if flag==False:
                #     res[i][j] = seg_orig[pos_0[0]][pos_0[1]]
            else:
                if label_orig[i][j]==1:
                    res[i][j] = seg_orig[i][j]

    for i in range(seg_orig.shape[0]):
        for j in range(label_orig.shape[1]):
            if label_orig[i][j]==0:
                res[i][j]=0

    filename = sys.argv[3]
    save_matrix_to_txt(res, filename)
    # print(f"二维矩阵已保存到文件 '{filename}' 中。")