import pymesh

# 加载网格
mesh1 = pymesh.formats.load_mesh(r"E:\FloorSG\code\S3DIS_total\data\area5\floor\5.obj")
mesh1 = pymesh.formats.load_mesh(r"E:\FloorSG\code\S3DIS_total\data\area5\floor\51.obj")

# 执行布尔做差
result = pymesh.boolean(mesh1, mesh2, operation="difference")

# 保存结果
result.save("result.obj")
