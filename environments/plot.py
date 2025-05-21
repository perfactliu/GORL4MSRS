import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator


def plot_and_save_3d_cubes(points, save_path, cube_size=1):
    """
    在三维直角坐标系中用红色半透明正方体表示点，并将结果保存为图像。

    参数:
    - points: 三维坐标点的集合，格式为 numpy 数组或列表，例如 [[x1, y1, z1], [x2, y2, z2], ...]
    - save_path: 图像保存路径，例如 "output/image.png"
    - cube_size: 正方体的边长，默认为 1
    """
    # 如果路径不存在，创建路径
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 遍历每个点并绘制正方体
    for point in points:
        x, y, z = point

        # 定义正方体的顶点
        vertices = np.array([
            [x - cube_size/2, y - cube_size/2, z - cube_size/2],
            [x + cube_size/2, y - cube_size/2, z - cube_size/2],
            [x + cube_size/2, y + cube_size/2, z - cube_size/2],
            [x - cube_size/2, y + cube_size/2, z - cube_size/2],
            [x - cube_size/2, y - cube_size/2, z + cube_size/2],
            [x + cube_size/2, y - cube_size/2, z + cube_size/2],
            [x + cube_size/2, y + cube_size/2, z + cube_size/2],
            [x - cube_size/2, y + cube_size/2, z + cube_size/2]
        ])

        # 定义正方体的面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]]
        ]

        # 绘制正方体
        ax.add_collection3d(Poly3DCollection(faces, facecolors='r', linewidths=1, edgecolors='k', alpha=0.5))

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 自动调整坐标轴范围
    # 计算坐标轴范围
    points_array = np.array(points)
    min_val = np.min(points_array) - 1  # 所有坐标的最小值
    max_val = np.max(points_array) + 1  # 所有坐标的最大值

    # 设置坐标轴范围
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])
    ax.set_box_aspect([1, 1, 1])  # 三个轴的比例为 1:1:1
    # 设置坐标轴刻度为整数
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.zaxis.set_major_locator(MultipleLocator(1))
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()  # 关闭图形，避免内存泄漏

    print(f"图像已保存至: {save_path}")


if __name__ == '__main__':
    # 示例调用
    points = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0]
    ]
    save_path = "output/3d_cubes.png"  # 保存路径
    plot_and_save_3d_cubes(points, save_path)
