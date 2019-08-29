from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth=0):
    if not point_list:
        return None

    k = len(point_list[0])  # 假定所有点的尺寸相同
    # 根据深度选择轴
    axis = depth % k

    # 根据轴对点的列表进行排序，并选择中间值作为轴元素
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2

    # 创建结点并构建子树
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )


def main():
    """构建kd树-案例"""
    point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
    tree = kdtree(point_list)
    print(tree)


if __name__ == '__main__':
    main()