import numpy as np

"""
angle_ROM:  关节活动度，以cb为参考线，计算直线ba与cb的夹角的角度值
"""
def angle_ROM(a, b, c):
    # # 向量
    # ba = b - a
    # cb = c - b
    # # 向量的模（长度）
    # ba_magnitude = np.linalg.norm(ba)
    # cb_magnitude = np.linalg.norm(cb)
    # if ba_magnitude == 0 or cb_magnitude == 0:  # 两点重合
    #     return None
    # # 向量的点积
    # dot_product = np.dot(ba, cb)
    # # 夹角的 cos 值
    # angle = np.arccos(dot_product / (ba_magnitude * cb_magnitude))
    # # 返回夹角的角度值
    # # return round(np.degrees(angle))  # 返回四舍五入后的整数
    # return int(np.ceil(np.degrees(angle)))  # 向上取整并转换为整数
    # # return np.degrees(angle)
    # 向量
    ba = b - a
    cb = c - b
    # 向量的模（长度）
    ba_magnitude = np.linalg.norm(ba)
    cb_magnitude = np.linalg.norm(cb)

    if ba_magnitude == 0 or cb_magnitude == 0:  # 两点重合
        return None

    # 向量的点积
    dot_product = np.dot(ba, cb)

    # 计算夹角的 cos 值，并确保值在合法范围 [-1, 1]
    cosine_angle = dot_product / (ba_magnitude * cb_magnitude)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # 限制在 -1 到 1 之间

    # 计算夹角
    angle = np.arccos(cosine_angle)

    # 返回夹角的角度值
    return int(np.ceil(np.degrees(angle)))


# lines_extension:  肢体de延长线
def lines_extension(a, b, c, len):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 计算向量 ba 和向量 cb
    ba = a - b
    cb = b - c

    # 计算 ba 和 cb 向量的模（长度）
    norm_ba = np.linalg.norm(ba)
    norm_cb = np.linalg.norm(cb)

    # 计算 ba 和 cb 的单位向量
    unit_vector_ba = ba / norm_ba
    unit_vector_cb = cb / norm_cb

    # 计算延长线的终点
    end_ba = b + len * unit_vector_ba
    end_cb = b + len * unit_vector_cb

    end_ba = tuple(np.round(end_ba).astype(int))
    end_cb = tuple(np.round(end_cb).astype(int))

    # angle_start = int(np.degrees(np.arctan2(unit_vector_cb[1], unit_vector_cb[0])))
    # angle_end = int(np.degrees(np.arctan2(unit_vector_ba[1], unit_vector_ba[0])))
    # 计算角度并确保在 [0, 360) 范围内
    angle_start = int(np.degrees(np.arctan2(unit_vector_cb[1], unit_vector_cb[0]))) % 360
    angle_end = int(np.degrees(np.arctan2(unit_vector_ba[1], unit_vector_ba[0]))) % 360

    # 如果 angle_start > angle_end，考虑顺时针绘制
    if angle_start > angle_end:
        angle_end += 360  # 增加360度以确保逆时针绘制

    return end_ba,end_cb,angle_start,angle_end
