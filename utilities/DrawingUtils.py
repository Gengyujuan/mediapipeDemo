import cv2
import numpy as np
from utilities.AngleLinesUtils import angle_ROM,lines_extension

import mediapipe as mp
from utilities.DrawingTools import DrawingTools

def draw_angle_lines(img, a, b, c, DrawingTools):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # 计算夹角的大小
    angle = angle_ROM(a, b, c)
    # 如果无法计算角度，则函数终止执行
    if angle is None:
        return None

    extension_len = DrawingTools.getExtension()

    end_ba,end_cb,angle_start,angle_end = lines_extension(a, b, c, extension_len)

    # 绘制延长线
    cv2.line(img, tuple(b), end_ba, DrawingTools.getTrajectoryColor(), DrawingTools.thickness)
    cv2.line(img, tuple(b), end_cb, DrawingTools.getTrajectoryColor(), DrawingTools.thickness)

    # 在 cb 延长线的末尾段绘制箭头
    arrow_thickness = 2  # 箭头的粗细
    cv2.arrowedLine(img, tuple(b), end_cb, DrawingTools.getTrajectoryColor(), arrow_thickness)
    cv2.arrowedLine(img, tuple(b), end_ba, DrawingTools.getTrajectoryColor(), arrow_thickness)

    # 绘制圆弧（夹角）===satrt===
    radius = DrawingTools.getRadius()

    # 计算夹角
    internal_angle = (angle_end - angle_start) % 360

    # 确保结束角度大于起始角度
    if angle_start > angle_end:
        angle_end += 360  # 调整结束角度以确保绘制正确

    # 根据内部角来设置绘制的弧线
    if internal_angle <= 180:
        # 直接绘制内角
        if angle_start > angle_end:
            # 如果是逆时针绘制，需要调整结束角度
            angle_end += 360
    else:
        # 如果内角大于 180 度，交换角度确保绘制的是内角弧线
        angle_start, angle_end = angle_end, angle_start + 360

    # 确保 b 的类型为整数
    center_b = tuple(b.astype(int))
    cv2.ellipse(img, center_b, (radius, radius), 0,angle_start,angle_end, DrawingTools.getAngleMarkColor(), 2)
    # 绘制圆弧（夹角）===end===

    # 绘制角度标签
    label_x = b[0] - 80
    label_y = b[1] - 80
    cv2.putText(img, f'{angle:.0f}', (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 2,DrawingTools.getAngleMarkColor(), 5)

    return angle


#=========================================初始化肢体模型===============================
mpDraw = mp.solutions.drawing_utils  # 初始化 Mediapipe 的绘图工具
mpPose = mp.solutions.pose  # 初始化 Mediapipe 的人体姿态识别解决方案
pose = mpPose.Pose()
PoseJoint = {
    "LEFT_ELBOW_FLEXION": (mpPose.PoseLandmark.LEFT_WRIST, mpPose.PoseLandmark.LEFT_ELBOW, mpPose.PoseLandmark.LEFT_SHOULDER),
    "LEFT_KNEE_FLEXION": (mpPose.PoseLandmark.LEFT_ANKLE, mpPose.PoseLandmark.LEFT_KNEE, mpPose.PoseLandmark.LEFT_HIP)
}


def drawlimbLines(img, connections):
    drawingTools = DrawingTools(20, 60)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)


    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        h, w, _ = img.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # print(f"connections:{connections}")
        # 确保连接的点是有效的
        # for connection in connections:
        a, b, c = connections  # 使用 PoseJoint 字典来获取连接点
        draw_angle_lines(img, points[a], points[b], points[c], drawingTools)