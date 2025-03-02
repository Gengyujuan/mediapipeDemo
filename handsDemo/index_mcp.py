import cv2
from utilities.DrawingTools import DrawingTools
from utilities.AngleLinesUtils import angle_ROM,lines_extension
import numpy as np
#===========================================================导入mediapipe模型 初始化模型===============================================================
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils  # 初始化 Mediapipe 的绘图工具
mpHands = mp.solutions.hands  # 初始化 Mediapipe 的手部检测解决方案
hands = mpHands.Hands(
    # static_image_mode=True
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
)  # static_image_mode=True 以便于处理单张图片
#将需要用到的手部关节点（参考目录名为appendix下的handsPoint图片）赋值给INDEX_FINGER_MCP变量
INDEX_FINGER_MCP = [
    (mpHands.HandLandmark.INDEX_FINGER_PIP, mpHands.HandLandmark.INDEX_FINGER_MCP, mpHands.HandLandmark.WRIST)
]
#==================================================================================================================================================

#===========================================================计算示指掌指关节活动角度===============================================================
def drawHandsLines(img,connections):
    drawingTools = DrawingTools(30,80)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将 BGR 图像转换为 RGB 格式，以适应 Mediapipe
    results = hands.process(imgRGB)  # 运用手部识别模型处理图像
    angle = 0
    # 如果检测到手部标志点
    if results.multi_hand_landmarks:
        # 遍历所有检测到的手部
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=handLms,
                connections=mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=mpDraw.DrawingSpec(color=drawingTools.getCoilaColor(), thickness=2, circle_radius=2),
                connection_drawing_spec=mpDraw.DrawingSpec(color=drawingTools.getConLineColor(), thickness=2)
            )
            # 获取关键点坐标
            landmarks = handLms.landmark
            # 将关键点坐标转换为像素值
            h, w, _ = img.shape
            points = []
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
            # 绘制每个关节点的夹角
            for (a, b, c) in connections:
                angle = draw_angle_lines(img, points[a], points[b], points[c], drawingTools)
    return  angle

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

#打开电脑的摄像头
cap = cv2.VideoCapture(0)

# connection = [HandJoint["THUMB_MCP"]]


# def drawHandsLines(img,connections):
#     drawingTools = DrawingTools(40,80)
#
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将 BGR 图像转换为 RGB 格式，以适应 Mediapipe
#     results = hands.process(imgRGB)  # 运用手部识别模型处理图像
#     angle = 0
#     # 如果检测到手部标志点
#     if results.multi_hand_landmarks:
#         # 遍历所有检测到的手部
#         for handLms in results.multi_hand_landmarks:
#             # filtered_landmarks = landmark_pb2.NormalizedLandmarkList()
#             # # 创建一个新的关键点列表，排除 INDEX_FINGER_DIP
#             # for idx, landmark in enumerate(handLms.landmark):
#             #     if idx != mpHands.HandLandmark.INDEX_FINGER_DIP:
#             #         filtered_landmarks.landmark.append(landmark)
#             #         # 使用过滤后的关键点列表绘制
#             # mpDraw.draw_landmarks(
#             #     image=img,
#             #     landmark_list=filtered_landmarks,
#             #     connections=mpHands.HAND_CONNECTIONS,
#             #     landmark_drawing_spec=mpDraw.DrawingSpec(color=drawingTools.getCoilaColor(), thickness=2,circle_radius=2),
#             #     connection_drawing_spec=mpDraw.DrawingSpec(color=drawingTools.getConLineColor(), thickness=2)
#             # )
#             mpDraw.draw_landmarks(
#                 image=img,
#                 landmark_list=handLms,
#                 connections=mpHands.HAND_CONNECTIONS,
#                 landmark_drawing_spec=mpDraw.DrawingSpec(color=drawingTools.getCoilaColor(), thickness=2, circle_radius=2),
#                 connection_drawing_spec=mpDraw.DrawingSpec(color=drawingTools.getConLineColor(), thickness=2)
#             )
#             # 获取关键点坐标
#             # landmarks = handLms.landmark
#
#             index_finger_dip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP]
#             index_finger_tip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
#             index_finger_pip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP]
#             index_finger_mcp = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
#
#
#             # 将关键点坐标转换为像素值
#             h, w, _ = img.shape
#             points = []
#             a = 0.1
#             b = 0.5
#
#
#             x1, y1 = int(index_finger_tip.x * w),int(index_finger_tip.y * h)
#             x2, y2 = int(index_finger_dip.x * w),int(index_finger_dip.y * h)
#             x0, y0 = int(index_finger_pip.x * w),int(index_finger_pip.y * h)
#
#
#             x3, y3 = int(index_finger_mcp.x * w), int(index_finger_mcp.y * h)
#
#             # new_x0 = (x0 - x3) * a + x0
#             # new_y0 = (y0 - y3) * a + y0
#             new_x0 = (x0 - x3) * a + x0
#             new_y0 = (y0 - y3) * a + y0
#
#             new_x = (x2 - (new_x0 + x1) / 2) * b + x2
#             new_y = (y2 - (new_y0 + y1) / 2) * b + y2
#
#
#             point1 = (x1, y1)
#             point2 = (new_x, new_y)
#             point3 = (new_x0, new_y0)
#
#             points.append(point1)
#             points.append(point2)
#             points.append(point3)
#
#             radius = 5  # 设置圆的半径大小
#             color = (0, 255, 0)
#             thickness = 9
#
#             cv2.circle(img, (int(x1), int(y1)), radius, color, -1)  # 绘制point1
#             cv2.circle(img, (int(new_x), int(new_y)), radius, color, -1)
#             cv2.circle(img, (int(new_x0), int(new_y0)), radius, color, -1)  # 绘制point3
#
#             # 将三个点用蓝色线条连接起来
#             cv2.line(img, (int(x1), int(y1)), (int(new_x), int(new_y)), color, thickness)
#             cv2.line(img, (int(new_x), int(new_y)), (int(new_x0), int(new_y0)), color, thickness)
#             cv2.line(img, (int(x3), int(y3)), (int(new_x0), int(new_y0)), color, thickness)
#
#             for a, b, c in zip(range(0, len(points), 3), range(1, len(points), 3), range(2, len(points), 3)):
#                 point_a = points[a]
#                 point_b = points[b]
#                 point_c = points[c]
#                 angle = draw_angle_lines(img,tuple(map(int, point_a)), tuple(map(int, point_b)), tuple(map(int, point_c)), drawingTools)
#     return  angle



count = 0
# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count += 1

    # 处理之后的图片
    angle = drawHandsLines(frame, INDEX_FINGER_MCP)

    # 显示处理后的图像
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()