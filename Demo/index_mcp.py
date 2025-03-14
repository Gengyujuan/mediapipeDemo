import cv2
from utilities.DrawingTools import DrawingTools
from utilities.DrawingUtils import draw_angle_lines
#===========================================================导入mediapipe模型 初始化模型===============================================================
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils  # 初始化 Mediapipe 的绘图工具
mpHands = mp.solutions.hands  # 初始化 Mediapipe 的手部检测解决方案
hands = mpHands.Hands(
    static_image_mode=False,#静态图像模式 True-适合处理单张图片，不进行跟踪 False-适合处理视频流，跟踪手部姿态变化
    max_num_hands=1, #图像中最多检测到的手部数量
    model_complexity=1,  #设置模型的复杂度（0：最小复杂度模型，计算量最小，但准确度也相对较低；1: 默认复杂度模型，是一个平衡的选择，准确性较高，同时计算量也适中。2: 最大复杂度模型，提供最高的准确度，但需要更多的计算资源。）
)
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
                 draw_angle_lines(img, points[a], points[b], points[c], drawingTools)


#打开电脑的摄像头
cap = cv2.VideoCapture(0)

count = 0
# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count += 1

    # 处理之后的图片
    drawHandsLines(frame, INDEX_FINGER_MCP)

    # 显示处理后的图像
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()