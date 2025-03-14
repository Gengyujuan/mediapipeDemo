import cv2
from utilities.DrawingUtils import drawlimbLines,PoseJoint
import time

#打开电脑的摄像头
cap = cv2.VideoCapture(0)
pTime = 0  # 循环之前初始化为0

count = 0
# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count += 1

    # 处理之后的图片
    drawlimbLines(frame, PoseJoint["LEFT_ELBOW_FLEXION"])

# ==============在图片上显示FPS=================
    # 循环之前pTime初始化为0
    # 每一次循环：
    cTime = time.time()  # 处理完一帧图像的时间
    fps = 1 / (cTime - pTime)  # 即为FPS
    pTime = cTime  # 重置起始时间
    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
# ===============================================
    # 显示处理后的图像
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()