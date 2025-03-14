import cv2
from utilities.DrawingUtils import drawlimbLines,PoseJoint

# 打开电脑的摄像头
cap = cv2.VideoCapture(0)

count = 0
# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count += 1

    # 处理之后的图片
    drawlimbLines(frame, PoseJoint["LEFT_KNEE_FLEXION"])

    # 显示处理后的图像
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()