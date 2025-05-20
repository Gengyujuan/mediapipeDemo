# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  # 存放检测到的手部关键点
  pose_landmarks_list = detection_result.hand_landmarks
  # 创建输入图像的一个副本，以便在副本上进行绘制操作，而不修改原始图像。
  annotated_image = np.copy(rgb_image)

  # 显示检测到的每一帧图像
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    #在图像上标记关键点
    # 建一个NormalizedLandmarkList对象，并将当前人手的关键点数据填充到这个对象中。每个关键点包含x, y, z坐标。
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
    # 在图像上绘制关键点后再返回图像
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style())
  return annotated_image

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# 将 RGB 图像转换为 BGR，因为 OpenCV 默认使用 BGR 格式
annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
# 显示图像
cv2.imshow('Annotated Image', annotated_image_bgr)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭所有窗口