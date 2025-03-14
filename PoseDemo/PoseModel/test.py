import requests

# 定义下载链接和目标文件名
# url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
# output_file = '../../HandsDemo/hand_landmarker.task'

url = 'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg'
output_file = '../../HandsDemo/image.jpg'

# 发送请求下载文件
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 将内容写入文件
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"文件已成功下载到 {output_file}")
else:
    print(f"下载失败，状态码：{response.status_code}")
