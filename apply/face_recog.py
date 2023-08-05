import csv
import time
import STESDK
import cv2
import numpy as np
from STESDK import FaceDetector, FaceTracker, LiveVerification, FaceAligner, FaceExtractor


# 人脸特征提取函数
def extract_feature(img, face_detector, face_aligner, face_extractor):
    # 步骤一：人脸检测
    rect = face_detector.detect(img)
    if len(rect) > 0:
        # 步骤二：获取人脸关键点
        keypoints = face_aligner.align(img, rect[0])
        # 步骤三：根据人脸关键点裁剪人脸位置图像
        crop_frame = face_aligner.crop_face(img, keypoints)
        # 步骤四：人脸特征提取
        feature = face_extractor.extract(crop_frame)
        return feature
    else:
        print('No face detected!')


# 加载已知人脸特征
def load_known_faces(csv_path):
    known_faces = {}
    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        for row in csv_reader:
            name = row[1]
            feature_vector = [float(x) for x in row[2][1:-1].split(',')]
            # 修改：每个名字都会关联一个特征向量列表，可以对应多种情况
            if name in known_faces:
                known_faces[name].append(feature_vector)
            else:
                known_faces[name] = [feature_vector]
    return known_faces



# 计算余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def main():
    # 加载已知人脸特征
    csv_path = r'/data/csv/dataset.csv'
    known_faces = load_known_faces(csv_path)

    # for name, feature_vectors in known_faces.items():
    #     print(name)
    #     for i, feature_vector in enumerate(feature_vectors):
    #         print(f"Feature vector {i + 1}: {feature_vector}")

    # 使用OpenCV的VideoCapture打开摄像头获取图片，设置视频源为0，设置分辨率为480
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建人脸检测、人脸对齐和人脸特征提取对象
    face_detector = FaceDetector()
    face_aligner = FaceAligner()
    face_extractor = FaceExtractor()
    # 创建人脸检测对象
    face_tracker = FaceTracker()

    # 活体检测
    live_verify = LiveVerification()

    frame_count = 0

    while True:
        # 从摄像头读取一帧图片
        ret, frame = cap.read()
        # 复制当前帧的图像，以便在其上绘制矩形，避免影响原图frame对象
        img = frame.copy()

        # 每隔 10 帧进行一次人脸检测
        if frame_count % 10 == 0:
            start_time = time.time()

            # 使用人脸检测器检测人脸，并返回人脸的矩形坐标，返回的坐标为左上角和右下角的坐标
            rects = face_detector.detect(img)


        # 更新帧计数器
        frame_count += 1

        # 如果检测到正好一个人脸
        if len(rects) >= 1:
            # 在图像上绘制一个矩形框，标识出人脸的位置
            face = cv2.rectangle(frame, (rects[0][0], rects[0][1]), (rects[0][2], rects[0][3]), (0, 0, 255), 1)

            # 进行活体检测
            res = live_verify.verify(frame)

            # res是一个列表，仅有一个数，res[0]是非活体的概率
            if res[0] < 0.4:
                text = "Live Face Detected"
                color = (0, 255, 0)  # 绿色

                # 提取当前帧的人脸特征
                current_feature = extract_feature(frame, face_detector, face_aligner, face_extractor)
                # print(current_feature)
                if current_feature is not None:
                    # 用于保存最佳匹配结果
                    best_match = None
                    best_similarity = -1

                    # 遍历已知人脸特征，与当前人脸特征进行相似度计算
                    for name, feature_vectors in known_faces.items():
                        for known_feature in feature_vectors:
                            similarity = cosine_similarity(current_feature, known_feature)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = name
                                # print(best_match, best_similarity)

                    # print(best_match, best_similarity)

                    # 如果相似度大于阈值，显示匹配到的人名
                    if best_similarity > 0.5:
                        text = f"Matched: {best_match}"
                        color = (0, 255, 0)  # 绿色

                        # 记录人脸检测结束时间
                        end_time = time.time()

                        # 计算并输出人脸检测所需的时间
                        face_detection_time = end_time - start_time
                        print(f"Face detection time: {face_detection_time:.2f} seconds")

                    else:
                        text = "No Match Found"
                        color = (0, 0, 255)  # 红色

            else:
                text = "No Live Face Detected"
                color = (0, 0, 255)  # 红色

            org = (100, 100)  # 文本的左下角坐标
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
            fontScale = 1  # 字体大小
            thickness = 2  # 字体粗细
            image = cv2.putText(face, text, org, font, fontScale, color, thickness)  # 在图像上绘制文本
            STESDK.imshow('result', image)  # 显示图像

        else:
            image = img.copy()  # 复制图像
            text = 'No Face Detected ！'  # 要显示的文本
            org = (100, 100)  # 文本的左下角坐标
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
            fontScale = 1  # 字体大小
            color = (0, 0, 255)  # 字体颜色，红色
            thickness = 2  # 字体粗细
            image = cv2.putText(image, text, org, font, fontScale, color, thickness)  # 在图像上绘制文本
            STESDK.imshow('result', image)  # 显示图像

        key = cv2.waitKey(30)  # 将按键值存储在变量key中
        if key & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
