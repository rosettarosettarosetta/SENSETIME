import cv2
import numpy as np
import onnxruntime
import time

CLASSES = ["person", "phone"]


class YOLOV5():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath, providers=['TensorrtExecutionProvider',
                                                                              'CUDAExecutionProvider',
                                                                              'CPUExecutionProvider'])
        print(self.onnx_session.get_providers())
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    # -------------------------------------------------------
    #   获取输入输出的名字
    # -------------------------------------------------------
    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    # -------------------------------------------------------
    #   输入图像
    # -------------------------------------------------------
    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    # -------------------------------------------------------
    #   1.cv2读取图像并resize
    #	2.图像转BGR2RGB和HWC2CHW
    #	3.图像归一化
    #	4.图像增加维度
    #	5.onnx_session 推理
    # -------------------------------------------------------

    def inference(self, frame):
        or_img = cv2.resize(frame, (640, 640))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #	置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep

def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def filter_box(org_box, conf_thres, iou_thres):  # 过滤box
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output

def box_in_box(inner_box, outer_box):
    x1, y1, x2, y2 = inner_box
    x3, y3, x4, y4 = outer_box

    # 检查内框是否在外框内
    if x1 > x3 and y1 > y3 and x2 < x4 and y2 < y4:
        return True

    # 检查内外框是否有重合区域
    if x1 < x4 and x3 < x2 and y1 < y4 and y3 < y2:
        return True

    return False

def draw(image, box_data, frame_counter, detection_time, output_file):
    frame = np.array(image)

    # 提取手机框
    phone_boxes = []
    for box in box_data:
        if box[5] == 1:
            phone_boxes.append(box[:4])

    # 检查人框是否与手机框相关联
    found = False
    for box in box_data:
        if box[5] == 0:
            person_box = box[:4]

            for phone_box in phone_boxes:
                px1 = int(phone_box[0])
                py1 = int(phone_box[1])
                px2 = int(phone_box[2])
                py2 = int(phone_box[3])
                print(f"手机框坐标：({px1}, {py1}), ({px2}, {py2})")
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(frame, 'phone', (px1, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if box_in_box(phone_box, person_box):
                    # 画框
                    x1 = int(person_box[0])
                    y1 = int(person_box[1])
                    x2 = int(person_box[2])
                    y2 = int(person_box[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, 'calling', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    found = True
                    with open(output_file, 'a') as f:
                        f.write(f"帧数：{frame_counter}, 推理时延: {detection_time:.2f}ms, 标签: ('calling'), 预测框坐标: ({px1}, {py1}), ({px2}, {py2}), 安全设备：商汤科技\n")
                    break
            if found:
                break

    if not found:
        with open(output_file, 'a') as f:
            f.write(f"帧数：{frame_counter}，推理时延: {detection_time:.2f}ms\n")

    return frame


if __name__ == "__main__":
    onnx_path = 'sim.onnx'
    conf_thres = 0.5  # 检测的置信度阈值
    iou_thres = 0.5  # NMS的IoU阈值
    output_file = "output.txt"  # output file name
    cap = cv2.VideoCapture('test.mp4')
    model = YOLOV5(onnx_path)

    frame_counter = 0
    while 1:
        rval, frame = cap.read()  # 读取视频帧
        if rval == False:
            break
        else:
            start_time = time.time()
            output, or_img = model.inference(frame)
            outbox = filter_box(output, conf_thres, iou_thres)
            frame_show = draw(or_img, outbox, frame_counter, float(time.time() - start_time) * 1000.0, output_file)
            frame_counter += 1

        cv2.imshow("show", frame_show)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

