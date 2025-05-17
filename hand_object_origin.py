import cv2
import mediapipe as mp
import torch

# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 加载YOLOv5模型（自动下载预训练权重）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 设置摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为RGB格式
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 手部检测
    hand_results = hands.process(image_rgb)
    hand_regions = []
    
    # 获取手部区域
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            # 计算手部边界框
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in landmarks.landmark]
            y_coords = [lm.y * h for lm in landmarks.landmark]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            
            # 扩展边界框范围（确保包含握持物品）
            expand = 30  # 可根据实际情况调整
            min_x = max(0, min_x - expand)
            max_x = min(w, max_x + expand)
            min_y = max(0, min_y - expand)
            max_y = min(h, max_y + expand)
            
            hand_regions.append((min_x, min_y, max_x, max_y))
            
            # 绘制手部关键点和边界框
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # 如果没有检测到手则跳过
    if not hand_regions:
        cv2.imshow('Handheld Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO物体检测
    results = model(image_rgb)
    detections = results.pandas().xyxy[0]  # 获取检测结果

    # 筛选有效检测
    for _, det in detections.iterrows():
        if det['confidence'] < 0.5:  # 置信度阈值
            continue
        
        # 解析检测结果
        x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
        label = f"{det['name']} {det['confidence']:.2f}"

        # 检查物体是否在手部区域内
        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        for (hx1, hy1, hx2, hy2) in hand_regions:
            # 判断物体中心是否在手部区域或物体与手部区域有显著重叠
            if (hx1 <= obj_center[0] <= hx2 and hy1 <= obj_center[1] <= hy2) or \
               (x1 < hx2 and x2 > hx1 and y1 < hy2 and y2 > hy1):
                # 绘制检测框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                break

    cv2.imshow('Handheld Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()