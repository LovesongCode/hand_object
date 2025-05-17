import cv2
import mediapipe as mp
import torch
import time
import datetime
import os

# 创建日志目录
log_dir = "hand_object_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 日志文件名
log_file = os.path.join(log_dir, f"hand_object_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 加载YOLOv5模型（自动下载预训练权重）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 设置摄像头
cap = cv2.VideoCapture(0)

# 记录物体状态的字典 {object_id: {'name': 名称, 'last_contact_time': 上次接触时间, 'in_hand': 是否在手中}}
tracked_objects = {}

# 用于生成唯一物体ID的计数器
next_object_id = 0

# 记录日志的函数
def log_event(event_type, object_name, duration=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if duration is not None:
        log_entry = f"{timestamp} - {event_type}: {object_name} (持续时间: {duration:.2f}秒)\n"
    else:
        log_entry = f"{timestamp} - {event_type}: {object_name}\n"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    print(log_entry.strip())

# 写入日志文件头
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"手部物体接触日志 - 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("-" * 60 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
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

    # 检测到的当前帧中的物体ID
    current_frame_objects = set()

    # YOLO物体检测
    results = model(image_rgb)
    detections = results.pandas().xyxy[0]  # 获取检测结果

    # 筛选有效检测
    for _, det in detections.iterrows():
        if det['confidence'] < 0.5:  # 置信度阈值
            continue
        
        # 解析检测结果
        x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
        object_name = det['name']
        label = f"{object_name} {det['confidence']:.2f}"
        
        # 检查物体是否在手部区域内
        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        object_in_hand = False
        
        for (hx1, hy1, hx2, hy2) in hand_regions:
            # 判断物体中心是否在手部区域或物体与手部区域有显著重叠
            if (hx1 <= obj_center[0] <= hx2 and hy1 <= obj_center[1] <= hy2) or \
               (x1 < hx2 and x2 > hx1 and y1 < hy2 and y2 > hy1):
                object_in_hand = True
                # 绘制检测框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                break
        
        # 生成与检测结果最匹配的对象ID（简单实现，实际应用可能需要更复杂的匹配算法）
        matched_object_id = None
        for obj_id, obj_info in tracked_objects.items():
            if obj_info['name'] == object_name:
                matched_object_id = obj_id
                break
        
        # 如果没有匹配到已有对象，创建新对象ID
        if matched_object_id is None:
            matched_object_id = next_object_id
            next_object_id += 1
            tracked_objects[matched_object_id] = {
                'name': object_name,
                'last_contact_time': current_time if object_in_hand else None,
                'in_hand': object_in_hand
            }
            
            if object_in_hand:
                log_event("开始接触", object_name)
        else:
            # 更新物体状态
            if object_in_hand and not tracked_objects[matched_object_id]['in_hand']:
                # 物体从不在手中变为在手中
                tracked_objects[matched_object_id]['in_hand'] = True
                tracked_objects[matched_object_id]['last_contact_time'] = current_time
                log_event("开始接触", object_name)
            elif not object_in_hand and tracked_objects[matched_object_id]['in_hand']:
                # 物体从在手中变为不在手中
                tracked_objects[matched_object_id]['in_hand'] = False
                last_time = tracked_objects[matched_object_id]['last_contact_time']
                duration = current_time - last_time if last_time is not None else 0
                log_event("结束接触", object_name, duration)
            
            # 如果物体仍在手中，更新最后接触时间
            if tracked_objects[matched_object_id]['in_hand']:
                tracked_objects[matched_object_id]['last_contact_time'] = current_time
        
        current_frame_objects.add(matched_object_id)
    
    # 检查是否有物体在当前帧中消失
    for obj_id in list(tracked_objects.keys()):
        if obj_id not in current_frame_objects and tracked_objects[obj_id]['in_hand']:
            # 物体在手中但在当前帧中消失了
            tracked_objects[obj_id]['in_hand'] = False
            last_time = tracked_objects[obj_id]['last_contact_time']
            duration = current_time - last_time if last_time is not None else 0
            log_event("离开视野", tracked_objects[obj_id]['name'], duration)
    
    # 显示已跟踪物体的信息
    y_offset = 30
    for obj_id, obj_info in tracked_objects.items():
        if obj_info['in_hand']:
            status_text = f"ID: {obj_id}, {obj_info['name']}: 手中"
            cv2.putText(frame, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30

    cv2.imshow('Handheld Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 程序结束前，检查是否有物体仍在手中
for obj_id, obj_info in tracked_objects.items():
    if obj_info['in_hand']:
        last_time = obj_info['last_contact_time']
        duration = time.time() - last_time if last_time is not None else 0
        log_event("程序结束时仍在接触", obj_info['name'], duration)

# 写入日志文件尾
with open(log_file, 'a', encoding='utf-8') as f:
    f.write("-" * 60 + "\n")
    f.write(f"记录结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

cap.release()
cv2.destroyAllWindows()