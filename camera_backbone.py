from ultralytics import YOLO
import cv2
import supervision as sv  # for tracking
import time
import numpy as np


model = YOLO(r"runs\detect\train5\weights\best.pt")

# Initialize tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.5,
    lost_track_buffer=50,
    minimum_matching_threshold=0.7,
    frame_rate=30
)

# Hi-Lo count mapping
HI_LO = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'j': -1, 'q': -1, 'k': -1, 'a': -1
}

running_count = 0
seen_cards = {}  # track_id -> card label
card_positions = {}
COOLDOWN_TIME = 5.0
MIN_CONFIDENCE = 0.5


def get_rank(label):
    label = label.lower()
    if label.startswith("10"):
        return "10"
    return label[0] 

def is_duplicate_card(xyxy, label, threshold=80):
    """Check if this detection is too close to an existing counted card within cooldown period"""
    center_x = (xyxy[0] + xyxy[2]) / 2
    center_y = (xyxy[1] + xyxy[3]) / 2
    current_time = time.time()
    
    # Clean up old positions that are outside cooldown period
    expired_tracks = []
    for track_id, pos_info in card_positions.items():
        stored_x, stored_y, stored_label, timestamp = pos_info
        if current_time - timestamp > COOLDOWN_TIME:
            expired_tracks.append(track_id)
    
    for track_id in expired_tracks:
        del card_positions[track_id]
    
    # Check remaining positions for duplicates
    for pos_info in card_positions.values():
        stored_x, stored_y, stored_label, timestamp = pos_info
        distance = np.sqrt((center_x - stored_x)**2 + (center_y - stored_y)**2)
        
        # If same card type is very close and within cooldown period, likely duplicate
        if distance < threshold and stored_label == label:
            time_since_count = current_time - timestamp
            if time_since_count < COOLDOWN_TIME:
                return True
    return False

cap = cv2.VideoCapture(1)
fps_start = time.time()

frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = sv.Detections.from_ultralytics(results[0])

    high_conf_mask = detections.confidence >= MIN_CONFIDENCE
    detections = detections[high_conf_mask]

    tracked = tracker.update_with_detections(detections)

    for xyxy, conf, cls_id, track_id in zip(tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        label = model.names[int(cls_id)]
        rank = get_rank(label)

        # Only count once per track ID
        if (track_id not in seen_cards and not is_duplicate_card(xyxy, label) and conf >= MIN_CONFIDENCE):
            seen_cards[track_id] = label

            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            current_time = time.time()
            card_positions[track_id] = (center_x, center_y, label, current_time)

            running_count += HI_LO.get(rank, 0)

        # Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calculate FPS
    frames += 1
    if frames % 10 == 0:
        fps = frames / (time.time() - fps_start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display running count
    cv2.putText(frame, f"Count: {running_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Card Counter (YOLOv8)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        seen_cards.clear()
        running_count = 0
        print("Count reset!")

cap.release()
cv2.destroyAllWindows()