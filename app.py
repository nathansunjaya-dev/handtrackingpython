import os
import urllib.request
import math
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Auto-download model file jika belum ada
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# Inisialisasi MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (0,17), (17,18), (18,19), (19,20)
]

cap = cv2.VideoCapture(0)

effects = ["edges", "thermal", "neon", "glitch", "invert", "night_vision"]
effect_index = 0

is_pinching_right = False 
is_pinching_left = False

# Toggle State & Debouncing Control for Clenched Fist
filter_toggled_on = False
is_fist_held = False

is_recording = False
out_writer = None
rec_start_time = 0

def apply_effect(image, effect_name):
    """Menerapkan filter visual pada seluruh frame tanpa distorsi"""
    if effect_name == "edges":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif effect_name == "thermal":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    elif effect_name == "neon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.applyColorMap(edges, cv2.COLORMAP_MAGMA)

    elif effect_name == "glitch":
        glitch = image.copy()
        glitch[:, :, 2] = np.roll(glitch[:, :, 2], 10, axis=1)
        glitch[:, :, 0] = np.roll(glitch[:, :, 0], -10, axis=1)
        return glitch

    elif effect_name == "invert":
        return cv2.bitwise_not(image)

    elif effect_name == "night_vision":
        green = image.copy()
        green[:, :, 0] = 0
        green[:, :, 2] = 0
        green[:, :, 1] = cv2.add(green[:, :, 1], 50) 
        return green

    return image

def is_fist_clenched(hand_landmarks, width, height):
    """Mengecek apakah tangan terkelepal (fist) berdasarkan jarak fingertip ke wrist (landmark 0)"""
    wrist = np.array([hand_landmarks[0].x * width, hand_landmarks[0].y * height])
    fingertip_ids = [8, 12, 16, 20]
    knuckle_ids = [5, 9, 13, 17]

    avg_tip_dist = sum(math.hypot(hand_landmarks[i].x * width - wrist[0], hand_landmarks[i].y * height - wrist[1]) for i in fingertip_ids) / 4.0
    avg_knuckle_dist = sum(math.hypot(hand_landmarks[i].x * width - wrist[0], hand_landmarks[i].y * height - wrist[1]) for i in knuckle_ids) / 4.0

    return avg_tip_dist < (avg_knuckle_dist * 1.25)

def draw_tactical_reticle(img, center, active=False, label=""):
    color = (0, 255, 0) if active else (255, 255, 0)
    x, y = center
    radius = 18 if active else 12
    
    cv2.circle(img, (x, y), radius, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
    
    l = 8
    cv2.line(img, (x - radius - l, y), (x - radius, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x + radius, y), (x + radius + l, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - radius - l), (x, y - radius), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y + radius), (x, y + radius + l), color, 1, cv2.LINE_AA)
    
    if label:
        cv2.putText(img, label, (x + radius + 10, y + 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def draw_corner_brackets(img, w, h):
    color = (255, 255, 255)
    length = 25
    thickness = 2
    
    cv2.line(img, (15, 15), (15 + length, 15), color, thickness)
    cv2.line(img, (15, 15), (15, 15 + length), color, thickness)
    cv2.line(img, (w - 15, 15), (w - 15 - length, 15), color, thickness)
    cv2.line(img, (w - 15, 15), (w - 15, 15 + length), color, thickness)
    cv2.line(img, (15, h - 15), (15 + length, h - 15), color, thickness)
    cv2.line(img, (15, h - 15), (15, h - 15 - length), color, thickness)
    cv2.line(img, (w - 15, h - 15), (w - 15 - length, h - 15), color, thickness)
    cv2.line(img, (w - 15, h - 15), (w - 15, h - 15 - length), color, thickness)

print("=== CYBERPUNK HUD INTERFACE ===")
print("1. Toggle Filter: Clench Right Hand (TOGGLE ON/OFF)")
print("2. Pinch Kanan (Jempol + Kelingking): Ganti Efek")
print("3. Pinch Kiri (Jempol + Kelingking) : Rekam / Stop")
print("4. Press [Q] : Keluar")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    hands_data = []
    currently_rendering = False

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = hand_landmarks[start_idx]
                p2 = hand_landmarks[end_idx]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)), 
                         (int(p2.x * w), int(p2.y * h)), (255, 100, 0), 1, cv2.LINE_AA)
            
            thumb = (int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h))
            index = (int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h))
            pinky = (int(hand_landmarks[20].x * w), int(hand_landmarks[20].y * h))
            
            hands_data.append({
                'wrist_x': hand_landmarks[0].x, 
                'thumb': thumb, 
                'index': index,
                'pinky': pinky,
                'raw_landmarks': hand_landmarks
            })

    # FITUR KEAMANAN: Cek ketersediaan DUA TANGAN
    if len(hands_data) == 2:
        hands_data.sort(key=lambda x: x['wrist_x'])
        left_hand = hands_data[0]
        right_hand = hands_data[1]

        # --- TOGGLE MECHANISM: Cek kepalan tangan kanan ---
        right_is_fist = is_fist_clenched(right_hand['raw_landmarks'], w, h)

        if right_is_fist:
            if not is_fist_held:
                filter_toggled_on = not filter_toggled_on  # Toggle status ON/OFF
                is_fist_held = True
        else:
            is_fist_held = False

        # --- Pinch Kanan (Filter Switcher) ---
        r_thumb, r_pinky = right_hand['thumb'], right_hand['pinky']
        dist_right = math.hypot(r_thumb[0] - r_pinky[0], r_thumb[1] - r_pinky[1])

        active_right = dist_right < 30
        draw_tactical_reticle(frame, r_pinky, active_right, "FX SWAP")

        if active_right:
            if not is_pinching_right:
                effect_index = (effect_index + 1) % len(effects)
                is_pinching_right = True
        else:
            is_pinching_right = False

        # --- Pinch Kiri (Record Switcher) ---
        l_thumb, l_pinky = left_hand['thumb'], left_hand['pinky']
        dist_left = math.hypot(l_thumb[0] - l_pinky[0], l_thumb[1] - l_pinky[1])

        active_left = dist_left < 30
        draw_tactical_reticle(frame, l_pinky, active_left, "REC TRIGGER")

        if active_left:
            if not is_pinching_left:
                is_recording = not is_recording
                is_pinching_left = True
                
                if is_recording:
                    rec_start_time = time.time()
                    filename = f"rec_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                else:
                    if out_writer is not None:
                        out_writer.release()
                        out_writer = None
        else:
            is_pinching_left = False

        # RENDER FILTER: Jika Toggle ON dan Ada 2 Tangan
        if filter_toggled_on:
            currently_rendering = True
            poly_pts = np.array([
                left_hand['index'],   
                right_hand['index'],  
                right_hand['thumb'],  
                left_hand['thumb']    
            ], dtype=np.int32)

            full_effect_frame = apply_effect(frame, effects[effect_index])

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_pts], 255)

            frame[mask == 255] = full_effect_frame[mask == 255]
            cv2.polylines(frame, [poly_pts], True, (255, 0, 255), 2, cv2.LINE_AA)

    # ==================== HUD OVERLAY DESIGN ====================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.line(frame, (0, 60), (w, 60), (255, 0, 255), 1)

    # Status Aktivasi HUD
    if filter_toggled_on and currently_rendering:
        status_str = "FILTER: ON"
        status_color = (0, 255, 0)
    elif filter_toggled_on and not currently_rendering:
        status_str = "PAUSED (NEED 2 HANDS)"
        status_color = (0, 165, 255)
    else:
        status_str = "FILTER: OFF (CLENCH RIGHT HAND TO TOGGLE)"
        status_color = (150, 150, 150)

    current_effect = effects[effect_index].upper()
    cv2.putText(frame, "MODE //", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, current_effect, (95, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, status_str, (w // 2 - 140, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    draw_corner_brackets(frame, w, h)

    # Recording Status
    if is_recording:
        if out_writer is not None:
            out_writer.write(frame)
        
        elapsed = int(time.time() - rec_start_time)
        mins, secs = divmod(elapsed, 60)
        timer_str = f"REC  {mins:02d}:{secs:02d}"

        rec_overlay = frame.copy()
        cv2.rectangle(rec_overlay, (w - 180, 15), (w - 25, 48), (0, 0, 150), -1)
        cv2.addWeighted(rec_overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (w - 180, 15), (w - 25, 48), (0, 0, 255), 1)

        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w - 160, 31), 6, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(frame, timer_str, (w - 142, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 3)

    cv2.imshow("Hand Tracking Interactive Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 9:  # TAB
        effect_index = (effect_index + 1) % len(effects)

# Cleanup
if out_writer is not None:
    out_writer.release()

cap.release()
cv2.destroyAllWindows()