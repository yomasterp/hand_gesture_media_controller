import cv2
import mediapipe as mp
import time
import pyautogui
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track previous positions
prev_x, prev_time = None, 0
cooldown = 1.5  # seconds between actions

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for natural mirror view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, c = img.shape
    current_time = time.time()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                # Get key points
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]

                # Measure distance between thumb and index finger
                dist = distance(thumb_tip[0], thumb_tip[1], index_tip[0], index_tip[1])

                # If thumb and index finger are close together, trigger Play/Pause
                if dist < 40:
                    if current_time - prev_time > cooldown:
                        pyautogui.press('space')  # Play/Pause
                        print("Play/Pause")
                        prev_time = current_time

                # Detect horizontal hand movement
                hand_center_x = lm_list[0][0]  # Wrist point as center

                if prev_x is not None:
                    move_x = hand_center_x - prev_x

                    if move_x > 80:  # Swiped right
                        if current_time - prev_time > cooldown:
                            pyautogui.press('right')  # Fast-forward
                            print("Fast Forward")
                            prev_time = current_time

                    elif move_x < -80:  # Swiped left
                        if current_time - prev_time > cooldown:
                            pyautogui.press('left')  # Rewind
                            print("Rewind")
                            prev_time = current_time

                prev_x = hand_center_x

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking Media Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
