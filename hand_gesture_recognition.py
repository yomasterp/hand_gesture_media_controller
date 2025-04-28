"""
Hand-Gesture Media Controller (v2)
──────────────────────────────────
Gestures
  • Thumb–index pinch      →  Play / Pause
  • Right-swipe (open hand)→  Fast-forward 10 s
  • Left-swipe  (open hand)→  Rewind       10 s
  • Up-swipe    (open hand)→  Volume Up
  • Down-swipe  (open hand)→  Volume Down

Press **q** in the OpenCV window to quit.
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# ─── Tunables ────────────────────────────────────────────────────────────────
PINCH_DIST      = 35      # px  – thumb/index distance for Play / Pause
SWIPE_THRESH_X  = 120     # px  – horizontal swipe sensitivity (∆x)
SWIPE_THRESH_Y  = 120     # px  – vertical   swipe sensitivity (∆y)
HISTORY_FRAMES  = 5       #      – frames kept to evaluate a swipe
ACTION_DELAY    = 0.60    # s   – debounce per action family

# ─── MediaPipe setup ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ─── Webcam ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# ─── Trackers ────────────────────────────────────────────────────────────────
pos_hist            = deque(maxlen=HISTORY_FRAMES)
last_playpause_time = 0
last_seek_time      = 0
last_volume_time    = 0


def dist(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    h, w, _ = frame.shape
    now = time.time()

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            # pixel-coordinates of landmarks
            lms = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
            thumb_tip, index_tip = lms[4], lms[8]
            wrist_x,   wrist_y   = lms[0]

            # ── Play / Pause (thumb–index pinch) ────────────────────────────
            if dist(thumb_tip, index_tip) < PINCH_DIST and (now - last_playpause_time) > ACTION_DELAY:
                pyautogui.press('space')
                print("Play / Pause")
                last_playpause_time = now
                pos_hist.clear()                 # reset swipe history
                continue                          # ignore swipe for this frame

            # ── Build swipe history ────────────────────────────────────────
            pos_hist.append((wrist_x, wrist_y))

            if len(pos_hist) == HISTORY_FRAMES:
                dx = pos_hist[-1][0] - pos_hist[0][0]
                dy = pos_hist[-1][1] - pos_hist[0][1]

                # ── Horizontal swipe → seek ──────────────────────────────
                if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESH_X and (now - last_seek_time) > ACTION_DELAY:
                    if dx > 0:
                        pyautogui.press('right')     # fast-forward
                        print("Fast-forward")
                    else:
                        pyautogui.press('left')      # rewind
                        print("Rewind")
                    last_seek_time = now
                    pos_hist.clear()
                    continue

                # ── Vertical swipe → volume ──────────────────────────────
                if abs(dy) > abs(dx) and abs(dy) > SWIPE_THRESH_Y and (now - last_volume_time) > ACTION_DELAY:
                    if dy < 0:
                        pyautogui.press('up')        # volume up
                        print("Volume Up")
                    else:
                        pyautogui.press('down')      # volume down
                        print("Volume Down")
                    last_volume_time = now
                    pos_hist.clear()
                    continue

            # Draw landmarks for visual feedback
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        pos_hist.clear()

    cv2.imshow("Hand-Gesture Media Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
