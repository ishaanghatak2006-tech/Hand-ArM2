from ursina import *
import cv2
import mediapipe as mp
import math
import numpy as np
import datetime
import os


app = Ursina()
window.color = color.black

os.makedirs("screenshots", exist_ok=True)

car = Entity(
    model='vintage_racing_car.glb',
    scale=1
)

try:
    min_b, max_b = car.model.get_tight_bounds()
except:
    min_b, max_b = car.get_tight_bounds()

center = (min_b + max_b) / 2
car.origin = center
car.position = Vec3(0, 0, 0)

size = max(
    max_b.x - min_b.x,
    max_b.y - min_b.y,
    max_b.z - min_b.z
)
if size > 0:
    car.scale = 3 / size

camera.position = Vec3(0, 0, -12)
camera.look_at(car.position)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
cap = cv2.VideoCapture(0)


def is_open_palm_relaxed(hand):
    lm = hand.landmark
    tips = [8, 12, 16]
    bases = [5, 9, 13]
    fingers_up = sum(lm[t].y < lm[b].y - 0.01 for t, b in zip(tips, bases))
    return fingers_up >= 2


def is_peace(hand):
    lm = hand.landmark
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y
            and lm[16].y > lm[14].y and lm[20].y > lm[18].y)


def is_thumbs_up(hand):
    lm = hand.landmark

    # Thumb must be HIGH above other thumb joints
    thumb_up = lm[4].y < lm[3].y < lm[2].y - 0.015
    folded = 0
    fingers = [(8, 5), (12, 9), (16, 13), (20, 17)]
    for tip, base in fingers:
        if lm[tip].y > lm[base].y + 0.01:  
            folded += 1

    return thumb_up and folded == 4


alpha = 0.25  # smoothing

smooth_rx = smooth_ry = 0
smooth_tx = smooth_ty = 0

# zoom
last_zoom_strength = None
smooth_zoom = 0
ZOOM_ALPHA = 0.35
ZOOM_STRENGTH = 50  # stronger zoom

# pause
PAUSE_FRAMES = 0
PAUSE_TARGET = 2  # faster pause

paused = False
screenshot_done = False

# memories
last_right_x = last_right_y = None
last_left_wrist_x = last_left_wrist_y = None


def take_screenshot():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"screenshots/snap_{now}"
    base.screenshot(prefix)
    print("Screenshot saved:", prefix)


def update():
    global paused, screenshot_done
    global last_right_x, last_right_y
    global last_left_wrist_x, last_left_wrist_y
    global smooth_rx, smooth_ry, smooth_tx, smooth_ty
    global last_zoom_strength, smooth_zoom
    global PAUSE_FRAMES

    ok, frame = cap.read()
    if not ok:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    left = right = None

    if res.multi_hand_landmarks and res.multi_handedness:
        for i, h in enumerate(res.multi_handedness):
            label = h.classification[0].label
            hand_lm = res.multi_hand_landmarks[i]
            if label == "Left":
                left = hand_lm
            else:
                right = hand_lm

    if left and is_open_palm_relaxed(left):
        PAUSE_FRAMES += 1
    else:
        PAUSE_FRAMES = 0#max(0, PAUSE_FRAMES - 1)

    paused = PAUSE_FRAMES >= PAUSE_TARGET


    if paused and right and is_thumbs_up(right):
        car.position = Vec3(0, 0, 0)
        car.rotation = Vec3(0, 0, 0)
        camera.position = Vec3(0, 0, -12)
        camera.look_at(car.position)
        print(" Reset!")

    if paused and right and is_peace(right) and not screenshot_done:
        take_screenshot()
        screenshot_done = True

    if not paused:
        screenshot_done = False


    if not paused:
        h, w, _ = frame.shape

        #ROTATION
        
        if right:
            ix = int(right.landmark[8].x * w)
            iy = int(right.landmark[8].y * h)

            if last_right_x is not None:
                dx, dy = ix - last_right_x, iy - last_right_y
                raw_rx = -dy * 2.0
                raw_ry = dx * 2.0
                smooth_rx = smooth_rx * (1-alpha) + raw_rx * alpha
                smooth_ry = smooth_ry * (1-alpha) + raw_ry * alpha
                car.rotation_x += smooth_rx
                car.rotation_y += smooth_ry

            last_right_x, last_right_y = ix, iy
        else:
            last_right_x = last_right_y = None


        #ZOOM

        if right:
            lm = right.landmark
            # pinch distance
            pd = math.dist(
                (lm[4].x, lm[4].y),
                (lm[8].x, lm[8].y)
            )
            # convert to strength
            pinch_strength = max(0.0, min(1.0, (pd - 0.02) / 0.15))
            pinch_strength = 1 - pinch_strength

            if last_zoom_strength is not None:
                delta = pinch_strength - last_zoom_strength
                zoom_amount = delta * ZOOM_STRENGTH
                zoom_amount *= abs(delta) * 2
                smooth_zoom = smooth_zoom * (1 - ZOOM_ALPHA) + zoom_amount * ZOOM_ALPHA
                new_z = camera.position.z - smooth_zoom
                new_z = max(-35, min(-3, new_z))
                camera.position = Vec3(camera.position.x, camera.position.y, new_z)

            last_zoom_strength = pinch_strength
        else:
            last_zoom_strength = None

        #TRANSLATION

        if left:
            wx = int(left.landmark[0].x * w)
            wy = int(left.landmark[0].y * h)

            if last_left_wrist_x is not None:
                dx = (wx - last_left_wrist_x) / w
                dy = (wy - last_left_wrist_y) / h

                raw_tx = dx * 15
                raw_ty = -dy * 15

                smooth_tx = smooth_tx * (1 - alpha) + raw_tx * alpha
                smooth_ty = smooth_ty * (1 - alpha) + raw_ty * alpha

                car.position += Vec3(smooth_tx, smooth_ty, 0)

            last_left_wrist_x, last_left_wrist_y = wx, wy
        else:
            last_left_wrist_x = last_left_wrist_y = None

    #cv2.imshow("Webcam", frame)
    #cv2.waitKey(1)



app.run()
cap.release()
cv2.destroyAllWindows()