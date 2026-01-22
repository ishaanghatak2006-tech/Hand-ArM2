from ursina import *
import cv2
import mediapipe as mp
import math
import numpy as np
import datetime
import os
import sys

selected_model = None
if len(sys.argv) > 1:
    selected_model = sys.argv[1]

app = Ursina()
window.color = color.black
os.makedirs("screenshots", exist_ok=True)

car = Entity(
    model=f"models/{selected_model}" if selected_model else "models/vintage_racing_car.glb",
    scale=1
)

try:
    min_b, max_b = car.model.get_tight_bounds()
except:
    min_b, max_b = car.get_tight_bounds()

center = (min_b + max_b) / 2
car.origin = center
car.position = Vec3(0, 0, 0)

size = max(max_b.x-min_b.x, max_b.y-min_b.y, max_b.z-min_b.z)
if size > 0:
    car.scale = 3 / size

camera.position = Vec3(0, 0, -12)
camera.look_at(car.position)


model_locked = False
camera_locked = False
view_mode = "free"


def toggle_model_lock():
    global model_locked
    model_locked = not model_locked
    model_btn.text = f"MODEL LOCK: {'ON' if model_locked else 'OFF'}"

def toggle_camera_lock():
    global camera_locked
    camera_locked = not camera_locked
    cam_btn.text = f"CAM LOCK: {'ON' if camera_locked else 'OFF'}"

def set_view(mode):
    global view_mode
    view_mode = mode
    camera_locked = True
    cam_btn.text = "CAM LOCK: ON"

    if mode == "front":
        camera.position = Vec3(0, 0, -12)
    elif mode == "side":
        camera.position = Vec3(12, 0, 0)
    elif mode == "top":
        camera.position = Vec3(0, 12, 0)
    elif mode == "iso":
        camera.position = Vec3(8, 6, -8)

    camera.look_at(car.position)

model_btn = Button("MODEL LOCK: OFF", scale=(0.25, 0.07), position=(-0.6, 0.45), on_click=toggle_model_lock)
cam_btn   = Button("CAM LOCK: OFF",   scale=(0.25, 0.07), position=(-0.6, 0.35), on_click=toggle_camera_lock)

Button("FRONT", scale=(0.12,0.06), position=(0.6,0.45), on_click=lambda: set_view("front"))
Button("SIDE",  scale=(0.12,0.06), position=(0.6,0.35), on_click=lambda: set_view("side"))
Button("TOP",   scale=(0.12,0.06), position=(0.6,0.25), on_click=lambda: set_view("top"))
Button("ISO",   scale=(0.12,0.06), position=(0.6,0.15), on_click=lambda: set_view("iso"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

def is_open_palm_relaxed(hand):
    lm = hand.landmark
    tips = [8,12,16]
    bases = [5,9,13]
    return sum(lm[t].y < lm[b].y - 0.01 for t,b in zip(tips,bases)) >= 2

def is_peace(hand):
    lm = hand.landmark
    return lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y > lm[14].y

def is_thumbs_up(hand):
    lm = hand.landmark
    return lm[4].y < lm[3].y < lm[2].y

alpha = 0.25
smooth_rx = smooth_ry = smooth_tx = smooth_ty = 0
last_rx = last_ry = last_lx = last_ly = None
last_zoom = None
smooth_zoom = 0

paused = False
PAUSE_FRAMES = 0

def update():
    global last_rx, last_ry, last_lx, last_ly, smooth_rx, smooth_ry
    global smooth_tx, smooth_ty, last_zoom, smooth_zoom, PAUSE_FRAMES

    ok, frame = cap.read()
    if not ok:
        return

    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    left = right = None
    if res.multi_hand_landmarks and res.multi_handedness:
        for i,h in enumerate(res.multi_handedness):
            if h.classification[0].label == "Left":
                left = res.multi_hand_landmarks[i]
            else:
                right = res.multi_hand_landmarks[i]

    if left and is_open_palm_relaxed(left):
        PAUSE_FRAMES += 1
    else:
        PAUSE_FRAMES = 0

    paused = PAUSE_FRAMES >= 2

    h, w, _ = frame.shape

    # rotation
    if right and not paused and not model_locked:
        ix = int(right.landmark[8].x * w)
        iy = int(right.landmark[8].y * h)
        if last_rx is not None:
            dx, dy = ix-last_rx, iy-last_ry
            smooth_rx = smooth_rx*(1-alpha) + (-dy*2)*alpha
            smooth_ry = smooth_ry*(1-alpha) + (dx*2)*alpha
            car.rotation_x += smooth_rx
            car.rotation_y += smooth_ry
        last_rx, last_ry = ix, iy
    else:
        last_rx = last_ry = None

    # zoom
    if right and not paused and not camera_locked:
        lm = right.landmark
        pd = math.dist((lm[4].x,lm[4].y),(lm[8].x,lm[8].y))
        strength = 1 - min(1,max(0,(pd-0.02)/0.15))
        if last_zoom is not None:
            delta = (strength-last_zoom)*40
            smooth_zoom = smooth_zoom*0.65 + delta*0.35
            camera.z = clamp(camera.z - smooth_zoom, -35, -3)
        last_zoom = strength
    else:
        last_zoom = None

    # translation
    if left and not paused and not model_locked:
        lx = int(left.landmark[0].x * w)
        ly = int(left.landmark[0].y * h)
        if last_lx is not None:
            dx = (lx-last_lx)/w
            dy = (ly-last_ly)/h
            smooth_tx = smooth_tx*(1-alpha) + dx*15*alpha
            smooth_ty = smooth_ty*(1-alpha) - dy*15*alpha
            car.position += Vec3(smooth_tx, smooth_ty, 0)
        last_lx, last_ly = lx, ly
    else:
        last_lx = last_ly = None

app.run()
cap.release()
cv2.destroyAllWindows()
