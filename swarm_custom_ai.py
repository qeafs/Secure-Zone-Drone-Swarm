#!/usr/bin/env python3
import asyncio
import threading
import math
import time
import cv2
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# AI Imports
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras

# MAVSDK Imports
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError

# ==========================================
# PART 1: ROS 2 CAMERA HELPER
# ==========================================
_ros_initialized = False
_ros_executor = None
_ros_thread = None

def _start_global_ros_thread():
    """Starts ONE background thread to handle ALL drones."""
    global _ros_initialized, _ros_executor, _ros_thread
    if _ros_initialized: return
    try: rclpy.init()
    except: pass 
    _ros_executor = MultiThreadedExecutor()
    _ros_initialized = True
    def _spin_loop(): _ros_executor.spin()
    _ros_thread = threading.Thread(target=_spin_loop, daemon=True)
    _ros_thread.start()

class CameraNode(Node):
    def __init__(self, topic_name):
        unique_id = str(abs(hash(topic_name)) % 100000)
        super().__init__(f'cam_node_{unique_id}')
        self.bridge = CvBridge()
        self._frame = None
        self._lock = threading.Lock()
        self.subscription = self.create_subscription(Image, topic_name, self._callback, 10)

    def _callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock: self._frame = cv_image
        except Exception: pass

    def get_latest_frame(self):
        with self._lock: return None if self._frame is None else self._frame.copy()

async def get_frames_generator(topic_name):
    _start_global_ros_thread()
    node = CameraNode(topic_name)
    _ros_executor.add_node(node)
    while True:
        frame = node.get_latest_frame()
        if frame is not None: yield frame
        else: await asyncio.sleep(0.05)
        await asyncio.sleep(0.01)

# ==========================================
# PART 2: MISSION PARAMETERS
# ==========================================
ALT_REL = 33.0            
POS_TOL = 3.0             
SYNC_WAIT_SEC = 6.0
SEPARATION = 200.0
GRID_LENGTH = 500.0 
GRID_WIDTH  = 150.0  
LANE_WIDTH  = 20.0 
GRID_ANGLE  = 70.0  

# ==========================================
# PART 3: AI LOGIC (LEADER VERSION)
# ==========================================
def run_prediction(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(img, dtype=np.float32)
    normalized_image_array = (img_array / 127.5) - 1.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data, verbose=0)
    return prediction

async def ai_worker(drone_name: str, topic_name: str):
    print(f"[{drone_name}] Loading Custom AI Model...")

    # PATHS
    model_path = "/home/mohammed/Desktop/my_model.h5"
    meta_path = "/home/mohammed/Desktop/metadata.json"
    
    # SAVE FOLDER
    save_folder = "/home/mohammed/Desktop/detected_cars"
    os.makedirs(save_folder, exist_ok=True)

    # CONFIGURATION
    TARGET_CLASS_INDEX = 0  
    CONFIDENCE_THRESHOLD = 0.80
    CAPTURE_COOLDOWN = 3.0 

    try:
        model = tf_keras.models.load_model(model_path, compile=False)
        with open(meta_path, "r") as f:
            data = json.load(f)
            labels = data.get("labels", ["Class 1", "Class 2"])
        print(f"[{drone_name}] Model Loaded! Watching for Index {TARGET_CLASS_INDEX}: {labels[TARGET_CLASS_INDEX]}")
    except Exception as e:
        print(f"[{drone_name}] CRITICAL AI ERROR: {e}")
        return

    loop = asyncio.get_running_loop()
    last_scan_time = 0
    scan_interval = 0.1
    last_save_time = 0
    show_saved_text_until = 0

    print(f"[{drone_name}] Connecting to camera topic: {topic_name}")

    async for frame in get_frames_generator(topic_name):
        current_time = time.time()
        
        # CHANGE 1: Check for LEADER here
        if drone_name == "LEADER":
            display_frame = frame.copy()
            
            if current_time - last_scan_time > scan_interval:
                last_scan_time = current_time
                try:
                    prediction = await loop.run_in_executor(None, lambda: run_prediction(model, frame))
                    
                    index = np.argmax(prediction)
                    class_name = labels[index]
                    confidence_score = prediction[0][index]

                    detected = (index == TARGET_CLASS_INDEX and confidence_score > CONFIDENCE_THRESHOLD)
                    
                    if detected:
                        print(f"🚨 [{drone_name}] Found {class_name} ({confidence_score*100:.1f}%)")
                        
                        text = f"{class_name}: {int(confidence_score*100)}%"
                        cv2.putText(display_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        h, w, _ = display_frame.shape
                        cv2.rectangle(display_frame, (0,0), (w,h), (0,0,255), 15)

                        if current_time - last_save_time > CAPTURE_COOLDOWN:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"car_{timestamp}.jpg"
                            full_path = os.path.join(save_folder, filename)
                            cv2.imwrite(full_path, frame) 
                            print(f"📸 SAVED: {full_path}")
                            last_save_time = current_time
                            show_saved_text_until = current_time + 1.0

                    else:
                        cv2.putText(display_frame, f"Scanning... ({labels[index]}: {int(confidence_score*100)}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                except Exception as e:
                    print(f"AI Prediction Error: {e}")

            if current_time < show_saved_text_until:
                 cv2.putText(display_frame, "IMAGE SAVED TO DESKTOP!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            cv2.imshow(f"Drone Vision: {drone_name}", display_frame)
            cv2.waitKey(1)

# ==========================================
# PART 4: FLIGHT LOGIC
# ==========================================
def rotate_point(n, e, angle_deg):
    rad = math.radians(angle_deg)
    new_n = n * math.cos(rad) - e * math.sin(rad)
    new_e = n * math.sin(rad) + e * math.cos(rad)
    return new_n, new_e

@dataclass(frozen=True)
class Cell:
    n: float
    e: float

def generate_smooth_grid(length, width, spacing, angle):
    raw_cells = []
    num_lines = int(width / spacing)
    current_e = -width / 2
    going_up = True
    for _ in range(num_lines + 1):
        start_n = 0.0 if going_up else length
        raw_cells.append(Cell(start_n, current_e))
        end_n = length if going_up else 0.0
        raw_cells.append(Cell(end_n, current_e))
        current_e += spacing
        going_up = not going_up
    return [Cell(*rotate_point(c.n, c.e, angle)) for c in raw_cells]

class OffboardManager:
    def __init__(self, drone):
        self.drone = drone
        self.target = PositionNedYaw(0, 0, 0, 0)
        self.running = False
        self._task = None
    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._loop())
    async def stop(self):
        self.running = False
        if self._task: await self._task
    def update(self, n, e, d, yaw):
        self.target = PositionNedYaw(n, e, d, yaw)
    async def _loop(self):
        while self.running:
            try: await self.drone.offboard.set_position_ned(self.target)
            except: pass
            await asyncio.sleep(0.05)

async def wait_arrival(d, target_n, target_e):
    home = await anext(d.telemetry.home())
    t_start = time.time()
    def ne_from(hlat, hlon, lat, lon):
        dy = (lat - hlat) * 111320.0
        dx = (lon - hlon) * 111320.0 * math.cos(math.radians(hlat))
        return dy, dx
    async for p in d.telemetry.position():
        cn, ce = ne_from(home.latitude_deg, home.longitude_deg, p.latitude_deg, p.longitude_deg)
        if math.hypot(target_n - cn, target_e - ce) < POS_TOL: return
        if time.time() - t_start > 300: return 

async def fly_one(name: str, addr: str, port: int, offset_n: float, offset_e: float, sync_t0: float, cam_topic: str):
    start_n, start_e = rotate_point(offset_n, offset_e, GRID_ANGLE)
    cells = generate_smooth_grid(GRID_LENGTH, GRID_WIDTH, LANE_WIDTH, GRID_ANGLE)

    d = System(port=port)
    print(f"[{name}] Connecting...")
    for _ in range(5):
        try: await d.connect(system_address=addr); break
        except: await asyncio.sleep(2)

    print(f"[{name}] Waiting GPS...")
    async for state in d.telemetry.health():
        if state.is_local_position_ok: break
        await asyncio.sleep(1)

    delay = max(0, sync_t0 - time.time())
    print(f"[{name}] Starting in {delay:.1f}s")
    await asyncio.sleep(delay)

    print(f"[{name}] Taking Off")
    await d.action.arm()
    await d.action.takeoff()
    await asyncio.sleep(8) 

    manager = OffboardManager(d)
    manager.update(0, 0, -ALT_REL, 0)
    await manager.start()

    # CHANGE 2: Assign AI Task to LEADER
    ai_task = None
    if name == "LEADER":
        ai_task = asyncio.create_task(ai_worker(name, cam_topic))

    for _ in range(5):
        try: await d.offboard.start(); break 
        except OffboardError:
            await asyncio.sleep(1)
            await d.offboard.set_position_ned(PositionNedYaw(0,0,-ALT_REL,0))

    print(f"[{name}] Starting Search Pattern")
    tgt_n = cells[0].n + start_n
    tgt_e = cells[0].e + start_e
    manager.update(tgt_n, tgt_e, -ALT_REL, 0)
    await wait_arrival(d, tgt_n, tgt_e)

    for i, wp in enumerate(cells):
        tgt_n = wp.n + start_n
        tgt_e = wp.e + start_e
        manager.update(tgt_n, tgt_e, -ALT_REL, 0)
        await wait_arrival(d, tgt_n, tgt_e)

    print(f"[{name}] Done. Landing.")
    if ai_task: ai_task.cancel()
    await manager.stop()
    await d.action.land()

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    t0 = math.ceil(time.time() + SYNC_WAIT_SEC)
    
    # Ensure this matches your ROS2 topic (usually 0 is leader)
    topic_base = "/world/baylandsnew/model/x500new_{}/link/camera_link/sensor/camera/image"

    tasks = [
        fly_one("LEADER", "udpin://0.0.0.0:14540", 50051, 0.0, 0.0, t0, topic_base.format(0)),
        fly_one("LEFT",   "udpin://0.0.0.0:14541", 50052, 0.0, -SEPARATION, t0, topic_base.format(1)),
        fly_one("RIGHT",  "udpin://0.0.0.0:14542", 50053, 0.0, SEPARATION, t0, topic_base.format(2)),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())