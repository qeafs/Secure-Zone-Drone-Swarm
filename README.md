# Autonomous UAV Swarm for Secure Zone Reconnaissance 🛸🚨

**Graduation Project - King Saud University**
* **Department:** Computer Engineering
* **Status:** Complete (Proof of Concept)

## 📖 Project Overview
This project simulates a swarm of autonomous Unmanned Aerial Vehicles (UAVs) designed to secure restricted zones without human intervention. The system utilizes **ROS 2**, **Gazebo**, and **Deep Learning** to execute coordinated "Search and Sweep" missions.

Unlike traditional patrols, this swarm flies a coordinated grid pattern and uses onboard **Computer Vision** to detect unauthorized vehicles in real-time. When a threat is detected, the system logs the incident and saves photographic evidence automatically.

## 🚀 Key Features
* **Swarm Autonomy:** Controls multiple drones simultaneously using `asyncio` and MAVSDK.
* **AI Threat Detection:** Custom TensorFlow model (CNN) detects vehicles with >80% confidence.
* **Auto-Documentation:** Automatically captures and saves timestamped images of intrusions.
* **Resilient Architecture:** Multi-threaded design separates flight logic from heavy AI processing to prevent lag.

## 🛠️ Technologies Used
* **Simulation:** ROS 2 (Humble/Foxy), Gazebo Garden/Classic.
* **Control:** MAVSDK (Python), PX4 Autopilot.
* **AI/Vision:** TensorFlow, Keras, OpenCV.

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Secure-Zone-Swarm.git](https://github.com/YOUR_USERNAME/Secure-Zone-Swarm.git)
   cd Secure-Zone-Swarm

   pip3 install -r requirements.txt

   ros2 run ros_gz_bridge parameter_bridge /world/baylandsnew/model/x500new_1/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image
