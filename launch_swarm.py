#!/usr/bin/env python3
import subprocess
import time
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# ✅ CHANGE THIS to the actual path of your PX4-Autopilot folder
PX4_DIR = os.path.expanduser("~/PX4-Autopilot") 

# ------------------------------------------------------------------
# COMMANDS (Exact versions you provided)
# ------------------------------------------------------------------
commands = [
    # DRONE 0 (Leader)
    f'cd {PX4_DIR} && PX4_GZ_WORLD=baylandsnew PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500new PX4_SIM_MAVSDK_UDP_PORT=14540 PX4_SIM_QGC_UDP_PORT=14550 ./build/px4_sitl_default/bin/px4 -i 0',

    # DRONE 1 (Left) - Z=0 as requested
    f'cd {PX4_DIR} && PX4_GZ_WORLD=baylandsnew PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500new PX4_GZ_MODEL_POSE="-0.04,-3.46,0,0,0,0" PX4_SIM_MAVSDK_UDP_PORT=14541 PX4_SIM_QGC_UDP_PORT=14551 ./build/px4_sitl_default/bin/px4 -i 1',

    # DRONE 2 (Right) - Z=0 as requested, Port 14543
    f'cd {PX4_DIR} && PX4_GZ_WORLD=baylandsnew PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500new PX4_GZ_MODEL_POSE="-1.10,1.62,0,0,0,0" PX4_SIM_MAVSDK_UDP_PORT=14543 PX4_SIM_QGC_UDP_PORT=14552 ./build/px4_sitl_default/bin/px4 -i 2'
]

def launch_terminals():
    print("🚀 Launching Swarm (Custom Config)...")
    
    for i, cmd in enumerate(commands):
        print(f"  -> Launching Drone {i} in new terminal...")
        
        # Opens a new GNOME Terminal, runs the command, and keeps bash open if it crashes
        subprocess.Popen([
            "gnome-terminal", "--", "bash", "-c", f"{cmd}; exec bash"
        ])
        
        # Wait 4 seconds between launches to allow Gazebo to initialize safely
        time.sleep(4)

    print("✅ Done! Check the new terminal windows.")

if __name__ == "__main__":
    launch_terminals()
