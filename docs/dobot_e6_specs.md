# Dobot E6 Magician Specifications

> **Reference**: Official DOBOT documentation and vendor specifications  
> **Last Updated**: 2026-01-02

## Hardware Specifications

### Physical Dimensions
- **Model**: DOBOT Magician E6 (6-Axis Collaborative Robot)
- **Weight**: ≤7.2 kg
- **Base Dimensions**: 162mm x 120mm x 103mm
- **Working Radius**: 450mm
- **Repeatability**: ±0.1mm

### Performance
- **Degrees of Freedom**: 6-axis
- **Payload Capacity**: 0.75kg (750g)
- **Maximum TCP Speed**: 0.5 m/s
- **Maximum Joint Speed**: 120°/s

### Power & Electronics
- **Input Power**: 100V~240V AC, 50/60 Hz
- **Rated Voltage**: 48V DC, 5A
- **Power Consumption**: 130W
- **IP Rating**: IP20

---

## Control Interfaces

### Communication
- **Primary**: TCP/IP (Ethernet)
- **Protocol**: Modbus TCP
- **Ports**: 2x Ethernet ports

### I/O Interface
**Arm Tip**:
- 2x Digital Inputs (DI)
- 2x Digital Outputs (DO)
- 1x 24V
- 1x GND

**Base**:
- 16x Digital Inputs (DI)
- 16x Digital Outputs (DO)
- 4x 24V
- 4x GND
- I/O Power: 24V, max 2A (0.5A per channel)

### External Interfaces
- 1x Emergency Stop (EMO)
- 1x ABZ Encoder
- 1x Power Connector

---

## Software & Programming

### Official Software
- **Control Software**: DobotStudio Pro
- **Supported OS**: Ubuntu 22.04 (for ROS2)

### Programming Support
- **ROS/ROS2**: ROS2 Humble (Official SDK available)
- **Languages**: Python, C++, C#, MATLAB, LabVIEW, Kotlin
- **Graphical Programming**: Scratch-type block programming
- **Lua Scripting**: Via DobotStudio Pro

### Special Features
- **Drag-to-Teach**: Proprietary trajectory replay technology
- **Collision Detection**: Built-in safety feature
- **Status Indicator**: LED ring for operational monitoring

---

## Action Space for VLA

### Joint Space
```python
# 6 revolute joints
joint_limits = {
    "joint_1": [-135, 135],   # degrees
    "joint_2": [-5, 85],
    "joint_3": [-10, 95],
    "joint_4": [-180, 180],
    "joint_5": [-90, 90],
    "joint_6": [-180, 180],
}
```

### Cartesian Space (TCP)
```python
# End-effector pose
tcp_workspace = {
    "x": [-450, 450],      # mm (working radius)
    "y": [-450, 450],
    "z": [0, 450],         # height above base
    "roll": [-180, 180],   # degrees
    "pitch": [-180, 180],
    "yaw": [-180, 180],
}
```

### Gripper
- **Type**: Pneumatic or servo gripper (optional end-effector)
- **Control**: Binary (open/close) or position control
- **Action Representation**: `gripper ∈ [0, 1]` (0=open, 1=closed)

---

## Operating Environment

### Physical Environment
- **Temperature**: 0°C to 40°C
- **Humidity**: 25% to 85% (non-condensing)
- **Workspace**: Desktop-sized (recommended 1m x 1m area)

### Safety Features
- Collision detection with automatic stop
- Emergency stop button
- Streamlined body design (reduce collision risk)
- LED status indicator (green=normal, red=error)

---

## VLA Integration Considerations

### Strengths for VLA Research
✅ **High Repeatability** (±0.1mm): Consistent baseline for learning  
✅ **ROS2 Native Support**: Easy integration with our pipeline  
✅ **Drag-to-Teach**: Efficient human demonstration collection  
✅ **Desktop Size**: Accessible for lab experiments  
✅ **TCP/IP Control**: Low-latency command transmission

### Limitations
⚠️ **Payload** (0.75kg): Cannot handle heavy objects  
⚠️ **Speed** (0.5 m/s max): Limited dynamic task capability  
⚠️ **Reach** (450mm): Small workspace compared to industrial robots  
⚠️ **No Force/Torque Sensor**: Cannot perform force-sensitive tasks  

### Recommended Task Types
1. **Light Object Manipulation**: Blocks, cups, small tools (< 500g)
2. **Precision Tasks**: Assembly, pick-and-place with tight tolerances
3. **Vision-based Tasks**: Color/shape recognition, sorting
4. **Language-conditioned Control**: Speed/style variations within safe limits

---

## ROS2 Integration

### Available ROS2 Packages
- **dobot_ros2**: Official ROS2 driver
- **moveit2_dobot**: MoveIt2 integration for motion planning
- **dobot_description**: URDF model for simulation

### Topic Structure (예상)
```bash
# Control
/dobot/joint_command        # JointTrajectory
/dobot/gripper_command      # GripperCommand

# Feedback
/dobot/joint_states         # JointState
/dobot/tcp_pose            # PoseStamped
/dobot/collision_status    # Bool

# Camera (if equipped)
/dobot/camera/image_raw    # Image (RGB)
```

---

## References
- [Official Dobot Website](https://www.dobot.cc/)
- [ROS2 Dobot SDK (GitHub)](https://github.com/Dobot-Arm/...)
- Vendor Specifications: Scan.co.uk, RobotShop, Unchained Robotics
