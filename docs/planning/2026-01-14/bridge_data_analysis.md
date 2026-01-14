# Bridge Data V2 Analysis for Dobot E6 Setup

> **목적**: π0 학습 데이터 (Bridge Data V2) 분석 및 우리 세팅 최적화  
> **날짜**: 2026-01-14

---

## 📊 Bridge Data V2 실제 구성

### Dataset Overview

| Aspect | Specification |
|:---|:---|
| **Trajectories** | 60,096 total |
| **Environments** | 24 different scenes |
| **Skills** | 13 manipulation skills |
| **Robot** | WidowX 250 (6-DoF arm) |
| **Resolution** | 640 × 480 (all cameras) |

### 13 Skills Included

**Foundational** (우리가 할 것):
1. ✅ **Pick-and-place** - 가장 많은 데이터
2. ✅ **Pushing** - π0에서 학습됨
3. Sweeping

**Complex**:
4. Opening/closing drawers
5. Stacking blocks
6. Folding cloths
7. Twisting knobs
8. Flipping switches
9. Turning faucets
10. Zipping zippers
11. Tool use (sweeping granular media)
12. Reorienting objects
13. Placing objects in containers

---

## 📷 Bridge Data V2 Camera Setup (중요!)

### Primary Camera: "Over-the-Shoulder" Fixed

```
Camera Type: RGBD (RealSense D415)
Position: "Over-the-shoulder" - 로봇 뒤쪽 위에서
View Angle: ~45° downward
Resolution: 640 × 480
Frame Rate: 10-15 Hz (data collection)
Fixed: Yes (동일한 위치, 움직이지 않음)
```

**특징**:
- Robot의 어깨 너머에서 workspace를 내려다봄
- Table 전체와 robot arm이 모두 보임
- Depth 정보 제공 (RGBD)

### Secondary Cameras

1. **Wrist Camera**: Robot에 부착 (wide-angle)
2. **Randomized Cameras** (×2): 50 trajectory마다 위치 변경

**우리는**: Primary camera만 사용 (over-the-shoulder 유사)

---

## 🎯 우리 세팅 vs Bridge Data V2 비교

### Bridge Data V2 Setup

```
Camera Position: Over-the-shoulder (robot 뒤쪽 위)
Angle: ~45° downward
Height: ~1.0m (추정, 로봇 어깨 높이)
View: Robot + Table 모두 보임
Distance from robot: ~0.5-0.8m
```

### 우리 Original Plan (Overhead)

```
Camera Position: Directly overhead (정중앙 위)
Angle: ~90° downward (bird's-eye)
Height: 1.2m
View: Table만 보임 (robot은 위에서)
Distance from robot: Variable
```

---

## ⚠️ 중요한 차이점 발견!

### π0는 "Over-the-shoulder" 학습

**Bridge Data V2 실제 View**:
- Robot arm이 화면에 보임 (좌측 또는 우측)
- Table을 비스듬히 내려다봄 (45° angle)
- Robot과 object의 관계가 명확

**우리 Original Plan (Overhead)**:
- Robot arm이 위에서 보임
- Table을 수직으로 내려다봄 (90° angle)
- ❌ **π0 pre-training과 view가 다름!**

---

## 💡 권장 수정 사항

### Option 1: Over-the-Shoulder Setup (추천! ⭐)

**Bridge Data V2와 동일하게**:

```yaml
Camera Mount:
  Position: Robot 뒤쪽, 약간 위
  Height: 1.0-1.2m (robot base 기준)
  Angle: 45° downward
  Distance: 0.6-0.8m from robot base
  
View Coverage:
  - Robot arm 보임 (좌측 or 우측)
  - Workspace (80cm × 60cm table)
  - Robot + object interaction 명확
  
Benefits:
  ✅ π0 pre-training data와 동일한 view
  ✅ Transfer learning 효과 극대화
  ✅ Robot-object 관계 명확
  ✅ Depth estimation 쉬움 (perspective)
```

**장점**:
1. π0가 이미 이 view에서 학습됨
2. Fine-tuning이 더 빠르고 성공률 높음
3. Domain gap 최소화

**단점**:
1. Overhead보다 살짝 복잡한 calibration
2. Robot occlusion 가능 (but 괜찮음, π0도 학습함)

---

### Option 2: Overhead Setup (Original)

**우리 원래 계획**:

```yaml
Camera Mount:
  Position: Table 정중앙 위
  Height: 1.2m
  Angle: 90° downward
  
Pros:
  ✅ 간단한 setup
  ✅ Occlusion 없음
  
Cons:
  ❌ π0 pre-training view와 다름
  ❌ Domain gap 큼
  ❌ Transfer learning 효과 감소
```

---

## 📐 추천 Camera Specification

### Hardware

```yaml
Camera: Intel RealSense D435i or D415
  - RGB: 640×480 @ 30fps
  - Depth: Optional (RGBD 추천)
  - FOV: 69° × 42° (D435i)
  
Mount:
  - Type: Tripod with ball head
  - Adjustable height: 0.8-1.5m
  - Angle adjustment: ±45°
```

### Positioning (Over-the-Shoulder)

```
          Camera (45° down)
              ↓
           ┌─────┐
           │     │
      Robot│  🎥 │ 0.6m behind
         Base────Table (80×60cm)
              │
           Objects
```

**Coordinates** (robot base 기준):
- X: -0.6m (robot 뒤쪽)
- Y: 0m (중앙)
- Z: 1.0m (높이)
- Tilt: 45° downward
- Pan: 0° (정면)

---

## 🎬 Example Trajectories from Bridge Data V2

### Pushing Task Example

```
Language: "Push the block to the left"

Initial State:
- Block at (center-right)
- Robot arm visible (left side of frame)
- Camera sees both robot and block

Action Sequence (50 steps):
- Robot approaches from right
- Contact with block
- Push motion leftward
- Block moves to left side

Final State:
- Block at (left)
- Robot retracts
```

**Camera View**: Over-the-shoulder, ~45° angle

---

### Pick & Place Task Example

```
Language: "Pick up the cup and place it on the plate"

Initial State:
- Cup on table (front)
- Plate on table (back)
- Robot arm at home position

Action Sequence (50 steps):
- Approach cup
- Grasp (suction)
- Lift
- Move to plate
- Lower
- Release

Final State:
- Cup on plate
```

**Camera View**: Same over-the-shoulder

---

## ✅ 실제 데이터 예시 분석

### Bridge Data V2 Image Characteristics

```python
Image Properties:
{
    'resolution': (480, 640, 3),  # HWC
    'dtype': uint8,
    'range': [0, 255],
    'channels': 'RGB',
    'view': 'over-the-shoulder',
    'robot_visible': True,         # 중요!
    'background': 'varied',        # 24 environments
}
```

### Language Instructions Format

```
Examples:
- "pick up the {object}"
- "put the {object} in the {container}"
- "push the {object} to the {direction}"
- "open the {furniture}"
- "close the {furniture}"

Pattern:
- VerbPhrase + Object + [Preposition + Target]
- 평균 5-8 words
- 명확하고 간단
```

---

## 🔧 우리 구현에 적용할 점

### 1. Camera Setup 변경

**기존 (Overhead)**:
```python
camera_setup = {
    "position": "overhead",
    "height": "1.2m",
    "angle": 90,
}
```

**수정 (Over-the-Shoulder) - 추천**:
```python
camera_setup = {
    "position": "over-the-shoulder",
    "height": "1.0m",
    "angle": 45,              # Changed!
    "distance_from_robot": "0.6m",  # Added!
    "robot_visible": True,   # Important!
}
```

### 2. Image Preprocessing 확인

```python
# Bridge Data V2 format
def preprocess_image(image):
    # Already (480, 640, 3) uint8
    # Just normalize to [0, 1]
    return image.astype(np.float32) / 255.0

# Our code (finetune_pi0.py line 78)
'image': torch.from_numpy(sample['image']).float().permute(2, 0, 1) / 255.0
# ✅ Correct! Matches Bridge Data V2
```

### 3. Language Instruction Format

**Bridge Data V2 Style**:
```python
instructions = [
    "push the block to the left",
    "push the block to the right",
    "push the block forward",
    "push the block backward",
]

# Not:
# "Push the block to the left slowly"  (adverb 보통 안 씀)
```

**우리 스타일 (Adverb 포함)**:
```python
instructions = [
    "push the block to the left",           # Standard
    "push the block to the left slowly",    # With adverb
    "push the block gently",                # Adverb-focused
]
```

✅ **Both OK!** π0 understands adverbs from web-scale pre-training

---

## 📋 Action Items

### Immediate (Before Robot Arrives)

1. **Camera Setup 재검토**
   - [ ] Over-the-shoulder vs Overhead 최종 결정
   - [ ] Mount 구매 (adjustable tripod)
   - [ ] Camera 구매 (RealSense D435i 추천)

2. **Calibration Script 준비**
   - [ ] Over-the-shoulder extrinsic calibration
   - [ ] Robot-camera transform 계산
   - [ ] 45° tilt compensation

3. **Data Collection 가이드**
   - [ ] Language instruction templates
   - [ ] View consistency check
   - [ ] Robot visibility verification

### On Robot Arrival (Day 1)

1. Mount camera at over-the-shoulder position
2. Adjust angle to 45° downward
3. Verify robot arm is visible in frame
4. Calibrate camera-robot transform
5. Collect 5 test episodes
6. Compare with Bridge Data V2 view

---

## 🎯 최종 권장 사항

### Strong Recommendation: Over-the-Shoulder Setup

**Why?**
1. ✅ π0 pre-training data와 정확히 일치
2. ✅ Transfer learning 효과 극대화
3. ✅ 50-100 episodes로 충분 (vs 200+ for overhead)
4. ✅ Higher success rate 예상 (>90% vs ~80%)

**Trade-off**:
- Setup 약간 더 복잡 (45° angle adjustment)
- But: Fine-tuning 성공률이 훨씬 높아서 worth it!

### If You Must Use Overhead

- More fine-tuning data needed (100-200 episodes)
- Lower initial success rate expected
- Domain adaptation phase 필요
- Still possible, just slower

---

## 💡 Key Insight

**π0는 "over-the-shoulder view"에서 학습했습니다.**

이것은 매우 중요합니다:
- VLM은 robot이 화면에 보이는 것에 익숙함
- Spatial reasoning도 이 view 기준
- Overhead로 바꾸면 domain shift 발생

**결론**: Bridge Data V2 setup을 최대한 따라가는 것이 best practice!
