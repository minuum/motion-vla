# π0 Dobot E6 Adaptation - Implementation Plan (2026-01-10)

> **목표**: π0 Task 선정 및 Dobot E6 적용  
> **참여**: 유빈, 민우  
> **기준일**: 2026-01-10

---

## 📋 미팅 결정 사항

### 할 일 목록

- [ ] **Jetson 설치** → 유빈
- [ ] **π0 Task 선정** (π0에서 했던 것 중 선택)
- [ ] **VLM (Frozen) + Action Head (Editing)** → 유빈, 민우
- [ ] **π0 → Dobot E6 Adaptation** → 유빈, 민우

---

## 🔧 구현 사항 (Hardware Constraints)

| 항목 | 설명 | 대응 |
|:---|:---|:---|
| **6-DoF** | Dobot E6 6축 제어 | π0 action space 수정 |
| **Grip Loss 제거** | Gripper 미사용 | Action dim 7→6 |
| **외부 카메라** | Overhead fixed | PaliGemma input 변경 |
| **파인튜닝** | Action head만 학습 | VLM freeze |
| **카메라 세팅** | 하늘에서 고정 | Bird's-eye view |

---

## 🎯 π0 Task 후보 평가

### π0 Original Tasks (Physical Intelligence)

| Task | π0 평가 | Dobot E6 실현성 | 추천 |
|:---|:---:|:---:|:---:|
| **Table Bussing** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ 추천 |
| **Box Assembly** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⚠️ 어려움 |
| **Laundry Folding** | ⭐⭐⭐⭐⭐ | ⭐ | ❌ 불가 |
| **Grocery Bagging** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⚠️ 복잡 |

### Wiping (기존 선정)

| Task | π0 적합성 | Dobot E6 실현성 | 추천 |
|:---|:---:|:---:|:---:|
| **Table Wiping** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **최우선** |

---

## 💡 Task 선정 권장: Table Wiping (유지)

### 선정 이유

1. **π0 강점 활용**
   - Contact-rich manipulation
   - 50Hz continuous control
   - Action chunking 효과적

2. **Dobot E6 적합**
   - 6-DoF 충분
   - Wiper tool 부착 (grip 불필요)
   - 작업 영역 내 (45cm reach)

3. **구현 현실성**
   - 환경 85% 완성 (기존 작업)
   - Isaac Sim 시뮬 준비됨
   - 데이터 수집 계획 완료

### π0 Table Bussing과의 비교

| Aspect | Table Bussing | Table Wiping |
|:---|:---|:---|
| **Object handling** | 다양한 물체 | 단일 wiper |
| **Gripper 필요** | ✅ 필요 | ❌ 불필요 |
| **Dobot 적합성** | ⚠️ Grip loss | ✅ 완벽 |
| **구현 난이도** | 높음 | 낮음 |
| **데이터 준비** | 미완성 | 완성 |

**결론**: Wiping 유지 (Grip loss 제거 요구사항에 적합)

---

## 🏗️ Architecture: VLM Frozen + Action Head Only

### π0 Original Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PaliGemma  │────▶│   Action    │────▶│   Robot     │
│    (VLM)    │     │    Head     │     │  Control    │
└─────────────┘     └─────────────┘     └─────────────┘
   3B params           ~50M params        6-DoF output
```

### Adaptation Strategy

```python
model = {
    # Frozen (학습 X)
    "vl_encoder": {
        "type": "PaliGemma-3B",
        "trainable": False,  # ❄️ Freeze
        "params": "3B",
    },
    
    # Trainable (학습 O)
    "action_head": {
        "type": "FlowMatchingDecoder",
        "trainable": True,   # 🔥 Fine-tune
        "params": "~50M",
        "output_dim": 6,     # 6-DoF (no gripper)
    },
}
```

### Action Space 변경

```python
# π0 Original (7-DoF + gripper)
action_original = [x, y, z, rx, ry, rz, gripper]  # dim=7

# Dobot E6 Adaptation (6-DoF, no gripper)
action_adapted = [x, y, z, rx, ry, rz]  # dim=6

# 변경 사항
changes = {
    "gripper": "제거",
    "output_dim": "7 → 6",
    "loss_function": "grip_loss 제거",
}
```

---

## 📷 Camera Setup: Overhead Fixed

### Configuration

```python
camera_setup = {
    "position": "overhead",     # 하늘에서
    "type": "fixed",            # 고정
    "height": "1.2m",           # 테이블 위
    "resolution": (640, 480),
    "fov": 60,                  # degrees
    "view": "bird's-eye",
}
```

### PaliGemma Input

```python
# Single camera input (simplified)
observation = {
    "image": (480, 640, 3),   # Overhead RGB
    "language": "Wipe the table gently",
}

# π0 original: multiple cameras
# Ours: single overhead camera (simplified)
```

---

## 📅 구현 계획

### Phase 1: 환경 준비 (Day 1-2)

- [ ] Jetson 설치 (유빈)
- [ ] Isaac Sim 환경 완성
- [ ] Camera 세팅 (overhead fixed)

### Phase 2: 모델 Adaptation (Day 3-5)

- [ ] Action head 수정 (6-DoF)
- [ ] Grip loss 제거
- [ ] VLM freeze 설정

### Phase 3: 학습 & 검증 (Day 6-10)

- [ ] Sim 데이터 수집 (342 eps)
- [ ] Action head fine-tuning
- [ ] Real robot 테스트 (38 eps)

---

## 📊 데이터셋 요약 (기존 계획 유지)

| Split | Sim | Real | Total |
|:---|:---:|:---:|:---:|
| Train | 274 | 30 | 304 |
| Val | 34 | 4 | 38 |
| Test | 34 | 4 | 38 |
| **Total** | **342** | **38** | **380** |

---

## ✅ 다음 단계

1. **Task 확정**: Table Wiping (권장) vs Table Bussing (대안)
2. **Jetson 설치**: 환경 구축
3. **Action Head 수정**: 6-DoF, grip loss 제거
4. **Camera 세팅**: Overhead fixed 구현

**결정 필요**: Table Wiping 유지? or π0 원본 task로 변경?
