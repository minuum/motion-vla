# Dataset Size Design: Episodes & Samples Analysis

> **날짜**: 2026-01-07  
> **목적**: π0 스타일 Flow-matching VLA를 위한 데이터셋 규모 설계

---

## 📊 Episodes vs Samples 구분

### 정의
- **Episode (궤적)**: 완전한 wiping 동작 (시작→끝)
  - 길이: 250 timesteps @ 50Hz = 5초
- **Sample (학습 데이터)**: Action chunking으로 추출된 훈련 포인트
  - 구성: (current obs, next 50 actions)
  - 추출: Episode 내에서 stride=10으로 샘플링

### Samples per Episode 계산

```python
episode_length = 250  # timesteps
chunk_size = 50       # predicted future actions
stride = 10           # sampling interval

samples = (250 - 50) / 10 = 20 samples/episode
```

---

## 🎯 최종 Dataset Size (논문 근거)

### 전체 비교

| Method | Type | Episodes | Samples (raw) | Samples (aug) | Setting |
|:---|:---|:---:|:---:|:---:|:---|
| **BridgeData Wiping** | Multi | ~3,000-5,000 | ~100K | - | Multi-robot |
| **Diffusion Policy** | Single | 205 | ~4,100 | - | Sim |
| **Relay-HER** | Single | 250 | ~5,000 | - | Real robot |
| **FERM** | Single | 20-80 | ~1,600 | - | Real robot |
| **우리 (최종)** | Single | **430** | **8,600** | **53,300** | Real robot |

---

## 📋 Task별 Episodes & Samples

### 최종 추천안 (Option 3)

| Task | Instruction | Episodes (Sim) | Episodes (Real) | **Total Episodes** | **Samples (raw)** | **Samples (aug 6.2x)** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **T1** | "Wipe the table" | 90 | 10 | **100** | **2,000** | **12,400** |
| **T2** | "Wipe gently" | 72 | 8 | **80** | **1,600** | **9,920** |
| **T3** | "Wipe firmly" | 72 | 8 | **80** | **1,600** | **9,920** |
| **T4** | "Wipe quickly" | 54 | 6 | **60** | **900*** | **5,580** |
| **T5** | "Wipe thoroughly" | 54 | 6 | **60** | **1,500**** | **9,300** |
| | **Total** | **342** | **38** | **380** | **7,600** | **47,120** |

*T4 quickly는 짧아서 (150 steps) → 15 samples/episode  
**T5 thoroughly는 길어서 (300 steps) → 25 samples/episode

### Augmentation 세부

```python
augmentation = {
    "temporal_crop": 2.0x,     # 다른 시작점
    "mirror_flip": 2.0x,       # 좌우 대칭
    "action_noise": 1.3x,      # Robustness
    "speed_jitter": 1.2x,      # ±10% playback
}

total_multiplier = 2 × 2 × 1.3 × 1.2 = 6.24x
```

---

## 📈 수정안 비교 (Episodes & Samples)

### Option 1: 보수적 (BridgeData 스타일)

| Task | Episodes | Samples (raw) | Samples (aug) | 수집 시간 |
|:---|:---:|:---:|:---:|:---:|
| T1-T5 | **780** | **15,600** | **97,200** | ~22시간 |

**평가**: ❌ 너무 많음, 시간 부족

---

### Option 2: 공격적 (Diffusion Policy 스타일)

| Task | Episodes | Samples (raw) | Samples (aug) | 수집 시간 |
|:---|:---:|:---:|:---:|:---:|
| T1-T5 | **270** | **5,400** | **33,600** | ~8시간 |

**평가**: ⚠️ 너무 적음, 성능 불안

---

### Option 3: 균형 (추천!) ⭐

| Task | Episodes | Samples (raw) | Samples (aug) | 수집 시간 |
|:---|:---:|:---:|:---:|:---:|
| T1-T5 | **380** | **7,600** | **47,120** | ~12시간 |

**평가**: ✅ 적절함!
- Relay-HER (250) 대비 1.5배
- Diffusion Policy (205) 대비 1.85배
- 수집 시간 현실적 (~12시간)

---

## 🔬 논문별 상세 분석

### Diffusion Policy (CoRL 2023)

```
Task: Push-T (simulation)
Episodes: 205
Samples: ~4,100 (20 samples/episode)
Control: 10Hz
Success: >90%
```

**우리와 비교**:
- 우리 380 episodes = **1.85배 많음** ✅
- 우리 50Hz = **5배 빠른 control** ✅

---

### Relay-HER (Real Robot)

```
Task: Sparse reward manipulation
Episodes: 250
Samples: ~5,000 (추정)
Success: 10/10 (100%)
Setting: Physical robot
```

**우리와 비교**:
- 우리 380 episodes = **1.52배 많음** ✅
- 동일하게 real robot ✅

---

### FERM (CoRL 2023)

```
Task: Sparse reward (from pixels)
Episodes: 20-80
Training time: 15-50 minutes
Success: High
Setting: Real robot
```

**우리와 비교**:
- 우리 380 episodes = **4.75~19배 많음** ✅
- 더 robustness 확보 ✅

---

### BridgeData V2

```
Task: Wiping (multi-task)
Estimated episodes: 3,000-5,000
Samples: ~100,000 (추정)
Control: 5Hz
Robot: WidowX
```

**우리와 비교**:
- 우리 380 episodes = **1/10 수준** ⚠️
- 하지만 우리는 **single task focused** ✅
- 우리 50Hz = **10배 빠른 control** ✅

---

## 💾 Storage Requirement

### Episode당 저장 공간

```python
single_episode = {
    "rgb": 250 × 480 × 640 × 3 × 1 byte = 230 MB,
    "robot_state": 250 × 6 × 4 bytes = 6 KB,
    "actions": 250 × 7 × 4 bytes = 7 KB,
}

total_per_episode ≈ 230 MB (uncompressed)
```

### 전체 Dataset 크기

```python
# 380 episodes
total_size = 380 × 230 MB = 87.4 GB (uncompressed)

# With compression (JPEG for images)
compressed_size = 87.4 GB × 0.3 = ~26 GB
```

**결론**: 약 **26GB** HDF5 파일

---

## ⏱️ 수집 시간 상세

### Simulation Data (342 episodes)

```python
# Automated trajectory generation
time_per_episode = 1분 (궤적 생성 + 저장)
total_time = 342 × 1분 = 342분 = 5.7시간

# Domain randomization overhead
setup_time = 2시간

total_sim = 5.7 + 2 = 7.7시간
```

### Real Robot Data (38 episodes)

```python
# Teleoperation
time_per_episode = 5분 (준비 + 실행 + 저장)
total_time = 38 × 5분 = 190분 = 3.2시간

# Setup & calibration
setup_time = 1.5시간

total_real = 3.2 + 1.5 = 4.7시간
```

### 총 수집 시간

```
Sim: 7.7시간
Real: 4.7시간
Total: 12.4시간
```

**현실성**: ✅ **2일 안에 완료 가능**

---

## 🎓 Training Compute Estimate

### Dataset Size

```python
episodes = 380
samples_raw = 7,600
samples_augmented = 47,120
```

### Training Time (A100 GPU)

```python
# π0-style flow-matching
batch_size = 48
total_samples = 47,120
iterations_per_epoch = 47,120 / 48 = 982

# 3-stage training
stage1_epochs = 20  # Sim warmup
stage2_epochs = 30  # Sim+Real
stage3_epochs = 20  # Real fine-tune

total_epochs = 70
total_iterations = 70 × 982 = 68,740

# A100 기준 (추정)
time_per_iteration = 0.5초
total_time = 68,740 × 0.5초 = 34,370초 = 9.5시간
```

**Training**: 약 **10시간** (A100 1장)

---

## 📊 최종 요약 (Episodes & Samples)

| Metric | Value | 근거 |
|:---|:---:|:---|
| **Total Episodes** | **380** | Relay-HER 1.5배 |
| **Sim Episodes** | 342 | 90% automation |
| **Real Episodes** | 38 | 10% teleoperation |
| **Raw Samples** | **7,600** | 20 samples/episode |
| **Augmented Samples** | **47,120** | 6.2x augmentation |
| **Storage** | 26 GB | Compressed HDF5 |
| **Collection Time** | 12.4 hours | 2일 작업 |
| **Training Time** | 9.5 hours | A100 1장 |

---

## ✅ 결론

### 왜 380 episodes + 47K samples인가?

1. **논문 근거**:
   - Relay-HER (250) → 실제 로봇 성공
   - Diffusion Policy (205) → Sim 고성능
   - 우리는 두 배 → 안전 마진

2. **Samples 효율**:
   - Raw 7.6K → 적당함
   - Augmented 47K → 충분함
   - Per task 9-12K → SOTA 수준

3. **현실성**:
   - 수집 12시간 → 2일 가능
   - 학습 10시간 → 하루 가능
   - 총 **1주일 완성** 가능!

**최종 추천**: **380 episodes (7.6K raw, 47K aug samples)** ✅
