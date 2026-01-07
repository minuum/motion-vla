# Dataset Terminology & Structure (용어 정의 및 구조)

> **목적**: Episode, Task, Sample 등 모든 개념의 명확한 정의  
> **날짜**: 2026-01-07

---

## 📚 핵심 용어 정의

### 1. Dataset (데이터셋)
**정의**: 전체 학습 데이터의 집합

```
Dataset = 모든 Task들의 모음
```

**구성**:
- 380 episodes (전체)
- 5 tasks로 구성

---

### 2. Task (태스크)
**정의**: 동일한 language instruction을 가진 episode들의 그룹

```
Task = Instruction이 같은 Episode들의 집합
```

**예시**:
```python
task_T1 = {
    "instruction": "Wipe the table",
    "adverb": "normal",
    "episodes": [ep_001, ep_002, ..., ep_100],  # 100개
}

task_T2 = {
    "instruction": "Wipe gently",
    "adverb": "gently",
    "episodes": [ep_101, ep_102, ..., ep_180],  # 80개
}
```

**핵심**: Task = **같은 지시문**, 다른 초기 조건

---

### 3. Episode (에피소드)
**정의**: 하나의 완전한 wiping 시도 (시작부터 끝까지)

```
Episode = Initial State + Execution + Final Result
```

**구성 요소**:
```python
episode = {
    # Meta info
    "task_id": "T1",
    "episode_id": "ep_001",
    
    # Initial state (매번 다름!)
    "dirt_distribution": {
        "pattern": "random",        # random/grid/cluster
        "count": 102,                # 95-105 범위
        "positions": [...],          # Unique!
        "initial_pixels": 11800,
    },
    
    # Execution (250 timesteps @ 50Hz)
    "trajectory": {
        "obs": (250, ...),           # 관찰 sequence
        "actions": (250, 7),         # 행동 sequence
        "wiper_path": (250, 3),      # Wiper 궤적
    },
    
    # Final result (episode 종료 시 측정)
    "result": {
        "final_pixels": 950,
        "cleaning_rate": 0.919,      # 91.9%
        "coverage": 0.88,
        "success": True,
    }
}
```

**핵심**: Episode = **하나의 시도**, 고유한 초기 조건 + 실행 + 결과

---

### 4. Dirt Distribution (먼지 분포)
**정의**: Episode의 초기 상태 (코코아 가루가 뿌려진 상태)

```
Dirt Distribution = 먼지의 초기 배치 상태
```

**속성**:
```python
dirt_distribution = {
    "pattern": "random",      # 패턴 종류
    "count": 102,             # 파티클 개수
    "positions": [            # 각 파티클 위치
        (0.12, 0.34, 0.405),
        (0.56, 0.21, 0.405),
        # ... 102개
    ],
}
```

**중요**: 
- ✅ Episode마다 **완전히 다름**
- ✅ Task가 같아도 dirt는 다름!

---

### 5. Trajectory (궤적)
**정의**: Episode 실행 중 로봇이 움직인 경로

```
Trajectory = 시간에 따른 Robot State의 sequence
```

**구성**:
```python
trajectory = {
    "timesteps": 250,            # 5초 @ 50Hz
    
    # State sequence
    "joint_positions": (250, 6), # 관절 위치
    "wiper_position": (250, 3),  # Wiper TCP 위치
    "velocities": (250, 6),      # 속도
    
    # Action sequence
    "actions": (250, 7),         # 다음 state를 위한 명령
}
```

**핵심**: Trajectory = **시간순 경로**, episode 실행의 흔적

---

### 6. Sample (샘플)
**정의**: Episode에서 Action Chunking으로 추출한 학습 데이터 포인트

```
Sample = (Current Observation, Future 50 Actions)
```

**구성**:
```python
sample = {
    # Input
    "obs_t": {
        "rgb": (480, 640, 3),    # 현재 시점 이미지
        "robot_state": (6,),     # 현재 joint positions
        "language": "Wipe gently",
    },
    
    # Output (학습 목표)
    "actions_future": (50, 7),   # t부터 t+49까지 50 steps
}
```

**추출 방법**:
```python
# 1 Episode (250 timesteps) → 20 Samples
for t in range(0, 200, 10):  # stride=10
    sample_i = {
        "obs_t": episode.obs[t],
        "actions_future": episode.actions[t:t+50],
    }
```

**핵심**: Sample = **학습용 단위**, episode에서 추출

---

## 🏗️ 계층 구조 (Hierarchical Structure)

```
Dataset (전체 데이터셋)
│
├─ Task T1 (100 episodes)
│  ├─ Episode 001
│  │  ├─ Dirt Distribution 001 (unique)
│  │  ├─ Trajectory 001
│  │  └─ Samples [s_001, s_002, ..., s_020]
│  │
│  ├─ Episode 002
│  │  ├─ Dirt Distribution 002 (different!)
│  │  ├─ Trajectory 002
│  │  └─ Samples [s_021, s_022, ..., s_040]
│  │
│  └─ Episode 100
│     ├─ Dirt Distribution 100
│     ├─ Trajectory 100
│     └─ Samples [...]
│
├─ Task T2 (80 episodes)
│  ├─ Episode 101
│  │  ├─ Dirt Distribution 101
│  │  ├─ Trajectory 101
│  │  └─ Samples [...]
│  └─ ...
│
└─ Task T3-T5 (...)
```

---

## 🔍 포함 관계 (Containment Relationship)

```
Dataset ⊃ Task ⊃ Episode ⊃ Sample
```

| 개념 | 포함 관계 | 개수 |
|:---|:---|:---:|
| **Dataset** | 모든 것 포함 | 1 |
| **Task** | Episodes 포함 | 5 |
| **Episode** | Samples 포함 | 380 |
| **Sample** | 최소 단위 | 7,600 |

---

## 📊 "같음 vs 다름" 비교표

### Task 간 (Between Tasks)

| 속성 | Task 간 비교 | 예시 |
|:---|:---:|:---|
| **Instruction** | ❌ 다름 | T1: "Wipe" vs T2: "Wipe gently" |
| **Adverb** | ❌ 다름 | Normal vs Gently |
| **Target velocity** | ❌ 다름 | 0.15 vs 0.05 m/s |

---

### Episode 간 (Within Same Task)

| 속성 | 같은 Task 내 Episode 간 | 예시 |
|:---|:---:|:---|
| **Instruction** | ✅ 같음 | 모두 "Wipe gently" |
| **Adverb** | ✅ 같음 | 모두 "gently" |
| **Dirt distribution** | ❌ **완전히 다름** | Random 102 vs Grid 97 |
| **Dirt positions** | ❌ **완전히 다름** | (0.1, 0.2) vs (0.5, 0.4) |
| **Trajectory** | ⚠️ 유사하지만 다름 | 전략 같지만 미세 조정 |
| **Cleaning rate** | ❌ 다름 | 91% vs 95% |

---

### Sample 간 (Within Same Episode)

| 속성 | 같은 Episode 내 Sample 간 | 예시 |
|:---|:---:|:---|
| **Episode** | ✅ 같음 | 모두 ep_001에서 추출 |
| **Dirt distribution** | ✅ 같음 | 동일한 초기 분포 |
| **Timestep** | ❌ 다름 | t=0 vs t=10 vs t=20 |
| **Observation** | ❌ 다름 | 다른 시점의 image |
| **Future actions** | ❌ 다름 | 다른 50-step window |

---

## 🎯 핵심 질문에 대한 답

### Q1: "같은 태스크 내에서의 dirt distribution이 다른 경우만 episode야?"

**답**: 아니요! Episode는 **더 넓은 개념**입니다.

```python
# Same task, different episodes
episode_A = {
    "task": "T1",
    "dirt_distribution": "random_102",  # 다름
    "trajectory": "zigzag_left_start",  # 다름
    "cleaning_rate": 0.91,              # 다름
}

episode_B = {
    "task": "T1",                       # 같음!
    "dirt_distribution": "cluster_97",  # 다름!
    "trajectory": "zigzag_center_start",# 다름!
    "cleaning_rate": 0.95,              # 다름!
}
```

**Episode를 구분하는 요소**:
1. ✅ Dirt distribution (다름)
2. ✅ Trajectory (다름)
3. ✅ 초기 robot pose (다를 수 있음)
4. ✅ 결과 (cleaning rate 등 다름)

**Episode = 하나의 완전한 시도**

---

### Q2: "같은 태스크 내에서의 dirt distribution이랑 trajectory가 다른 경우만 episode야?"

**답**: 맞습니다! 더 정확히는:

```
Episode = Unique (Dirt Distribution + Initial Conditions + Execution)
```

**세부 설명**:

```python
# Task T1: "Wipe the table" - 100 episodes

# Episode 1
ep_001 = {
    "dirt": {"pattern": "random", "positions": [...]},  # Unique set 1
    "initial_pose": "left_corner",
    "trajectory": execute_wiping(),  # 결과 궤적 1
}

# Episode 2 (다른 초기 조건!)
ep_002 = {
    "dirt": {"pattern": "cluster", "positions": [...]}, # Unique set 2
    "initial_pose": "center",
    "trajectory": execute_wiping(),  # 결과 궤적 2
}

# → 다른 Episode!
```

**Episode를 다르게 만드는 것**:
- Dirt distribution (항상 다름)
- Initial robot configuration (다를 수 있음)
- 실행 중 noise/variation
- 결과 (cleaning rate 등)

---

## 📐 구체적 예시

### Dataset 구조 예시

```
Wiping VLA Dataset
├─ Task T1: "Wipe the table" (100 episodes)
│  │
│  ├─ Episode 001
│  │  ├─ Dirt: random_102_seed42
│  │  ├─ Trajectory: 250 timesteps
│  │  ├─ Result: 91% cleaning
│  │  └─ 20 Samples
│  │
│  ├─ Episode 002
│  │  ├─ Dirt: cluster_97_seed84  (다른 분포!)
│  │  ├─ Trajectory: 250 timesteps (다른 경로!)
│  │  ├─ Result: 95% cleaning      (다른 결과!)
│  │  └─ 20 Samples
│  │
│  └─ Episode 100
│     └─ ...
│
├─ Task T2: "Wipe gently" (80 episodes)
│  ├─ Episode 101
│  │  ├─ Dirt: random_99_seed11   (T1과 완전히 다름)
│  │  ├─ Trajectory: slow motion  (gently → 느림)
│  │  └─ 20 Samples
│  └─ ...
│
└─ Task T3-T5
   └─ ...
```

---

## 🔢 카운팅 (Counting)

### 전체 구조

| Level | Count | Description |
|:---|:---:|:---|
| **Dataset** | 1 | 전체 |
| **Tasks** | 5 | T1, T2, T3, T4, T5 |
| **Episodes** | 380 | 5 tasks에 분산 |
| **Samples** | 7,600 | 380 × 20 avg |

### Task별 분포

| Task | Instruction | Episodes | Samples |
|:---|:---|:---:|:---:|
| T1 | "Wipe the table" | 100 | 2,000 |
| T2 | "Wipe gently" | 80 | 1,600 |
| T3 | "Wipe firmly" | 80 | 1,600 |
| T4 | "Wipe quickly" | 60 | 900 |
| T5 | "Wipe thoroughly" | 60 | 1,500 |

---

## 🎨 시각적 비교

### Episode의 고유성

```
Episode 001: 
  Dirt: ●●●   ●    ●●  (random)
  Path: ────────────→
  Result: 91%

Episode 002:
  Dirt: ●●●●●●       (cluster)
  Path:    ↓↓↓↓↓↓
  Result: 95%

Episode 003:
  Dirt: ● ● ● ● ● ●  (grid)
  Path: ～～～～～～～ (zigzag)
  Result: 89%
```

**모두 같은 Task (T1)이지만, 완전히 다른 Episode!**

---

## ✅ 최종 정리

### 명확한 정의

```
1. Dataset
   └─ 전체 데이터 집합

2. Task
   └─ 같은 instruction을 가진 episode 그룹
   └─ 예: "Wipe gently" (80 episodes)

3. Episode
   └─ 하나의 완전한 wiping 시도
   └─ Unique: Dirt distribution + Execution + Result
   └─ 예: ep_001 (random dirt, 91% cleaning)

4. Dirt Distribution
   └─ Episode의 초기 상태
   └─ 매 episode마다 다름!

5. Trajectory
   └─ Episode 실행 중 로봇 경로
   └─ 매 episode마다 다름!

6. Sample
   └─ Episode에서 추출한 학습 단위
   └─ 1 episode → 20 samples
```

### 포함 관계

```
Dataset (1)
  ⊃ Task (5)
    ⊃ Episode (380)
      ⊃ Sample (7,600)
```

### 핵심

**Episode = Task + Unique Initial State + Execution + Result**

- Task는 같아도, Dirt와 Trajectory는 매번 다름!
- 380 episodes = 380개의 서로 다른 wiping 시도
- Domain Randomization으로 Generalization 확보!

---

---

이제 명확한가요? 😊

---

## ⚠️ 중요: Episode vs Chunk 명확화

### ❌ 흔한 오해

```
Chunk = Episode? NO!
```

### ✅ 정확한 관계

```
Episode (전체 시도):
├─ 250 timesteps (5초 @ 50Hz)
└─ 하나의 완전한 wiping 동작

Chunk (예측 윈도우):
├─ 50 timesteps (1초 @ 50Hz)
└─ Episode 내의 일부 actions
```

---

## 🔍 Episode 내 Chunk 추출

### Episode = 전체 trajectory

```python
episode = {
    "timesteps": 250,
    "actions": (250, 7),  # 전체 250 steps
    "duration": 5.0,  # 초
}
```

### Chunk = Sliding Window

```python
# Episode에서 20개 chunks 추출
chunks = []
for t in range(0, 200, 10):  # stride=10
    chunk = episode.actions[t:t+50]  # 50-step window
    chunks.append(chunk)

# Result: 20 overlapping chunks from 1 episode
```

---

## 📊 시각적 비교

### Episode (250 steps)

```
Episode:
|================================================|
0         50        100       150       200    250
                 전체 5초
```

### Chunks (50-step windows)

```
Chunk 1:  [0────50)
          |=====|

Chunk 2:     [10───60)
             |=====|

Chunk 3:        [20───70)
                |=====|

...

Chunk 20:                          [200──250)
                                   |=====|
```

**Overlap**: 각 chunk는 40 steps씩 겹침!

---

## 🎯 용도의 차이

| 개념 | 용도 | 크기 |
|:---|:---|:---:|
| **Episode** | 데이터 수집 단위 | 250 steps |
| **Chunk** | 모델 예측 단위 | 50 steps |
| **Sample** | 학습 데이터 단위 | (obs, chunk) 쌍 |

---

## 🔢 카운팅

```
1 Episode (250 steps)
  → 20 Chunks (overlapping 50-step windows)
    → 20 Samples (for training)

380 Episodes
  → 7,600 Chunks
    → 7,600 Samples
```

---

## 💡 핵심

**Episode ≠ Chunk!**

- Episode는 **전체 시도** (250 steps)
- Chunk는 **예측 단위** (50 steps)
- **1 Episode 안에 20개 chunks 포함됨**
- Overlapping window로 더 많은 학습 데이터 확보!

---

**최종 정리**: Episode는 큰 그릇, Chunk는 그 안의 작은 조각들!

