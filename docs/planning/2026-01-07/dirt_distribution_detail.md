# Dirt Distribution 상세 정의

> **목적**: Pattern Category vs Instance의 명확한 구분  
> **날짜**: 2026-01-07

---

## 🎲 Dirt Distribution의 2-Level 구조

### Level 1: Pattern Category (패턴 종류)
**정의**: 분포의 **전략** 또는 **스타일**

```
Pattern Category = 어떤 방식으로 뿌릴 것인가
```

**3가지 Category**:
1. **Random**: 무작위 분포
2. **Grid**: 격자 분포
3. **Cluster**: 집중 분포 (여러 군집)

---

### Level 2: Instance (실제 분포)
**정의**: 매 episode마다 **물리적으로 다른** 실제 코코아 분포

```
Instance = 실제로 뿌려진 고유한 위치 집합
```

**핵심**: 같은 category라도 매번 instance는 다름!

---

## 📊 구체적 예시

### Random Category의 Instance들

```python
# Episode 1: Random pattern, Seed 42
instance_001 = {
    "category": "random",
    "seed": 42,
    "positions": [
        (0.12, 0.34, 0.405),
        (0.56, 0.21, 0.405),
        (0.78, 0.45, 0.405),
        # ... 102 particles
    ],
    "unique_id": "random_102_seed42",
}

# Episode 2: Random pattern, Seed 84 (다른 instance!)
instance_002 = {
    "category": "random",  # 같은 category
    "seed": 84,            # 다른 seed
    "positions": [
        (0.23, 0.15, 0.405),  # 완전히 다른 위치!
        (0.41, 0.52, 0.405),
        (0.67, 0.33, 0.405),
        # ... 97 particles
    ],
    "unique_id": "random_97_seed84",
}
```

**같은 "random" category, 완전히 다른 instance!**

---

### Grid Category의 Instance들

```python
# Episode 3: Grid pattern, 10×10
instance_003 = {
    "category": "grid",
    "grid_size": (10, 10),
    "positions": [
        (0.05, 0.05, 0.405),  # Uniform spacing
        (0.14, 0.05, 0.405),
        (0.23, 0.05, 0.405),
        # ... 100 particles (exact grid)
    ],
    "unique_id": "grid_100_10x10",
}

# Episode 4: Grid pattern, 9×11 (다른 grid size!)
instance_004 = {
    "category": "grid",    # 같은 category
    "grid_size": (9, 11),  # 다른 configuration
    "positions": [
        (0.06, 0.04, 0.405),  # 다른 spacing
        (0.15, 0.04, 0.405),
        # ... 99 particles
    ],
    "unique_id": "grid_99_9x11",
}
```

---

### Cluster Category의 Instance들

```python
# Episode 5: Cluster pattern, 5 clusters
instance_005 = {
    "category": "cluster",
    "num_clusters": 5,
    "cluster_centers": [
        (0.2, 0.2),
        (0.6, 0.3),
        (0.4, 0.5),
        (0.7, 0.1),
        (0.3, 0.4),
    ],
    "positions": [...],  # Around these centers
    "unique_id": "cluster_102_5centers_seed11",
}

# Episode 6: Cluster pattern, 3 clusters (다른 instance!)
instance_006 = {
    "category": "cluster",  # 같은 category
    "num_clusters": 3,      # 다른 cluster 개수
    "cluster_centers": [
        (0.3, 0.3),  # 완전히 다른 위치
        (0.5, 0.2),
        (0.4, 0.4),
    ],
    "positions": [...],
    "unique_id": "cluster_98_3centers_seed22",
}
```

---

## 🔄 Real Robot vs Simulation

### Simulation: Reproducible (재현 가능)

```python
# Sim에서는 seed로 exact 재현 가능
np.random.seed(42)
dirt_sim = spawn_random_dirt()
# → 매번 같은 위치 (debugging용)

# But for training, seed를 매번 바꿈!
for episode_id in range(342):
    seed = 1000 + episode_id  # Unique seed
    dirt = spawn_random_dirt(seed=seed)
```

---

### Real Robot: Non-Reproducible (재현 불가능)

```python
# Real robot은 물리적으로 뿌림
def sprinkle_cocoa_powder():
    # 사람이 손으로 뿌림
    # → 매번 완전히 다름!
    # → seed로 재현 불가능
    
    # 결과: Vision-based measurement만 가능
    dirt_pixels = count_dirt_pixels(initial_image)
    return {
        "category": "random",  # 대략적 분류
        "measured_pixels": dirt_pixels,
        "positions": None,  # 알 수 없음 (vision만)
    }
```

**Real robot**: Instance는 unique하지만 정확한 위치는 모름!

---

## 📐 Episode마다의 Variation

### 같은 Task 내에서

```python
task_T1 = "Wipe the table"

# 100 episodes in T1
episodes_T1 = [
    # Random pattern variations
    {"category": "random", "instance": "seed42"},
    {"category": "random", "instance": "seed84"},
    {"category": "random", "instance": "seed126"},
    # ... 33 more random
    
    # Grid pattern variations
    {"category": "grid", "instance": "10x10"},
    {"category": "grid", "instance": "9x11"},
    {"category": "grid", "instance": "11x9"},
    # ... 30 more grid
    
    # Cluster pattern variations
    {"category": "cluster", "instance": "5centers_seed11"},
    {"category": "cluster", "instance": "3centers_seed22"},
    {"category": "cluster", "instance": "7centers_seed33"},
    # ... 31 more cluster
]

# Total: 100 unique instances
# → 100% different dirt distributions!
```

---

## 🎯 Category vs Instance 비교

| Aspect | Pattern Category | Instance |
|:---|:---|:---|
| **정의** | 분포 스타일 | 실제 위치 집합 |
| **개수** | 3개 (random/grid/cluster) | 380개 (episode마다) |
| **재현성** | 개념적 | Sim: 가능, Real: 불가 |
| **용도** | Domain randomization 전략 | 실제 데이터 |

---

## 💡 왜 Category가 필요한가?

### Without Category (category 없이)

```python
# 380 episodes, all completely random
for i in range(380):
    dirt = spawn_random_dirt(seed=i)
    
# Problem: 너무 random → grid 패턴 학습 못 함
```

### With Category (category 사용)

```python
# 3 categories × ~126 instances each
categories = ["random", "grid", "cluster"]

for i in range(380):
    category = categories[i % 3]  # Balanced
    dirt = spawn_dirt(category, seed=i)
    
# Benefit: 모든 패턴 고르게 학습!
```

---

## 📊 분포 전략 (Distribution Strategy)

### Balanced Category Distribution

```python
# 380 episodes × 균등 분배
category_distribution = {
    "random": 127 episodes,   # 33%
    "grid": 127 episodes,     # 33%
    "cluster": 126 episodes,  # 33%
}

# Task별로도 균등
task_T1_distribution = {
    "random": 33 episodes,
    "grid": 33 episodes,
    "cluster": 34 episodes,
}
```

---

## 🔬 Instance 생성 예시

### Random Instance Generation

```python
def generate_random_instance(seed, count=100):
    np.random.seed(seed)
    
    # Table size
    w, h = 0.8, 0.6
    
    # Random positions within table
    positions = []
    for _ in range(count):
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        z = 0.405  # Just above table
        positions.append((x, y, z))
    
    return {
        "category": "random",
        "instance_id": f"random_{count}_seed{seed}",
        "positions": positions,
    }

# Usage
instance_1 = generate_random_instance(seed=42, count=102)
instance_2 = generate_random_instance(seed=84, count=97)
# → Completely different positions!
```

---

### Grid Instance Generation

```python
def generate_grid_instance(grid_size=(10, 10)):
    rows, cols = grid_size
    w, h = 0.8, 0.6
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = (i + 0.5) * (w / rows)
            y = (j + 0.5) * (h / cols)
            z = 0.405
            positions.append((x, y, z))
    
    return {
        "category": "grid",
        "instance_id": f"grid_{rows}x{cols}",
        "positions": positions,
    }

# Usage
instance_3 = generate_grid_instance((10, 10))  # 100 particles
instance_4 = generate_grid_instance((9, 11))   # 99 particles
# → Different grid layouts!
```

---

### Cluster Instance Generation

```python
def generate_cluster_instance(num_clusters=5, seed=42):
    np.random.seed(seed)
    w, h = 0.8, 0.6
    
    # Random cluster centers
    centers = []
    for _ in range(num_clusters):
        cx = np.random.uniform(0.1*w, 0.9*w)
        cy = np.random.uniform(0.1*h, 0.9*h)
        centers.append((cx, cy))
    
    # Distribute particles around centers
    positions = []
    particles_per_cluster = 100 // num_clusters
    
    for cx, cy in centers:
        for _ in range(particles_per_cluster):
            # Gaussian around center
            x = np.random.normal(cx, 0.05)  # σ=5cm
            y = np.random.normal(cy, 0.05)
            z = 0.405
            positions.append((x, y, z))
    
    return {
        "category": "cluster",
        "instance_id": f"cluster_{num_clusters}centers_seed{seed}",
        "positions": positions,
        "centers": centers,
    }

# Usage
instance_5 = generate_cluster_instance(5, seed=11)
instance_6 = generate_cluster_instance(3, seed=22)
# → Different cluster configurations!
```

---

## ✅ 최종 정리

### Dirt Distribution의 완전한 정의

```
Dirt Distribution = Pattern Category + Unique Instance

1. Pattern Category (3 types):
   - Random: 무작위 분포
   - Grid: 격자 분포
   - Cluster: 집중 분포

2. Instance (380 unique):
   - 매 episode마다 다른 실제 위치
   - Sim: Seed로 재현 가능
   - Real: 물리적으로 매번 다름 (재현 불가)
```

### Episode의 Uniqueness

```python
episode = {
    "dirt_category": "random",  # High-level pattern
    "dirt_instance": {          # Low-level actual positions
        "seed": 42,
        "positions": [...],     # 102 unique positions
    },
    "trajectory": [...],
    "result": 0.91,
}

# 380 episodes = 380 unique instances!
```

---

**핵심**: Category는 전략, Instance는 실제!
