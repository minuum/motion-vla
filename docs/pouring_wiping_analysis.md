# Pouring & Wiping Task 심층 조사

> **작성일**: 2026-01-02  
> **목적**: Pouring과 Wiping 태스크의 기존 연구, 구현 방법, 어려움, 평가 메트릭 분석

---

## 1. Pouring Task 상세 분석

### 1.1 기존 연구 사례

#### 주요 연구 프로젝트
| 연구/시스템 | 접근법 | 핵심 기술 |
|:---|:---|:---|
| **PourNet** (RL-based) | Deep RL + NMPC | Fluid dynamics 학습 없이 end-to-end |
| **PourIt!** | Visual Closed-loop | 실시간 비전 피드백으로 pouring 제어 |
| **UW Liquid Pouring Dataset** | Perception + Reasoning | Pixel-level liquid label 사용 |
| **VLA Pouring** (GPT-4V) | Vision-Language | Force/Torque → Vision으로 점도 추론 |

#### 핵심 발견
- ✅ **VLA 적용 사례 있음**: 언어로 "pour milk into cup" 같은 지시 가능
- ✅ **Learning-based 성공**: RL, Self-supervised learning 모두 효과적
- ⚠️ **Velocity Profile 중요**: Slosh control에 속도 프로파일 최적화 필수
- ⚠️ **Sim2Real Gap 큼**: Fluid simulation이 어려움

---

### 1.2 구현 방법론

#### A. Perception (액체 인식)
```python
# 비전 기반 액체 인식 방법
methods = {
    "Volume Estimation": "RGB-D로 액체 부피 추정",
    "Surface Tracking": "액체 표면 검출 및 추적",
    "Level Detection": "컵 채움 정도 측정 (±mm 정확도)",
    "Boundary Detection": "액체 경계 및 형상 변화 감지"
}
```

**Dobot E6 적용**:
- ✅ Wrist camera로 컵 내부 관찰 가능
- ✅ Overhead camera로 전체 작업 공간 모니터링
- ❌ Force/Torque 센서 없음 → Vision으로 대체 필수

---

#### B. Control Strategy

**1. Velocity Profile 기반 제어**
```python
# Slosh 최소화를 위한 속도 프로파일
def optimal_pouring_profile(adverb):
    if adverb == "slowly":
        return {
            "tilt_speed": 5°/s,         # 천천히 기울이기
            "max_tilt": 45°,            # 최대 각도 제한
            "deceleration": smooth      # 부드러운 감속
        }
    elif adverb == "quickly":
        return {
            "tilt_speed": 20°/s,
            "max_tilt": 60°,
            "deceleration": abrupt
        }
```

**2. Visual Feedback Control**
- Real-time: 50Hz 비전 시스템으로 액체 Level 추적
- Closed-loop: Level 목표치 도달 시 자동 중지
- Adaptive: 액체 흐름 속도에 따라 기울기 동적 조절

---

### 1.3 Isaac Sim Fluid Simulation

#### 가능한 방법
| 방법 | 장점 | 단점 | Dobot E6 적용성 |
|:---|:---|:---|:---:|
| **Particle System** | 물리 기반, 시각적 리얼 | 느림, 파라미터 튜닝 어려움 | ✅ 가능 |
| **Simplified Fluid** | 빠름, 안정적 | 정확도 낮음 | ✅ 프로토타입용 |
| **Cylinder Animation** | 매우 빠름 | 물리 없음 (높이만 변경) | ⚠️ 학습용 부적합 |

**Isaac Sim 구체적 방법**:
```python
# Particle System 설정
particle_system = ParticleSystem()
particle_set = ParticleSet(sampler="FluidVolumeSampler")
particle_set.set_properties(
    viscosity=0.001,      # 물
    density=1000,         # kg/m³
    particle_size=0.005,  # 5mm
    color=(0.2, 0.5, 0.8)
)
```

**주의사항**:
- Sim2Real Gap 매우 큼 (액체 물리는 시뮬 어려움)
- Domain Randomization 필수: 점도, 밀도, flow rate 변화
- **대안**: 구슬(beads)로 먼저 검증, 나중에 물

---

### 1.4 평가 메트릭

| 메트릭 | 정의 | 측정 방법 | 목표치 |
|:---|:---|:---|:---:|
| **Success Rate** | 목표량 ±10% 도달 | Vision: 컵 채움 % 계산 | >80% |
| **Spillage** | 쏟은 액체 비율 | Vision: 외부 액체 픽셀 카운트 | <5% |
| **Pouring Time** | 시작~끝 시간 | 타이머 | Adverb 상관관계 |
| **Smoothness** | Jerk 표준편차 | 관절 가속도 로그 | <0.5 |
| **Level Accuracy** | 목표 vs 실제 | Vision: mm 단위 | ±5mm |

**Adverb Verification**:
```python
# "Slowly" vs "Quickly" 검증
assert avg_tilt_speed["slowly"] < 0.3 * avg_tilt_speed["quickly"]
assert spillage["slowly"] < spillage["quickly"]
```

---

### 1.5 Dobot E6 구현 실현 가능성

#### 장점
- ✅ 페이로드 충분 (컵 + 250ml 물 = ~300g)
- ✅ 반복 정밀도 ±0.1mm로 Level 제어 가능
- ✅ Wrist camera 장착으로비전 피드백 가능

#### 단점
- ❌ Force/Torque 센서 없음 → 액체 무게 변화 감지 불가
- ❌ 점도 고려 어려움 (시각만으로 제한적)
- ⚠️ Isaac Sim fluid 정확도 문제

#### 우회 전략
1. **구슬(Beads) 먼저**: Sim2Real Gap 감소, 물리 단순화
2. **Vision-only**: 센서 없이 비전만으로 Level 추정
3. **Pre-calibrated Profiles**: 물 특정 점도에 대해 미리 최적화

**난이도 재평가**: ⭐⭐⭐⭐ → ⭐⭐⭐⭐⭐ (매우 도전적, 하지만 가능)

---

## 2. Wiping Task 상세 분석

### 2.1 기존 연구 사례

#### 주요 연구 프로젝트
| 연구 | 접근법 | 핵심 기술 |
|:---|:---|:---|
| **Google Research** | Vision + RL | Visual observations → Force control |
| **Adaptive Wiping (RL)** | Deep RL | 다양한 표면 곡률/마찰 적응 |
| **Imitation Learning Wiping** | IL + FT Sensor | 사람 시연 → Force profile 학습 |
| **Deep Predictive Learning** | Teleoperation | 미래 이미지 예측 + Impedance |

#### 핵심 발견
- ✅ **Contact-rich 대표 태스크**: 많은 연구가 집중
- ✅ **Force Control 필수**: Impedance, Admittance control 필요
- ✅ **Learning 효과적**: RL/IL 모두 성공 사례 多
- ⚠️ **F/T Sensor 중요**: 대부분 Force-Torque 센서 사용

---

### 2.2 구현 방법론

#### A. Force Control Strategies

**1. Impedance Control**
```python
# End-effector를 mass-spring-damper로 모델링
class ImpedanceController:
    def __init__(self, stiffness, damping):
        self.K = stiffness   # N/m
        self.D = damping     # Ns/m
    
    def compute_force(self, pos_error, vel_error):
        # F = K * Δx + D * Δv
        return self.K * pos_error + self.D * vel_error
```

**2. Variable Impedance** (Adverb 적용!)
```python
adverb_impedance = {
    "gently": {"K": 50, "D": 10},   # 낮은 강성 = 부드럽게
    "firmly": {"K": 200, "D": 50},  # 높은 강성 = 강하게
}
```

**Dobot E6 문제**:
- ❌ F/T Sensor 없음 → Direct force measurement 불가
- ✅ **Position-based Implicit Force**: 위치로 압력 간접 제어

**우회 방법**:
```python
# Position-based force control (센서 없이)
def implicit_force_control(target_surface_height, compliance):
    # 테이블 표면보다 약간 아래로 목표 설정
    target_z = surface_height - compliance  # compliance = 2mm
    # 로봇이 테이블과 접촉 → 자연스럽게 압력 발생
    robot.move_to(x, y, target_z)
```

---

#### B. Perception

**1. Dirt/Spill Detection**
```python
# Vision-based 먼지 감지
def detect_dirt(image):
    # Thresholding으로 먼지/오염 픽셀 분리
    dirt_mask = (image < dirty_threshold)
    dirt_pixels = count(dirt_mask)
    clean_percentage = 1 - (dirt_pixels / total_pixels)
    return clean_percentage
```

**2. Surface Reconstruction**
- RGB-D로 테이블 표면 3D 재구성
- 높이 변화 감지 → Adaptive trajectory

---

### 2.3 Learning Approaches

#### Reinforcement Learning
- **Reward**: Clean percentage 증가
- **State**: RGB image + Wiper position
- **Action**: Continuous wiping trajectory + Stiffness

#### Imitation Learning
- 사람 Teleoperation 시연 수집
- Position + (Implicit) Force profile 학습
- Real-time adaptation with vision feedback

---

### 2.4 Isaac Sim 구현

#### Wiping 환경 구성
```python
# Isaac Sim wiping task
class WipingTask:
    def __init__(self):
        self.table = create_table(size=(0.8, 0.6))
        self.dirt = scatter_particles(num=100, area=0.3)
        self.wiper = attach_to_robot(shape="rect", size=(0.05, 0.03))
    
    def check_clean(self):
        # Dirt particles와 wiper collision 체크
        cleaned = count_collisions(self.wiper, self.dirt)
        return cleaned / len(self.dirt)
```

**장점**:
- ✅ Contact physics 시뮬 가능 (PhysX)
- ✅ Particle-based dirt representation
- ✅ Sim2Real Gap이 Pouring보다 작음

---

### 2.5 평가 메트릭

| 메트릭 | 정의 | 측정 방법 | 목표치 |
|:---|:---|:---|:---:|
| **Cleaning Rate** | 제거된 먼지 비율 | Vision: Before/After 픽셀 차이 | >90% |
| **Wiping Time** | 작업 완료 시간 | 타이머 | <30s |
| **Coverage** | Wiped 영역 비율 | Trajectory heatmap | >95% |
| **Force Consistency** | 압력 균일성 | Position variance (indirect) | σ < 2mm |
| **Collision Count** | 과도한 충돌 | PhysX collision events | 0 |

---

### 2.6 Dobot E6 구현 실현 가능성

#### 장점
- ✅ **Position-based control 가능**: F/T 없어도 OK
- ✅ **Sim2Real Gap 작음**: Contact physics가 액체보다 안정적
- ✅ **Wiper 제작 쉬움**: 3D 프린팅 또는 스폰지 부착

#### 단점
- ⚠️ Force feedback 부족 → 압력 제어 정밀도 떨어짐
- ⚠️ 표면 높이 변화 감지 어려움 (Depth camera 필요)

#### 구현 전략
1. **Fixed Surface**: 평평한 테이블만 (높이 변화 X)
2. **Vision-based Adaptation**: RGB로 먼지 위치 파악
3. **Compliance via Soft Wiper**: 부드러운 재질로 압력 자연 분산

**난이도 재평가**: ⭐⭐⭐ → ⭐⭐⭐ (Pouring보다 쉬움)

---

## 3. 두 Task 비교 종합

### 3.1 비교 테이블

| 항목 | Pouring | Wiping |
|:---|:---|:---|
| **기존 연구 성숙도** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (더 많음) |
| **VLA 적용 사례** | ✅ 있음 | ✅ 많음 |
| **Sim2Real Gap** | ⭐⭐⭐⭐⭐ (매우 큼) | ⭐⭐⭐ (중간) |
| **Dobot E6 적합성** | ⚠️ (센서 부족) | ✅ (충분) |
| **구현 난이도** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **참신성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Flow-matching 필수도** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 3.2 권장사항

#### Option A: **Wiping 먼저** (안전한 선택)
**이유**:
- ✅ 구현 상대적으로 쉬움
- ✅ Position-based control로 충분
- ✅ Sim2Real Gap 작음
- ✅ 빠른 성과 가능

**리스크**: 참신성이 Pouring보다 낮음

---

#### Option B: **Pouring 도전** (High Risk High Return)
**이유**:
- ✅ 기존 VLA가 거의 안 함 (참신성 최고)
- ✅ Flow-matching의 진가 발휘
- ✅ 성공 시 Top Conference 확실

**리스크**:
- ❌ Isaac Sim fluid simulation 어려움
- ❌ Sim2Real Gap 클 가능성
- ❌ 시간 많이 소요

---

#### Option C: **Hybrid 전략** (추천 ⭐⭐⭐⭐⭐)
1. **Phase 1**: Wiping 먼저 구현 (2주)
   - Flow-matching pipeline 검증
   - Dobot E6 제어 익숙해지기
   - **Deliverable**: Workshop paper

2. **Phase 2**: Pouring 도전 (4주)
   - 구슬로 시작 → 물로 확장
   - Wiping 경험 활용
   - **Deliverable**: Full Conference paper

---

## 4. 다음 단계

### Immediate (이번 주)
1. **Wiping Task 프로토타입**
   - Isaac Sim 환경 구축
   - Dobot E6 + Wiper end-effector 모델
   - Simple dirt detection (particle-based)

2. **Pouring Feasibility Test**
   - Isaac Sim Particle System 테스트
   - 구슬 pouring 시뮬레이션
   - Sim2Real Gap 사전 평가

### Short-term (2주)
1. Wiping Task 완성
2. Pouring은 병행 연구 (낮은 우선순위)

**최종 결정 필요**: Wiping 먼저 vs Pouring 도전 vs Hybrid
