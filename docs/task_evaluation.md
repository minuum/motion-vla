# VLA Task Suite 평가 및 조합 분석

> **작성일**: 2026-01-02  
> **목적**: 3가지 VLA 태스크의 이론적 타당성 평가 및 필요 데이터 가짓수 산출

---

## 1. Task 평가 Framework

### 평가 기준
1. **학술적 참신성**: 기존 VLA 연구와의 차별점
2. **구현 난이도**: 데이터 수집, 학습, 평가의 어려움
3. **성공 가능성**: Sim2Real transfer 위험도 고려
4. **연구 기여도**: 논문 출판 가능성

### 난이도 척도
- ⭐ (매우 쉬움) ~ ⭐⭐⭐⭐⭐ (매우 어려움)

---

## 2. 카테고리 기반 Task 후보 선정

### Category 1: 조작 스킬 타입 (Manipulation Skills)

| 스킬 | 설명 | Dobot E6 적합성 | 필요 장비 | 선정 여부 |
|:---|:---|:---:|:---|:---:|
| **Pick** | 물체 집기 | ✅ 매우 적합 | Gripper | ⭐⭐⭐⭐⭐ 채택 |
| **Place** | 물체 놓기 | ✅ 매우 적합 | - | ⭐⭐⭐⭐⭐ 채택 |
| **Push** | 물체 밀기 | ✅ 적합 | - | ⭐⭐⭐⭐ 채택 |
| **Pour** | 액체 따르기 | ⚠️ 제한적 | 컵, 액체 | ⭐⭐ 보류 (센서 부족) |
| **Flip** | 뒤집기 | ⚠️ 제한적 | 특수 그리퍼 | ⭐ 보류 (정밀도 부족) |
| **Stack** | 쌓기 | ✅ 적합 | 블록 | ⭐⭐⭐⭐ 후보 |
| **Slide** | 밀어서 이동 | ✅ 적합 | 평평한 표면 | ⭐⭐⭐ 후보 |
| **Insert** | 끼우기 | ⚠️ 도전적 | 구멍이 있는 물체 | ⭐⭐ 보류 (정밀도 요구) |

**선정 이유**:
- **Pick, Place, Push**: Dobot E6의 ±0.1mm 반복 정밀도로 충분히 수행 가능
- **Stack, Slide**: 추가 태스크로 확장 가능, 새로운 스킬 학습 검증
- **Pour, Insert**: Force/Torque 센서 없어 실패 위험 높음

---

### Category 2: 언어 제어 차원 (Language Control Dimensions)

| 차원 | 예시 | VLA 기여도 | 구현 난이도 | 선정 여부 |
|:---|:---|:---:|:---:|:---:|
| **What** (물체) | "Pick the **red** cup" | ⭐⭐ | ⭐ | ✅ 기본 |
| **Where** (위치) | "Place it **on the left**" | ⭐⭐ | ⭐ | ✅ 기본 |
| **How** (스타일) | "Move **carefully**" | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ 핵심 |
| **When** (타이밍) | "Stop **when** it touches" | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ 보류 |
| **Correction** | "Move it **more to the right**" | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ 핵심 |
| **Sequence** | "**First** pick, **then** place" | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ 후보 |

**선정 이유**:
- **How (Adverb)**: 기존 VLA가 다루지 않은 영역, 참신성 최고
- **Correction**: Real-time HRI에 필수, 학술적 가치 매우 높음
- **When (타이밍)**: Contact sensing 없어 구현 어려움
- **Sequence**: 이미 기존 VLA가 다룸, 차별성 낮음

---

### Category 3: 물체 속성 (Object Properties)

| 속성 | 변형 가능성 | Sim2Real 난이도 | 선정 여부 |
|:---|:---:|:---:|:---:|
| **Rigid (강체)** | Cube, Sphere, Cylinder | ⭐ | ⭐⭐⭐⭐⭐ 필수 |
| **Color** | Red, Blue, Green, Yellow | ⭐ | ⭐⭐⭐⭐⭐ 필수 |
| **Size** | Small (5cm), Medium (10cm) | ⭐⭐ | ⭐⭐⭐⭐ 채택 |
| **Weight** | Light (<300g), Heavy (300-750g) | ⭐⭐⭐ | ⭐⭐⭐ 후보 |
| **Deformable** | 천, 스폰지 | ⭐⭐⭐⭐⭐ | ⭐ 제외 (Sim 어려움) |
| **Articulated** | 서랍, 문 | ⭐⭐⭐⭐ | ⭐⭐ 제외 (복잡도) |

**Payload 제약 고려**:
- Dobot E6: 최대 0.75kg → 무거운 물체는 신중히 선택
- **권장**: 300g 이하 물체 위주 (컵, 블록, 작은 공구)

---

### Category 4: 난이도 레벨 (Difficulty Levels)

| Level | Task 예시 | 성공률 예상 | 학습 Episodes | 선정 |
|:---:|:---|:---:|:---:|:---:|
| **L1 (Easy)** | Reach fixed target | 95%+ | 50 | ✅ Warmup |
| **L2 (Medium)** | Pick & Place static object | 80-90% | 200 | ⭐⭐⭐⭐⭐ 핵심 |
| **L3 (Hard)** | Pick with Adverb control | 70-85% | 300 | ⭐⭐⭐⭐⭐ 핵심 |
| **L4 (Very Hard)** | Real-time correction | 60-75% | 100 (사람) | ⭐⭐⭐⭐ 후보 |
| **L5 (Extreme)** | Dynamic object catching | <50% | 1000+ | ❌ 제외 |

**구현 전략**:
1. L1 → L2 → L3 순차 구현 (Curriculum Learning)
2. L4는 L3 성공 후 추가

---

### Category 5: 평가 메트릭 카테고리 (Evaluation Metrics)

| 메트릭 | 측정 방법 | 적용 Task | 우선순위 |
|:---|:---|:---|:---:|
| **Success Rate** | Goal region 도달 여부 | All tasks | ⭐⭐⭐⭐⭐ |
| **Execution Time** | 시작~끝 시간 | Adverb task | ⭐⭐⭐⭐ |
| **Collision Count** | 충돌 감지 횟수 | All tasks | ⭐⭐⭐⭐ |
| **Trajectory Smoothness** | Jerk 표준편차 | Adverb task | ⭐⭐⭐ |
| **Correction Latency** | 피드백~반응 시간 | Correction task | ⭐⭐⭐⭐⭐ |
| **Generalization** | 새 물체 성공률 | All tasks | ⭐⭐⭐⭐ |

---

### 최종 선정 Task Suite (우선순위 기반)

| 순위 | Task Name | 스킬 조합 | 언어 차원 | 난이도 | 비고 |
|:---:|:---|:---|:---|:---:|:---|
| **1** | Pick & Place + Adverb | Pick, Place | What, Where, How | L3 | 핵심 태스크 |
| **2** | Push with Adverb | Push | What, Where, How | L2 | 확장 스킬 |
| **3** | Real-time Correction | Pick/Place | Correction | L4 | 참신성 최고 |
| **4** | Stack Blocks | Pick, Place, Stack | What, Where | L3 | 추가 검증용 |

---

## 3. Task 1: Pick-and-Place with Language Variations

### 학술적 근거
- **기존 연구**: RT-2, OpenVLA, Octo 등 대부분 Pick & Place 포함
- **우리의 차별점**:
  1. **Adverb conditioning** (기존 연구는 "What"만 다룸, 우리는 "How"도)
  2. **Small-scale robot** (Dobot E6)에서의 VLA 검증
  3. **Sim2Real with limited real data** (Real 50 demos만으로 transfer)

### 실제 데이터 조합 가짓수

#### Isaac Sim에서 생성 가능한 조합
| 변수 | 가짓수 | 설명 |
|:---|:---:|:---|
| **물체 종류** | 4 | Cube, Cylinder, Sphere, Box |
| **물체 색상** | 4 | Red, Blue, Green, Yellow |
| **초기 위치** | 5x5 = 25 | 테이블 그리드 |
| **목표 위치** | 3 | Left/Center/Right platform |
| **Adverb** | 3 | carefully, quickly, normal |

**총 조합 수**: 4 x 4 x 25 x 3 x 3 = **3,600 가지**

#### 실제 수집 계획
- **Sim**: 200 episodes (약 5.5% 샘플링)
- **Real**: 50 episodes (대표적인 조합만)

### 구현 난이도 평가
| 항목 | 난이도 | 근거 |
|:---|:---:|:---|
| Isaac Sim 환경 구축 | ⭐⭐ | Dobot E6 URDF 제공, 기존 예제 많음 |
| 데이터 수집 (Sim) | ⭐ | Scripted policy로 자동화 |
| 데이터 수집 (Real) | ⭐⭐⭐ | Drag-to-teach 반복 작업 필요 |
| 학습 안정성 | ⭐⭐ | Flow-matching은 비교적 안정적 |
| Sim2Real Gap | ⭐⭐⭐⭐ | 물리적 속성 차이, 그리퍼 불확실성 |

**총 난이도**: ⭐⭐⭐ (중간)

### 연구 기여도
- **Novelty**: ⭐⭐⭐ (Adverb control이 참신하지만 task는 기본적)
- **Impact**: ⭐⭐⭐⭐ (Small robot VLA는 실용적 가치 높음)
- **논문 가능성**: ⭐⭐⭐⭐ (Conference급 가능, Workshop 확실)

---

## 3. Task 2: Real-time Language Correction

### 학술적 근거
- **기존 연구**: 
  - **ExTraCT** (Frontiers in Robotics): 언어 → 궤적 수정 함수
  - **IRP** (RSS): Residual Policy Learning
- **우리의 차별점**:
  1. **Real-time feedback** (기존은 사전 계획 단계)
  2. **Flow-matching + Residual** 결합 (Novel architecture)
  3. **Human-in-the-loop dataset** (사람이 직접 개입하는 데이터)

### 실제 데이터 조합 가짓수

#### Correction 시나리오 조합
| 변수 | 가짓수 | 설명 |
|:---|:---:|:---|
| **Base Task** | 3 | Reach, Pick, Place |
| **Correction 방향** | 6 | Left, Right, Up, Down, Forward, Back |
| **Correction 시점** | 3 | Early (20%), Mid (50%), Late (80%) |
| **Correction 강도** | 2 | Small (±3cm), Large (±8cm) |

**총 조합 수**: 3 x 6 x 3 x 2 = **108 가지**

#### 실제 수집 계획
- **Sim**: 불가능 (사람의 판단이 필요)
- **Real Human-in-the-loop**: 100 trials
  - 10명 x 10 trials = 100 (조합 중 91%를 커버)

### 구현 난이도 평가
| 항목 | 난이도 | 근거 |
|:---|:---:|:---|
| 데이터 수집 | ⭐⭐⭐⭐⭐ | 사람이 실시간으로 개입해야 함 |
| 언어 피드백 인코딩 | ⭐⭐ | CLIP/BERT로 간단히 처리 |
| Residual Head 학습 | ⭐⭐ | 구조 단순, 데이터만 있으면 OK |
| Real-time 성능 | ⭐⭐⭐⭐ | 50Hz 유지 어려움 (VL Encoder 병목) |
| 평가 메트릭 정량화 | ⭐⭐⭐ | "얼마나 잘 수정했나?" 모호함 |

**총 난이도**: ⭐⭐⭐⭐ (어려움)

### 연구 기여도
- **Novelty**: ⭐⭐⭐⭐⭐ (Real-time correction은 거의 연구 안 됨)
- **Impact**: ⭐⭐⭐⭐ (HRI 관점에서 매우 중요)
- **논문 가능성**: ⭐⭐⭐⭐⭐ (Top-tier Conference 가능, CoRL/ICRA)

**⚠️ 리스크**: 데이터 수집 난이도가 매우 높아 프로젝트 지연 가능

---

## 4. Task 3: Adverb-Conditioned Speed Control

### 학술적 근거
- **기존 연구**:
  - **Language-to-Velocity Mapping** (CMU, arXiv)
  - **Motion Style Transfer** (Dartmouth)
- **우리의 차별점**:
  1. **VLA에 Style Token 통합** (기존은 별도 모듈)
  2. **Auto-labeling pipeline** (속도 → 부사 자동 매핑)
  3. **실제 로봇 검증** (기존은 대부분 Sim만)

### 실제 데이터 조합 가짓수

#### Adverb Style 조합
| 변수 | 가짓수 | 설명 |
|:---|:---:|:---|
| **부사 종류** | 4 | carefully, quickly, steadily, normal |
| **Base Task** | 5 | Reach, Pick, Place, Push, Move |
| **물체/목표** | 8 | Task 1과 동일한 물체 조합 |

**총 조합 수**: 4 x 5 x 8 = **160 가지**

#### 실제 수집 계획
- **Sim**: 200 episodes (125% 오버샘플링으로 모든 조합 커버)
- **Real**: 20 episodes (부사당 5개, 대표 태스크만)

### Auto-labeling 전략

#### 속도 기반 부사 분류
```python
def classify_adverb(trajectory):
    avg_velocity = compute_avg_velocity(trajectory)
    jerk_std = compute_jerk_std(trajectory)
    
    if avg_velocity < 0.15 and jerk_std < 0.05:
        return "carefully"
    elif avg_velocity > 0.35:
        return "quickly"
    elif jerk_std < 0.03:
        return "steadily"
    else:
        return "normal"
```

**Auto-labeling 정확도 예상**: 85-90% (수동 검증 필요)

### 구현 난이도 평가
| 항목 | 난이도 | 근거 |
|:---|:---:|:---|
| Isaac Sim 환경 | ⭐ | Task 1과 공유 |
| Auto-labeling 스크립트 | ⭐⭐ | 간단한 휴리스틱 |
| Style Token 통합 | ⭐⭐ | Architecture 수정 필요 |
| 학습 안정성 | ⭐⭐ | Multi-task learning 수준 |
| 평가 (속도 상관관계) | ⭐ | Pearson r 계산하면 됨 |

**총 난이도**: ⭐⭐ (쉬움)

### 연구 기여도
- **Novelty**: ⭐⭐⭐ (Adverb control은 새롭지만 개념은 기존 연구 확장)
- **Impact**: ⭐⭐⭐ (실용적이지만 혁신적이진 않음)
- **논문 가능성**: ⭐⭐⭐ (Workshop 급, Task 1과 함께 묶어야 Conference)

---

## 5. 종합 비교 및 우선순위

### 비교 테이블
| Task | 난이도 | 참신성 | 기여도 | 데이터 필요량 | 우선순위 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Task 1: Pick & Place** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 250 episodes | **P0** |
| **Task 2: Correction** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 100 episodes | **P1** |
| **Task 3: Adverb** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 220 episodes | **P1** |

### 추천 구현 전략

#### Phase 1 (Week 1-4): Task 1 + Task 3 통합
- **이유**: 둘 다 Isaac Sim 데이터로 학습 가능
- **목표**: Pick & Place with Adverb Control 완성
- **Deliverable**: Conference Workshop 논문 (4-6 pages)

#### Phase 2 (Week 5-8): Task 2 추가
- **이유**: 데이터 수집이 오래 걸림
- **목표**: Real-time Correction 기능 검증
- **Deliverable**: Full Conference Paper (8 pages)

---

## 6. 필요 데이터셋 요약

### 총 데이터 요구량
| 출처 | Task 1 | Task 2 | Task 3 | 합계 |
|:---|:---:|:---:|:---:|:---:|
| **Isaac Sim** | 200 | 0 | 200 | 400 episodes |
| **Real Teleoperation** | 50 | 0 | 20 | 70 episodes |
| **Human Correction** | 0 | 100 | 0 | 100 episodes |
| **총계** | 250 | 100 | 220 | **570 episodes** |

### 데이터 수집 예상 시간
- **Sim (자동)**: 400 episodes x 2 min/episode = **13.3 hours**
- **Real Teleoperation**: 70 episodes x 5 min/episode = **5.8 hours**
- **Human Correction**: 100 trials x 3 min/trial = **5 hours**
- **총계**: **~24 hours** (실제로는 디버깅 등 포함 2-3배 소요)

---

## 7. 결론 및 권장사항

### 최우선 작업 (이번 주)
1. ✅ **Task 1 + Task 3 통합 구현**
   - 둘 다 Sim 데이터로 학습 가능
   - 난이도 낮고 빠른 성과 가능
2. ⏸️ **Task 2는 후순위**
   - Human-in-the-loop 데이터 수집이 병목
   - 초기 성과 이후 추가

### 위험 요소
1. **Sim2Real Gap**: Domain randomization 필수
2. **Real Robot 접근성**: Dobot E6가 항상 사용 가능한지 확인 필요
3. **사람 참여자 모집**: Task 2를 위한 10명 확보

### 논문 출판 전략
- **Option A (안전)**: Task 1 + 3만으로 Workshop 논문
- **Option B (도전)**: Task 1 + 2 + 3 통합하여 Main Conference (CoRL, ICRA)
