# Task Selection Rationale: Why Wiping?

> **목적**: 5개 태스크 평가 및 Wiping 선택의 과학적 근거  
> **날짜**: 2026-01-07  
> **결론**: Wiping (Phase 1) → Drawing (Optional) → Pouring (Phase 2)

---

## 🎯 Task Selection Criteria

### 프로젝트 목표 (Motion VLA with π0)

```
1. Flow-matching의 강점 활용 (50Hz continuous control)
2. Adverb-conditioned control ("gently", "firmly")
3. Dobot E6 + 흡착 그리퍼로 실현 가능
4. 빠른 구현 (workshop paper 목표)
```

### 평가 기준 (5점 척도)

| Criterion | Weight | 설명 |
|:---|:---:|:---|
| **π0 적합성** | ⭐⭐⭐⭐⭐ | Flow-matching 필수성 |
| **물리적 실현성** | ⭐⭐⭐⭐⭐ | 흡착 그리퍼 제약 |
| **구현 속도** | ⭐⭐⭐⭐ | 시간 제약 (4일 목표) |
| **차별성** | ⭐⭐⭐⭐ | 기존 VLA 대비 novelty |
| **실용성** | ⭐⭐⭐ | Real-world impact |

---

## 📋 5개 Task 상세 평가

### 1. Wiping (닦기) 🥇

#### 개요
```
Task: "Wipe the table {adverb}"
Tool: 5cm × 3cm wiper pad (suction-attached)
Dirt: Cocoa powder (20g, random distribution)
```

#### 평가 점수

| Criterion | Score | 근거 |
|:---|:---:|:---|
| **π0 적합성** | ⭐⭐⭐⭐⭐ | Contact-rich → 50Hz 필수 |
| **물리적 실현성** | ⭐⭐⭐⭐⭐ | 흡착으로 wiper 고정 완벽 |
| **구현 속도** | ⭐⭐⭐⭐⭐ | 4일 안에 완성 가능 |
| **차별성** | ⭐⭐⭐⭐ | Adverb control (novel) |
| **실용성** | ⭐⭐⭐⭐ | 청소 로봇 응용 |

**Total**: 23/25 (92%)

---

#### 왜 π0에 최적인가?

**1. Contact-rich Manipulation**
```
Problem: 표면 접촉 시 실시간 압력 조절 필요
Solution: 50Hz flow-matching
  - Discrete VLA (1-15Hz): 끊김
  - π0 (50Hz): 부드러운 압력 modulation
```

**2. Continuous Velocity Control**
```
"Wipe gently": 0.05 m/s  (매우 느림)
"Wipe normally": 0.15 m/s
"Wipe firmly": 0.30 m/s  (빠름)

Discrete: 3-5단계만 가능
π0 Flow-matching: 연속적 속도 제어 (정밀함)
```

**3. Action Chunking 효과**
```
50-step lookahead (1초 예측)
→ Table geometry 이해
→ Coverage 최적화
```

---

#### 흡착 그리퍼 적합성: 완벽 (⭐⭐⭐⭐⭐)

**Wiper 부착 방식**:
```
Suction cup (φ16mm)
    ↓ (vacuum)
Wiper mount (cylindrical, φ15mm)
    ↓ (rigid attachment)
Wiper pad (5cm × 3cm, 30g)
```

**Forces 분석**:
```
Wiping force: ~2-3N (horizontal)
Suction force: 20N (vertical)
Shear capacity: 6N (φ16mm)

Safety factor: 6N / 3N = 2.0× ✅
→ 안전하게 고정 가능!
```

**장점**:
- ✅ 가벼움 (30g < 750g payload)
- ✅ Shear force 범위 내
- ✅ 진동 없음 (안정)

---

#### 구현 속도: 4일 (⭐⭐⭐⭐⭐)

**Day-by-Day Plan**:
```
Day 1 (완료): Environment setup (85%)
  - DirtSimulator (PhysX particles)
  - VisionMetrics (HSV detection)
  - WipingEnv (Isaac Sim)

Day 2: Robot controller + Demo
  - Position control interface
  - Zigzag trajectory
  - First wiping video

Day 3-4: Data collection
  - 342 sim episodes (automated)
  - 38 real episodes (teleoperation)

Total: 4일 → Workshop paper 가능!
```

---

#### 차별성: Adverb Control (⭐⭐⭐⭐)

**기존 연구 (Google Wiping)**:
```
Task: "Wipe the table"
Control: Single velocity
Output: Cleaning rate >90%
```

**우리 (Motion VLA)**:
```
Task: "Wipe the table {gently/firmly/quickly}"
Control: Adverb-conditioned dynamics
Output: Cleaning rate + Velocity correlation

차별점:
- "How" dimension 추가
- Continuous velocity 제어 (π0 강점)
- Multi-objective (quality vs speed)
```

---

### 2. Drawing (그리기) 🥈

#### 개요
```
Task: "Draw a {shape} {adverb}"
Tool: Pen holder (3D printed)
Surface: Paper on table
```

#### 평가 점수

| Criterion | Score | 근거 |
|:---|:---:|:---|
| **π0 적합성** | ⭐⭐⭐⭐ | Smooth trajectory 필요 |
| **물리적 실현성** | ⭐⭐⭐ | Pen holder 제작 필요 |
| **구현 속도** | ⭐⭐⭐⭐ | 2-3일 (Wiping 코드 재사용) |
| **차별성** | ⭐⭐⭐ | RTC 검증 (correction) |
| **실용성** | ⭐⭐ | 응용 제한적 |

**Total**: 17/25 (68%)

---

#### 문제점

**1. Vacuum Hysteresis**
```
Problem: 진공 ON/OFF 200-500ms 지연
Task: "Draw staccato" (끊어 그리기)
  → Pen up/down이 느림
  → "Staccato" 표현 불가능

Workaround: Continuous drawing만 (제한적)
```

**2. Force Control**
```
Drawing pressure: 0.5-2N 필요
Suction control: 불가능 (ON/OFF만)
→ Position-based implicit force 사용
  (부정확함)
```

**결론**: **Optional** (RTC 검증용으로만 추가)

---

### 3. Pouring (따르기) 🥉

#### 개요
```
Task: "Pour water {slowly/carefully}"
Setup: Cup → Cup transfer
Challenge: Sim2Real gap (fluid physics)
```

#### 평가 점수

| Criterion | Score | 근거 |
|:---|:---:|:---|
| **π0 적합성** | ⭐⭐⭐⭐⭐ | Velocity profile 정밀 제어 |
| **물리적 실현성** | ⭐⭐⭐ | 가능하지만 어려움 |
| **구현 속도** | ⭐⭐ | 2-3주 (Sim2Real 난제) |
| **차별성** | ⭐⭐⭐⭐⭐ | 기존 VLA 불가능 |
| **실용성** | ⭐⭐⭐⭐⭐ | 주방 로봇 |

**Total**: 19/25 (76%)

---

#### 장점: 최고 Novelty

**기존 VLA (OpenVLA, RT-2)**:
```
Discrete action (1-15Hz)
→ Pouring 불가능!
  (액체 출렁임 제어 못 함)
```

**π0 (우리)**:
```
Continuous control (50Hz)
→ Angular velocity profile 정밀 제어
  "Slowly": 2°/s
  "Carefully": 1°/s (매우 느림)
→ 출렁임 최소화!
```

---

#### 단점: 구현 시간

**Sim2Real Gap**:
```
Simulation: Isaac Sim particle system
  - 구슬로 근사 (1주)
  - 물 시뮬 (PhysX fluid, 2주)

Real Robot:
  - 센서 부족 (유량계 없음)
  - Vision-based level detection
  - Calibration 필요

Total: 2-3주 → Phase 2로 연기
```

**결론**: **Phase 2** (Week 3-6)

---

### 4. Card Dealing (카드 분배) ❌

#### 개요
```
Task: "Deal cards to players"
Action: Pick card → Place at position
Challenge: Aerodynamic peeling
```

#### 평가 점수

| Criterion | Score | 근거 |
|:---|:---:|:---|
| **π0 적합성** | ⭐⭐⭐ | 단순한 Pick & Place |
| **물리적 실현성** | ⭐ | **물리적 불가능** |
| **구현 속도** | N/A | 실현 불가 |
| **차별성** | ⭐⭐ | 낮음 |
| **실용성** | ⭐ | 제한적 |

**Total**: REJECTED

---

#### 치명적 결함: Aerodynamic Peeling

**물리 분석**:
```
Card specs:
- Size: 6cm × 9cm
- Weight: 1.8g
- Suction area: π × (0.8cm)² = 2cm²

Forces @ 0.5 m/s:
- Drag force: 0.15N
- Holding force: 2 cm² × 10 kPa = 0.2N

Safety margin: 0.2N - 0.15N = 0.05N (25%)
→ 너무 작음!

At 0.6 m/s:
- Drag > Holding
→ 카드 날아감! ❌
```

**실험적 증거**:
- Paper airplane 원리와 동일
- 0.5 m/s 이상에서 박리 관찰됨

**결론**: **물리적으로 불가능** → 제외

---

### 5. Stacking (쌓기) ❌

#### 개요
```
Task: "Stack blocks {carefully/precariously}"
Challenge: Vacuum hysteresis + Force control
```

#### 평가 점수

| Criterion | Score | 근거 |
|:---|:---:|:---|
| **π0 적합성** | ⭐⭐⭐⭐ | Precision placement |
| **물리적 실현성** | ⭐ | **Hysteresis 문제** |
| **구현 속도** | N/A | 실현 불가 |
| **차별성** | ⭐⭐⭐ | "Precariously" novel |
| **실용성** | ⭐⭐⭐ | 물류 로봇 |

**Total**: REJECTED

---

#### 치명적 결함: Vacuum Hysteresis

**문제**:
```
"Stack precariously" (위태롭게 쌓기)
→ Block을 살짝만 놓고 빠져야 함

Vacuum OFF:
- 압력 해제: 200-500ms 지연
- 잔류 진공: 50-100ms
- Total: 250-600ms

이 시간 동안:
- Block이 gripper에 붙어있음
- 다음 block 위치 틀어짐
→ "Precariously" 제어 불가능! ❌
```

**대안 고려**:
```
"Stack normally" (정상적으로)?
→ Yes, 가능
But: π0 강점 활용 못 함
  (단순 Pick & Place)
→ 차별성 없음
```

**결론**: **제외** (π0 장점 활용 못 함)

---

## 📊 최종 점수 비교

| Task | π0 | 실현성 | 속도 | 차별성 | 실용성 | **Total** | 결정 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Wiping** | 5 | 5 | 5 | 4 | 4 | **23** (92%) | ✅ Phase 1 |
| Drawing | 4 | 3 | 4 | 3 | 2 | 17 (68%) | ⚠️ Optional |
| Pouring | 5 | 3 | 2 | 5 | 5 | 19 (76%) | ✅ Phase 2 |
| Card Dealing | 3 | 1 | - | 2 | 1 | - | ❌ Rejected |
| Stacking | 4 | 1 | - | 3 | 3 | - | ❌ Rejected |

---

## 🎯 선택 논리

### Wiping을 Phase 1로 선택한 이유

**1. Risk-Minimization (위험 최소화)**
```
물리적 제약: 없음 (완벽히 해결됨)
구현 시간: 4일 (최단)
성공 확률: ~95% (매우 높음)

→ Workshop paper 확실히 나옴!
```

**2. π0 강점 활용**
```
Contact-rich: 50Hz control 필수
Adverb: Continuous velocity
Action chunking: Coverage optimization

→ π0만 할 수 있는 것!
```

**3. 점진적 확장**
```
Wiping (Phase 1)
  → Code base 확립
    → Drawing (Optional, 2-3일 추가)
      → Pouring (Phase 2, 2-3주)

→ Incremental risk management
```

---

### Pouring을 Phase 2로 연기한 이유

**장점**:
- ⭐⭐⭐⭐⭐ Novelty (기존 VLA 불가능)
- ⭐⭐⭐⭐⭐ π0 적합성
- ⭐⭐⭐⭐⭐ 실용성

**단점**:
- ⭐⭐ 구현 시간 (2-3주)
- Sim2Real gap 큼
- 실패 risk 30%

**결론**:
```
Wiping 성공 후 → Pouring 시작
  - Wiping으로 검증된 pipeline
  - Conference paper 목표
  - 충분한 시간 (3-6주)
```

---

### Card Dealing & Stacking 제외 이유

**과학적 근거**:

1. **Card Dealing**: 공기역학
   ```
   Drag force (0.5 m/s) ≈ Holding force
   → 25% safety margin
   → 실용 불가능
   ```

2. **Stacking**: 진공 히스테리시스
   ```
   Release delay: 250-600ms
   → "Precariously" 제어 불가
   → π0 장점 활용 못 함
   ```

**실용적 판단**:
```
시간 낭비 risk > 잠재적 benefit
→ 명확히 제외
```

---

## 💡 전략적 의사결정

### Timeline Optimization

```
Week 1-2: Wiping (확실한 성과)
  ├─ Day 1-4: Implementation
  └─ Day 5-10: Data + Training

Week 3: Decision Point
  ├─ Wiping 성공? → Pouring 시작
  └─ Wiping 실패? → Drawing으로 pivot

Week 4-6: Conference Paper
  ├─ Pouring 성공? → Top conference
  └─ Pouring 실패? → Workshop (Wiping만)
```

---

### Risk Management

| Phase | Task | Success Prob | Fallback |
|:---|:---|:---:|:---|
| **Phase 1** | Wiping | 95% | N/A (확실) |
| **Optional** | Drawing | 80% | Skip |
| **Phase 2** | Pouring | 70% | Workshop (Wiping) |

**전략**:
- Phase 1: Low-risk, high-certainty
- Phase 2: High-risk, high-reward

---

## ✅ 결론

### Wiping이 최선의 선택인 이유

**1. 과학적 타당성**
- ✅ π0 강점 100% 활용
- ✅ 물리적 제약 완벽 해결
- ✅ 차별성 확보 (adverb control)

**2. 실행 가능성**
- ✅ 4일 구현 (가장 빠름)
- ✅ 성공 확률 95%
- ✅ Workshop paper 확실

**3. 확장 가능성**
- ✅ Drawing 추가 (2-3일)
- ✅ Pouring으로 확장 (Phase 2)
- ✅ Code reusability 높음

**4. 전략적 가치**
- ✅ 빠른 검증 (proof of concept)
- ✅ Risk minimization
- ✅ 점진적 확장 가능

---

### 최종 Roadmap

```
Phase 1 (확정): Wiping
  - Timeline: Week 1-2
  - Goal: Workshop paper
  - Success: >95% probability

Optional: Drawing
  - Timeline: +2-3 days
  - Condition: If Wiping succeeds
  - Goal: RTC validation

Phase 2 (조건부): Pouring
  - Timeline: Week 3-6
  - Condition: Wiping success + 충분한 시간
  - Goal: Conference paper
  - Success: ~70% probability
```

---

**최종 결정**: **Wiping First, Pouring Next, Others Skip**

이 전략으로 최소 1개(Wiping), 최대 2개(Wiping+Pouring) 태스크 완성 가능!
