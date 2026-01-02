# Motion VLA 프로젝트 브리핑 (2026-01-02)

## 📊 프로젝트 현황 요약

### 완료된 작업
1. ✅ **기술 스택 구현** (4개 Core Component)
   - VisionLanguageEncoder (PaliGemma/OpenVLA)
   - FlowActionExpert (π0-style flow-matching)
   - ResidualCorrectionHead (IRP 기반)
   - StyleController (Adverb 매핑)

2. ✅ **하드웨어 선정 및 스펙 조사**
   - Dobot E6 Magician (6축, 450mm reach, 0.75kg payload)
   - ROS2 Humble 지원 확인
   - Action space 정의 (7-dim)

3. ✅ **Task 선정 Framework 수립**
   - 5가지 카테고리 기반 체계적 분류
   - Dobot E6 제약 조건 반영
   - 최종 4개 Task Suite 선정

---

## 🎯 핵심 연구 방향: "Motion-Aware VLA"

### 차별화 포인트
기존 VLA (RT-2, OpenVLA)는 **"What"(무엇을)과 "Where"(어디에)**만 다룸.  
우리는 **"How"(어떻게) + "Correction"(실시간 수정)**을 추가하여 **Motion-Aware VLA** 구축.

### 학술적 기여
1. **Adverb-Conditioned Control**: 부사로 모션 스타일 제어 (carefully, quickly 등)
2. **Real-time Language Correction**: 동작 중 언어 피드백으로 궤적 수정
3. **Small-scale Robot VLA**: 산업용 대형 로봇이 아닌 Desktop robot에서 검증

---

## 📋 최종 선정 Task Suite (우선순위)

### Task 1: Pick & Place with Adverb Control (P0)
**목표**: 언어 지시로 물체를 집어 배치하되, 부사로 속도/스타일 제어

**예시 Instructions**:
- "Pick up the red cup **carefully**" → 속도 0.25 m/s
- "Place it on the left **quickly**" → 속도 0.5 m/s
- "Move the blue block **steadily**" → Jerk 최소화

**데이터 조합**: 3,600가지 (실제 수집 250 episodes)
- 물체 4종 x 색상 4가지 x 위치 25 x 목표 3 x 부사 3

**구현 난이도**: ⭐⭐⭐ (중간)
**연구 기여도**: ⭐⭐⭐⭐ (높음, Workshop 논문 가능)

---

### Task 2: Push with Adverb (P0-예비)
**목표**: Task 1과 동일하지만 "Push" 스킬 추가

**이유**: 
- Pick & Place만으로는 스킬 다양성 부족
- Push는 구현 쉬우면서 새로운 interaction 패턴 검증

**데이터 조합**: ~200가지
**난이도**: ⭐⭐ (쉬움)

---

### Task 3: Real-time Language Correction (P1)
**목표**: 로봇 동작 중 "Move right", "Slower" 같은 피드백으로 궤적 즉시 수정

**Correction Commands**:
| 명령 | Delta Action |
|:---|:---|
| "Move right" | dx = +0.05m |
| "Higher" | dz = +0.05m |
| "Slower" | velocity × 0.5 |

**데이터 조합**: 108가지 (실제 수집 100 episodes)
**난이도**: ⭐⭐⭐⭐ (어려움, 사람 참여 필수)
**연구 기여도**: ⭐⭐⭐⭐⭐ (매우 높음, Top Conference 가능)

**⚠️ 리스크**: Human-in-the-loop 데이터 수집 병목

---

### Task 4: Stack Blocks (P2)
**목표**: 검증용 추가 태스크, "Stack 3 blocks **carefully**"

**이유**: Generalization 검증, 새로운 스킬 조합

---

## 🗂️ 카테고리 기반 선정 근거

### Category 1: 조작 스킬
- ✅ **채택**: Pick, Place, Push (Dobot E6 적합)
- ⚠️ **보류**: Pour, Insert (센서/정밀도 부족)
- 📝 **후보**: Stack, Slide

### Category 2: 언어 제어 차원
- ⭐⭐⭐⭐⭐ **How (Adverb)**: 참신성 최고
- ⭐⭐⭐⭐⭐ **Correction**: HRI 가치 높음
- ⭐⭐ **When (타이밍)**: 센서 부족으로 제외

### Category 3: 물체 속성
- ✅ Rigid objects (Cube, Sphere, Cylinder)
- ✅ 4가지 색상 (Red, Blue, Green, Yellow)
- ✅ 2가지 크기 (Small 5cm, Medium 10cm)
- ❌ Deformable (Sim2Real 어려움)

### Category 4: 난이도
- **L2 (Medium)**: Pick & Place → 80-90% 성공률
- **L3 (Hard)**: Adverb Control → 70-85%
- **L4 (Very Hard)**: Correction → 60-75%

### Category 5: 평가 메트릭
1. Success Rate (필수)
2. Execution Time (Adverb 검증)
3. Collision Count (안전성)
4. Correction Latency (반응성)

---

## 📊 데이터 요구량 총정리

| Task | Isaac Sim | Real Teleoperation | Human-in-the-loop | 합계 |
|:---|:---:|:---:|:---:|:---:|
| Task 1 (Pick & Place + Adverb) | 200 | 50 | 0 | 250 |
| Task 2 (Push) | 100 | 20 | 0 | 120 |
| Task 3 (Correction) | 0 | 0 | 100 | 100 |
| Task 4 (Stack) | 100 | 20 | 0 | 120 |
| **총계** | 400 | 90 | 100 | **590 episodes** |

### 예상 소요 시간
- Sim 자동 생성: 400 x 2분 = **13.3시간**
- Real Teleoperation: 90 x 5분 = **7.5시간**
- Human Correction: 100 x 3분 = **5시간**
- **총계**: ~26시간 (실제 50시간 예상, 디버깅 포함)

---

## 🚀 구현 전략 (2단계)

### Phase 1 (Week 1-4): Task 1 + Task 2
**목표**: Pick & Place + Push with Adverb Control

**이유**:
- 둘 다 Isaac Sim 데이터로 학습 가능
- 빠른 성과 도출 (Workshop 논문)

**Deliverable**: 
- 4-6 page Workshop paper
- Live demo video

---

### Phase 2 (Week 5-8): Task 3 추가
**목표**: Real-time Correction 기능 통합

**이유**:
- 참신성 최고 (Top Conference 가능)
- 하지만 데이터 수집 어려움

**Deliverable**:
- 8 page Full Conference paper (CoRL, ICRA, IROS 목표)

---

## ⚠️ 주요 리스크 및 대응

| 리스크 | 확률 | 영향도 | 대응 방안 |
|:---|:---:|:---:|:---|
| **Sim2Real Gap** | 높음 | 높음 | Domain randomization 강화 |
| **Real Robot 접근성** | 중간 | 높음 | Dobot E6 사용 스케줄 사전 확보 |
| **Human 참여자 모집** | 중간 | 중간 | Task 3는 Phase 2로 연기 |
| **학습 불안정** | 낮음 | 중간 | Pre-trained VLM 활용 |

---

## 📚 관련 문서
- `docs/dobot_e6_specs.md`: 로봇 상세 스펙
- `docs/implementation_plan.md`: 8주 구현 로드맵
- `docs/task_evaluation.md`: Task 평가 상세 분석
- `docs/architecture.md`: 시스템 아키텍처

---

## 🎓 논문 출판 전략

### Option A (안전): Workshop 논문
- **Target**: NeurIPS/ICML Workshop, CoRL Workshop
- **Content**: Task 1 + Task 2 (Pick, Place, Push with Adverb)
- **Timeline**: Week 4 완료 → 5월 제출

### Option B (도전): Main Conference
- **Target**: CoRL 2026, ICRA 2027, IROS 2026
- **Content**: Task 1 + 2 + 3 (Correction 포함)
- **Timeline**: Week 8 완료 → 9월 제출

**권장**: Option A 먼저 진행 → 성공 시 Option B 확장

---

## 다음 단계 (즉시 실행 가능)

1. **Isaac Sim 환경 구축 협업**
   - Dobot E6 URDF import
   - 작업 테이블 설정 (800x600mm)
   
2. **데이터 수집 스크립트 작성**
   - `scripts/generate_sim_data.py`
   - `scripts/collect_real_demos.py`

3. **End-to-End Pipeline 통합**
   - 4개 Component를 `MotionVLAPipeline` 클래스로 통합
   - Dummy data 검증
