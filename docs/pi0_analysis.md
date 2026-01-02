# π0 vs 기존 VLA 비교 분석 및 태스크 재정의

> **작성일**: 2026-01-02  
> **목적**: π0의 실제 적용 사례 분석 → 우리 프로젝트에 최적화된 태스크 선정

---

## 1. π0가 선택한 Task와 그 이유

### π0의 대표 Task
| Task | 설명 | 왜 선택되었는가 |
|:---|:---|:---|
| **Laundry Folding** | 빨래 개기 | **Deformable object** 조작의 극한, 무한한 초기 상태 → 일반화 능력 검증 |
| **Table Bussing** | 식탁 치우기 | **Emergent strategy** (접시 털기, 분류), Multi-object handling |
| **Grocery Bagging** | 장보기 봉투 담기 | 다양한 물체 크기/무게, **Sequencing** 능력 검증 |
| **Box Assembly** | 상자 조립 | Multi-step, **정밀 조작** 요구 |

### 선택 기준 (Physical Intelligence의 철학)
1. **복잡도 (Complexity)**: 단순 반복 불가, 상황 판단 필요
2. **손재주 (Dexterity)**: 미세한 힘 조절, 고주파 제어
3. **일반성 (Generalization)**: 특정 물체가 아닌 "옷", "접시" 같은 카테고리 전체
4. **실용성 (Relatability)**: 사람들이 공감하는 귀찮은 일
5. **Emergent Behavior**: 학습되지 않은 전략이 자연스럽게 나타남

---

## 2. π0 vs RT-2 vs OpenVLA 핵심 차이

### 비교 테이블
| 항목 | π0 | RT-2 | OpenVLA |
|:---|:---|:---|:---|
| **Action Output** | ✅ Continuous (Flow-matching) | ❌ Discrete Tokens | ❌ Discrete Tokens |
| **Control Frequency** | **50Hz** | 1Hz | 5-15Hz |
| **적합한 태스크** | Long-horizon, **multi-step** 작업 | Single-instruction 일반화 | Multi-object, single-step |
| **강점** | **Smooth trajectory**, Real-time dexterity | Web knowledge 활용, 추론 능력 | Multi-task, 빠른 fine-tuning |
| **약점** | 데이터 많이 필요 | 느림, 부드러운 동작 어려움 | 마찬가지로 smooth motion 약함 |

### Flow-matching의 결정적 이점
1. **Smoothness**: 끊기지 않는 부드러운 궤적 (Jerk 최소화)
2. **Real-time**: 50Hz = 20ms per action → Reactive control 가능
3. **Efficiency**: Diffusion보다 ~85% 빠른 추론 속도
4. **Precision**: Continuous output → 정밀한 힘/속도 제어

---

## 3. Flow-matching을 활용하기 최적인 Task 특성

### ✅ Flow-matching이 빛나는 Task
1. **Deformable Object Manipulation** (옷, 천, 케이블)
   - 이유: 부드러운 힘 조절 필수, Discrete action으로는 끊김
   
2. **Contact-rich Manipulation** (밀기, 쓸기, 문지르기)
   - 이유: 연속적인 힘 피드백, 50Hz로 실시간 반응
   
3. **Long-horizon Multi-step** (요리, 청소, 조립)
   - 이유: 여러 primitive의 부드러운 전환
   
4. **Fine-grained Speed Control** (부사 제어!)
   - 이유: Continuous velocity → "carefully" = 0.25 m/s 정밀 제어

5. **Dynamic Interaction** (쏟기, 따르기, 흔들기)
   - 이유: 속도/가속도 프로파일이 결과에 직접 영향

### ❌ Flow-matching이 불필요한 Task
1. **Static Pick & Place** (단순 집기/놀기)
   - OpenVLA로도 충분, Flow-matching 오버킬
   
2. **Waypoint Navigation** (지점 이동만)
   - 궤적이 단순해서 discrete도 OK

---

## 4. Dobot E6로 구현 가능한 "π0-style" Task 후보

### 🎯 최종 제안: "Flow-matching의 강점을 활용한 Task Suite"

#### **Task A: Contact-rich Manipulation** ⭐⭐⭐⭐⭐
**예시**: "Wipe the table **gently**", "Push debris **carefully** toward the edge"

**π0 강점 활용**:
- Continuous force control → 테이블 손상 없이 민감하게 조절
- Adverb → 속도/압력 직접 매핑 (Flow-matching의 continuous output 활용)

**Dobot E6 적합성**: ✅ (그리퍼 대신 wiper 부착)

**데이터**: Sim에서 다양한 테이블 표면, 먼지 패턴 생성 가능

---

#### **Task B: Pouring with Style Control** ⭐⭐⭐⭐⭐
**예시**: "Pour water **slowly**", "Fill the cup **carefully** without spilling"

**π0 강점 활용**:
- Flow-matching → 속도 프로파일 정밀 제어 (쏟지 않기)
- 50Hz → 실시간 컵 기울기 조절

**Dobot E6 적합성**: ⚠️ (센서 부족하지만 시각 기반으로 가능)

**차별점**: **기존 VLA가 못 하는 태스크!** (Discrete tokenization으로는 불가능)

---

#### **Task C: Sequential Folding (Simplified Laundry)** ⭐⭐⭐⭐
**예시**: "Fold the towel **neatly**"

**π0 강점 활용**:
- Deformable object (천 수건)
- Multi-step: Grasp → Align → Fold → Press
- Smooth transition between steps

**Dobot E6 적합성**: ✅ (수건 크기 제한, 300g 이하)

**데이터**: π0처럼 무한한 초기 상태 → Generalization 극한 검증

---

#### **Task D: Real-time Correction (기존 유지)** ⭐⭐⭐⭐⭐
**예시**: 동작 중 "Slower", "Gentler" 피드백

**π0 강점 활용**:
- 50Hz → 실시간 반응 (RT-2는 1Hz라 불가능)
- Continuous output → Delta velocity 즉시 적용

---

## 5. 기존 계획 vs π0-기반 계획 비교

### 기존 계획 (일반 VLA 접근)
| Task | 차별점 | π0 활용도 |
|:---|:---|:---:|
| Pick & Place + Adverb | Adverb 제어 | ⭐⭐ (OpenVLA로도 가능) |
| Push with Adverb | 스킬 확장 | ⭐⭐⭐ (Contact-rich) |
| Correction | Real-time | ⭐⭐⭐⭐⭐ |

### π0 기반 신규 계획
| Task | 차별점 | π0 활용도 | 기존 VLA와 차별성 |
|:---|:---|:---:|:---|
| **Contact-rich Wiping** | Continuous force | ⭐⭐⭐⭐⭐ | ✅ High |
| **Pouring with Style** | Speed profile | ⭐⭐⭐⭐⭐ | ✅ **Very High** |
| **Towel Folding** | Deformable | ⭐⭐⭐⭐ | ✅ High |
| **Real-time Correction** | 50Hz reactive | ⭐⭐⭐⭐⭐ | ✅ Very High |

---

## 6. 최종 권장사항

### 추천 전략: "π0의 강점에 집중"

#### Phase 1 (P0): **Pouring + Wiping**
- **Pouring**: 기존 VLA가 못 하는 영역, 논문 임팩트 최고
- **Wiping**: Contact-rich manipulation 검증
- **공통점**: 둘 다 **Continuous velocity control** 필수

#### Phase 2 (P1): **Towel Folding**
- π0 대표 Task의 Simplified version
- Deformable object handling 검증

#### Phase 3 (P2): **Real-time Correction**
- 모든 Task에 적용 가능한 General feature

---

### 변경 이유
1. ❌ **기존**: Pick & Place는 너무 basic, OpenVLA도 잘함
2. ✅ **신규**: Pouring/Wiping은 **Flow-matching 필수**, 차별화 극대화
3. ✅ **논문 가치**: "우리만 할 수 있는 것"을 보여줘야 Top Conference

---

## 7. 구현 난이도 재평가

### Pouring Task
| 항목 | 난이도 | 대응 방안 |
|:---|:---:|:---|
| 센서 부족 (힘/유량) | ⭐⭐⭐⭐ | Vision-based: 컵 채워진 정도 인식 |
| Sim2Real Gap (액체 물리) | ⭐⭐⭐⭐⭐ | Isaac Sim의 Particle system 활용 |
| 안전성 (쏟을 위험) | ⭐⭐⭐ | 물 대신 구슬, 나중에 물 |

**총 난이도**: ⭐⭐⭐⭐ (도전적이지만 가치 있음)

### Wiping Task
| 항목 | 난이도 | 대응 방안 |
|:---|:---:|:---|
| End-effector 교체 | ⭐⭐ | Wiper 제작 (3D 프린팅) |
| Force control | ⭐⭐⭐ | Position-based implicit force |
| 평가 메트릭 (얼마나 깨끗한지) | ⭐⭐⭐ | Vision: 먼지 픽셀 카운트 |

**총 난이도**: ⭐⭐⭐ (적당함)

---

## 결론

**π0를 쓰려면 π0만 할 수 있는 것을 해야 합니다.**

- ❌ Pick & Place는 모든 VLA가 하는 것
- ✅ **Pouring**, **Wiping**, **Folding**은 **Flow-matching 없이는 어려운 Task**
- ✅ 우리의 차별점: "**어떻게(How)**" → Continuous control의 정수

**다음 단계**: 이 방향으로 진행할지 결정 후, Pouring/Wiping 환경 구축 시작
