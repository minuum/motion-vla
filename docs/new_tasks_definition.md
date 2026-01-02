# New VLA Task Proposals

기존 VLA 연구의 한계(정밀 제어, 스타일 부재)를 극복하기 위해, 다음과 같은 **Motion 중심의 신규 태스크 2가지**를 제안합니다.

---

## Task 1: Language-Guided Trajectory Correction (LGTC)
### 1. 정의 (Definition)
로봇이 동작을 수행하는 도중(On-the-fly), 인간의 언어적 피드백을 받아 **실시간으로 궤적(Trajectory)을 수정**하는 태스크입니다.
> 예시: 로봇이 물건을 집으러 갈 때 "조금 더 오른쪽으로"라고 말하면, 현재 궤적에서 즉시 오른쪽으로 편향된 새로운 궤적을 생성해야 함.

### 2. 입/출력 (Input/Output)
- **Input**:
    - Current Observation (RGB Image + Proprioception)
    - Instruction (Initial: "Cup을 집어라", Feedback: "더 오른쪽으로")
    - History (지난 $T$ 스텝의 joint positions)
- **Output**:
    - Delta Action ($\Delta q$): 현재 계획된 궤적 대비 수정량 (Residual Action)

### 3. 데이터셋 요구사항 (Data Requirements)
- **기존 데이터셋(BridgeData V2, OpenX) 활용 불가**: 대부분 성공 궤적만 존재하며, "수정(Correction)" 데이터가 없음.
- **수집 전략**:
    1.  **Simulation (Isaac Gym/Sim)**: 정상 궤적에 노이즈를 주어 실패하게 만든 후, 이를 복구(Correction)하는 궤적을 생성하고, 해당 수정 조작에 맞는 언어 텍스트("왼쪽으로", "위로")를 자동 라벨링.
    2.  **Human-in-the-loop**: 텔레오퍼레이션 중 운영자가 의도적으로 궤적을 틀고, 다시 수정하는 데이터를 수집.

### 4. 평가 메트릭 (Metrics)
- **Correction Success Rate (CSR)**: 피드백 후 $N$초 이내에 목표 궤적으로 복귀했는지 비율.
- **Reaction Time**: 언어 명령 입력 후 궤적 변화가 시작되기까지의 시간 (Latency).

---

## Task 2: Adverb-Conditioned Motion Control (ACMC)
### 1. 정의 (Definition)
단순한 목표 달성을 넘어, **부사(Adverb)**가 지시하는 **동작의 스타일(Style)과 속성(Dynamics)**을 반영하여 로봇을 제어하는 태스크입니다. (What + **How**)

### 2. 주요 Adverb Classes
| Class | 물리적 의미 (Physical Mapping) | 사용 예시 |
| :--- | :--- | :--- |
| **Carefully / Gently** | 속도($v$) 감소, 가속도($a$) 제한, 그리퍼 힘($F$) 최소화 | 깨지기 쉬운 물컵, 두부 집기 |
| **Quickly / Rush** | 속도($v$) 증가, 가속도($a$) 허용치 최대, 동작 간소화 | 긴급 정지, 던지기 |
| **Steadily** | 진동(Jerk) 최소화, 손목 고정 | 물이 꽉 찬 컵 옮기기 |

### 3. 입/출력 (Input/Output)
- **Input**: RGB Image + Instruction (e.g., "Pour the water **carefully**")
- **Output**: Action Token + **Style Token** (Stiffness, Max Velocity Limit 등의 파라미터 제어)

### 4. 구현 및 데이터 전략
- **Style Injection**: 기존 데이터셋의 궤적(Trajectory)을 리샘플링하여 속도를 조절하거나, 필터를 적용해 부드럽게 만든 후, 해당 스타일에 맞는 텍스트("Slowly", "Smoothly")를 Augmentation하여 학습.
- **Contrastive Learning**: 동일한 Task(컵 집기)를 수행하지만 서로 다른 스타일(빠름 vs 느림)을 가진 비디오 쌍을 통해 VLA가 스타일의 차이를 학습하도록 유도.
