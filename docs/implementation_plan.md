# Motion VLA Implementation Plan

앞서 제안된 두 가지 태스크(LGTC, ACMC)를 구현하기 위한 단계별 실행 계획입니다.

## Phase 1: 기반 환경 구축 (Weeks 1-2)
가장 시급한 것은 "데이터 확보"의 어려움을 해결하는 시뮬레이션 환경 구축입니다.
- [ ] **Simulation Environment**:
    - **Isaac Lab (formerly Orbit)** 추천: GPU 가속 기반으로 대량의 궤적 데이터 생성 가능.
    - 대안: **Robomimic / PyBullet** (가볍고 설정이 쉬움, 초기 검증용).
- [ ] **Data Generation Pipeline**:
    - "Standard Trajectory"를 생성하는 Oracle Agent 구현.
    - 궤적에 Random Noise를 주입하고, 이를 다시 원복(Recovery)하는 과정을 녹화하여 **Correction Dataset** 자동 생성.

## Phase 2: 베이스 모델 선정 및 파이프라인 (Weeks 3-4)
- [ ] **Base Model 선정**: **OpenVLA (7B)**
    - 이유: 오픈소스 중 가장 성능이 좋고(Standard SOTA), Llama 2 기반이라 언어 이해도가 높음 -> "Adverb" 이해에 유리.
    - 경량화 필요 시: **Octo-Small** 고려.
- [ ] **Fine-tuning Setup**:
    - **LoRA (Low-Rank Adaptation)** 설정: VRAM 효율성을 위해 전체 파라미터가 아닌 LoRA 튜닝 사용.
    - **Action Head 수정**: 기존 Discrete Token Head 대신, 연속적인 값을 예측하거나 Style Token을 추가로 예측하도록 Head 구조 변경 실험.

## Phase 3: Task 구현 및 실험 (Weeks 5-8)

### Track A: Language-Guided Correction (Task 1)
1.  기존 OpenX 데이터셋으로 Pre-training 된 모델 로드.
2.  Phase 1에서 생성한 `(Noisy Traj + "Right", Corrected Traj)` 데이터셋으로 LoRA Fine-tuning.
3.  Evaluation: 시뮬레이터에서 로봇을 움직이다가 중간에 개입 명령을 내렸을 때 궤적 변화 측정.

### Track B: Adverb-Conditioned Control (Task 2)
1.  BridgeData V2 등 기존 데이터를 **속도/가속도 기반 필터링**하여 `Slow`, `Fast`, `Jerky` 등으로 분류(Auto-labeling).
2.  Instruction에 해당 부사를 붙여 학습 ("Pick up coke" -> "Pick up coke **quickly**").
3.  Evaluation: 동일한 "Pick up" 명령에 대해 부사에 따라 소요 시간과 가속도 프로파일이 달라지는지 확인.

---

## 🚀 Immediate Action Items (Today)
당장 오늘(남은 1시간) 실행할 구체적인 작업입니다.

1.  **Dependencies 설치**: `requirements.txt` 작성 및 설치 (PyTorch, Transformers, Accelerate 등).
2.  **프로젝트 구조화**: `src/motion_vla/` 패키지 생성 및 `model.py` (OpenVLA 로딩 코드) 껍데기 작성.
3.  **데이터 스키마 정의**: 학습에 사용할 데이터셋 포맷(JSON/HDF5) 설계 (`docs/data_schema.md`).
