# Deep Dive: Language-Guided Trajectory Correction & Motion Style Control

## Background
VLA 모델의 한계(discrete tokenization으로 인한 jittering, 실시간 반응성 부족)를 극복하려면, 기존 연구에서 이미 검증된 방법론을 활용하는 것이 중요합니다. 본 문서는 웹 조사를 통해 발견한 **관련 선행 연구**와 **구현 가능한 접근법**을 심층 분석합니다.

---

## Analysis: Task 1 - Language-Guided Trajectory Correction

### 1.1 주요 관련 논문/프레임워크
| 논문/프로젝트 | 핵심 아이디어 | 방법론 | 적용 가능성 |
| :--- | :--- | :--- | :--- |
| **ExTraCT** (Frontiers in Robotics, 2023) | LLM을 이용해 자연어 피드백을 궤적 수정 함수로 변환 | Modular 구조: Language Understanding + Trajectory Adapter | ⭐⭐⭐⭐⭐ 직접 적용 가능 |
| **Iterative Residual Policy (IRP)** (RSS) | Delta dynamics를 학습하여 기존 궤적을 점진적으로 개선 | Model-based RL: $\Delta q = f(q_{prev}, g_{target})$ | ⭐⭐⭐⭐⭐ 우리 Residual 개념과 완벽히 일치 |
| **Diffusion Trajectory-guided Policy** (arXiv) | Vision-Language Model로 Diffusion 기반 궤적 생성 후 Policy 가이드 | Generative model + Policy distillation | ⭐⭐⭐ 계산 비용 높음, 초기엔 IRP 우선 |

### 1.2 선택한 구현 전략: **IRP (Iterative Residual Policy)**
- **선정 이유**:
    1.  우리가 제안한 "Residual Action ($\Delta q$)" 개념과 이론적으로 정확히 일치합니다.
    2.  기존 Oracle/Nominal Policy 위에 Correction Layer만 학습하면 되므로 데이터 효율적입니다.
    3.  Fine-tuning이 아닌 **Add-on Module** 형태라서 OpenVLA 같은 거대 모델을 건드리지 않아도 됩니다.

- **구체적 구현 방안**:
    ```python
    # 의사 코드 (Pseudo-code)
    class ResidualCorrectionHead(nn.Module):
        """
        입력: 현재 관측(Obs), 언어 피드백(Lang), 이전 궤적(History)
        출력: Delta Action (Δq)
        """
        def forward(self, obs, lang_correction, traj_history):
            # 1. VLM으로 언어 인코딩
            lang_embed = self.vlm_encoder(lang_correction)  # "오른쪽으로" -> [768,]
            
            # 2. 궤적 히스토리 인코딩 (Temporal Transformer)
            traj_embed = self.temporal_encoder(traj_history)  # (T, D) -> [512,]
            
            # 3. Fusion + Residual Prediction
            fused = torch.cat([obs, lang_embed, traj_embed], dim=-1)
            delta_action = self.mlp(fused)  # [D_action]
            return delta_action
    ```

### 1.3 데이터 수집 전략 (IRP 논문 기반)
IRP 논문은 "Noisy Trajectory"를 생성하고 이를 복구하는 과정을 자동으로 수집했습니다.
- **우리 적용안**:
    1.  Isaac Sim에서 성공 궤적 $\tau_{success}$ 생성.
    2.  중간 스텝에 Gaussian Noise 주입 -> $\tau_{noisy}$.
    3.  Correction Controller(예: MPC)가 복구 -> $\tau_{corrected}$.
    4.  Noise 방향을 분석하여 자동으로 언어 라벨 생성 (예: $\Delta x > 0$ → "왼쪽으로", $\Delta z > 0$ → "위로").

---

## Analysis: Task 2 - Adverb-Conditioned Motion Control

### 2.1 주요 관련 연구
| 연구 | 핵심 기법 | 물리적 매핑 | 적용 가능성 |
| :--- | :--- | :--- | :--- |
| **Language-to-Velocity Control** (arXiv, CMU) | Adverb를 속도/가속도 제약 조건으로 해석 | "Slowly" → $v_{max} = 0.3 \cdot v_{nominal}$ | ⭐⭐⭐⭐⭐ 즉시 구현 가능 |
| **Verb-Adverb Motion Style** (Dartmouth) | Verb(동작) + Adverb(스타일)을 분리하여 Interpolation | Style Latent Space 학습 | ⭐⭐⭐ 연구적 가치 높으나 복잡함 |
| **MotionGlot** (Brown Univ.) | Motion을 언어처럼 취급, Tokenize하여 변환 | Seq2Seq Transformer | ⭐⭐ 우리 목적과는 다소 거리 있음 |

### 2.2 선택한 구현 전략: **Language-to-Velocity Constraint Mapping**
- **선정 이유**:
    1.  가장 직관적이고 물리적으로 해석 가능합니다.
    2.  기존 데이터에 "Post-processing"을 통해 Adverb를 자동 라벨링할 수 있습니다.
    3.  VLA 출력에 **별도의 Style Token**을 추가하면 됩니다.

- **구현 예시**:
    ```python
    # Adverb -> Dynamics Parameter 매핑
    ADVERB_MAPPING = {
        "carefully": {"v_scale": 0.5, "a_max": 0.3, "jerk_limit": 0.1},
        "quickly":   {"v_scale": 1.5, "a_max": 2.0, "jerk_limit": 1.0},
        "steadily":  {"v_scale": 0.8, "a_max": 0.5, "jerk_limit": 0.05},
    }
    
    def apply_adverb_style(trajectory, adverb):
        """주어진 궤적에 부사 스타일을 적용"""
        params = ADVERB_MAPPING[adverb]
        # 속도 스케일링
        trajectory.velocity *= params["v_scale"]
        # 가속도 클리핑
        trajectory.acceleration = np.clip(
            trajectory.acceleration, 
            -params["a_max"], 
            params["a_max"]
        )
        return trajectory
    ```

### 2.3 자동 라벨링 전략
기존 BridgeData V2, OpenX 등의 데이터셋에는 Adverb 라벨이 없습니다. 자동 생성 방안:
1.  **속도 기반 분류**:
    - 평균 속도 $< 0.1$ m/s → "slowly"
    - 평균 속도 $> 0.5$ m/s → "quickly"
2.  **Jerk(가속도 변화율) 기반**:
    - Jerk 표준편차 $< 0.05$ → "steadily"
    - Jerk 표준편차 $> 0.2$ → "roughly"
3.  **Instruction Augmentation**:
    - 원본: "Pick up the cup"
    - 증강: "Pick up the cup **carefully**" (자동 분류된 스타일 부사 추가)

---

## Findings

### 핵심 발견사항
1.  **IRP는 우리 Task 1의 Perfect Match입니다**: Delta Dynamics 학습 방식이 Residual Policy와 동일하며, 논문에서 이미 noisy trajectory 복구 실험을 수행했습니다.
2.  **Adverb-to-Velocity Mapping은 검증된 접근법입니다**: CMU 연구에서 이미 "slowly", "gently" 등을 속도 제약으로 매핑하여 성공했습니다.
3.  **Human-in-the-loop 데이터는 드뭅니다**: 대부분 연구가 시뮬레이션 자동 생성 또는 Augmentation에 의존하고 있어, 우리도 Isaac Sim + Auto-labeling 전략이 타당합니다.

### 구현 우선순위
| 우선순위 | 항목 | 예상 소요 시간 | 근거 |
| :---: | :--- | :--- | :--- |
| **P0** | IRP 기반 Residual Correction Head 구현 | 1-2주 | 핵심 차별화 요소, 논문 레퍼런스 명확 |
| **P1** | Adverb Mapping + Auto-labeling 파이프라인 | 1주 | 데이터 증강의 핵심, 구현 난이도 낮음 |
| **P2** | Isaac Sim 환경 구축 및 Noisy Traj 생성 | 2주 | 데이터 인프라, 병렬 처리 가능 |

---

## Conclusion

### 다음 단계 (Next Actions)
1.  **[문서화]** IRP 논문 정독 후 `docs/irp_paper_review.md` 작성 (수식 및 알고리즘 정리).
2.  **[코드]** `src/motion_vla/residual_head.py` 구현 시작 (IRP 기반 아키텍처).
3.  **[실험]** BridgeData V2 일부를 다운로드하여 속도 기반 Adverb Auto-labeling 스크립트 작성 및 검증.

### 참고 문헌
- ExTraCT: [Frontiers in Robotics and AI, 2023](https://www.frontiersin.org/articles/...)
- IRP (Iterative Residual Policy): [Robotics: Science and Systems](https://roboticsproceedings.org/...)
- Language-to-Velocity Mapping: [arXiv, CMU](https://arxiv.org/...)
