# IRP (Iterative Residual Policy) 논문 요약

> **작성일**: 2026-01-02  
> **출처**: Robotics: Science and Systems (RSS)  
> **논문 제목**: Iterative Residual Policy: Learning to Correct Policy Predictions  

## Background
기존 로봇 제어 방식은 "완벽한 Policy"를 처음부터 학습하려 했으나, 이는 다음과 같은 문제가 있습니다:
- 샘플 비효율성: 완벽한 정책 학습에 수만~수십만 episodes 필요.
- Generalization 부족: 학습 당시 보지 못한 물체/환경에서 실패.
- Catastrophic Forgetting: Fine-tuning 시 이전 지식 상실.

IRP는 이를 해결하기 위해 **"기존 Policy(Nominal Policy)의 실수를 수정하는 Residual Policy"**를 학습합니다.

---

## Analysis: 핵심 알고리즘

### 수식 정의
1.  **Nominal Policy**: 초기 정책 $\pi_0(a|s)$ (예: Pre-trained VLA, Heuristic Controller)
2.  **Residual Policy**: $\pi_R(\Delta a | s, \tau_{history})$ → "수정량"만 예측
3.  **최종 Action**: $a_{final} = a_0 + \Delta a$

### 학습 방식
IRP는 다음과 같은 **Delta Dynamics Model**을 학습합니다:
$$
s_{t+1} = f(s_t, a_0 + \Delta a) \approx s_t + \Delta s
$$

여기서 $\Delta s$는 "작은 수정이 상태에 미치는 영향"이며, 이를 반복적으로 적용하여 목표 상태 $s_{goal}$에 도달합니다.

**알고리즘 의사코드**:
```python
# IRP 학습 단계
for episode in range(num_episodes):
    # 1. Nominal Policy로 초기 궤적 생성
    traj_nominal = rollout(pi_0, env)
    
    # 2. 목표와의 차이 계산
    delta_goal = goal_state - traj_nominal[-1]
    
    # 3. Residual Policy로 수정량 예측
    for t in range(T):
        delta_a = pi_R(obs[t], delta_goal, traj_nominal[:t])
        action[t] = nominal_action[t] + delta_a
    
    # 4. 새로운 궤적으로 학습
    traj_corrected = rollout_with_actions(env, action)
    loss = MSE(traj_corrected[-1], goal_state)
    update(pi_R, loss)
```

---

## Findings

### 우리 프로젝트 적용 시 장점
1.  **데이터 효율성**: Nominal Policy(예: OpenVLA)가 "대략적인" 동작만 수행하고, Residual Head가 "언어 피드백 기반 미세 조정"만 담당하므로 적은 데이터로 학습 가능.
2.  **모듈화**: OpenVLA 전체를 Fine-tuning하지 않고 Residual Head만 추가하면 되므로 VRAM 효율적.
3.  **실시간 Correction**: Iterative하게 수정량을 누적 적용하므로, "오른쪽으로" 같은 피드백을 여러 번 반영 가능.

### 우리가 추가로 해야 할 것
- **Language Grounding**: 원본 IRP는 Goal State $s_{goal}$을 직접 주지만, 우리는 "오른쪽으로"같은 **언어**를 $\Delta goal$ 벡터로 변환해야 합니다.
    - 해결책: CLIP/BERT로 언어 임베딩 후, MLP로 Target Delta 예측.

---

## Conclusion

IRP는 우리 Task 1(Language-Guided Trajectory Correction)의 **이론적 기반**으로 완벽하게 적용 가능합니다. 다음 단계로 `ResidualCorrectionHead` 구현을 시작하고, Isaac Sim에서 Nominal + Residual 구조를 검증해야 합니다.

### 참고 구현 레포지토리
- [IRP Official Code (추정)](https://github.com/.../irp) ← 실제 존재 여부 확인 필요
- 대안: 직접 논문 수식 기반 구현 (`src/motion_vla/models/residual_head.py`)
