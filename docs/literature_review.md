# VLA Research Trend & Gap Analysis

## 1. 최신 VLA 모델 동향 (State-of-the-Art)
최근 Vision-Language-Action (VLA) 모델들은 RT-2(Google), OpenVLA(Stanford), Octo(Berkeley) 등을 중심으로 발전하고 있으며, 주로 **"Generalist Policy"** (다양한 로봇/태스크를 하나의 모델로 수행)에 집중하고 있습니다.

| 모델 | 특징 | Action Space | 주요 한계점 |
| :--- | :--- | :--- | :--- |
| **RT-2** | VLM(PaLM-E 등)을 로봇 데이터로 Fine-tuning | Discrete (Tokens) | 추론 속도가 느림(1-3Hz), 정밀 제어 부족 |
| **OpenVLA** | Llama 2 + SigLIP 기반, 오픈소스 SOTA | Discrete (Tokens) | Action Tokenization으로 인한 동작 끊김(Jittering) |
| **Octo** | Diffusion Policy 기반의 Generalist 모델 | Continuous | 계산 비용이 매우 높음, 언어 이해도(Reasoning)는 LLM보다 낮음 |

## 2. 주요 기술적 한계 (Research Gaps)

### A. **Fine-grained Control & Motion Quality** (정밀 제어 및 모션 품질)
대부분의 VLA 모델은 Action을 0~255 사이의 **이산화된 토큰(Discretized Tokens)**으로 예측합니다.
- **문제점**: 이로 인해 로봇의 움직임이 뚝뚝 끊기거나(Stuttering), 정밀한 조작(바늘 꿰기, 액체 따르기 등)에서 성능이 저하됩니다.
- **기존 해결책**: Diffusion Policy 등을 사용하지만, 이는 연산량이 많아 실시간(Real-time) 반응성이 떨어지는 Trade-off가 있습니다.

### B. **Frequency & Real-time Interaction** (실시간성)
거대 언어 모델(LLM) 기반 VLA는 추론 속도가 느려(3~5Hz), 고속 제어나 인간과의 실시간 상호작용(Human-Robot Interaction)에 취약합니다.
- **Gap**: 실행 도중 사람의 개입("잠깐 멈춰", "조금 더 왼쪽으로")에 즉각 반응하기 어렵습니다.

### C. **Adverbial Instruction Understanding** (부사적 지시 이해 부재)
현재 VLA는 "Pick up the cup" 같은 **What(무엇을)**에 집중합니다.
- **Gap**: "Pick up the cup **carefully**" 또는 "**quickly**"와 같이 **How(어떻게)** 동작해야 하는지에 대한 연구는 거의 없습니다. (Motion Style Transfer 부재)

## 3. Motion VLA가 집중해야 할 영역 (Niche)
위 분석을 바탕으로 우리 프로젝트(`Motion VLA`)는 다음 영역을 공략해야 합니다.

1.  **Language-Guided Trajectory Adaptation**: 초기 계획된 경로를 언어 피드백으로 **실시간 수정**하는 능력.
2.  **Adverb-Conditioned Control**: "천천히", "조심스럽게" 등의 부사를 **Joint Velocity/Acceleration 또는 Stiffness** 제어로 매핑하는 능력.
3.  **Residual Policy Learning**: 무거운 VLA가 High-level Goal(웨이포인트)만 주고, 가벼운 Motion Policy가 Low-level 제어를 담당하여 **속도와 지능을 모두 잡는 구조**.
