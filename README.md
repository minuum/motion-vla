# Motion VLA: Vision-Language-Action for Fine-grained Motion Control

> **π0-style Flow-matching VLA for Dobot E6 Manipulator**  
> Focus: "How" (Adverb control) + Real-time Correction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.0-green.svg)](https://developer.nvidia.com/isaac-sim)

---

## 🎯 Project Overview

Motion VLA는 **π0 스타일의 Flow-matching 아키텍처**를 활용하여 로봇 조작에서 **"어떻게(How)"** 차원을 제어하는 VLA 시스템입니다. 기존 VLA(RT-2, OpenVLA)가 다루지 못한 **Adverb-Conditioned Control**과 **Real-time Language Correction**을 구현합니다.

### Key Features
- 🌊 **Flow-matching Action Expert**: 50Hz continuous action generation
- 🗣️ **Adverb Control**: "carefully", "quickly" 등으로 속도/스타일 제어
- ⚡ **Real-time Correction**: 동작 중 언어 피드백으로 궤적 즉시 수정
- 🤖 **Dobot E6 Magician**: Desktop-sized 6-axis manipulator (450mm reach, 0.75kg payload)

---

## 📁 Project Structure

```
motion-vla/
├── docs/                           # 📚 Documentation
│   ├── pi0_analysis.md            # π0 vs RT-2/OpenVLA 비교
│   ├── pouring_wiping_analysis.md # Task 심층 분석
│   ├── task_evaluation.md         # 카테고리 기반 Task 평가
│   ├── project_briefing.md        # 프로젝트 전체 요약
│   ├── architecture.md            # 시스템 아키텍처
│   ├── dobot_e6_specs.md          # 로봇 상세 스펙
│   ├── literature_review.md       # VLA 문헌 조사
│   ├── deep_dive_analysis.md      # IRP, Adverb 연구
│   ├── irp_paper_summary.md       # IRP 논문 요약
│   ├── new_tasks_definition.md    # Task 정의 (구버전)
│   ├── data_schema.md             # HDF5 데이터 스키마
│   └── implementation_plan.md     # 구현 계획 (구버전)
├── src/
│   └── motion_vla/
│       ├── models/                # 🧠 Core Components
│       │   ├── vl_encoder.py          # Vision-Language Encoder
│       │   ├── flow_action_expert.py  # Flow-matching Expert
│       │   ├── residual_head.py       # Real-time Correction
│       │   └── style_controller.py    # Adverb Style Control
│       ├── data/                  # 📊 Data Pipeline
│       ├── training/              # 🎓 Training Scripts
│       └── inference/             # 🚀 Inference Server
├── scripts/                       # 🛠️ Utility Scripts
│   ├── verify_vl_encoder.py
│   └── verify_flow_expert.py
├── tests/                         # ✅ Unit Tests
├── configs/                       # ⚙️ Configuration Files
└── requirements.txt
```

---

## 📖 Documentation Guide

### 시작하기
1. **[Project Briefing](docs/project_briefing.md)** - 프로젝트 전체 요약 (5분 읽기)
2. **[Dobot E6 Specs](docs/dobot_e6_specs.md)** - 하드웨어 상세 정보

### 연구 배경
3. **[π0 Analysis](docs/pi0_analysis.md)** - π0 vs 기존 VLA 비교, Task 선정 근거
4. **[Task Evaluation](docs/task_evaluation.md)** - 카테고리 기반 Task 후보 평가
5. **[Pouring & Wiping Analysis](docs/pouring_wiping_analysis.md)** - 두 태스크 심층 조사

### 기술 문서
6. **[Architecture](docs/architecture.md)** - 3-stage 파이프라인 설계
7. **[Literature Review](docs/literature_review.md)** - VLA 관련 문헌 조사
8. **[Deep Dive Analysis](docs/deep_dive_analysis.md)** - IRP, Language-to-Velocity 연구

---

## 🎯 Current Task Selection (Updated)

### ✅ Phase 1 (Week 1-2): **Wiping with Adverb Control**
**Task**: "Wipe the table **gently**" / "Push debris **firmly**"

**Why**:
- ✅ Flow-matching의 Continuous force control 활용
- ✅ Dobot E6 적합 (Position-based implicit force)
- ✅ Sim2Real Gap 관리 가능
- ✅ 빠른 성과 (Workshop 논문)

**Metrics**:
- Cleaning Rate (>90%)
- Wiping Time (Adverb correlation)
- Coverage (>95%)

---

### 🔄 Phase 2 (Week 3-6): **Pouring with Style Control**
**Task**: "Pour water **slowly**" / "Fill cup **carefully**"

**Why**:
- ✅ 기존 VLA가 못 하는 영역 (참신성 최고)
- ✅ Velocity profile 정밀 제어 필수
- ✅ Top Conference 타겟

**Challenge**:
- ⚠️ Isaac Sim fluid simulation 어려움
- ⚠️ Sim2Real Gap 큼
- **Strategy**: 구슬로 시작 → 물로 확장

---

## 🏗️ Implementation Status

### ✅ Completed
- [x] VisionLanguageEncoder (PaliGemma/OpenVLA support)
- [x] FlowActionExpert (ODE-based flow-matching)
- [x] ResidualCorrectionHead (LGTC task)
- [x] StyleController (ACMC task)
- [x] Project documentation (12 docs)
- [x] Task selection framework

### 🔄 In Progress
- [ ] Isaac Sim wiping environment
- [ ] Dobot E6 ROS2 integration
- [ ] Data collection pipeline

### 📅 Planned
- [ ] End-to-end training
- [ ] Sim2Real experiments
- [ ] Benchmark evaluation

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.10+
# CUDA 11.8+ (for PyTorch)
# Isaac Sim 4.0+ (optional, for simulation)
```

### Installation
```bash
git clone https://github.com/minuum/motion-vla.git
cd motion-vla
pip install -r requirements.txt
```

### Quick Test
```bash
# Verify VL Encoder (requires PaliGemma download)
python scripts/verify_vl_encoder.py

# Verify Flow Expert (no download needed)
python scripts/verify_flow_expert.py
```

---

## 📊 Research Timeline

| Week | Milestone | Deliverable |
|:---:|:---|:---|
| **1-2** | Wiping Task (Isaac Sim) | Workshop demo |
| **3-4** | Real robot integration | Wiping success |
| **5-6** | Pouring Task (Beads) | Feasibility test |
| **7-8** | Evaluation & Paper | Conference submission |

---

## 📝 Citation

```bibtex
@misc{motion-vla-2026,
  title={Motion VLA: Adverb-Conditioned Control and Real-time Correction for Vision-Language-Action Models},
  author={minuum},
  year={2026},
  url={https://github.com/minuum/motion-vla}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **π0 (Physical Intelligence)** - Flow-matching architecture inspiration
- **OpenVLA** - VLA baseline
- **Isaac Sim (NVIDIA)** - Simulation environment
- **Dobot** - E6 Magician robot platform

---

## 📮 Contact

- **Author**: minuum
- **Email**: minwool0357@gmail.com
- **GitHub**: [@minuum](https://github.com/minuum)

---

**Last Updated**: 2026-01-02
