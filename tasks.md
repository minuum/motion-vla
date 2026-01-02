<!-- id: task-sprint-1 -->
# 2시간 스프린트 계획 (VLA Research & Planning)

이 문서는 기존에 존재하지 않는 **새로운 VLA(Motion 중심) 태스크**를 정의하고, 이를 구현하기 위한 구체적인 계획을 수립하는 것을 목표로 합니다.

## 1. VLA 트렌드 및 Gap 분석 ✅ COMPLETED <!-- id: step-research -->
- [x] 최신 VLA (OpenVLA, Octo, RT-2)의 한계점 분석 (특히 Motion/Control 측면) <!-- id: step-gap-analysis -->
- [x] "Motion VLA"로서 차별화 가능한 연구 주제(Niche) 발굴 (e.g., 실시간 궤적 수정, 언어 기반 속도/스타일 제어) <!-- id: step-niche-finding -->
- [x] 관련 논문/레포지토리 조사 및 `docs/literature_review.md` 작성 <!-- id: step-lit-review -->
- [x] 심층 조사 완료: IRP, ExTraCT, Language-to-Velocity 매핑 발견 (`docs/deep_dive_analysis.md`) <!-- id: step-deep-dive -->

## 2. 신규 VLA 태스크 정의 ✅ COMPLETED <!-- id: step-task-def -->
- [x] **Task 1**: Language-Guided Trajectory Correction (LGTC) 정의 완료 <!-- id: step-task1 -->
- [x] **Task 2**: Adverb-Conditioned Motion Control (ACMC) 정의 완료 <!-- id: step-task2 -->
- [x] 각 태스크별 입력/출력, 평가 메트릭 정의 완료 (`docs/new_tasks_definition.md`) <!-- id: step-task-spec -->

## 3. 구현 전략 수립 ✅ COMPLETED <!-- id: step-planning -->
- [x] **데이터셋 요구사항**: HDF5 스키마 정의 완료 (`docs/data_schema.md`) <!-- id: step-data-req -->
- [x] **모델 아키텍처**: OpenVLA + IRP Residual Head 선정 완료 <!-- id: step-arch-design -->
- [x] **Action Plan**: 8주 로드맵 작성 완료 (`docs/implementation_plan.md`) <!-- id: step-roadmap -->

---

## 🚀 Phase 2: 구체적 구현 시작 (Next Sprint)

### Task 1 구현: IRP 기반 Residual Correction
- [ ] **[P0-1]** IRP 논문 (RSS) 정독 및 수식 정리 → `docs/irp_paper_summary.md` 작성 <!-- id: impl-irp-paper -->
- [ ] **[P0-2]** `src/motion_vla/models/residual_head.py` 구현: ResidualCorrectionHead 클래스 <!-- id: impl-residual-head -->
- [ ] **[P0-3]** Isaac Lab 환경 구축: 간단한 "Reach" 태스크 설정 <!-- id: impl-isaac-env -->
- [ ] **[P0-4]** Noisy Trajectory 생성 스크립트: `scripts/generate_noisy_traj.py` <!-- id: impl-noisy-gen -->
- [ ] **[P0-5]** 언어 자동 라벨링 로직 구현 (Delta 분석 기반) <!-- id: impl-auto-label -->

### Task 2 구현: Adverb Style Control
- [ ] **[P1-1]** BridgeData V2 다운로드 (일부 샘플만, ~100 demos) <!-- id: impl-download-data -->
- [ ] **[P1-2]** 속도/Jerk 기반 Adverb 자동 분류 스크립트: `scripts/adverb_labeling.py` <!-- id: impl-adverb-script -->
- [ ] **[P1-3]** Instruction Augmentation 파이프라인 구현 <!-- id: impl-augment -->
- [ ] **[P1-4]** Adverb Style Token을 Action Head에 추가하는 코드 수정 <!-- id: impl-style-token -->
