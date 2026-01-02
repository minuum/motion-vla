# Motion VLA Project Rules

## 1. Research & Documentation
- **논문 비교 분석**: 항상 **표(Table)** 형식으로 정리하고, 출처(논문 제목, 섹션)를 명확히 표기합니다.
- **기술적 주장**: 코드나 논문의 구체적 근거를 함께 제시해야 합니다.
- **분석 리포트**: Markdown 형식 (Background -> Analysis -> Findings -> Conclusion 구조 준수).
- **정량적 메트릭**: 실험 결과나 모델 비교 시 수치(Episodes, Tasks, Success Rate 등)를 우선 제시합니다.

## 2. Code & Experiments
- **주석**: Python 분석 스크립트 및 주요 코드의 주석은 **한국어**로 작성합니다.
- **Git**: 작업 전 항상 `git status`로 현재 상태를 확인합니다.
- **대용량 파일**: 50MB 이상 파일은 `.gitignore`에 추가하거나 Git LFS 사용을 제안합니다.

## 3. Communication Style
- **언어**: 기본 **한국어** 사용.
- **용어**: 기술 용어는 영어 원문 유지 (e.g., LoRA, VLM, Fine-tuning, Sim2Real).
- **설명**: 복잡한 개념은 구체적 예시나 다이어그램(Mermaid 등)을 활용합니다.

## 4. Project Context
- **작업 디렉토리**: `/home/billy/26kp/motion-vla`
- **주요 관심 분야**:
    - Vision-Language-Action (VLA) Models
    - Motion Generation & Control
    - Sim2Real Transfer
- **문서 위치**: 모든 문서는 `docs/` 디렉토리에 체계적으로 정리합니다.

## 5. Workflow Preferences
- **프로세스**: 논문/자료 분석 → 표 작성 → 코드 검증 → 리포트 작성.
- **논증 구조**: 주장(Claim) → 근거(Reasoning) → 데이터(Data).
- **팩트 체크**: 불확실한 내용은 추측하지 않고 원본 소스 확인 후 답변합니다.

## 6. Documentation Naming Rules
- **카테고리**: `setup`, `architecture`, `simulation`, `meeting`, `experiments` 등으로 분류.
- **파일명**: `카테고리.md` (메인) 또는 `카테고리_세부내용.md` (예: `setup_environment.md`).
- **날짜**: 문서 내용 내부에 기록 (파일명에는 지양, 필요시만 사용).
