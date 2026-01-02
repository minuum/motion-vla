<!-- id: task-sprint-1 -->
# 2시간 스프린트 계획 (Motion VLA Setup)

이 문서는 프로젝트 초기 2시간 동안 수행할 수 있는 "환경 설정 및 기반 마련" 작업을 정의합니다.

## 1. 프로젝트 초기화 (0-30분) <!-- id: step-init -->
- [x] Git Repository 초기화 (`git init`) <!-- id: step-git-init -->
- [x] 기본 디렉토리 구조 생성 (`docs`, `src`, `scripts`, `tests`) <!-- id: step-dirs -->
- [x] `.gitignore` 및 `README.md` 작성 <!-- id: step-basic-files -->
- [x] 프로젝트 룰 문서화 (`docs/project_rules.md`) <!-- id: step-rules -->
- [ ] Python 가상환경 설정 및 의존성 관리 파일 (`requirements.txt` or `pyproject.toml`) 생성 <!-- id: step-env -->

## 2. 문서 체계화 (30-60분) <!-- id: step-docs -->
- [ ] `docs/architecture.md`: 전체 시스템 아키텍처 초안 작성 (Mermaid 다이어그램 활용) <!-- id: step-arch-doc -->
- [ ] `docs/setup.md`: 개발 환경 셋업 가이드 작성 <!-- id: step-setup-doc -->
- [ ] `docs/research_plan.md`: 향후 연구/개발 방향성 정리 <!-- id: step-research-doc -->

## 3. 기본 코드 스켈레톤 작성 (60-90분) <!-- id: step-skeleton -->
- [ ] `src/motion_vla/__init__.py` 패키지 구조 생성 <!-- id: step-pkg-init -->
- [ ] `scripts/test_env.py`: CUDA/Torch 등 환경 검증 스크립트 작성 <!-- id: step-test-script -->

## 4. 검토 및 계획 수정 (90-120분) <!-- id: step-review -->
- [ ] 초기 셋업 점검 <!-- id: step-check -->
- [ ] 다음 스프린트 계획 수립 <!-- id: step-next-plan -->
