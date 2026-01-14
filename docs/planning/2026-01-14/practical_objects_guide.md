# Bridge Data V2 → Dobot E6 실전 활용 가이드

> **목적**: Bridge Data V2 태스크/오브젝트 중 우리가 바로 쓸 수 있는 것들  
> **기준**: Dobot E6 + 흡착 그리퍼 (φ16mm), 간단한 setup

---

## 🎯 추천 태스크 (난이도순)

### ⭐⭐⭐ 1순위: Pushing (No Gripper)

**왜 최고인가**:
- ✅ Gripper 동작 불필요 (6-DoF만)
- ✅ 실패해도 안전 (물체 떨어지지 않음)
- ✅ π0에서 많이 학습됨
- ✅ 데이터 수집 가장 빠름

**태스크 예시**:
```
1. "Push the block to the left"
2. "Push the block to the right"
3. "Push the block forward"
4. "Push the block to the center"
5. "Push the block to the edge"
```

**필요한 오브젝트** (쉬운 순):
| Object | 크기 | 무게 | 구입처 | 가격 | 추천도 |
|:---|:---|:---|:---|:---:|:---:|
| **Wooden blocks** | 5×5×5cm | 50-100g | 다이소, 문구점 | ₩5,000 | ⭐⭐⭐⭐⭐ |
| **Plastic cups** | φ7cm × 10cm | 20-50g | 다이소 | ₩3,000 | ⭐⭐⭐⭐⭐ |
| **Small boxes** | 8×6×4cm | 30-80g | 택배 박스 재활용 | 무료 | ⭐⭐⭐⭐ |
| **Toy cars** | 10×5×4cm | 80-150g | 장난감 가게 | ₩10,000 | ⭐⭐⭐⭐ |
| **Sponges** | 10×7×3cm | 10-30g | 다이소 | ₩2,000 | ⭐⭐⭐ |

**Setup**:
```
Table: 80cm × 60cm (일반 책상)
Objects: 3-5개 블록
Initial position: Random
Target: Marked zones (테이프로 표시)
```

---

### ⭐⭐ 2순위: Pick & Place (Gripper 사용)

**장점**:
- ✅ π0에서 가장 많이 학습 (foundational task)
- ✅ VLA benchmark standard
- ✅ 실용성 높음

**단점**:
- ⚠️ Gripper 동작 필요 (7-DoF)
- ⚠️ 물체 떨어뜨릴 수 있음
- ⚠️ 흡착 실패 가능 (표면에 따라)

**태스크 예시**:
```
1. "Pick up the block and place it on the plate"
2. "Move the cup to the right"
3. "Put the block in the bowl"
4. "Stack the block on top"
```

**필요한 오브젝트** (흡착 그리퍼 적합성순):

| Object | 표면 | 무게 | 흡착 성공률 | 추천도 |
|:---|:---|:---|:---:|:---:|
| **Smooth blocks** | 매끄러움 | 50-100g | 95% | ⭐⭐⭐⭐⭐ |
| **Plastic containers** | 평평함 | 30-80g | 90% | ⭐⭐⭐⭐⭐ |
| **Cardboard boxes** | 약간 거침 | 20-60g | 85% | ⭐⭐⭐⭐ |
| **Plastic bottles** | 곡면 | 50-100g | 70% | ⭐⭐⭐ |
| **Fabric items** | 부드러움 | <50g | 60% | ⭐⭐ |

**Setup**:
```
Table: Same (80×60cm)
Objects: 5-8개 아이템
Containers: 2-3개 그릇/접시
Markers: Target positions
```

---

### ⭐ 3순위: Simple Stacking

**장점**:
- ✅ Bridge Data V2 포함
- ✅ 난이도 조절 가능

**단점**:
- ⚠️ Precision 필요
- ⚠️ 실패 시 retry 어려움
- ⚠️ 물체 크기 중요

**태스크 예시**:
```
1. "Stack the red block on the blue block"
2. "Put the small box on top"
```

**필요한 오브젝트**:
```
- Flat-top blocks (10cm × 10cm)
- Stable base (무거운 블록)
- Lightweight top (가벼운 블록)
```

**추천하지 않는 이유**:
- Vacuum hysteresis issue (release delay)
- Precariously stacking 불가능 (미팅에서 제외했던 이유)

---

## 🧊 추천 오브젝트 세트

### Beginner Set (₩20,000)

**바로 시작 가능**:

```yaml
Pushing Objects:
  - Wooden blocks (5개): ₩5,000
  - Plastic cups (10개): ₩3,000
  - Small cardboard boxes (재활용): 무료

Pick & Place Objects:
  - Smooth plastic containers (5개): ₩7,000
  - Plastic plates/bowls (3개): ₩5,000
  
Total: ~₩20,000 (다이소)
```

### Advanced Set (₩50,000)

**다양한 시나리오**:

```yaml
추가 구성:
  - Toy kitchen items: ₩15,000
  - Various sized blocks: ₩10,000
  - Textured objects: ₩5,000
  
Total: ~₩50,000
```

---

## 📦 오브젝트 구매 가이드

### 다이소 추천 아이템

| 품목 | 규격 | 가격 | 용도 | 위치 |
|:---|:---|:---:|:---|:---|
| **원목 블록** | 5cm 큐브 | ₩5,000 | Pushing, Stacking | 장난감 코너 |
| **플라스틱 컵** | 다양한 사이즈 | ₩3,000 | Pushing, Pick&Place | 주방 용품 |
| **사각 용기** | 10×8×5cm | ₩7,000 | Pick&Place | 수납 용품 |
| **접시/그릇** | φ15-20cm | ₩5,000 | Target platform | 주방 용품 |
| **스펀지** | 10×7cm | ₩2,000 | Pushing (soft) | 청소 용품 |

### 온라인 구매 (쿠팡/11번가)

```
검색어: "어린이 원목 블록 세트"
가격대: ₩15,000-30,000
장점: 다양한 크기/색상
```

---

## 🎨 오브젝트 특성별 분류

### By Surface (흡착 성공률)

**Best** (95%+):
- Smooth wooden blocks
- Glossy plastic containers
- Laminated cardboard

**Good** (80-95%):
- Matte plastic
- Regular cardboard
- Painted wood

**Poor** (<80%):
- Fabric/cloth
- Foam
- Very small objects (<3cm)

### By Weight (Dobot E6 Payload: 6kg)

**Light** (<100g) - Best for Pushing:
- Plastic cups
- Cardboard boxes
- Sponges

**Medium** (100-500g) - Best for Pick&Place:
- Wooden blocks
- Plastic containers
- Toy items

**Heavy** (>500g) - Not Recommended:
- Metal objects
- Glass
- Large containers

### By Shape (Manipulation 난이도)

**Easy** (Cubic/Rectangular):
- Blocks
- Boxes
- Containers

**Medium** (Cylindrical):
- Cups
- Bottles
- Cans

**Hard** (Irregular):
- Toys with complex shapes
- Cloth/fabric
- Deformable objects

---

## 🏗️ Workspace Setup

### Minimal Setup (₩10,000)

```yaml
Table: 
  - 기존 책상 활용
  - 80cm × 60cm 이상
  - 높이 70-80cm

Surface:
  - White paper/foam board (배경)
  - Tape markers (target zones)
  
Lighting:
  - 기존 실내등
  - LED 스탠드 추가 (₩10,000)

Total: ₩10,000 (테이프 + LED)
```

### Recommended Setup (₩50,000)

```yaml
Table:
  - IKEA LINNMON (100×60cm): ₩20,000
  - 또는 기존 책상

Lighting:
  - LED ring light: ₩30,000
  - 일정한 조명 (그림자 최소화)

Surface:
  - White poster board: ₩5,000
  - Grid markers (optional)

Total: ~₩50,000
```

---

## 📋 실전 태스크 리스트

### Pushing (30가지 Variations)

**Direction-based** (10개):
```
1. "push the block left"
2. "push the block right"
3. "push the block forward"
4. "push the block backward"
5. "push the block to the center"
6. "push the block to the corner"
7. "push the block to the edge"
8. "push the block diagonally"
9. "push the block in a circle"
10. "push the block away"
```

**Object-based** (10개):
```
11. "push the red block left"
12. "push the small cup forward"
13. "push the blue box right"
14. "push the large block to the edge"
15. "push the yellow cube to the center"
...
```

**Adverb-based** (10개):
```
21. "push the block slowly"
22. "push the block gently"
23. "push the block carefully"
24. "push the block quickly"
25. "push the block firmly"
...
```

### Pick & Place (30가지)

**Simple** (10개):
```
1. "pick up the block"
2. "place the block on the plate"
3. "move the cup to the right"
4. "put the block in the bowl"
5. "place the block on the left side"
...
```

**With Targets** (10개):
```
11. "pick the red block and place it on the blue plate"
12. "move the small cup into the large bowl"
13. "put the cube in the container"
...
```

**Complex** (10개):
```
21. "pick the block and place it gently on the plate"
22. "carefully move the cup to the edge"
23. "stack the small block on the large one"
...
```

---

## 🎯 우선순위 추천

### Week 1: Pushing Only

**Why**:
- Setup 가장 간단
- 실패해도 안전
- 50 episodes면 충분

**Shopping List**:
```
✅ Wooden blocks (5개): ₩5,000
✅ Plastic cups (5개): ₩3,000
✅ LED lamp: ₩10,000
Total: ₩18,000
```

**Expected Outcome**:
- 3일 만에 working system
- >85% 성공률

### Week 2-3: Pick & Place (Optional)

**If Pushing succeeds**:

**Shopping List**:
```
✅ Smooth containers (5개): ₩7,000
✅ Plates/bowls (3개): ₩5,000
Total: ₩12,000 추가
```

**Expected Outcome**:
- 1주일 추가 작업
- >80% 성공률

---

## 💡 Pro Tips

### 오브젝트 선택 기준

1. **크기**: 5-10cm (φ16mm 흡착 그리퍼 기준)
2. **무게**: 50-200g (안정적)
3. **표면**: 매끄러울수록 좋음
4. **색상**: 다양한 색 (언어 instruction용)
5. **모양**: Cubic > Cylindrical > Irregular

### 피해야 할 것

❌ **너무 작은 것** (<3cm): 인식 어려움
❌ **너무 가벼운 것** (<20g): 밀 때 날아감
❌ **투명한 것**: Vision 어려움
❌ **Deformable**: Cloth, balloon 등
❌ **표면 거친 것**: Foam, sandpaper

### 구매 전 체크리스트

- [ ] 흡착 가능한 평평한 면이 있는가?
- [ ] Robot workspace (45cm radius) 안에 들어가는가?
- [ ] 카메라에 잘 보이는가? (색상 대비)
- [ ] 여러 개 구매 가능한가? (variation)
- [ ] 저렴한가? (실험용)

---

## ✅ 최종 추천

### 당장 시작하려면

**Pushing Set** (₩8,000):
```
다이소 방문:
- [ ] 원목 블록 5개 (₩5,000)
- [ ] 플라스틱 컵 5개 (₩3,000)

= 총 ₩8,000
```

**이것만 있으면**:
- 50 episodes 수집 가능
- Bridge Data V2 Pushing transfer
- 3일 내 working system

### 완벽하게 준비하려면

**Full Set** (₩30,000):
```
다이소:
- [ ] 원목 블록 세트 (₩10,000)
- [ ] 플라스틱 용기 세트 (₩7,000)
- [ ] 컵/그릇 (₩5,000)
- [ ] LED 스탠드 (₩10,000)

= 총 ₩32,000
```

**이것으로**:
- Pushing + Pick&Place
- 다양한 variation
- Publication-ready data

---

**Bottom Line**: 
**₩8,000**만 투자하면 **바로 시작** 가능합니다! 🚀
