# Motion VLA Data Schema

## HDF5 Structure
학습 효율성을 위해 **HDF5** 포맷을 기본으로 사용합니다. (Robomimic / OpenX 호환성 고려)

```text
dataset.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── agentview_rgb     (T, H, W, 3)  # uint8
│   │   │   ├── eye_in_hand_rgb   (T, H, W, 3)  # uint8
│   │   │   ├── joint_positions   (T, D)        # float32
│   │   │   └── ee_pose           (T, 7)        # float32 (pos + quat)
│   │   ├── actions/              (T, D)        # float32 (Next joint pos or delta)
│   │   ├── rewards/              (T,)
│   │   ├── dones/                (T,)
│   │   └── language_instruction  (String Attribute)
│   ├── demo_1/
│   └── ...
└── mask/
    ├── train/                    (List of demo keys)
    └── valid/                    (List of demo keys)
```

## Custom Attributes for Motion VLA
신규 태스크를 위해 각 데모(Group)에 다음 Attribute나 Dataset이 추가됩니다.

1.  **`motion_style` (Attribute)**:
    - Type: `String`
    - Values: `"careful"`, `"fast"`, `"jerky"`, `"normal"`
    - Usage: Task 2 (Adverb Control) 학습 시 Condition으로 사용.

2.  **`correction_label` (Dataset)**:
    - Type: `(T,) int8`
    - Values: `0` (Normal), `1` (Correction Start), `2` (Correction End)
    - Usage: Task 1 (Trajectory Correction)에서 수정 구간을 식별하기 위함.
