# 저가형 단안 카메라 파지 로봇 — SO-101 + AmazingHand

RGB 카메라 **한 대**로 책상 위 물체를 **어디서, 몇 도로** 집을지 예측하고,
저가형 **SO-101 로봇팔 + Pollen AmazingHand**로 실제로 집는 포트폴리오 프로젝트입니다.
깊이 센서·모션캡처 없이 시중 부품만 씁니다.

```
카메라 프레임 ─▶ GR-ConvNet ─▶ (u, v, θ, 폭, 품질)          이미지 속 파지
            └─▶ 픽셀 → 관절각 캘리브레이션 ─▶ 팔 5관절 목표각
            └─▶ 접근 ─▶ 손 닫기 ─▶ 들어 올리기
```

> **사용 모델은 GR-ConvNet입니다.** 저장소는 [GG-CNN](https://github.com/dougsm/ggcnn)(Morrison et al., RSS 2018)
> 코드를 포크해 시작했고, 그 위에 GR-ConvNet(`models/grconvnet.py`)을 추가했습니다.
> GG-CNN은 초기에 비교용으로만 학습했습니다(RGB 입력 IoU 0.854 vs GR-ConvNet 0.955).
> `train_ggcnn.py`, `eval_ggcnn.py` 같은 파일 이름은 포크 원본 이름을 그대로 쓴 것이고, GR-ConvNet도 이 스크립트로 학습합니다.
> 원본 GG-CNN 설명은 [`docs/README_ggcnn_original.md`](docs/README_ggcnn_original.md)에 있습니다.

## 결과 요약

| 항목 | 결과 |
|---|---|
| 파지 모델 (GR-ConvNet, RGB만) | Cornell 5-fold 교차검증 IoU **0.903 ± 0.054** |
| 학습 모델 vs 기하학(PCA) 파지 각도 | 30° 이내 일치 **84% vs 67%** |
| 카메라 캘리브레이션 | 재투영 오차 0.22~0.27 px, 테이블 평면 복원 평균 **0.08 mm** |
| 시뮬레이션 전체 파이프라인 | 카메라 → 모델 → 3D → IK → 파지, **4/4** |
| MuJoCo 강화학습 (355 ml 캔) | 서 있는 캔 **79%** (스크립트 42%), 누운 캔 **100%** |
| **실물 파지** (SO-101 + AmazingHand) | 자동 모드 **3/5**, 오늘 전체 **6/15** — 물체 중심 보정 + 팀원 캘리브레이션 |

자세한 내용과 그림은 **[RESULTS.md](RESULTS.md)**, 현재 진행 상황과 남은 일은
**[HANDOFF.md](HANDOFF.md)**에 있습니다.

<p align="center">
  <img src="docs/figures/rl_can_upright.gif" width="45%"> <img src="docs/figures/rl_can_lying.gif" width="45%"><br>
  <em>강화학습 정책이 355 ml 캔을 쥐는 모습 (왼쪽: 서 있는 캔, 오른쪽: 누운 캔). 손가락 마찰만으로 들고 있습니다.</em>
</p>

## 새 컴퓨터에서 시작하기

`pretrained/`에 학습된 모델이 들어 있어서, 저장소를 받은 뒤 바로 예측과 시뮬레이션 재생까지 됩니다.

| 폴더 | 내용 |
|---|---|
| `pretrained/grconvnet_rgb/` | 배포 GR-ConvNet 가중치(`weights.pt`, RGB, Cornell 학습), `MODEL.md` |
| `pretrained/ppo_can/` | 학습된 잔차 PPO 정책(`model.zip`, `vecnormalize.pkl`) |
| `pretrained/cam.json` | 이 책상 카메라의 내부·외부 파라미터 (카메라가 다르면 `camera_calib.py`로 다시) |
| `pretrained/real_grasp_logs/` | 실물 파지 시도 기록(CSV) |

```bash
git clone https://github.com/vpfmtldk/Grasp-Project.git && cd Grasp-Project
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt            # 정확한 버전은 requirements-tested.txt (Python 3.14, Windows)

# 1) 예측 (카메라 없이 사진 한 장)
python predict_grasp.py --network pretrained/grconvnet_rgb/weights.pt --image <사진>

# 2) 강화학습 정책 재생 / 평가 (MuJoCo)
python -m robot.rl.eval_policy --model pretrained/ppo_can --view --mode mixed
python -m robot.rl.eval_policy --model pretrained/ppo_can --episodes 80 --mode upright

# 3) 시뮬레이션 장면 만들기 (robot/rl/so101_rlhand_can.xml 는 저장소에 포함돼 있음)
python robot/rl/build_can_scene.py
```

**하드웨어가 없으면 여기까지만 됩니다.** 실물 로봇은 다음이 이 로봇·이 책상에 맞춰져 있어서 그대로는 안 됩니다.

- 시리얼 포트 `COM9`(팔) · `COM8`(손), 카메라 번호 2 → `robot/robot_control.py`의 `Config`와 `--camera`
- 서보 내부 영점과 카메라 위치 → `robot/calib/handeye_teammate.json`은 **같은 팔·같은 카메라 자리**에서만 유효합니다. 다른 로봇이면 캘리브레이션을 다시 해야 합니다.
- 실물 파지: `python grasp_and_execute.py --backend real --network pretrained/grconvnet_rgb/weights.pt --grip power --auto` (자세한 사용법은 `HANDOFF.md`)

학습 데이터(Cornell 등)는 포함하지 않습니다. 재학습하려면 [Cornell](http://pr.cs.cornell.edu/grasping/rect_data/data.php)을 받아 `train_ggcnn.py`로 학습하세요.

## 설계 원칙

| 선택 | 이유 |
|---|---|
| **RGB 단안만** (깊이 카메라 없음) | 비싼 센서를 빼고, 높이는 "물체는 테이블 위에 있다"는 평면 가정으로 구합니다 |
| **고정 카메라** (eye-to-hand) | 한 번 캘리브레이션하면 세션 내내 유효합니다 |
| **SO-101 + AmazingHand** | 저가형 탁상 구성 |
| **손은 1자유도처럼** 사용 | 연구의 초점은 인식입니다. 손은 열기/닫기 명령 하나로 움직입니다 |

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `models/grconvnet.py` | GR-ConvNet 구조 (이 포크에 추가) |
| `train_ggcnn.py`, `run_train.ps1`, `run_cv.ps1` | 학습, 중단 시 자동 재개, 5-fold 교차검증 |
| `predict_grasp.py` | 이미지 → 파지 `{x, y, angle, width, quality}` |
| `pixel_to_world.py` | 파지 픽셀 → 3D 자세 (광선과 테이블 평면의 교점) |
| `grasp_geometry.py`, `compare_grasp_angles.py` | 학습 모델 vs 기하학 각도 비교 (실험 2) |
| `camera_calib.py`, `extrinsic_click.py` | 카메라 내부·외부 파라미터 캘리브레이션 |
| `robot/robot_control.py` | SO-101(STS3215) + AmazingHand(SCS) Feetech 서보 드라이버 |
| `robot/calib/` | 실물 캘리브레이션: 픽셀 → 관절각 (자동 수집, 풀기, 팀원 캘리브레이션 변환) |
| `robot/sim/` | MuJoCo: SO-101 + 공식 AmazingHand (운동학 파지 데모) |
| `robot/rl/` | MuJoCo 강화학습: 힘이 전달되는 AmazingHand + 캔 파지 (잔차 PPO) |
| `grasp_and_execute.py` | 전체 루프: 이미지 → 파지 → 관절각 → 이동 → 쥐기 → 들기 |

## 실행 예시

```bash
# 이미지 한 장에서 파지 예측
python predict_grasp.py --network output/models/final_grconvnet_rgb1_d0/weights.pt --image <사진>

# 시뮬레이션 전체 파이프라인 (운동학 파지)
python grasp_and_execute.py --backend sim --network output/models/final_grconvnet_rgb1_d0/weights.pt --trials 4

# 강화학습: 학습 / 평가 / 뷰어
python -m robot.rl.train_ppo --steps 6000000 --mode mixed
python -m robot.rl.eval_policy --episodes 80 --mode upright
python -m robot.rl.eval_policy --view --mode mixed
```

환경: Windows, Python 3.14 venv, PyTorch, MuJoCo 3.x, gymnasium, stable-baselines3.
