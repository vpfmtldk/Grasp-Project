# 진행 상황 / 인계 메모 (2026-09-30 저녁 기준)

저가형 SO-101 + AmazingHand에서 깊이 센서 없이 RGB 단안으로 파지를 검출·실행하는 프로젝트입니다.
정리된 결과는 **[RESULTS.md](RESULTS.md)**, 이 파일은 작업용 인계 메모입니다.

## 1. 파지 모델 (Cornell 사전학습) — 완료

**사용 모델은 GR-ConvNet입니다.** GG-CNN(포크 원본 모델)은 초기에 비교용으로만 학습했습니다.

| 모델 | 입력 | 폴더 | 검증 IoU (89장, 증강 없음) |
|---|---|---|---|
| GG-CNN | 깊이 | `output/models/260901_1924_training_example/` epoch 29 | 0.798 |
| GG-CNN | RGB | `output/models/260902_1343_ggcnn_rgb1_d0/` epoch 22 | 0.854 |
| **GR-ConvNet** | **RGB** | `output/models/260903_1112_grconvnet_rgb1_d0/` **epoch 13** | **0.955** |

- **배포 모델 = `output/models/final_grconvnet_rgb1_d0/weights.pt`** (Cornell 95%로 학습, epoch 22, 폴더의 `MODEL.md` 참고).
  `eval_ggcnn.load_network`가 가중치 키로 구조를 자동 판별합니다.
- 5-fold 교차검증: 0.944 / 0.831 / 0.843 / 0.944 / 0.955 → **평균 0.903, 표준편차 0.054**.
  단일 분할 0.955는 낙관적이므로 평균 ± 표준편차를 인용합니다.
- RGB만으로 깊이 입력과 비슷하거나 더 좋습니다 → "깊이 센서 불필요" 주장의 근거.
- 팀원 이미지 103장으로 파인튜닝을 시도했으나 IoU가 0.21을 넘지 못했습니다(라벨 문제로 추정, 미해결).

## 2. 비전 파이프라인 — 완료

```
카메라 프레임 -> predict_grasp.py -> (u, v, theta, width, quality)
             -> pixel_to_world.py -> 로봇 기준 xyz + yaw + 폭
```

카메라 캘리브레이션(`camera_calib.py`, `extrinsic_click.py`) 결과는 `output/cam.json`.
실험 2(학습 vs 기하학 각도)는 RESULTS.md 2장.

## 3. 로봇 드라이버 — 동작함

`robot/robot_control.py` (`Config`, `SO101`, `AmazingHand`)

| | 포트 | 서보 | 비고 |
|---|---|---|---|
| 팔 | COM9 | STS3215, id 1~5 (pan, lift, elbow, wrist_flex, wrist_roll) | 1M baud, protocol_end 0 |
| 손 | COM8 | SCS 계열, id 1~8 (검지 1·2 / 중지 3·4 / 약지 5·6 / 엄지 7·8) | protocol_end 1 |

알아둘 것:
- **scservo_sdk 바이트 순서가 전역 변수**입니다. 팔과 손을 한 프로세스에서 쓰면 서로 통신을 깨뜨리므로
  `FeetechBus._sync_end()`로 매 통신 전에 다시 설정합니다.
- **Torque_Limit(레지스터 48)이 0으로 출하**됩니다. 연결 시 값을 씁니다.
- 토크를 끈 동안에도 Goal_Position이 남아 있어, 켜기 전에 현재 위치로 맞춥니다(급발진 방지).
- **관찰된 증상**: 내 스크립트로 wrist_roll(id5)에 토크를 켜면 id2·4·5가 곧바로 응답을 끊고(이후 읽으면 토크 꺼짐),
  어깨(id2)가 풀려 팔이 처질 수 있습니다. 토크 한계 200, 다른 서보 무부하, 케이블 교체 후에도 같았습니다(3회).
  **원인은 확정하지 못했습니다.** "id5 서보 고장"은 추정일 뿐이고, 전원/배선 구간, 서보 설정(모드·보호 값),
  내 테스트 절차 자체의 문제일 수도 있습니다. 다른 도구로는 id5가 동작한다는 보고가 있어 **재확인이 필요**합니다.
  안전 때문에 `Config.arm_disabled_ids = [5]`로 기본 비활성이고, 켜려면 `--with-roll` 또는 목록에서 5를 뺍니다.
  확인은 `python -m robot.roll_check`(단계별, 이상 시 즉시 중단)와 읽기 전용 레지스터 비교로.
- 각도 기준은 **팀원(LeRobot) 기준**입니다: 모든 관절 `home_steps = 2048`, 팀원 실측 소프트 리밋, 팀원 파킹 자세.

## 4. 실물 캘리브레이션 · 파지 — 동작함

**캘리브레이션 = 팀원 결과** (`robot/calib/handeye_teammate.json`, 22점, LOO 1.23°, 손끝 평균 12 mm).
`robot/team_fk.py`가 손끝을 25 mm 띄워 테이블을 누르지 않게 합니다. 우리 자체 수집 데이터는 옛 각도
기준이라 `*_OLDFRAME`으로 보관만 합니다.

```
# 한 번 (확인 창: READY -> PLAN -> HOVER)
python grasp_and_execute.py --backend real --network output/models/final_grconvnet_rgb1_d0/weights.pt --grip power
# 반복 측정 (RESULT에서 s/f 입력, CSV 기록)
python grasp_and_execute.py --backend real --network ... --grip power --real-trials 10
# 자동 모드 (물체만 옮기면 알아서 잡고 카메라로 판정)
python grasp_and_execute.py --backend real --network ... --grip power --auto
# 중단 후 복귀 (손 펴기 -> 수직으로 들기 -> 파킹)
python -m robot.recover
```

- 물체는 **세로로**(손목이 못 돌아서), **초록 상자 가운데 쪽**에 놓습니다. 테이블에는 체커보드 등 무늬를 두지 않습니다.
- 결과: 자동 모드 3/5, 오늘 전체 6/15. 가운데는 성공, 상자 좌우 가장자리는 실패가 많습니다. 자세한 표는 RESULTS.md 8장.
- 기록: `output/grasp_runs/trials_*.csv`, `auto_*.csv` (사진은 삭제함).

## 5. 시뮬레이션

| 폴더 | 내용 |
|---|---|
| `robot/sim/` | 공식 AmazingHand(운동학 손가락)로 전체 파이프라인 데모. RESULTS.md 5장 |
| `robot/rl/` | 힘이 전달되는 AmazingHand + 355 ml 캔 잔차 PPO. RESULTS.md 6장 |

강화학습 최종 모델: `output/rl/ppo_can/` (서 있는 캔 79%, 누운 캔 100%).
이전 시도 v1~v5는 `output/rl/ppo_can_v*`에 보관.

```
python -m robot.rl.view_hand                        # 장면을 뷰어로 (슬라이더로 관절 조작)
python -m robot.rl.eval_policy --view --mode mixed  # 학습된 정책 재생
```

## 6. 다음 단계

1. **wrist_roll(id5) 서보 교체** → `roll_check.py` → 파지 각도대로 손목 회전(물체 방향 제약 해제).
2. 상자 가장자리 오차: 가장자리 실패 위치로 오프셋 보정표 만들기 또는 팀원 캘리브레이션 점 보강(`--append`).
3. 물체별(작은 물체 3~5 cm, 캔) 성공률 20회 이상 측정 → 이 프로젝트의 핵심 숫자.
4. 강화학습: 가득 찬 캔(약 370 g), 다른 물체, 실물 적용.

## 7. 인프라 메모

- `run_train.ps1` / `run_cv.ps1`: `--save-folder` + `ckpt_last.pt`로 자동 재개. 중단되면 같은 명령을 다시 실행.
- 이 노트북은 **Modern Standby**라 유휴 상태에서 백그라운드 작업이 20~60분 뒤 죽을 수 있습니다.
  긴 작업은 전원 연결, 절전 해제 상태에서 돌립니다.
- 카메라는 2번(USB, `cv2.CAP_DSHOW` 필요). 한글 경로 이미지는 `cv2.imread` 대신 `np.fromfile` + `cv2.imdecode`.
