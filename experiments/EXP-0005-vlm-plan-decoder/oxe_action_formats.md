# Open X-Embodiment action formats vs LIBERO (EXP-0005 stage 0)

Survey of 2026-11-02 (subagent, every row from a fetched source; spot-checked by the main session: sailor / mutex / bc_z action descriptions in the TFDS `features.json`, openpi `docs/norm_stats.md` L62–69). Not a literature claim; re-check before quoting in a paper.

**Target (LIBERO, verified in the installed packages):** robosuite `OSC_POSE`, `control_delta: true`, 20 Hz; 7-d = delta pos (3) + delta rot (3, axis-angle) in [−1, 1], scaled to ≤ ±0.05 m / ±0.5 rad per step; gripper −1 open / +1 close.

## Sources

- **[MIX]** Octo `octo/data/oxe/oxe_dataset_mixes.py` @ 241fb35, `OXE_MAGIC_SOUP` L46–73.
- **[CFG]** Octo `oxe_dataset_configs.py` (`ActionEncoding` L29–38; `EEF_POS` is only a label: some datasets under it carry velocities or absolute poses).
- **[TR]** Octo `oxe_standardization_transforms.py` (function line ranges below).
- **[DU]** Octo `octo/data/utils/data_utils.py`: `binarize_gripper_actions` L296, `rel2abs_gripper_actions` L363 (+1 close / −1 open → 1 open / 0 closed), `invert_gripper_actions` L392, `relabel_actions` L396 (action[:6] = state[t+1] − state[t]). After Octo standardization: gripper 1 = open, 0 = close.
- **[SH]** OXE spreadsheet (robotics-transformer-x.github.io → `docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g`, gid 0, CSV): Robot, # Episodes, Action Space, Control Frequency.
- **[TFDS]** `steps/action` descriptions in `storage.googleapis.com/gresearch/robotics/<name>/0.1.0/features.json` (bc_z and droid: 1.0.1).
- **[LR]** `huggingface.co/datasets/lerobot/<name>/raw/main/meta/info.json` (v3.0; action names generic `motor_0..6`).

## Magic soup ([MIX] L47–72) + DROID

| Dataset (weight) | Robot [SH] | Action [TFDS] + Octo transform [TR] | Rotation | Gripper raw → Octo | Hz SH / LR | Eps SH / LR |
|---|---|---|---|---|---|---|
| fractal20220817_data (0.541) | Google Robot (mobile) | commanded EEF displacement, base frame; TR L42–65 | rpy displacement, base frame | continuous position → `rel2abs` | 3 / — | 73,499 / — |
| kuka (0.834) | Kuka iiwa | world_vector, rotation_delta (no description); TR L68–101 | unverified | `rel2abs` | 10 / — | 580,392 / — |
| bridge_dataset (1.0) | WidowX | relabelled from consecutive states (TR L27–39) | difference of Euler state angles | `binarize` | 5 / — | 25,460 / — |
| taco_play (2.0) | Franka | relative, base frame (x, y, z, yaw, pitch, roll); TR L104–126 | Euler delta, yaw-pitch-roll | TFDS −1 open / 1 close vs Octo comment "+1 = open": **conflict** | 15 / 15 | 3,242 / 3,603 |
| jaco_play (1.0) | Jaco 2 | world_vector only, rotation zero-filled (TR L129–148) | none | `rel2abs` | 10 / 10 | 976 / 1,085 |
| berkeley_cable_routing (1.0) | Franka | velocity XYZ; TR L151–166 | angular velocity about z | none (0) | 10 / 10 | 1,482 / 1,647 |
| roboturk (2.0) | Sawyer | world_vector, rotation_delta (no description); TR L169–190 | unverified | clip, invert | 10 / 10 | 2,144 / 1,995 |
| nyu_door_opening (1.0) | Hello Stretch (mobile) | velocity XYZ; TR L193–213 | angular velocity xyz | `rel2abs` | 3 / 3 | 435 / 484 |
| viola (2.0) | Franka | world_vector, rotation_delta (no description); TR L216–240 | unverified | clip, invert | 20 / — | 135 / — |
| berkeley_autolab_ur5 (2.0) | UR5 | delta XYZ; TR L243–268 | delta rpy | 1 close / −1 open / 0 no change → `rel2abs` | 5 / 5 | 896 / 1,000 |
| toto (1.0) | Franka | SH "joint position" vs TFDS EEF-delta fields: **conflict**; TR L271–284 | unverified | not inverted | 30 / 30 | 901 / 1,003 |
| language_table (0.1) | xArm | 2D delta setpoint, zero-padded (TR L287–310) | none | fixed open | 10 / — | 442,226 / — |
| stanford_hydra (2.0) | Franka | 3 pos delta, 3 Euler delta, 1 close; TR L359–379 | Euler delta | invert | 10 / 10 | 550 / 570 |
| austin_buds (1.0) | Franka | 6 EEF delta pose, 1 gripper position; TR L382–397 | delta, representation unverified | clip, invert | **20 / 5** | 50 / 50 |
| nyu_franka_play (3.0) | Franka | 15-d (joint vel + EE delta xyz + rpy + gripper + terminate), Octo keeps EE part (TR L400–419) | rpy delta | clip | **3 / 5** | 456 / 456 |
| furniture_bench (0.1) | Franka | pos + quaternion velocities; TR L433–454 | quat velocity → Euler (meaning unclear) | clip, invert | 10 / — | 5,100 / — |
| ucsd_kitchen (2.0) | xArm | EEF pos + orientation (delta not stated); TR L468–471 | unverified | none | **2 / 5** | 150 / 150 |
| austin_sailor (1.0) | Franka | 3 ee relative pos, 3 ee relative rotation, 1 gripper; TR L487–502 | relative, representation unverified | clip, invert | 20 / 20 | 250 / 240 |
| austin_sirius (1.0) | Franka | same as sailor; TR L505–520 | same | clip, invert | 20 / 20 | 600 / 559 |
| bc_z (0.2) | Google Robot (mobile) | 3D position delta, first of 10 future steps (TR L523–545) | **axis-angle delta** | absolute closure → invert | 10 / — | 39,350 / — |
| dlr_edan_shared_control (1.0) | DLR EDAN | EEF pos + Euler zxy (delta not stated); TR L704–716 | Euler zxy | invert | 5 / 5 | 100 / 104 |
| iamlab_cmu_pickup_insert (1.0) | Franka | EEF pos + quaternion (reads absolute); TR L744–762 | quat → Euler | not inverted | 20 / 20 | 520 / 631 |
| utaustin_mutex (1.0) | Franka | 6 EEF delta pose, 1 gripper position; TR L781–796 | delta, representation unverified | clip, invert | 20 / 20 | 1,500 / 1,500 |
| berkeley_fanuc_manipulation (2.0) | Fanuc Mate | [dx, dy, dz], [droll, dpitch, dyaw]; TR L799–815 | rpy delta | from state, invert | 10 / 10 | 415 / 415 |
| cmu_stretch (1.0) | Hello Stretch (mobile) | ee pos + rot (delta not stated); TR L847–857 | unverified | binary | **10 / 5** | 135 / 135 |
| DROID (not in soup) | Franka | no Octo config; LR default action = 7 joints + gripper, `action.original` = x, y, z, roll, pitch, yaw, gripper | absolute/delta unverified | unverified | 15 / 15 | 92,233 / 95,658 |

None of the soup datasets is bimanual. No LeRobot port: fractal, kuka, bridge, viola, language_table, furniture_bench, bc_z.

## pi0.5 / LIBERO

- pi0.5 paper (arXiv 2504.16054, §IV-C): pretraining includes "the open-source OXE dataset … an extended version of the dataset used by π0"; the HTML contains no "LIBERO".
- openpi @ 215abfb: `pi05_base` assets hold norm stats for arx, arx_mobile, droid, fibocom_mobile, franka, trossen, trossen_mobile, ur5e, ur5e_dual (no libero). `docs/norm_stats.md` L62–67: pi's own action space is joint angles + gripper in [0, 1] (0 open, 1 closed), 20 Hz for Franka / UR5e; L69: DROID used joint velocities at 15 Hz. `pi05_libero` fine-tunes from `pi05_base` on `physical-intelligence/libero` with LIBERO's delta actions as is (config.py L744–766, L332).
- LIBERO is in neither the OXE spreadsheet nor Octo's mixtures.
- LeRobot / openpi LIBERO ports declare fps 10 (openpi converter `fps=10`, L49); whether frames are subsampled is unverified. Our data (`yifengzhu-hf/LIBERO-datasets`, replayed) is 20 Hz.
- FAST+ card (`physical-intelligence/fast` L16, L29): trained on 1M real robot action sequences; recommends 1 s chunks quantile-normalised to [−1, 1].

## Unverified

LIBERO's absence from pi05_base pretraining (not stated either way); which OXE subsets pi0.5 used; rotation representation / frame for austin_buds/sailor/sirius/mutex, viola, kuka, roboturk; absolute vs delta for ucsd_kitchen, cmu_stretch, dlr_edan; toto's action space; DROID cartesian frame; taco_play gripper sign; rate where SH and LR disagree.
