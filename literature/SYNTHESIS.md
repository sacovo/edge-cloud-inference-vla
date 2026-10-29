# Synthesis: Slow Remote Model → Latent → Small Fast On-Robot Policy

**A rapid scoping review, 2026-10-27.** The protocol is in [`PROTOCOL.md`](PROTOCOL.md). Per-theme annotated bibliographies are in [`themes/`](themes/), and all 147 deduplicated references are in [`references.bib`](references.bib). Every arXiv ID was checked against the arXiv API by `scripts/verify_bib.py`.

> **Read-scope caveat.** Many 2025–26 entries were read via abstract or an automated full-text summary. Numbers in this file are copied from the theme files, which give a locator for each. Re-check against the PDF before quoting any number in a paper. Spot-checked by hand against full text: AsyncVLA, CloudEdgeVLA, VLA-ULAP.

---

## 1. One-paragraph answer

The architecture you describe exists and is getting active work in 2025–26. **Nearly all of that work is on arms or ground robots with desktop GPUs or 30 W Jetsons, slow models of 1–8B, fixed or bounded synthetic delays, and a single camera plus proprioception on the fast side.**

The closest systems are:
- **AsyncVLA** (8.26B on a workstation at ≤5 Hz → 76M edge adapter on a Jetson Orin, WiFi, delays up to 6 s; [01 §4.22](themes/01-dual-system-vla.md))
- **CloudEdgeVLA** (OpenVLA-7B hidden states → light edge head with fresh vision; 63.8–78.0% LIBERO success with a 40-step delay window; [02](themes/02-latency-async-edge-cloud.md))
- **Figure Helix** (7B at 7–9 Hz → 80M at 200 Hz via a single latent vector, both onboard; [01](themes/01-dual-system-vla.md))

Nobody has yet done any of these:
1. a **size sweep of the fast model** conditioned on a VLM latent
2. **explicit latent-age conditioning** under realistic jitter and packet loss
3. a fast policy fusing **IMU + camera + torque + a remote latent**
4. any of this on a **drone or other battery- and weight-limited platform**
5. a ~27B slow model
6. compression of the **cloud→robot (downlink)** latent

Those gaps line up with your research question almost one-to-one.

---

## 2. Evidence map (by research question)

### RQ1: Slow/fast architectures and the interface between them ([01](themes/01-dual-system-vla.md))

| Interface type | Examples | Payload | Notes |
|---|---|---|---|
| Continuous latent vector or token(s) | Helix, HiRT, LCB, OpenHelix, DP-VLA, DuoCore-FS, AsyncVLA, CloudEdgeVLA | 1 vector to 8×1024 (AsyncVLA) or L×D hidden states | Most compact, closest to the proposed design |
| Language subtask | Hi Robot, π0.5, Gemini Robotics 1.5 | Text | Interpretable and very low bandwidth, but coarse. Low level still needs a capable (≥ hundreds of M) policy |
| Geometric intermediate (2D path, waypoints) | HAMSTER, AsyncShield (Theme 2) | ~tens of points | Can be **re-projected into the current frame**, which is a cheap fix for staleness (AsyncShield) |
| Action candidates / coarse actions | RoboDual, Hume, Libra-VLA, RTC-style chunks | Action chunk | Fast policy refines or corrects (A2C2, EXPO-FT) |
| No link at all (local predictor replaces calls) | VLA-ULAP (7.4M local) | none | Removes 45–77% of cloud calls. A useful **lower-bound baseline** |

Rates in practice: the slow model runs at 1–10 Hz and the fast model at 8–200 Hz. Where it was ablated, FiS-VLA found a slow:fast ratio of 1:4 best among 1:1 to 1:8. **Caution signal:** OpenHelix found its MLLM latent barely changes with visual input, so calling the slow model every 60 steps did not hurt. On many benchmarks the latent may carry only task identity. That makes a benchmark look easy for a small fast model and would inflate any "how small can it be" conclusion (see §4, threat T1).

### RQ2: Latency, asynchrony, staleness ([02](themes/02-latency-async-edge-cloud.md))

Families of staleness compensation (details in Theme 2 §3):
1. **Delay augmentation in training.** Train with random or fixed offsets between slow input and fast step. Used by Helix, RoboDual, DuoCore-FS, AsyncVLA, Agarwal/Parkour locomotion. This is the universal baseline.
2. **Action-prefix conditioning / inpainting.** RTC, Training-time RTC, REMAC, Legato, Action ControlNet.
3. **Forecasting.** Roll the state or observation forward to when the action will apply: VLASH, FutureRTC.
4. **Fast correction heads.** A small per-step network on fresh observations over a stale chunk or latent: A2C2, Real-Time EXPO-FT, CloudEdgeVLA. **The independent comparison (Agouzoul 2026) ranks A2C2-style correction best at medium-to-high delay.** This is the pattern closest to "small local model + stale cloud embedding".
5. **Geometric re-projection** of stale waypoints with the robot's own pose history: AsyncShield.
6. **Explicit time encoding.** Chronos (time-RoPE), LAG-Fusion (staleness weights), CfC/S5 (Δt as input). These have appeared only in 2026 preprints, and none is used on the cloud latent.

Network realism: only AsyncShield (heavy-tail spikes, 15% loss, 2 s outages), SPO (±30 ms jitter) and CALF do more than fixed delays. CALF reports that **jitter and loss hurt more than constant latency**. Disconnection fallback is essentially unevaluated.

### RQ3: How small can the onboard model be? ([03](themes/03-small-onboard-policies.md))

Demonstrated floors by capability (Theme 3 §6):

| Capability | Smallest shown | Hardware / power |
|---|---|---|
| Quadrotor inner-loop control | 2,084-param GRU (RAPTOR) | Flight-controller MCU |
| Stabilization / reactive control | MLP 2×20 to 2×64 at 100–500 Hz | STM32 Cortex-M4 |
| Visual reactive navigation | 6.4 kB CNN, 1.5 M MACs, 160 fps | GAP8, **34 mW** |
| Agile racing control (perception factored out) | MLP 2×128 | Jetson TX2 CPU |
| Manipulation diffusion policy | 2.65M (pruned) | Orin NX, 38 ms |
| Generalist manipulation transformer | 27M (Octo-S) | — |
| Language-guided drone flight, fully onboard | 86M semantic model + small CNN policy | Orin Nano |
| Drone VLA with task results | 0.45B: 46.6% vs 84.1% at 4B (Exp2VLA) | 4B does not fit an Orin NX |

**Interpretation.** Control fits in 10³–10⁴ parameters, and visual reactivity fits in kilobytes. Semantics has not been shown on board below ~86M, and VLAs below ~0.5B lose heavily. Running a VLA on board costs 30–50 W for 6–10 Hz. So the semantic part is the one worth offloading. The open question is whether a **remote embedding can stand in for the onboard semantic module**, pushing the onboard model down toward the 1–10M "visuomotor + control" regime. The best single data points so far are RoboDual (20M fast model on 7B latent) and VLA-ULAP (7.4M, no latent). No size curve exists.

### RQ4: Multi-rate sensor fusion ([04](themes/04-multirate-sensor-fusion.md))

- **Dominant pattern:** each modality has its own encoder running at its native rate. The fast loop holds the latest slow feature, and training randomizes update periods and latency (MResT 5/20/75 Hz; legged policies with depth at 10 Hz and the policy at 50 Hz). This is the natural default for IMU 100 Hz / torque 60 Hz / camera 20 Hz / latent 5 Hz.
- **Rate matters:**
  - MResT: single-rate 4.2% vs multi-rate 73.6% on dynamic tasks.
  - FoAR and LAG-Fusion: performance drops when the force rate drops.
  - Counter-signal (EventOA): high-rate sensors give no benefit when model latency dominates. Sensor rate and model size have to be designed together.
- **Force/torque** works best fused **late** (near the action head), compressed to one history token (ForceVLA, TA-VLA).
- **Modality collapse:** policies ignore a sparse modality unless forced (FACTR curriculum, ImplicitRDP). Expect the fast model to over-rely on whichever of {cloud latent, camera} is most predictive and ignore IMU/torque.
- **Taxonomy of 11 alignment approaches**, A–K in Theme 4 §5. Continuous-time recurrent models (CfC, S5) take Δt natively and run in O(1) per update, which suits MCUs. They have no multi-modal robot evaluation yet.

### RQ5: Latent-conditioned shrinking and testbeds ([05](themes/05-latent-conditioning-and-testbeds.md))

- **Closest analogues for "the latent lets the student be small":**
  - RMA: an 8-D extrinsics latent at 10 Hz feeds a 100 Hz base policy. This is structurally the same rate split, with a privileged latent instead of a VLM latent.
  - HiRT: 7B latent → 35M/150M.
  - RoboDual: 7B → 20M.
  - BC-Z and RT-1: FiLM conditioning, 35M total.
- **Only explicit student-size sweep:** Rusu et al. 2016 (Atari). A 62k-param student keeps 84% of teacher score. Nothing comparable exists for VLM-conditioned robot policies.
- **Confound:** VLA-Adapter (0.5B backbone) and DEM (no VLM) reach near-SOTA on LIBERO/RoboCasa. On those benchmarks a big slow model may not be needed at all, so an experiment must show the 27B latent adds value (threat T1).
- **Testbeds:**
  - Built-in per-sensor rates: robosuite (`sampling_rate`, `delayer`, wrist F/T; underlies LIBERO and RoboCasa) and Isaac Lab (`update_period`, Delayed PD actuator, IMU).
  - Drones: Aerial Gym (Isaac Gym, IMU noise model, BSD-3) and gym-pybullet-drones (Crazyflie, Betaflight SITL).
  - Network delay and jitter on the latent must be injected in user code in every simulator.

---

## 3. Gap list (deduplicated across themes)

| # | Gap | Evidence of absence | Priority for this project |
|---|---|---|---|
| G1 | **No fast-policy size sweep under a fixed VLM latent.** How performance scales with fast-model params × latent dim × latent rate | 01 §5.3, 03 §5.3, 05 §6.1 | **Core RQ** |
| G2 | **No explicit latent-age input under realistic jitter, loss and outage.** Systems use fixed or bounded integer delays; CloudEdgeVLA deliberately omits age | 01 §5.2, 02 §5.1/5.4, 04 §6.3 | High; cheap to test |
| G3 | **No IMU + torque + camera + remote latent fusion** in one fast policy; no IMU-rate sweep | 04 §6.1–6.2, 02 §5.6 | High |
| G4 | **No embedded or aerial edge for latent-conditioned dual systems.** Drones either run everything offboard (RaceVLA, Exp2VLA) or everything onboard (SINGER) | 01 §5.5, 02 §5.3, 03 §5.1 | High (drone variant) |
| G5 | **Downlink (cloud→robot) latent compression and rate limits unstudied.** Only uplink compression exists (ComVLA, Nakanoya) | 05 §6.2, 02 §5.2 | Medium |
| G6 | **No disconnection/fallback behaviour**, and no robustness to a dropped latent | 02 §5.5, 05 §6.4 | Medium; safety-relevant |
| G7 | **Energy per action rarely measured** outside nano-drone work | 03 §5.2, 01 §5.6 | Medium; needed for the "battery-limited" claim |
| G8 | **Slow models are 1–8B, none ~27B.** Unknown whether a larger VLM gives a richer or more transferable latent | 01 §5.1, 05 §6.6 | Medium |
| G9 | **What does the latent actually encode?** OpenHelix suggests task identity only | 01 §5.4 | High; determines whether results generalize |

---

## 4. Threats to validity for the planned experiments

- **T1, benchmark too easy / latent too static.** If the latent encodes only task identity (OpenHelix) and small policies already solve the benchmark (VLA-Adapter, DEM on LIBERO), a size sweep shows "tiny is fine" for the wrong reason. *Mitigation:* include a no-latent baseline (VLA-ULAP-style), a task-ID-only baseline (one-hot or language embedding), and tasks that need scene-level reasoning mid-episode (instruction changes, distractor shifts, dynamic objects).
- **T2, modality collapse.** The fast model may ignore IMU/torque when the latent plus camera is enough. *Mitigation:* tasks where high-rate signals are necessary (contact, disturbance rejection, agile flight); modality dropout or FACTR-style curriculum; report per-modality ablations.
- **T3, synthetic delay ≠ network.** Fixed integer delays overstate robustness (CALF). *Mitigation:* use delay distributions recorded from a real WiFi/5G link (heavy tail, loss, outages) as in AsyncShield.
- **T4, size confounded with pretraining** (03 §5.4). *Mitigation:* the same architecture family and training recipe across sizes; report pretrained vs from-scratch separately.

---

## 5. Implications: a candidate experimental programme

Ordered so each step is cheap and de-risks the next.

1. **E0: Reproduce a reference point.**
   - Take a LIBERO (robosuite) setup.
   - Precompute embeddings from a ~27B VLM (e.g. Gemma-3-27B or Qwen-VL-32B class) per frame offline, so VLM cost is decoupled from the sweep.
   - Train small fast policies conditioned on them at a fixed latent period.
   - Compare with the published RoboDual, HiRT and CloudEdgeVLA numbers.
2. **E1: Size × latent sweep (G1, G9), now [EXP-0001](../experiments/EXP-0001-latent-taskid-vs-state/README.md).**
   - Hypothesis: a large fast policy uses z only as a task ID. Small or **blind** fast policies need z to carry scene state, so a task-ID substitute fails below some size, and staleness starts to matter.
   - Grid: local vision {none, tiny, full} × head size {~10k … 50M} × conditioning {none, task-ID, z at episode start, z refreshed, delayed oracle state, fresh oracle state} × latent period {1, 4, 10, 20 steps}.
   - Tasks: manipulation (robosuite/LIBERO), graded by scene dynamics (static, contact, one-off event, continuous motion).
   - Anchors: RMA (8-D), BC-Z (512-D), RoboDual (20M), HiRT (35M); blind-policy precedent in legged locomotion (Agarwal 2022, Robot Parkour).
3. **E2: Staleness and network realism (G2, G6).**
   - Variants: (a) delay augmentation only; (b) plus an explicit latent-age input (Δt embedding, CfC-style); (c) plus an A2C2-style correction head.
   - Conditions: recorded jitter and loss traces, and outages, testing fallback when the latent is missing.
4. **E3: Multi-rate fusion (G3).**
   - Move to Isaac Lab, which has per-sensor `update_period` and IMU.
   - Inputs: IMU 100 Hz, torque 60 Hz, camera 20 Hz and latent 5 Hz, through native-rate encoders + hold (Theme 4 approach A/B) vs a continuous-time recurrent core (approach I).
   - Ablate each modality's rate.
5. **E4 (optional, later): Drone variant (G4, G7).** Manipulators are the primary platform; drones are one possible deployment target.
   - Aerial Gym or gym-pybullet-drones.
   - Fast policy sized for Orin Nano and for MCU (GAP8/STM32 class), conditioned on the remote VLM latent.
   - Measure energy per action (J) and flight-relevant latency.
6. **E5 (optional): Downlink compression (G5).** Quantize or sparsify the latent and measure the bits/s vs success curve.

---

## 6. Must-read shortlist (one per role)

| Role | Paper | Why |
|---|---|---|
| Closest system (ground robot, WiFi, Jetson) | AsyncVLA, Hirose et al. 2026, arXiv:2602.13476 | Nearly the exact architecture |
| Closest system (manipulation, delay sweep) | CloudEdgeVLA, Peng et al. 2026, arXiv:2608.00569 | Baseline to extend with age input and multi-rate sensing |
| Industrial template | Figure Helix 2025 (blog) | 7B→80M latent, train-time offset matched to deployed latency |
| Smallest fast model on a VLM latent | RoboDual, Bu et al. 2024, arXiv:2410.08001 | 20M specialist |
| Correction-head pattern | A2C2 (arXiv:2509.23224), with Agouzoul 2026 (arXiv:2605.08168) | Best at mid/high delay in an independent comparison |
| Rate-split precedent | RMA, Kumar et al. 2021 | 8-D latent at 10 Hz feeding a 100 Hz policy |
| Multi-rate fusion | MResT, Saxena et al. 2023 | Strongest single-rate vs multi-rate evidence |
| Drone size floor | RAPTOR 2025; Tiny-PULP-Dronets 2022 | What fits on an MCU |
| Testbed docs | robosuite sensors; Isaac Lab (arXiv:2511.04831) | Per-sensor rates and delay injection |

---

## 7. Limitations of this review

- **One AI-assisted screener**, no dual screening, and search mostly via web search and arXiv. Venue databases (IEEE Xplore, ACM) were not queried directly. Coverage of non-English work is unknown.
- **Preprint-heavy.** Most 2025–26 sources are Tier 2 (arXiv), and several key 2026 findings (CloudEdgeVLA, Chronos, LAG-Fusion, Agouzoul) are not peer-reviewed.
- Some numbers came from an automated full-text reader; locators are given in the theme files. Theme 4 flags the Chronos Fig. 3 per-condition numbers as possibly inconsistent.
- The 2026 literature is moving fast. Re-run the searches in [`SEARCH_LOG.md`](SEARCH_LOG.md) before writing a paper.

*AI disclosure: search, screening, extraction and synthesis were done with Claude (Anthropic) agents through Claude Code and the ARS deep-research plugin, under the author's direction. References were checked for existence by script. Content claims need author verification.*

## 8. Addendum (2026-10-29): Theme 7, learned filtering and Kalman-inspired mechanisms

Added after the first EXP-0001 A1-teacher results; see `themes/07-learned-filtering-in-policies.md` (50 sources, all verified).

- **Each mechanism has a precedent; the combination has none.** Age-indexed plans with variable update periods ([jenelten2024dtc] Fig. 7A), zero-initialised residuals on chunked and VLA bases ([ankile2025imitation], [yuan2025policy]), Kalman-structured learned cells ([becker2019recurrent], [shaj2023multi] Eq. 6), the χ² innovation test ([willsky1976survey] Sec. VII) and innovation-triggered communication ([trimpe2017distributed] Eq. 13, 19). None of the included sources combines them for a tiny policy tracking an ageing plan from a learned slow model.
- **New gap G10: innovation-gated trust in a remote plan.** Learned trust signals come from the policy's own outputs ([agia2024unpacking], [xu2025faildetect], [yang2026tud]). The residual between plan-implied and measured force/torque is used for compliance ([ding2026cwwam]) or to gate learning ([gao2026impact]), never to down-weight a stale slow-model plan or trigger a re-query.
- **New gap G11: residual feedback around an ageing plan.** In every residual-policy paper the base action is fresh (Theme 7, Sec. 1); learned delay handling excludes out-of-order arrival and jitter, and explicit forward prediction hurts under random delay ([wang2024addressing] Sec. 5).
- **Implication:** these gaps define the draft EXP-0001 amendment A1-teacher v2 (`experiments/EXP-0001-latent-taskid-vs-state/DRAFT-amendment-A1-teacher-v2.md`); Theme 7 Sec. 3 lists the changes the literature suggests.

