# Theme 3 (RQ3): How small can on-device robot policies get?

Efficient and small VLAs, compression of VLAs for edge hardware, and embedded learned controllers, with emphasis on drones.

- **Search date:** 2026-10-27 (searched up to the present)
- **Screener:** one AI agent (rapid review shortcut, see PROTOCOL.md)
- **Read scope legend:** `full_text` = arXiv HTML / PMC / PDF read (through a fetch summary or extracted text); `abstract_only` = arXiv abstract page or publisher landing page only; `search_only` = existence and key claims confirmed only through search-engine snippets of the official page (the page itself refused connection).
- **Number policy:** parameters, Hz, latency and watts are reported only when the source states them, with a locator. Otherwise "not stated". Numbers marked "≈" are read off a chart.

---

## 1. Search log

| # | Query / action | Tool | Hits used |
|---|---|---|---|
| 1 | Seed check: arXiv 2409.12514 (TinyVLA) abs + HTML, tiny-vla.github.io | WebFetch | 1 |
| 2 | Seed check: arXiv 2506.01844 (SmolVLA) abs + HTML | WebFetch | 1 |
| 3 | Seed check: arXiv 2506.10100 (EfficientVLA) abs + HTML | WebFetch | 1 |
| 4 | Seed check: ai.stanford.edu/blog/minivla (connection refused twice) → `"MiniVLA" OpenVLA smaller ai.stanford.edu blog` | WebFetch, WebSearch | 1 (search_only) |
| 5 | Seed check: DeepMind blog "Gemini Robotics On-Device" | WebFetch | 1 |
| 6 | Seed check: Swift. Nature page redirects to login → `Champion-level drone racing ... PMC` → PMC10468397 full text | WebSearch, WebFetch | 1 |
| 7 | Seed check: arXiv 2311.13081 (Learning to fly in seconds) abs + HTML | WebFetch | 1 |
| 8 | Seed check: arXiv 1805.01831 (PULP-Dronet, 64 mW) | WebFetch | 1 |
| 9 | Seed check: Chahine et al. Sci. Robot. (science.org returned 403) → `"Robust flight navigation out of distribution with liquid neural networks"` → MIT CSAIL PDF, text extracted with pdftotext | WebSearch, WebFetch, Bash | 1 |
| 10 | Seed check: arXiv 2503.02572 (RaceVLA), 2503.01378 (CognitiveDrone), 2501.05014 (UAV-VLA) abs + HTML | WebFetch | 2 (+1 excluded) |
| 11 | Seed check: arXiv 1811.08027 (Neural Lander), 1903.04628 (Sim-to-(Multi)-Real, abs + ar5iv) | WebFetch | 1 (+1 excluded) |
| 12 | Seed check: arXiv 2407.02405 (Tiny-PULP-Dronets) abs + HTML | WebFetch | 1 |
| 13 | `vision-language-action model Jetson Orin deployment quantization real-time edge arXiv 2025` | WebSearch | 2 (LiteVLA-Edge, FoldQuantVLA) |
| 14 | `BitVLA 1-bit vision-language-action model arXiv memory` → abs + HTML | WebSearch, WebFetch | 1 |
| 15 | `distilling vision-language-action model into small policy onboard drone arXiv 2025 2026` → `Exp2VLA ...` → abs + HTML | WebSearch, WebFetch | 1 |
| 16 | `nano-drone vision-based navigation GAP9 transformer foundation model onboard 2025 arXiv Crazyflie` | WebSearch | 0 (+1 excluded: GAP9Shield) |
| 17 | `end-to-end quadrotor control from pixels reinforcement learning agile flight onboard arXiv 2024 2025` → 2406.12505 abs + RSS PDF (pdftotext) | WebSearch, WebFetch, Bash | 1 |
| 18 | Octo (2405.12213 HTML), π0 (2410.24164 HTML), HPT (2409.20537 abs + HTML); `"Octo ..." Robotics: Science and Systems 2024 proceedings` for venue | WebFetch, WebSearch | 3 |
| 19 | `diffusion policy Jetson Orin real-time inference benchmark robot latency power arXiv` → 2607.12659 (EagleVLA), 2606.28529 (Speedup Paradox), 2608.03938 (8 GB bimanual) abs + HTML | WebSearch, WebFetch | 3 |
| 20 | `LightDP lightweight diffusion policy Jetson Orin NX pruning distillation arXiv` → 2508.00697 HTML | WebSearch, WebFetch | 1 |
| 21 | `RAPTOR foundation policy quadrotor control Eschmann arXiv parameters` → 2509.11481 HTML | WebSearch, WebFetch | 1 |
| 22 | arXiv 1909.11236 (Learning to Seek / tinyRL) abs + HTML; venue search | WebFetch, WebSearch | 1 |
| 23 | arXiv 2110.05113 (Learning high-speed flight in the wild) abs + HTML | WebFetch | 1 |
| 24 | `Molchanov Sim-to-(Multi)-Real IROS 2019 ...` (venue) | WebSearch | 0 (venue check) |
| 25 | `VLM high-level guidance drone onboard lightweight policy hierarchical navigation ...` → SINGER (MSL page + 2509.18610 HTML), GRaD-Nav++ abs, VLN on the Fly abs | WebSearch, WebFetch | 1 (+2 excluded) |
| 26 | arXiv 2609.13984 (What Makes an Efficient VLA?) abs + HTML (from search 4 results) | WebFetch | 1 |

Approximate totals: 26 search/fetch actions, about 40 distinct records seen, 35 records checked at source, **30 included**, **5 verified-but-excluded**, **0 unverifiable**. The count is above the 15–25 target because the seed list alone gave about 20 eligible items, and the drone/VLA deployment papers from 2026 add direct evidence for RQ3.

## 2. Inclusion notes

- Included: work that reports, for an on-device or edge-deployable learned policy, at least one of parameter count, FLOPs/MACs, latency/Hz, hardware or power, or that measures what is lost when a policy is shrunk (scaling or compression ablations).
- Large drone VLAs that run **offboard** (RaceVLA, CognitiveDrone, Exp2VLA's real deployment) are included as negative evidence: they show that 0.45–7B VLAs do not yet fit drone compute budgets, which is the motivation for our slow/fast split.
- Excluded with reason (section 7): no onboard policy, compute location unclear, or abstract-only with no RQ3 numbers.
- Tier: Tier 1 = peer-reviewed venue confirmed (on arXiv comments/journal-ref, publisher page or proceedings); Tier 2 = preprint; Tier 3 = blog / tech report.

## 3. Comparison table

"n.s." = not stated in what was read. Power is the policy/compute power unless noted.

| Paper | Year | Platform | Params | Control / inference Hz | Hardware | Power | Inputs | Task |
|---|---|---|---|---|---|---|---|---|
| RAPTOR (Eschmann et al.) | 2025 | 10 quadrotors, 32 g–2.4 kg | **2,084** (3-layer GRU, hidden 16) | n.s. (uses <10% of MCU compute) | Crazyflie STM32, PX4, Betaflight, M5StampFly FCs | n.s. | state (pos/vel/att/ang. vel) + recurrent | attitude/position control, trajectory tracking |
| Learning to Seek / tinyRL (Duisterhof et al.) | 2021 | Crazyflie 2.x nano drone | MLP 2×20 hidden (count n.s.); stack 20.5 kB | 100 Hz | STM32F405 Cortex-M4 | drone avg 7.81→8.52 W with system (+0.71 W) | light sensor + multiranger | light-source seeking, obstacle avoidance |
| Sim-to-(Multi)-Real (Molchanov et al.) | 2019 | Crazyflie + other quads | MLP 2×64, tanh (count n.s.) | 500 Hz loop; NN 0.8 ms | STM32F405 | n.s. | 18-D state | low-level stabilization (state→thrust) |
| Learning to Fly in Seconds (Eschmann et al.) | 2024 | Crazyflie | n.s. (supplement) | n.s. (sim 100 Hz) | MCU (Crazyflie) | n.s. | state | RPM-level trajectory tracking |
| Tiny-PULP-Dronets (Lamberti et al.) | 2022 | Crazyflie 2.1 + AI-deck | model **6.4 kB** min (vs 320 kB); 1.5 M MACs | up to 160 fps; 89 fps energy mode | GAP8 SoC | 34 mW avg; 0.38 mJ/frame | grayscale camera | steering + collision probability |
| PULP-Dronet (Palossi et al.) | 2019 | Crazyflie 2.0 (27 g) | DroNet CNN (count n.s. in abstract) | 6 fps (peak 18 fps) | GAP8 | **64 mW** avg (3.5% of drone power) | grayscale camera | end-to-end visual navigation |
| Swift (Kaufmann et al.) | 2023 | 5" racing quad | policy MLP 2×128 (count n.s.); U-Net gate detector | VIO 100 Hz; policy ~8 ms on CPU | Jetson TX2 | n.s. | VIO state + gate corners (camera, IMU) | champion-level racing |
| Learning High-Speed Flight in the Wild (Loquercio et al.) | 2021 | quadrotor | MobileNet-V3 + MLPs (count n.s.) | ~24 Hz (38.9 ms) | Jetson TX2 | n.s. | depth, velocity, attitude, goal dir | forest/urban high-speed avoidance |
| Liquid networks / CfC (Chahine et al.) | 2023 | DJI M300 RTK | CNN + CfC/LTC; counts in table S3 (not read) | n.s. | Jetson TX2 (Manifold 2) GPU | n.s. | RGB (gimbal camera) | fly-to-target under distribution shift |
| Agile Flight from Pixels (Geles et al.) | 2024 | racing quad | 3-layer CNN + MLP 2×512 (count n.s.); SwinV2-B detector | video 60 Hz; detector 4 ms | **offboard** RTX 3090 | n.s. | FPV video (no state est.) | gate racing up to 40 km/h |
| SINGER (Adang et al.) | 2026 | 5" cinewhoop | CLIPSeg ViT-B/16 86M + small CNN policy (count n.s.) | policy 20 Hz; CLIPSeg 12 Hz | Jetson Orin Nano 8 GB | n.s. | RGB, IMU, optical flow | language-guided navigation |
| Exp2VLA (Dang et al.) | 2026 | drone (sim + real) | π0.5 4B; SmolVLA 0.45B | loop 26.5 ms / 23.2 ms on desktop GPU | RTX 4080S; real flights **offboard** (Orin NX insufficient) | n.s. | RGB + language | language-conditioned navigation |
| RaceVLA (Serpiva et al.) | 2025 | 8" racing drone | OpenVLA 7B + LoRA | **4 Hz** (0.25 s) | **offboard** RTX 4090 | n.s. | FPV image + language | VLA racing |
| CognitiveDrone (Lykov et al.) | 2025 | sim (Gazebo/ArduPilot) | OpenVLA 7B (+ Qwen2.5-VL 7B reasoner) | VLA 10 Hz; reasoner ~2 Hz | A100 (training); deployment n.s. | n.s.; ~10/20 GB memory | image + language | cognitive UAV tasks |
| TinyVLA (Wen et al.) | 2024 | manipulators | 422M / 740M / 1.3B | 14 ms/action (H) | A6000 | n.s. | RGB + language | manipulation |
| SmolVLA (Shukor et al.) | 2025 | SO100 arms | 450M (~100M action expert) | n.s.; async inference | consumer GPU / CPU | n.s. | RGB + state + language | manipulation |
| MiniVLA (Belkhale & Sadigh) | 2024 | LIBERO | 1B | n.s. | n.s. | n.s. | RGB (multi-image) + language | manipulation |
| Octo (Octo team) | 2024 | 9 robot platforms | 27M (S), 93M (B) | n.s. | n.s. | n.s. | images + language/goal | manipulation |
| π0 / π0-small (Black et al.) | 2024 | 7+ robots | 3.3B; π0-small 470M | 50 Hz chunks; 73 ms inference | RTX 4090 | n.s. | images, state, language | dexterous manipulation |
| HPT (Wang et al.) | 2024 | multi-embodiment | trunk 3.1M–1.1B | n.s. | n.s. | n.s. | proprio + vision | pretraining scaling |
| EffVLA scaling study (Sun et al.) | 2026 | LIBERO, SO-ARM101 | sweep up to >3.75B | 39.2 ms/chunk | RTX 5090 | n.s. | RGB + language | manipulation |
| BitVLA (Wang et al.) | 2025 | LIBERO + real arm | 3.0B ternary; 1.4 GB | 73 ms | RTX 3050 Ti laptop (4 GB) fits; timing on A-series | n.s. | RGB + language | manipulation |
| EfficientVLA (Yang et al.) | 2025 | SIMPLER | CogACT ~7.63B, pruned | 1.93× faster | A40 | n.s. | RGB + language | manipulation |
| LiteVLA-Edge (Williams et al.) | 2026 | sim closed loop | SmolVLM-256M-based, 4-bit | **6.6 Hz** (150.5 ms) | Jetson AGX Orin 64 GB | n.s. | RGB + language | timing feasibility only |
| EagleVLA / Jetson-PI (Yang et al.) | 2026 | LIBERO | π0.5 | **10.3 Hz** (271 ms vs 1421 ms naive) | Jetson Orin 32/64 GB, 30/50 W modes | power mode 30–50 W | RGB + language | manipulation |
| FoldQuantVLA (Ho et al.) | 2026 | LIBERO, SimplerEnv, real | GR00T N1.7 etc., W4A4/W8A8 | 1.20–1.33× faster on Orin | Jetson AGX Orin, Ada GPUs | n.s. | RGB + language | manipulation |
| 8 GB bimanual ACT (Singh et al.) | 2026 | bimanual arms | ACT (count n.s.) | INT8 12.65 ms/inference | Jetson Orin Nano Super 8 GB | n.s. | multi-camera + joints | bimanual manipulation |
| LightDP (Wu et al.) | 2025 | arms + edge | DP-T 8.97M→2.65M; MDT-V 22.5M→7.45M | Orin NX 244.7→37.7 ms | Jetson Orin NX, iPhone 13 | n.s. | RGB + state | manipulation |
| Speedup Paradox (Wang et al.) | 2026 | LIBERO, RoboCasa, UR5e | MLP-Mixer … π0.5 | varies | Jetson AGX Orin 15/30/50 W, desktop GPUs | power modes 15/30/50 W | varies | speed–quality trade-off |
| Gemini Robotics On-Device (Google DeepMind) | 2025 | ALOHA, Franka, Apollo | n.s. | n.s. | "on-device" (n.s.) | n.s. | images + language | dexterous manipulation |

---

## 4. Annotated entries

### 4.1 Drone and nano-drone embedded controllers

#### E1. RAPTOR: A Foundation Policy for Quadrotor Control
- **Citation:** Eschmann, J., Albani, D., & Loianno, G. (2025). *RAPTOR: A Foundation Policy for Quadrotor Control.* arXiv:2509.11481 (v2, 2026-04-06). https://arxiv.org/abs/2509.11481
- **Read scope:** full_text (arXiv HTML v2)
- **Relevance:** Smallest demonstrated cross-platform drone control policy, and a distillation of 1000 teachers into one tiny student. Speaks to RQ3 and RQ5.
- **Key findings:** A 3-layer GRU policy with **2,084 parameters** (hidden dim 16) adapts zero-shot to 10 real quadrotors (32 g–2.4 kg) across PX4, Betaflight, Crazyflie and M5StampFly. It uses "<10% of the available computational power" on the MCU. Crazyflie tracking RMSE 0.19 m vs 0.17/0.15 m for a dedicated single-platform policy (HTML, results section).
- **Method:** 1000 SAC teacher policies on sampled dynamics, then meta-imitation distillation (about 1.9 h post-training) into a recurrent student that infers the dynamics in context.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** (author-acknowledged, Limitations section) sim-to-real gaps from flight-controller firmware delays; policy "overindexes on the linear velocity observations"; randomization limited to thrust-to-weight ≤5; no trajectory lookahead. (reader-inferred) No visual input, so the tiny size covers only the inner state-feedback loop.
- **Relevance to our setup:** Shows that the innermost drone loop needs only about 2k parameters, so nearly all fast-model capacity can go to perception and fusing the conditioning signal.

#### E2. Learning to Seek / Tiny Robot Learning (tinyRL) for source seeking on a nano quadcopter
- **Citation:** Duisterhof, B. P., Krishnan, S., Cruz, J. J., Banbury, C. R., Fu, W., Faust, A., de Croon, G. C. H. E., & Janapa Reddi, V. (2021). *Tiny Robot Learning (tinyRL) for Source Seeking on a Nano Quadcopter.* ICRA 2021, pp. 7242–7248. arXiv:1909.11236 ("Learning to Seek …"). https://arxiv.org/abs/1909.11236 ; venue: https://research.tudelft.nl/en/publications/tiny-robot-learning-tinyrl-for-source-seeking-on-a-nano-quadcopte
- **Read scope:** full_text (arXiv HTML v6)
- **Relevance:** An end-to-end deep-RL navigation policy on a Cortex-M4 with measured power; the TinyML size-floor reference.
- **Key findings:** Feed-forward policy with two hidden layers of 20 nodes runs at **100 Hz** on the STM32F405. The source-seeking stack uses 20.5 kB. Average drone power rose from 7.81 W to 8.52 W with the system added, cutting hover endurance by 35.5 s. 94% success in cluttered environments vs 80% prior; "almost three times less additional power" than a competing learning-based nano-drone approach (abstract; HTML results).
- **Method:** Deep RL in simulation with application-specific low-dimensional observations (light-sensor readings, multiranger distances). Not vision.
- **Quality tier:** Tier 1 (ICRA 2021)
- **Method weaknesses:** (reader-inferred) Low-dimensional, hand-designed observations; the task is simpler than visuomotor control. The power delta includes sensor hardware, not only inference.
- **Relevance to our setup:** A reference point for how small a learned reactive layer can be when a higher layer supplies a well-shaped observation, which is the role our conditioning embedding would play.

#### E3. Sim-to-(Multi)-Real: Transfer of low-level robust control policies to multiple quadrotors
- **Citation:** Molchanov, A., Chen, T., Hönig, W., Preiss, J. A., Ayanian, N., & Sukhatme, G. S. (2019). *Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors.* IROS 2019. arXiv:1903.04628. https://arxiv.org/abs/1903.04628
- **Read scope:** full_text (ar5iv HTML)
- **Relevance:** An early demonstration of a learned state→motor policy running on a Crazyflie MCU.
- **Key findings:** MLP with 2 hidden layers × 64 tanh units, 18-D state input → 4 thrusts. Auto-generated C function in the Crazyflie firmware on the STM32F405. NN evaluation "about 0.8 ms" inside a 500 Hz loop (ar5iv, experiments section).
- **Method:** RL (PPO-style) in simulation with domain randomization and noise models; zero-shot transfer to several quadrotors.
- **Quality tier:** Tier 1 (IROS 2019; venue confirmed by search, USC publication page)
- **Method weaknesses:** (author-acknowledged, ar5iv discussion) limited to quadrotors within a bounded range of physical parameters; no edge cases such as very high thrust-to-weight or large CoM offsets.
- **Relevance to our setup:** Confirms that 500 Hz learned inner loops fit a 168 MHz MCU, well above our ≥20 Hz action requirement.

#### E4. Learning to Fly in Seconds
- **Citation:** Eschmann, J., Albani, D., & Loianno, G. (2024). *Learning to Fly in Seconds.* IEEE Robotics and Automation Letters. arXiv:2311.13081. https://arxiv.org/abs/2311.13081
- **Read scope:** full_text (arXiv HTML; architecture constants in supplement not read)
- **Relevance:** Shows that MCU-deployable end-to-end RPM control can be trained in seconds; a precursor of RAPTOR.
- **Key findings:** Sim-to-real for direct RPM control "after only 18 seconds of training on a consumer-grade laptop." The simulator runs "about 5 months of flight per second" on a laptop GPU. Deployed on a Crazyflie "under real-time guarantees" (abstract). Network size and deployed rate are not stated in the main text read (sim 100 Hz).
- **Method:** Asymmetric actor-critic, curriculum, highly optimized simulator (RLtools).
- **Quality tier:** Tier 1 (RA-L)
- **Method weaknesses:** (author-acknowledged, conclusion) the design space of RL quadrotor control is "still sparsely explored"; no adaptation to battery level or wind. (reader-inferred) State-only inputs.
- **Relevance to our setup:** Fast retraining of the inner loop makes it practical to retrain the fast model per airframe while the slow model stays fixed.

#### E5. Tiny-PULP-Dronets: squeezing neural networks for nano-drones
- **Citation:** Lamberti, L., Niculescu, V., Barciś, M., Bellone, L., Natalizio, E., Benini, L., & Palossi, D. (2022). *Tiny-PULP-Dronets: Squeezing Neural Networks for Faster and Lighter Inference on Multi-Tasking Autonomous Nano-Drones.* IEEE AICAS 2022. arXiv:2407.02405. https://arxiv.org/abs/2407.02405
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** Directly measures what is lost when a visuomotor CNN is shrunk 50× on a 100 mW-class processor.
- **Key findings:** Model size down to **6.4 kB** from 320 kB (50×), MACs 1.5 M from 41 M (27×). GAP8 throughput 89 fps in energy-efficient mode (vs 10 fps) and up to 160 fps (vs 19 fps); 11.3 ms at 34 mW average, 0.38 mJ/frame. Steering regression held; collision accuracy 0.88 at the smallest configuration with "6% at most" degradation; flight performance comparable (HTML results).
- **Method:** Channel scaling (γ 0.125–1.0) and removal of residual bypass branches; deployment on GAP8.
- **Quality tier:** Tier 1 (AICAS 2022, per arXiv comments)
- **Method weaknesses:** (author-acknowledged, results discussion) speedup is sub-linear (27× fewer MACs → 8.5× speedup) because of data-marshaling overhead. (reader-inferred) Closed-loop evaluation in a limited set of indoor environments.
- **Relevance to our setup:** Strong evidence that vision features for reactive drone control can live in kilobytes. Shrinking 50× cost little accuracy because the original net was over-parameterized (up to 92% inactive neurons).

#### E6. A 64 mW DNN-based visual navigation engine for autonomous nano-drones (PULP-Dronet)
- **Citation:** Palossi, D., Loquercio, A., Conti, F., Flamand, E., Scaramuzza, D., & Benini, L. (2019). *A 64mW DNN-based Visual Navigation Engine for Autonomous Nano-Drones.* IEEE Internet of Things Journal. arXiv:1805.01831. https://arxiv.org/abs/1805.01831
- **Read scope:** abstract_only
- **Relevance:** First fully onboard end-to-end CNN navigation on a 27 g drone, with a power figure.
- **Key findings:** 64 mW average, 3.5% of the nano-aircraft power envelope; 6 fps real-time, 18 fps peak; GAP8 on a Crazyflie 2.0 (abstract).
- **Method:** DroNet CNN (steering + collision) ported to the GAP8 parallel ultra-low-power SoC.
- **Quality tier:** Tier 1 (IEEE IoT-J)
- **Method weaknesses:** not assessed (read scope: abstract_only)
- **Relevance to our setup:** Sets the milliwatt-class power floor for onboard visual policies, two orders of magnitude below Jetson-class compute.

#### E7. Champion-level drone racing using deep reinforcement learning (Swift)
- **Citation:** Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., & Scaramuzza, D. (2023). *Champion-level drone racing using deep reinforcement learning.* Nature, 620(7976), 982–987. https://doi.org/10.1038/s41586-023-06419-4 (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)
- **Read scope:** full_text (PMC)
- **Relevance:** The strongest performance result with a very small control policy plus a separate perception network: a split of perception and control on board.
- **Key findings:** Control policy is a two-layer perceptron with 128 nodes per layer (LeakyReLU), "approximately 8 ms" per inference on the Jetson TX2 CPU. The gate detector is a six-level U-Net on 384×384 grayscale images, ~40 ms on the TX2. T265 VIO at 100 Hz; Betaflight low-level controller on STM32. Beat human champions (fastest lap 0.16 s faster than the best human) (Methods).
- **Method:** Model-free RL in simulation with non-parametric empirical noise models fitted from real data; onboard VIO + gate detection + Kalman filter feed the policy.
- **Quality tier:** Tier 1 (Nature)
- **Method weaknesses:** (author-acknowledged, Discussion) assumes the environment's appearance matches training; cannot recover after crashes; less robust than humans to lighting changes. (reader-inferred) Fixed known track; power not reported.
- **Relevance to our setup:** Evidence that a two-layer, 128-unit MLP (parameter count not stated) suffices for agile control when perception is factored out. The heavy part is perception (U-Net), which is what our slow model would partly replace.

#### E8. Learning high-speed flight in the wild
- **Citation:** Loquercio, A., Kaufmann, E., Ranftl, R., Müller, M., Koltun, V., & Scaramuzza, D. (2021). *Learning high-speed flight in the wild.* Science Robotics, 6(59). arXiv:2110.05113. https://arxiv.org/abs/2110.05113
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** Onboard sensorimotor policy on a Jetson TX2 at ~24 Hz, near our ≥20 Hz requirement.
- **Key findings:** MobileNet-V3 depth backbone + MLPs; forward pass 38.9 ms on the TX2, "an update rate of about 24 Hz." Inputs: depth image, velocity, attitude, goal direction. Consistent success at 5–7 m/s; 60% at 10 m/s in forest (HTML, experiments).
- **Method:** Privileged imitation of a sampling-based expert in simulation, with sensor-noise simulation for zero-shot transfer.
- **Quality tier:** Tier 1 (Science Robotics)
- **Method weaknesses:** (author-acknowledged, discussion) at ≥10 m/s feasible solutions need long-horizon temporal consistency and sampling becomes intractable; sim-to-real dynamics mismatch from aerodynamics. (reader-inferred) Parameter count and power not reported.
- **Relevance to our setup:** A concrete Jetson TX2 budget: a MobileNet-scale visual policy reaches ~24 Hz, so our fast model must stay at or below this class on similar hardware.

#### E9. Robust flight navigation out of distribution with liquid neural networks
- **Citation:** Chahine, M., Hasani, R., Kao, P., Ray, A., Shubert, R., Lechner, M., Amini, A., & Rus, D. (2023). *Robust flight navigation out of distribution with liquid neural networks.* Science Robotics, 8(77), eadc8892. https://doi.org/10.1126/scirobotics.adc8892 (PDF: https://cap.csail.mit.edu/sites/default/files/research-pdfs/Robust%20flight%20navigation%20out%20of%20distribution%20with%20liquid%20neural%20networks.pdf)
- **Read scope:** full_text (PDF text extracted; supplementary tables S3/S4 not read)
- **Relevance:** Compact continuous-time recurrent controllers (CfC/LTC) for vision-based drone navigation, with onboard inference.
- **Key findings:** A shared CNN backbone produces 128-D features fed to a recurrent unit (CfC, LTC/NCP, LSTM and others compared). Inference runs onboard a DJI M300 RTK on the Jetson TX2 GPU of a DJI Manifold 2 (TensorFlow 2.3), sending velocity setpoints to the flight controller (Materials and Methods, "Drone hardware" and "System design"). Liquid networks outperformed other recurrent models in fly-to-target tasks under distribution shift. Per-model parameter counts are in table S3 (not read); control rate not stated in the main text read.
- **Method:** Imitation learning from human-flown sequences with augmentation and closed-loop synthetic sequences; hyperparameters chosen with TPE/Optuna.
- **Quality tier:** Tier 1 (Science Robotics)
- **Method weaknesses:** (reader-inferred) Low-speed velocity-setpoint task on a large enterprise drone; parameter comparison relegated to the supplement; power not reported.
- **Relevance to our setup:** Argues that a small recurrent head with good temporal inductive bias can be robust out of distribution. A candidate head for the fast model, fusing a stale slow-model embedding with fresh frames.

#### E10. Demonstrating agile flight from pixels without state estimation
- **Citation:** Geles, I., Bauersfeld, L., Romero, A., Xing, J., & Scaramuzza, D. (2024). *Demonstrating Agile Flight from Pixels without State Estimation.* Robotics: Science and Systems (RSS) 2024. arXiv:2406.12505. https://arxiv.org/abs/2406.12505 (PDF: https://rpg.ifi.uzh.ch/docs/RSS24_Geles.pdf)
- **Read scope:** full_text (PDF text extracted)
- **Relevance:** A counter-design. All perception and control run **offboard** on a ground PC over a video link, with latency figures.
- **Key findings:** FPV video to a ground station at 60 Hz (1280×720) with 33 ms transmission latency (Introduction). Detector is a SwinV2-B, 4 ms on an RTX 3090 with TensorRT. Policy: 3-layer CNN on 84×84 gate masks + MLP actor 2×512. Up to 40 km/h and 2 g.
- **Method:** Asymmetric actor-critic RL with gate-edge abstraction (up to 400 M interactions/day); detector trained on non-deployment images.
- **Quality tier:** Tier 1 (RSS 2024)
- **Method weaknesses:** (author-acknowledged, Discussion/Limitations) the controller remembers only three past actions; success "drops drastically" when no gate is visible for several frames; a recurrent architecture is suggested. (reader-inferred) Relies on a low-latency, high-bandwidth analog/digital FPV link that may not hold under real network jitter.
- **Relevance to our setup:** Shows that offloading the whole policy can work at 33 ms link latency. Our case (5 Hz slow model, jittery link) needs an onboard fast loop, which this paper avoids by assuming a near-real-time link.

#### E11. SINGER: An onboard generalist vision-language navigation policy for drones
- **Citation:** Adang, M., Low, J., Shorinwa, O., & Schwager, M. (2026). *SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones.* ICRA 2026. arXiv:2509.18610. https://arxiv.org/abs/2509.18610 ; https://msl.stanford.edu/bibliography/adang2026singeronboardgeneralistvisionlanguage.html
- **Read scope:** full_text (arXiv HTML v1)
- **Relevance:** The closest drone analogue to our design that runs fully on board: a language-grounded semantic model at low rate conditions a small fast visuomotor policy, all on an Orin Nano.
- **Key findings:** CLIPSeg (ViT-B/16, 86M) produces semantic heatmaps at up to 12 Hz onboard a Jetson Orin Nano 8 GB, run asynchronously. A lightweight CNN policy (SV-Net-derived) outputs collective-thrust and body-rate commands at **20 Hz**. Platform: 5" cinewhoop with Pixracer, ZED Mini (672×376), ARK Flow. Beats a velocity-controlled semantic-guidance baseline by reaching the query 23.33% more often (abstract; HTML, system section).
- **Method:** Gaussian-splatting language-embedded simulator, RRT-inspired multi-trajectory expert, ~0.7–1 M labeled examples, imitation learning.
- **Quality tier:** Tier 1 (ICRA 2026)
- **Method weaknesses:** (author-acknowledged, Limitations) lack of collision avoidance and reliance on low-resolution images. (reader-inferred) Policy parameter count not stated; the semantic model is far smaller than a 27B VLM.
- **Relevance to our setup:** Direct template of a slow semantic map (12 Hz) feeding a fast 20 Hz drone policy on an Orin Nano 8 GB (power not stated). Replace CLIPSeg with a remote 27B model's embedding.

#### E12. Exp2VLA: Enabling vision-language-action for drone navigation from expert demonstrations
- **Citation:** Dang, V. H., Rajendran, K., Sayar, E., & Kayacan, E. (2026). *Exp2VLA: Enabling Vision-Language-Action for Drone Navigation from Expert Demonstrations.* arXiv:2607.03146. https://arxiv.org/abs/2607.03146
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** A head-to-head of 4B vs 0.45B VLAs on drone navigation, with the finding that the small one fails in complex scenes and the large one cannot run on board.
- **Key findings:** π0.5 (4B; 3B PaliGemma + 300M flow-matching expert) vs SmolVLA (0.45B). Loop latency 26.53 ms / 12.90 GB VRAM vs 23.22 ms / 6.74 GB on an RTX 4080 Super. Success: single cube 84.1% vs 46.6%; multi-object 65.0% vs 15.0%; unseen pairs 51.7% (π0.5). Real deployment used a mobile workstation because "the onboard Orin NX lacks sufficient computational resources" (HTML, experiments).
- **Method:** Distil RL/teleop/controller experts into demonstration data and fine-tune VLAs; sim-to-sim and simulation-in-the-loop tests.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** (author-acknowledged, conclusion) scarce aerial data, moderate generalization drop, no standard drone-VLA benchmark. (reader-inferred) Only two model sizes; no compressed or quantized variants tried on the Orin NX.
- **Relevance to our setup:** Direct evidence for our split. Shrinking a VLA to 0.45B to fit on board costs 38–50 points of success, which favours keeping semantics remote and making only the fast loop small.

#### E13. RaceVLA: VLA-based racing drone navigation with human-like behaviour
- **Citation:** Serpiva, V., Lykov, A., Myshlyaev, A., Khan, M. H., Abdulkarim, A. A., Sautenkov, O., & Tsetserukou, D. (2025). *RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour.* IROS 2025. arXiv:2503.02572. https://arxiv.org/abs/2503.02572
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** "OpenVLA on a drone", in practice run offboard. Quantifies the rate a 7B VLA delivers.
- **Key findings:** OpenVLA-7B with LoRA rank 32, served from an RTX 4090 ground server over a Flask API: **4 Hz**, 0.25 s per frame. Drone: 8" frame with SpeedyBee F405 (ArduPilot), T265, Intel NUC for localization. Average 1.04 m/s, max 2.02 m/s (HTML, system section; abstract).
- **Method:** Fine-tune OpenVLA on a custom racing-drone dataset; velocity commands via MAVROS.
- **Quality tier:** Tier 1 (IROS 2025, per arXiv comments)
- **Method weaknesses:** (author-acknowledged, conclusion) "increasing the drone's speed will require reducing the model's inference time"; weaker visual/physical generalization in unpredictable scenes. (reader-inferred) No onboard fast loop, so speed is capped by the 4 Hz rate.
- **Relevance to our setup:** A baseline showing what happens without a fast onboard model: 4 Hz VLA control limits a racing drone to about 1–2 m/s.

#### E14. CognitiveDrone: A VLA model and evaluation benchmark for real-time cognitive task solving in UAVs
- **Citation:** Lykov, A., Serpiva, V., Khan, M. H., Sautenkov, O., Myshlyaev, A., Tadevosyan, G., Yaqoot, Y., & Tsetserukou, D. (2025). *CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs.* arXiv:2503.01378. https://arxiv.org/abs/2503.01378
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** A dual-rate drone VLA (7B VLA at 10 Hz + 7B VLM reasoner at ~2 Hz), a sim-only analogue of slow/fast.
- **Key findings:** OpenVLA-based 7B VLA at 10 Hz; Qwen2.5-VL 7B reasoning module at ~2 Hz. Success: racing baseline 31.3%, CognitiveDrone 59.6%, CognitiveDrone-R1 77.2%. Memory ~20 GB with the reasoner vs ~10 GB alone (HTML).
- **Method:** Trained on >8,000 simulated trajectories in Gazebo + ArduPilot with velocity setpoints.
- **Quality tier:** Tier 2 (preprint; "IEEE conference pending")
- **Method weaknesses:** (author-acknowledged) ~20 GB memory makes embedded deployment impractical. (reader-inferred) Simulation only; claims of "real-time deployability on UAV hardware" are not backed by an onboard measurement.
- **Relevance to our setup:** Adding a slow reasoner improves hard cognitive tasks by +17.6 points, which supports keeping a large slow model in the loop. The 10 Hz fast part is still 7B and offboard.

### 4.2 Small and efficient VLAs, scaling evidence

#### E15. TinyVLA: Towards fast, data-efficient vision-language-action models for robotic manipulation
- **Citation:** Wen, J., Zhu, Y., Li, J., Zhu, M., Wu, K., Xu, Z., Liu, N., Cheng, R., Shen, C., Peng, Y., Feng, F., & Tang, J. (2024). *TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation.* arXiv:2409.12514. https://arxiv.org/abs/2409.12514 ; https://tiny-vla.github.io
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** Sub-1.5B VLAs with a diffusion head that match or beat OpenVLA.
- **Key findings:** Variants S/B/H = 422M / 740M / 1.3B total (101M / 138M / 143M trainable) vs OpenVLA 7.2B. Per-action latency on an A6000: TinyVLA-H 14 ms vs OpenVLA 292 ms ("20 times less"). Real single-arm tasks 94.0% vs OpenVLA 68.3%; bimanual 44.5% vs 0%; MetaWorld-50 31.6% vs Diffusion Policy 10.5% (HTML, experiments).
- **Method:** Small pretrained multimodal backbone + diffusion policy decoder; no robot-data pretraining stage; LoRA-style fine-tuning.
- **Quality tier:** Tier 2 (preprint; venue not confirmed on the pages read)
- **Method weaknesses:** (author-acknowledged, Sec. IV-F) OpenVLA does slightly better on some spatial-generalization distances, attributed to large-scale pretraining. (reader-inferred) Timing only on a desktop A6000; no embedded measurement.
- **Relevance to our setup:** Suggests that a 0.4–1.3B model with a diffusion head is a plausible ceiling for a capable onboard model on Orin-class hardware, though not on a nano-drone.

#### E16. SmolVLA: A vision-language-action model for affordable and efficient robotics
- **Citation:** Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., Alibert, S., Cord, M., Wolf, T., & Cadene, R. (2025). *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.* arXiv:2506.01844. https://arxiv.org/abs/2506.01844
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** A 450M VLA that runs on consumer GPU/CPU with asynchronous inference. Size versus π0 and OpenVLA.
- **Key findings:** 450M parameters (~100M action expert). LIBERO average 87.3% vs π0 (3.3B) 86.0% and OpenVLA (7B) 76.5%. Real SO100: 78.3% vs π0 61.7%. Async inference cut task time from 13.75 s to 9.7 s and gave 19 vs 9 pick-place cycles in fixed time (HTML, results).
- **Method:** SmolVLM backbone with layer skipping and reduced visual tokens; flow-matching action expert; trained on ~23k community trajectories.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** (author-acknowledged, Sec. 5.1) single-embodiment pretraining (SO100); small dataset; sub-0.5B size may limit scaling; OCR-oriented backbone; short-horizon tasks only.
- **Relevance to our setup:** On a single embodiment a 0.45B VLA can match a 3.3B one, but E12 shows the same model degrading badly on drone navigation, so the result is task-dependent.

#### E17. MiniVLA: A better VLA with a smaller footprint
- **Citation:** Belkhale, S., & Sadigh, D. (2024, December 8). *MiniVLA: A Better VLA with a Smaller Footprint.* Stanford AI Lab Blog. https://ai.stanford.edu/blog/minivla/
- **Read scope:** search_only (the blog refused connection twice; content confirmed through search-engine extracts of the official page and the Stanford-ILIAD Hugging Face model card https://huggingface.co/Stanford-ILIAD/minivla-libero90-prismatic)
- **Relevance:** A 1B OpenVLA derivative showing that a 7× size cut can raise success.
- **Key findings:** 7B → 1B (Qwen2.5 0.5B LLM, OpenVLA's ViT kept). LIBERO-90 62% → 82% with VQ action chunking and multi-image inputs (wrist + history) (search extract of the blog).
- **Method:** Prismatic VLM with a smaller LLM; vector-quantized action chunks.
- **Quality tier:** Tier 3 (blog)
- **Method weaknesses:** not assessed (read scope: search_only). (reader-inferred) Gains mix size reduction with input/output changes, so size alone is confounded.
- **Relevance to our setup:** Action representation (chunking) and inputs matter more than raw size in the 1–7B range.

#### E18. Octo: An open-source generalist robot policy
- **Citation:** Octo Model Team: Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., et al. (2024). *Octo: An Open-Source Generalist Robot Policy.* Robotics: Science and Systems XX. arXiv:2405.12213. https://arxiv.org/abs/2405.12213 ; https://www.roboticsproceedings.org/rss20/p090.html
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** A generalist policy at 27M/93M parameters, with a size ablation.
- **Key findings:** Octo-Small 27M, Octo-Base 93M. "Performance of Octo improves with larger model sizes"; Base is more robust to scene variation and makes fewer premature grasps (HTML, ablations). Fine-tuning takes ~5 h on one A5000.
- **Method:** Transformer over image/language tokens with a diffusion action head; 800k OXE trajectories.
- **Quality tier:** Tier 1 (RSS 2024)
- **Method weaknesses:** (reader-inferred) Size ablation covers only about 3× (27M→93M); embedded latency not reported in the parts read.
- **Relevance to our setup:** A 27–93M generalist is a realistic fast-model size class for Orin Nano, and performance still rises with size in this range.

#### E19. π0: A vision-language-action flow model for general robot control
- **Citation:** Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164. https://arxiv.org/abs/2410.24164
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** Includes a size and initialization ablation (π0-small) and rate figures.
- **Key findings:** π0 3.3B (3B PaliGemma + 300M action expert); π0-small 470M without VLM initialization performs "far below" π0 out of the box. 50-step action chunks; 50 Hz control for dexterous tasks; 73 ms on-board inference on an RTX 4090 (HTML).
- **Method:** VLM backbone + flow-matching action expert; cross-embodiment pretraining.
- **Quality tier:** Tier 2 (arXiv version read; venue not confirmed here)
- **Method weaknesses:** (reader-inferred) The π0-small comparison confounds size with the absence of VLM pretraining.
- **Relevance to our setup:** Much of the large-model advantage comes from web-scale VLM priors, which our slow 27B model supplies. The fast model then need not hold them.

#### E20. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers (HPT)
- **Citation:** Wang, L., Chen, X., Zhao, J., & He, K. (2024). *Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers.* NeurIPS 2024. arXiv:2409.20537. https://arxiv.org/abs/2409.20537
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** One of few explicit policy-size scaling curves (1M → 1B).
- **Key findings:** Trunk sizes 3.1M / 12.6M / 50.5M / 226.8M / 1.1B (Table 1). Validation loss falls with size until it plateaus; the gains need matching data and compute, otherwise the plateau comes earlier; depth and width scale about equally (Fig. 7).
- **Method:** Embodiment-specific stems/heads + shared transformer trunk, pretrained on 52 datasets.
- **Quality tier:** Tier 1 (NeurIPS 2024)
- **Method weaknesses:** (reader-inferred) The scaling curve is on validation loss, not closed-loop success.
- **Relevance to our setup:** Fast-model capacity beyond ~50–200M gives diminishing returns unless data scales too.

#### E21. What makes an efficient VLA? Navigating action-head design, scaling, and latency (EffVLA)
- **Citation:** Sun, L., Xia, G., Li, F., Ren, L., Cui, X., Zhang, H., Feng, F., Zhang, K., Zhan, K., Xie, Y., Wang, J., & Deng, C. (2026). *What Makes an Efficient VLA? Navigating Action-Head Design, Scaling, and Latency.* arXiv:2609.13984. https://arxiv.org/abs/2609.13984
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** A controlled scaling sweep of VLA modules against latency.
- **Key findings:** Vision 86M–~1B, LLM 0.5B–7B, action head ~0.35B–0.65B. "Scaling any module past the ll configuration [≈3.75B total] leaves accuracy flat while latency keeps climbing." Copying the last LLM layers into the action head is "the single largest lever, at no latency cost." EffVLA ≈3.75B, 39.2 ms/chunk on an RTX 5090 bf16 (HTML).
- **Method:** Fixed SigLIP2/Qwen2.5 families, grid over module scales and head designs; LIBERO, LIBERO-Plus, SO-ARM101.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** (author-acknowledged, Limitations) short-horizon simulation, single seed per cell (five-seed check on key cells), one real arm and 50 trials, causal mechanism open.
- **Relevance to our setup:** Accuracy saturates near 3–4B on LIBERO, so a 27B slow model is justified only by generalization or reasoning, not in-domain control.

#### E22. Gemini Robotics On-Device
- **Citation:** Parada, C. / Google DeepMind (2025, June 24). *Gemini Robotics On-Device brings AI to local robotic devices.* Google DeepMind Blog. https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/
- **Read scope:** abstract_only (blog page)
- **Relevance:** Industry evidence on what an on-device VLA loses relative to the flagship.
- **Key findings:** Generalization scores (≈, read from chart): visual 0.74 vs 0.75, semantic 0.72 vs 0.73, action 0.52 vs 0.60 for on-device vs flagship. Adapts with 50–100 demos; ALOHA, Franka FR3, Apollo. **No parameter count, latency or hardware disclosed.**
- **Method:** Not disclosed.
- **Quality tier:** Tier 3 (blog)
- **Method weaknesses:** not assessed (read scope: abstract_only). (reader-inferred) Undisclosed size and hardware make the result non-reproducible.
- **Relevance to our setup:** The on-device model loses most on *action* generalization (≈−0.08) and almost nothing on visual/semantic generalization, so action generalization may be what a small onboard model gives up first.

### 4.3 Compression and edge deployment of VLAs and diffusion policies

#### E23. BitVLA: 1-bit vision-language-action models for robotics manipulation
- **Citation:** Wang, H., Xiong, C., Wang, R., & Chen, X. (2025). *BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation.* arXiv:2506.07530. https://arxiv.org/abs/2506.07530
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** Extreme weight quantization of a full VLA.
- **Key findings:** Ternary 3.0B model, 1.4 GB memory (11.0× smaller than OpenVLA-OFT's 15.4 GB). LIBERO 96.0% vs OpenVLA-OFT 97.1%. 73 ms latency, 4.4× lower than OpenVLA-OFT; fits an RTX 3050 Ti laptop GPU (4 GB) (HTML).
- **Method:** BitNet b1.58 2B4T backbone; Quantize-then-Distill for the 1.58-bit vision encoder.
- **Quality tier:** Tier 2 (preprint, "work in progress")
- **Method weaknesses:** (author-acknowledged, conclusion) ~1M-sample robotics pretraining may be insufficient for strong generalization. (reader-inferred) No ternary-kernel speedups measured on embedded hardware.
- **Relevance to our setup:** Memory, not FLOPs, is often the binding constraint on board. 1.4 GB for a 3B VLA would fit Orin Nano RAM, but not MCU-class drones.

#### E24. EfficientVLA: Training-free acceleration and compression for VLA models
- **Citation:** Yang, Y., Wang, Y., Wen, Z., Zhongwei, L., Zou, C., Zhang, Z., Wen, C., & Zhang, L. (2025). *EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models.* arXiv:2506.10100. https://arxiv.org/abs/2506.10100
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** Layer pruning + visual-token pruning + diffusion-head caching. Quantifies the loss.
- **Key findings:** On CogACT (~7.63B; LLM ~6.74B): 1.93× speedup, FLOPs down to 28.9%, −0.6% SIMPLER success. Language module 134.5 → 58.9 ms; action module 51.5 → 26.2 ms on an A40 (HTML).
- **Method:** Inter-layer-redundancy LLM pruning, task-aware visual token selection, cached diffusion features.
- **Quality tier:** Tier 2 (venue not confirmed on pages read)
- **Method weaknesses:** (author-acknowledged, Limitations) training-free design caps compression; fixed caching intervals trade speed against quality; validated mainly on CogACT.
- **Relevance to our setup:** Applies mainly to the slow 27B side (token pruning to raise its ~5 Hz), not to the fast model.

#### E25. LiteVLA-Edge: Quantized on-device multimodal control for embedded robotics
- **Citation:** Williams, J., Gupta, K. D., George, R., & Sarkar, M. (2026). *LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics.* arXiv:2603.03380. https://arxiv.org/abs/2603.03380
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** A fully offline sub-1B VLA on a Jetson with a latency measurement.
- **Key findings:** SmolVLM-256M-derived backbone, 4-bit GGUF on llama.cpp; 150.5 ms mean end-to-end (≈6.6 Hz), jitter SD 0.13 ms on a Jetson AGX Orin 64 GB (Orin NX also mentioned) in ROS 2 (HTML).
- **Method:** FP32 image-to-action supervised fine-tuning, then post-training quantization; all 42 layers on GPU; n_ctx 512.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** (author-acknowledged, Limitations) no robot-task benchmarking, one hardware platform, simulated closed loop only.
- **Relevance to our setup:** Even a 256M VLA reaches only ~6.6 Hz on AGX Orin, below our ≥20 Hz target. Onboard VLM-style models cannot be the fast loop.

#### E26. EagleVLA (formerly Jetson-PI): Towards onboard real-time robot control via foresight-aligned asynchronous inference
- **Citation:** Yang, Z., Wang, Q., Wang, Y., Guo, X., Yu, B., Liu, S., Xu, J., Dong, H., & Li, M. (2026). *EagleVLA: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference.* CoRL 2026 (per arXiv comments). arXiv:2607.12659. https://arxiv.org/abs/2607.12659
- **Read scope:** full_text (arXiv HTML)
- **Relevance:** The best reported onboard π0.5 rate on Jetson, plus asynchronous inference (overlaps RQ2).
- **Key findings:** Naive PyTorch π0.5 on Jetson Orin: 1420.8 ms; EagleVLA: 271.1 ms total, **10.30 Hz** control. 14.71× vs naive PyTorch, 9.20× vs vla.cpp. Tested on Orin 32 GB / 64 GB at 30 W and 50 W modes. LIBERO +14.8 points vs VLASH (92–98% per suite) (HTML).
- **Method:** System optimizations + foresight-aligned asynchronous inference.
- **Quality tier:** Tier 1 (CoRL 2026, per arXiv comments; proceedings not yet checked)
- **Method weaknesses:** (author-acknowledged) the onboard platform remains constrained and gains may not close the gap to high-end GPUs as model scale increases. (reader-inferred) LIBERO only (simulation).
- **Relevance to our setup:** A 4B-class VLA reaches ~10 Hz at 30–50 W. That is too heavy and power-hungry for a small drone, but it is a credible onboard "slow" fallback for larger robots.

#### E27. FoldQuantVLA: Native low-bit quantization of VLA models via consistent folding
- **Citation:** Ho, H. T., Nguyen, K. D., et al. (2026). *FoldQuantVLA: Native Low-Bit Quantization of Vision-Language-Action Models via Consistent Folding.* arXiv:2609.24433. https://arxiv.org/abs/2609.24433
- **Read scope:** abstract_only
- **Relevance:** W4A4 VLA quantization measured on Jetson AGX Orin.
- **Key findings:** 1.20–1.33× speedup on Orin (1.25–1.52× desktop) over floating point. On GR00T N1.7 real-robot tasks, selective W8A8 for language attention lifts success from 80.0% (uniform W4A4) to 92.5% over 80 trials/task at +1 ms (abstract).
- **Method:** PTQ with channel scaling, block Hadamard transforms, per-token dynamic quantization, TensorRT plugins.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** not assessed (read scope: abstract_only)
- **Relevance to our setup:** 4-bit quantization gives only ~1.2–1.3× on Orin, which is small. Quantization alone will not bridge a 5–10× rate gap.

#### E28. Bimanual manipulation within an 8 GB budget: zero-copy sensing and quantized ACT on an entry-level Jetson
- **Citation:** Singh, E., Samuel, E., Reneau, A., Schmeelk, R., & Gandhi, Y. (2026). *Bimanual Manipulation Within an 8 GB Budget: Zero-Copy Sensing and Quantized ACT on an Entry-Level Jetson.* arXiv:2608.03938. https://arxiv.org/abs/2608.03938
- **Read scope:** abstract_only
- **Relevance:** A small action-chunking policy on the cheapest current Jetson, with a quantization loss measurement.
- **Key findings:** Jetson Orin Nano Super 8 GB. ACT on TensorRT: FP16 17.93 ms (6.4×), INT8 12.65 ms (9.0×); success 19/20 (full precision), 18/20 (FP16), 19/20 (INT8). Zero-copy sensing lowers peak single-core CPU from 98.0% to 77.0% and worst-case latency from 117.31 to 101.52 ms (abstract).
- **Method:** TensorRT conversion, quantization, zero-copy camera pipeline; ACT vs Diffusion Policy.
- **Quality tier:** Tier 2 (preprint; GTRI)
- **Method weaknesses:** not assessed (read scope: abstract_only). (reader-inferred) 20 trials per condition is too few to resolve small differences.
- **Relevance to our setup:** The best match for our fast model's compute class. A chunked transformer policy needs 12.65 ms per inference (≈79 Hz, derived by us from the stated latency) on an 8 GB Orin Nano with no measurable success loss from INT8. The bottleneck shifts to sensing I/O, which matters for multi-rate fusion (RQ4).

#### E29. On-device diffusion transformer policy for efficient robot manipulation (LightDP)
- **Citation:** Wu, Y., Wang, H., Chen, Z., Pang, J., & Xu, D. (2025). *On-Device Diffusion Transformer Policy for Efficient Robot Manipulation.* ICCV 2025. arXiv:2508.00697. https://arxiv.org/abs/2508.00697 ; https://openaccess.thecvf.com/content/ICCV2025/html/Wu_On-Device_Diffusion_Transformer_Policy_for_Efficient_Robot_Manipulation_ICCV_2025_paper.html
- **Read scope:** full_text (arXiv HTML v1)
- **Relevance:** Pruning + consistency distillation of diffusion policies, with Jetson and phone latencies and the loss they cause.
- **Key findings:** DP-T 8.97M → 2.65M params; MDT-V 22.52M → 7.45M. 100 → 4 sampling steps. Orin NX (DP-T) 244.68 → 37.69 ms; iPhone 13 DP-T 90.6 → 2.72 ms. Push-T 77.2% → 72.4%; CALVIN avg length 4.52 → 3.44 (most compressed); LIBERO 78.5% → 77.9% (HTML).
- **Method:** Learnable pruning with SVD-initialized Bernoulli gates, retraining, consistency distillation.
- **Quality tier:** Tier 1 (ICCV 2025)
- **Method weaknesses:** (author-acknowledged, conclusion) focuses on diffusion policies; newer VLAs not explored. (reader-inferred) Long-horizon CALVIN loses about 24% of chain length, so compression is not free on long tasks.
- **Relevance to our setup:** A few-million-parameter diffusion head reaches ~27 Hz on an Orin NX after compression, a candidate fast-model action head. Long-horizon losses argue for letting the slow model handle long horizons.

#### E30. The speedup paradox: rethinking inference speed–quality trade-off in embodied tasks
- **Citation:** Wang, Y., Chen, J., Li, Y., Dong, S., Yang, H., Liu, Y., & Jia, H. (2026). *The Speedup Paradox: Rethinking Inference Speed-Quality Trade-off in Embodied Tasks.* arXiv:2606.28529. https://arxiv.org/abs/2606.28529
- **Read scope:** full_text (arXiv HTML, via fetch summary; per-setting numbers not extracted)
- **Relevance:** Measures how compression changes closed-loop success and completion time across Jetson power modes.
- **Key findings:** Faster per-chunk inference can *lengthen* task time in static tasks (degraded actions need more cycles), while moderate lossy lightweighting can *raise* success in dynamic tasks by reducing staleness. The optimum shifts with hardware: Jetson AGX Orin at 15/30/50 W and RTX 3090/4070/6000 Ada; MLP-Mixer, diffusion policies, π0.5, DynamicVLA, Cosmos-Policy; real UR5e (abstract; HTML).
- **Method:** TISED decomposition of task-level effects of quantization, pruning and async inference.
- **Quality tier:** Tier 2 (preprint)
- **Method weaknesses:** (reader-inferred) Per-configuration numbers not captured in this read; dynamic-task results depend on the benchmark's dynamics.
- **Relevance to our setup:** For a drone (a highly dynamic task), a smaller, faster but less accurate fast model may beat a larger, slower one. Policy size should be chosen in closed loop, not by offline accuracy.

---

## 5. Theme gaps

1. **No paper conditions a tiny onboard drone controller on a remote large-model embedding.** SINGER (E11) is closest but runs an 86M semantic model on board. RaceVLA, CognitiveDrone and Exp2VLA (E12–E14) run the whole VLA offboard with no fast onboard loop.
2. **Power is rarely reported.** Only the nano-drone work (E2, E5, E6) gives watts or milliwatts. Jetson studies report power *modes* (E26, E30), not measured draw or energy per action. No VLA paper reports joules per action.
3. **Few closed-loop size-vs-success curves.** HPT (E20) uses validation loss; Octo (E18) spans only 27–93M; EffVLA (E21) starts at ~1B+. No sweep covers 10k–100M parameters on visuomotor drone tasks.
4. **Size is confounded with pretraining** (π0-small, E19; MiniVLA, E17), so "what is lost by shrinking" is rarely isolated.
5. **Quantization speedups on Jetson are modest** (1.2–1.3×, E27), while memory savings are large (E23). There are no MCU-class results for any language-conditioned policy.
6. **Drone VLA evaluation is mostly simulation or slow flight** (≤2 m/s in RaceVLA). No shared benchmark exists, as Exp2VLA notes.

## 6. Size floor: smallest demonstrated policies per task class

| Task class | Smallest demonstrated policy | Rate / hardware / power | Source |
|---|---|---|---|
| Quadrotor inner-loop control (state → motors), cross-platform | 2,084-parameter 3-layer GRU | MCU, <10% of compute | RAPTOR (E1) |
| Quadrotor stabilization, single platform family | MLP 2×64 | 500 Hz loop, 0.8 ms on STM32F405 | Molchanov et al. (E3) |
| Low-dim reactive navigation (source seeking) | MLP 2×20, 20.5 kB stack | 100 Hz on Cortex-M4; +0.71 W system | Duisterhof et al. (E2) |
| Monocular visual navigation on nano-drone | CNN of 6.4 kB, 1.5 M MACs | up to 160 fps, 34 mW on GAP8 | Tiny-PULP-Dronets (E5); 64 mW PULP-Dronet (E6) |
| Agile racing (perception factored out) | MLP 2×128 control policy | ~8 ms on Jetson TX2 CPU | Swift (E7) |
| Visuomotor obstacle avoidance at speed | MobileNet-V3 + MLP | ~24 Hz on Jetson TX2 | Loquercio et al. (E8) |
| Language-guided drone navigation, fully onboard | 86M CLIPSeg + small CNN policy | policy 20 Hz, semantics 12 Hz, Orin Nano 8 GB | SINGER (E11) |
| Manipulation, diffusion policy | 2.65M (pruned DP-T) | 37.7 ms on Orin NX | LightDP (E29) |
| Manipulation, generalist transformer | 27M (Octo-Small) | n.s. | Octo (E18) |
| Manipulation, VLA | 256M-class (LiteVLA-Edge, timing only); 422M TinyVLA-S; 450M SmolVLA with task results | 6.6 Hz on AGX Orin (LiteVLA-Edge) | E25, E15, E16 |
| Drone VLA (language → velocity) with task results | 0.45B SmolVLA: large losses vs 4B (46.6% vs 84.1%) | offboard; Orin NX insufficient | Exp2VLA (E12) |

**Summary.** Learned *control* fits in about 10³–10⁴ parameters on an MCU at 100–500 Hz. Learned *visual reactive* policies fit in kilobytes at tens of milliwatts. *Semantic or language* competence has not been shown below ~86M parameters on board, and VLA-style models below ~0.5B lose heavily on drone tasks (E12) while running only at ~6–10 Hz on 30–50 W Jetsons (E25, E26). The evidence therefore supports a design in which the slow remote model carries semantics and the onboard model is a ≤10–100M visuomotor head conditioned on its embedding. No study has yet tested this directly (gap 1).

## 7. Verified but excluded / unverified

| Item | Status | Reason |
|---|---|---|
| UAV-VLA (Sautenkov et al., HRI 2025, arXiv:2501.05014) | verified, excluded | Offline mission/waypoint generation from satellite imagery with GPT/VLM; no onboard policy or RQ3 numbers. Better suited to RQ1 (language/waypoint interface). |
| Neural Lander (Shi et al., ICRA 2019, arXiv:1811.08027) | verified, excluded | Learned residual dynamics in a nonlinear controller; network size, rate and compute location not stated in the abstract read. |
| GAP9Shield (Müller, Kartsch, Benini, ERF 2024, arXiv:2407.13706) | verified, excluded | Hardware module (150 GOPS, <100 mW), not a learned policy. Useful as a hardware reference only. |
| GRaD-Nav++ (Chen et al., RA-L 11(2) 2026, arXiv:2506.14009) | verified, excluded | Fully onboard drone VLA, but the abstract gives no sizes, rates or hardware; full text not read. Candidate for a second pass. |
| VLN on the Fly (Tayar et al., arXiv:2609.20191) | verified, excluded | Onboard modular VLN stack with a quantized VLM and an RL controller, but the abstract gives no params, Hz or hardware. Candidate for a second pass. |
| Unverifiable items | none | All seed items resolved to a real arXiv ID, DOI or official page. |
