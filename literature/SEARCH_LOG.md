# Search Log (merged from theme files)

Each section is copied verbatim from its theme file's "Search log" section. Search date: 2026-10-27. Re-run these queries to update the review.

---

## 01-dual-system-vla

| # | Query / action | Source | Approx. hits | Notes |
|---|---|---|---|---|
| 1 | Seed list verification: direct fetch of arXiv abs/HTML pages for 22 seed IDs | arXiv | 22 fetched | All seed arXiv IDs resolved |
| 2 | `dual-system VLA asynchronous slow reasoning fast action latent arXiv 2025` | Web search | ~9 | Found Libra-VLA (2604.24921), DuoCore-FS (2512.20188) |
| 3 | `hierarchical VLA cloud edge slow VLM fast policy onboard robot latency 2026 arXiv` | Web search | ~9 | Found AsyncVLA (2602.13476), VLA-ULAP (2609.18663), Agile-VLA (2603.22899) |
| 4 | `Gemini Robotics On-Device local low-latency inference DeepMind` | Web search | ~9 | Official blog located |
| 5 | `GR00T N1.5 NVIDIA research page Eagle VLM frozen System 2` | Web search | ~9 | Mostly secondary wikis; official page fetched directly |
| 6 | `"dual process" VLA "System 1" "System 2" robot manipulation L-Sys2 latent arXiv 2024` | Web search | ~10 | Found DP-VLA (2410.15549) |
| 7 | `fast and slow thinking robot policy hierarchical VLA latent token stale asynchronous ablation fast policy size 2026` | Web search | ~9 | Found StreamVLA (2602.01100), Fast ECoT (2506.07639) |
| 8 | `VLA latent conditioning lightweight policy drone quadrotor slow VLM fast controller arXiv` | Web search | ~9 | No drone dual-system paper with a latent interface found; mostly surveys and RL-control work |
| 9 | Official pages: figure.ai/news/helix, research.nvidia.com/labs/gear/gr00t-n1_5, deepmind.google Gemini Robotics On-Device blog | Web | 3 fetched | All verified |

Totals: about 65 search hits screened, 34 candidate records fetched, **27 included** (25 core + 2 background), **7 verified but excluded as off-focus**, **2 not verified (unverifiable claims)**.


---

## 02-latency-async-edge-cloud

All searches were run on 2026-10-27 with the WebSearch tool (Google-like engine, US). "Hits" means the result links returned (about 9–10 per query). Relevant hits are the ones taken forward to verification.

| # | Query | Hits | Relevant hits taken forward |
|---|---|---|---|
| 1 | Real-time execution of action chunking flow policies arXiv | 9 | RTC (2506.07339) |
| 2 | training-time action conditioning real-time chunking VLA latency arXiv 2025 | 9 | TT-RTC (2512.05964), REMAC (2601.20130), PI blog |
| 3 | SmolVLA asynchronous inference arXiv | 10 | SmolVLA (2506.01844), async comparison (2605.08168) |
| 4 | VLASH asynchronous VLA inference future state aware | 9 | VLASH (2512.01031) |
| 5 | A2C2 action chunking correction head latency VLA arXiv | 9 | A2C2 (2509.23224), Action ControlNet (2606.25985) |
| 6 | FogROS2 cloud robotics ROS 2 offloading arXiv | 9 | FogROS2 (2205.09778), FogROS2-PLR (2410.05562) |
| 7 | edge-cloud collaborative vision-language-action model inference robot 2025 2026 arXiv | 9 | RAPID/ActECC (2603.07949), RoboECC (2603.20711), Beyond LLM Serving (2610.05062), Cross-platform scaling (2509.11480) |
| 8 | Bidirectional Decoding action chunking closed-loop resampling arXiv | 9 | BID (2408.17355) |
| 9 | Streaming flow policy action trajectories as flows arXiv | 9 | SFP (2505.21851) |
| 10 | acting in delayed environments non-stationary Markov policies arXiv | 10 | Derman et al. (2101.11992) |
| 11 | speculative decoding vision-language-action model acceleration arXiv | 10 | Spec-VLA (2507.22424); HeiSD (not verified) |
| 12 | Understanding Asynchronous Inference Methods for VLA 2605.08168 | 9 | confirms 2605.08168 |
| 13 | RoboECC edge-cloud collaborative deployment VLA arXiv | 9 | CloudEdgeVLA (2608.00569), RoboECC |
| 14 | VLA network latency jitter packet loss robustness remote inference robot policy | 9 | SPO (2603.19418), FutureRTC (2607.24008), AsyncShield (2604.24086), CALF (2603.12543) |
| 15 | Real-time chunking Physical Intelligence blog | 10 | PI blog (Tier 3 companion to RTC) |
| 16 | CALF communication-aware learning framework ... | 9 | CALF (2603.12543) |
| 17 | Reinforcement learning with random delays Bouteiller arXiv | 10 | DCAC (2010.02966) |
| 18 | DeeR-VLA dynamic inference early exit ... | 10 | DeeR-VLA (2411.02359) |
| 19 | delay-aware action chunking VLA latency embedding conditioning variable delay 2026 arXiv | 9 | Real-Time EXPO-FT (2609.18207), Legato (2602.12978), DEFLECT (not verified) |
| 20 | "FogROS2" ICRA 2023 IEEE Xplore | 9 | venue/DOI for FogROS2 |
| 21 | "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" RSS 2023 | 10 | venue for ACT |

ACT (2304.13705) and Diffusion Policy (2303.04137) came from the seed list and were checked directly on arXiv.

**Verification.** Every included entry was checked on its arXiv abs page (title, authors, date, comments). Most were also read on the arXiv HTML full text or the PDF. HTML full texts were read through a summarising fetch tool, so all numbers come from that extraction. Each one should be spot-checked against the PDF before it is cited in a paper.

**Read-scope labels used below:**
- `full_text*`: full HTML read through the summariser.
- `partial_full_text`: PDF text grepped for specific passages.
- `abstract_only`


---

## 03-small-onboard-policies

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


---

## 04-multirate-sensor-fusion

| # | Query / action | Tool | Useful hits |
|---|---|---|---|
| 1 | "Reactive Diffusion Policy slow-fast visual-tactile arXiv" | WebSearch | 2503.02881 (RSS 2025) |
| 2 | "ForceVLA force-aware mixture of experts VLA" | WebSearch | 2505.22159 (NeurIPS 2025) |
| 3 | "torque-aware vision-language-action TA-VLA" | WebSearch | 2509.07962 (CoRL 2025); TaF-VLA 2601.20321 and AT-VLA 2605.07308 surfaced |
| 4 | "FoAR force-aware reactive policy" | WebSearch | 2411.15753 (RA-L 2025) |
| 5 | "FACTR force-attending curriculum" | WebSearch | 2502.17432 (RSS 2025); FACTR 2 2606.12406 surfaced |
| 6 | Agarwal et al. egocentric vision locomotion | WebSearch + PDF | 2211.07638 (CoRL 2022) |
| 7 | Miki et al. perceptive locomotion | WebSearch + ar5iv | 2201.08117 (Sci. Robot. 2022) |
| 8 | Robot Parkour Learning; Extreme Parkour | WebSearch + PDF | 2309.05665 (CoRL 2023); 2309.14341 (ICRA 2024) |
| 9 | MResT multi-resolution sensing | WebSearch + HTML | 2401.14502 (CoRL 2023) |
| 10 | ViNL | WebSearch + HTML | 2210.14791 (ICRA 2023) |
| 11 | TLIO learned inertial odometry | WebSearch + ar5iv | 2007.01867 (RA-L 2020) |
| 12 | "multi-rate asynchronous sensor fusion transformer robot policy ... 2025" (extended) | WebSearch | LAG-Fusion 2607.17257; DeMUSE 2602.19764 |
| 13 | mTAN; CfC; S5 irregular sampling | WebSearch + abs/ar5iv | 2101.10318; 2106.13898; 2208.04933 |
| 14 | Liquid networks drone flight (Chahine) | WebSearch + PDF | Sci. Robot. 8, eadc8892 (2023) |
| 15 | Swift champion drone racing | WebSearch + PMC | Nature 620 (2023), PMC10468397 |
| 16 | Selective sensor fusion neural VIO | WebSearch + ar5iv | 1903.01534 (CVPR 2019) |
| 17 | "ImplicitRDP slow-fast force vision 2025 2026" | WebSearch + HTML | 2512.10946 |
| 18 | "event camera learned control policy IMU quadrotor" | WebSearch + HTML | 2411.03303; 2410.14685 surfaced |
| 19 | "robot policy timestamp embedding asynchronous multimodal ... sampling rates" | WebSearch | Chronos Policy 2610.13929 (via deeplearn.org, then confirmed on arXiv) |
| 20 | Loquercio high-speed flight in the wild | WebSearch + ar5iv | 2110.05113 (Sci. Robot. 2021) |
| 21 | See, Hear, and Feel | arXiv abs + ar5iv | 2212.03858 (CoRL 2022) |
| 22 | HiP; MT-ACT/RoboAgent; Hierarchical Diffusion Policy | WebSearch | 2309.08587; 2309.01918; 2403.03890 (verified, then excluded as off-theme) |

Counts: about 22 query/fetch rounds, about 40 candidate records seen, 24 included, 9 verified-but-excluded or not assessed (Section 7).


---

## 05-latent-conditioning-and-testbeds

| # | Query / action | Tool | Hits screened | Kept |
|---|---|---|---|---|
| 1 | `distilling vision-language-action model into small student policy 2025 arXiv` | WebSearch | 9 | RPD 2503.05833, ActDistill 2511.18082 (LVLM2P 2505.11221 later excluded) |
| 2 | `task-oriented semantic communication robot policy embedding compression edge cloud arXiv` | WebSearch | 9 | Nakanoya et al. 2011.03216 |
| 3 | `HyPoGen hypernetwork policy generation ICLR 2025` | WebSearch | 9 | HyPoGen (ICLR proceedings), HyperPPO 2309.16663 |
| 4 | `frozen VLM embeddings conditioning small diffusion policy latent from VLM lower frequency arXiv robot` | WebSearch | 9 | Subedi et al. 2506.14507 |
| 5 | `vision-language-action split inference compress latent features transmitted over wireless bandwidth robot arXiv 2025` | WebSearch | 9 | ComVLA 2609.07838, DEM 2609.18374 |
| 6 | `robot policy goal conditioning dropout classifier-free guidance diffusion policy language robustness missing goal` | WebSearch | 9 | none kept (CFG-DP 2510.09786 checked, then excluded) |
| 7 | `student network size versus performance teacher-student distillation locomotion policy ablation parameters` | WebSearch | 10 | X-Distill 2601.11269 |
| 8 | `Make-An-Agent generalizable policy network generator behavior-prompted diffusion arXiv` | WebSearch | 9 | Make-An-Agent 2407.10973 |
| 9 | `ablation VLM latent update interval stale conditioning fast policy success rate dual-system asynchronous arXiv 2025 2026` | WebSearch | 10 | UniFS 2606.22794, DuoCore-FS 2512.20188 |
| 10 | Seed-list verification: 1511.06295, 1912.12294, 1710.06542, 2010.11251, 2107.04034, 2110.05113, 2405.14853, 2412.09858, 1903.01973, 2405.04798, 2410.11758, 2505.06111, 2212.06817, 2202.02005, 2410.05273, 2410.08001, 2509.09372, 2211.15457 | WebFetch (arXiv abs; ar5iv for 2107.04034, 1511.06295, 2212.06817, 2410.05273, 2202.02005) | 18 | 18 |
| 11 | `Isaac Lab arXiv 2025 "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning"` | WebSearch | 9 | Isaac Lab 2511.04831 |
| 12 | `LeRobot arXiv paper Hugging Face open-source library end-to-end robot learning SO-100` | WebSearch | 10 | LeRobot 2602.22818 |
| 13 | `OpenUAV OpenFly UAV vision-language navigation benchmark realistic simulator arXiv` | WebSearch | 9 | OpenUAV 2410.07087 (OpenFly 2502.18041 not fetched, so unverified) |
| 14 | Testbed verification: 2103.02142 (+ar5iv), 2309.12825, 2503.01471, 2009.00563, 1705.05065 (+ar5iv), 2410.00425, 2306.03310, 2112.03227, 2405.05941, 2406.02523, 1910.10897, 2308.06735, 2406.14240, 2511.04831 (HTML) and repos/docs: gym-pybullet-drones, OmniDrones, aerial_gym_simulator (+docs), ManiSkill, SimplerEnv, AirSim, IsaacLab (+sensor docs), flightmare, CALVIN, LIBERO, Metaworld, robocasa, PX4-Autopilot, PX4 simulation docs, robosuite sensor docs | WebFetch | 29 | 17 testbeds |

**Totals:** 30 Part A sources and 17 Part B testbeds verified and included. 12 items are listed under "Unverified / excluded".

