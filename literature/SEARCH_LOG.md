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

## 07-learned-filtering-in-policies

Search date: 2026-10-29.

Rows are prefixed by slice: A = learned filters + delays, B = innovation gating + event triggers, C = plan + learned feedback.

| # | Query / action | Tool | Useful hits |
|---|---|---|---|
| A1 | Read `literature/PROTOCOL.md`, Theme 4 (format), Theme 2 bib titles (to avoid duplicates) | Read/grep | Theme 2 already holds Derman21 (delayed MDP) and Ramstedt21 (DCAC, random delays): not repeated here |
| A2 | arXiv API id_list batch: 2107.10043, 1605.07148, 1805.11122, 1805.08975, 1511.05121, 1605.06432, 1905.07357, 1811.04551, 2012.14313, 2010.10201, 2310.18534, 2111.11344, 1806.02426, 2005.05440, 1810.07286, 2403.12309, 2205.05569 | curl + arXiv API | All 17 resolved. Metadata copied from API (titles, authors, comments/journal_ref/DOI) |
| A3 | Crossref `query.bibliographic`: "Update with out-of-sequence measurements in tracking: exact solution" | Crossref API | 10.1109/taes.2002.1039398 (TAES 38(3) 2002); SPIE 2000 conference version 10.1117/12.392006 |
| A4 | Crossref: "Incorporation of time delayed measurements in a discrete-time Kalman filter" | Crossref API | 10.1109/cdc.1998.761918 (37th CDC, vol. 4, pp. 3972–3977) |
| A5 | Crossref: "Optimal update with out-of-sequence measurements" (Zhang, Li, Zhu) | Crossref API | 10.1109/tsp.2005.847830 (verified, excluded) |
| A6 | Crossref: "One-step solution for the multistep out-of-sequence-measurement problem in tracking" | Crossref API | 10.1109/taes.2004.1292140 (TAES 40(1) 2004) |
| A7 | Crossref: "Markov decision processes with delays and asynchronous cost collection" | Crossref API | 10.1109/tac.2003.809799 (verified, excluded) |
| A8 | Crossref: "Learning and planning in environments with delayed feedback" | Crossref API | 10.1007/s10458-008-9056-7 (AAMAS journal 18(1) 2008) |
| A9 | "neural network Kalman filter out-of-sequence measurements delayed measurements learning arXiv" | WebSearch | arXiv:2201.08020 (LSTM estimator with age-of-information input); Stone Soup OOSM docs (grey, not used) |
| A10 | "Addressing Signal Delay in Deep Reinforcement Learning ICLR 2024 arXiv" | WebSearch + ICLR proceedings page | ICLR 2024 proceedings entry (no arXiv version found, see #12) |
| A11 | "learned policy observation delay compensation robot state history replay predictor sim-to-real latency" | WebSearch | TSRL (Sandha et al., CoRL 2020, PMLR 155); DA-DP, KRLST/LDN soft-robot delay compensation (2504.12428) surfaced, not assessed |
| A12 | arXiv search `all:"signal delay" AND all:reinforcement`; `ti:"Sim2Real Transfer" AND ti:delays` | arXiv API | No arXiv copy of Wang24 or Sandha20. Surfaced 2603.01553 (State-Action Inpainting Diffuser, same group as Wang24) and 2209.15216 (time delay in sim2real) |
| A13 | ICLR proceedings page for Wang24; PMLR v155 page for Sandha20 | WebFetch | Title, authors, venue, abstract confirmed |
| A14 | Full-text PDF download + pdftotext for 12 arXiv papers + Wang24 + Sandha20 | curl/pdftotext | 14 full texts |
| A15 | Crossref works/<doi> for KalmanNet (TSP 70:1532–1547), Kloss (Auton. Robots 45(4):561–578), Bar-Shalom02, Larsen98, Walsh08 | Crossref API | Venues confirmed; Crossref holds no abstracts for the IEEE/Springer items |
| A16 | Semantic Scholar by DOI (abstracts) | S2 API | Larsen98: open-access pointer to DTU Orbit; Walsh08/Bar-Shalom02 abstracts elided |
| A17 | DTU Orbit record for Larsen98 | WebFetch | Abstract retrieved; PDF download blocked (bot check) |
| A18 | Bar-Shalom 2002 / 2004 abstracts | WebSearch (×2), SPIE and IEEE pages (WebFetch returned empty) | Abstract text only via search snippets |
| A19 | MathWorks "Handle out-of-sequence measurements with filter retrodiction" | WebFetch | Vendor doc comparing reprocessing vs retrodiction (grey, not included; cited in notes only) |
| A20 | Walsh et al. abstract | WebSearch | Abstract summary via indexer (Springer page redirects to login) |
| A21 | Venue checks: PMLR v97 (PlaNet, RKN), v80 (DVRL), v162 (CRU), v155 (ac-RKN), RSS 2018 index (DPF), NeurIPS 2016 (BKF), NeurIPS 2023 (MTS3) proceedings index pages | curl + grep | All 8 titles found on the official proceedings index |
| A22 | "Karamzade ... RLC 2024 OR Reinforcement Learning Journal"; RLJ volume page | WebSearch + curl | RLJ vol. 5, pp. 2123–2139, 2024 confirmed on rlj.org |
| A23 | dblp API venue lookups | curl | Blocked (bot challenge), so replaced by #21 |
| B1 | Read `literature/PROTOCOL.md`, Theme 4 entries and bib, and the bib keys of all themes (to avoid duplicates) | Bash | Format reference. DP-VLA (event-triggered slow calls) is already in Theme 1, and ActECC replanning is already in Theme 2 |
| B2 | "adaptive inference frequency vision-language-action model when to re-query trigger replanning 2025 arXiv" | WebSearch (extended) | 2604.04161 (AAC), 2510.01389 (INSIGHT), 2608.06434, 2604.02965 |
| B3 | "learned forward model predict wrist force torque anomaly detection manipulation failure slip LSTM-VAE" | WebSearch | 2011.05817 (FINO-Net), 2509.26308 (Grambow et al.), 2209.06019 (proactive slip) |
| B4 | "event-triggered learning Solowjow Trimpe arXiv; event-based state estimation innovation triggering Trimpe D'Andrea" | WebSearch | 1904.03042, 1511.05223, TAC 2014 variance-based triggering |
| B5 | arXiv API batch (16 IDs): 1904.03042, 1511.05223, 1809.05152, 2107.10043, 1711.00614, 2011.05817, 2509.26308, 2410.04640, 2503.08558, 2604.04161, 2608.06434, 2304.12046, 2307.01928, 2510.01389, 2209.06019, 2604.02965 | curl, arXiv API | All 16 resolved; titles and authors copied |
| B6 | Crossref bibliographic search: Tabuada 2007; Heemels/Johansson/Tabuada 2012; Trimpe & D'Andrea 2014; Mehra & Peschon 1971; Bar-Shalom et al.; Haddadin et al. 2017; Wolpert et al. 1995; Willsky 1976 | curl, Crossref | All 8 DOIs resolved |
| B7 | "learning when to query large model planner robot fast policy decides to call slow VLM uncertainty-triggered arXiv" | WebSearch (extended) | 2610.05025 (TUD), 2608.23224 (TOWN-VLA), 2606.20458, 2606.16995 |
| B8 | "proprioceptive prediction error detect slip or collision learned forward dynamics model ... residual torque neural network" | WebSearch | Only general hits; led to the Heo et al. RA-L 2019 Crossref lookup |
| B9 | arXiv API: 2610.05025, 2608.23224, 2606.20458, 2606.16995; Crossref: Heo et al. 2019 | curl | Resolved |
| B10 | Full text: TUD 2610.05025v2, Sentinel 2410.04640v2, FAIL-Detect 2503.08558v3 | WebFetch (arXiv HTML) | Read |
| B11 | Full text: AAC 2604.04161v2, TOWN-VLA 2608.23224v1, Event-triggered learning (ar5iv 1904.03042), DRL for ETC (ar5iv 1809.05152) | WebFetch | Read. TOWN-VLA excluded because its trigger is oracle/label-based |
| B12 | Crossref venue checks: Baumann CDC 2018, Honda ICRA 2024, Park RA-L 2018, FINO-Net IROS 2021, FAIL-Detect RSS 2025; PMLR v229 (KnowNo) and v270 (Sentinel) index pages | curl | All venues confirmed |
| B13 | Full text: Park LSTM-VAE (arXiv HTML), Grambow et al. (arXiv HTML), KalmanNet (arXiv HTML), Honda et al. (arXiv HTML) | WebFetch | Read |
| B14 | "Failure Prediction at Runtime for Generative Robot Policies" | WebSearch + arXiv API | 2510.09459 (FIPER, NeurIPS 2025) verified, not assessed |
| B15 | Crossref + Semantic Scholar abstracts for the classical DOI works | curl | Abstract only for Wolpert 1995. IEEE/Elsevier abstracts elided. Open-access links for Tabuada, Heemels (KTH DiVA), Haddadin (Sapienza postprint), Willsky (NASA NTRS tech report) |
| B16 | Downloaded Tabuada (UCLA PDF) and Willsky (NASA NTRS ESL-P-633 tech-report version) and ran pdftotext | curl + pdftotext | Read. The Haddadin postprint and the Heemels DiVA link returned HTML or a bot wall, so they were not read |
| B17 | doi.org → ieeexplore pages for Trimpe 2014, Heemels 2012, Haddadin 2017 | WebFetch | Empty pages. IEEE scraping was not pursued (blocked by a permission rule) |
| B18 | arXiv PDF of 1511.05223 (Trimpe), via pdftotext | WebFetch + pdftotext | Read. The PDF states the IET CTA DOI. Crossref confirms the retitled published version |
| B19 | "Heemels Johansson Tabuada 2012 ... pdf" and the bibbase page | WebSearch + WebFetch | No abstract or full text found; 2009.12783 (ML-in-ETC survey) surfaced |
| B20 | "robot manipulation learned model predicts expected force torque ... prediction error policy re-planning arXiv" | WebSearch (extended) | 2610.30117 (CW-WAM), 2606.10818 (IMPACT), 2606.12406 (FACTR 2), 2608.21592 |
| B21 | Full text: CW-WAM 2610.30117, IMPACT 2606.10818; arXiv API for both; KnowNo abstract page | WebFetch + curl | Read |
| C1 | Read `PROTOCOL.md`, Theme 4 format; grep all arXiv IDs and keys in `themes/*.md` for overlap | Bash | Overlap list (≈150 IDs); A2C2, RTC, SFP, VLASH, AsyncShield, RDP already covered |
| C2 | grep themes for "residual\|smith\|tube\|DCT\|B-spline\|movement primitive\|waypoint" | Bash | Only A2C2 / ACNet / FutureRTC residual heads (Theme 2); Neural Lander excluded in Theme 3 |
| C3 | "Policy Decorator residual policy model-agnostic online refinement large policy arXiv" | WebSearch | 2412.13630 (ICLR 2025) |
| C4 | "demonstration-efficient guided policy search imitation of robust tube MPC Tagliabue arXiv" | WebSearch | 2109.09910 (ICRA 2022 submission); 2306.00286 (T-RO 2024); 2210.10127 |
| C5 | "DTC deep tracking control model-based planning reinforcement learning legged Jenelten arXiv" | WebSearch | 2309.15462 (Sci. Robot. 2024) |
| C6 | "BEAST B-spline encoded action sequence tokenizer imitation learning arXiv" | WebSearch | 2506.06072 (NeurIPS 2025) |
| C7 | "learned Smith predictor neural network delay compensation robot teleoperation reinforcement learning arXiv" | WebSearch | 2605.15480 (residual RL, stochastic delay); 2504.12428 (learned Smith predictor) |
| C8 | "residual policy on top of vision-language-action model actions lightweight correction real-time arXiv 2025" | WebSearch | PLD 2511.00091; VLA-Corrector; J-PARC 2607.01804 |
| C9 | "high-level policy waypoints low-level learned tracking controller hierarchical manipulation ... plan-conditioned" | WebSearch | HiWET 2602.06341, LaCoLoco, 2506.22827 (surfaced only) |
| C10 | "delay-aware model-based reinforcement learning continuous control arXiv Chen" | WebSearch | 2005.05440 (Neurocomputing 2021) |
| C11 | "ProDMP probabilistic dynamic movement primitives deep learning arXiv; movement primitive diffusion" | WebSearch | ProDMP 2210.01531; MPD 2312.10008 (RA-L 2024) |
| C12 | "continuous-time action chunk spline evaluated arbitrary time inference latency robot policy arXiv 2026" | WebSearch | ChunkFlow 2607.12992; RTC / Legato (already in Theme 2) |
| C13 | arXiv API batch metadata for 18 candidate IDs (+2509.19301 separately) | curl | All 19 resolved; titles/authors/comments/DOIs recorded |
| C14 | Crossref bibliographic queries for venues (Johannink, Ankile, Koller, Tagliabue, Chen, Kaufmann, Scheikl, Silver) | curl | DOIs for ICRA 2019, ICRA 2025, ICRA 2026 (ResFiT surfaced here), CDC 2018, T-RO 2024, Neurocomputing 2021, RSS 2020, RA-L 2024; no venue for Silver et al. |
| C15 | dblp search API for venues | curl | HTTP 429 (rate limited), abandoned |
| C16 | OpenReview search: RT-Trajectory, Policy Decorator, BEAST, PLD | curl | ICLR 2024 spotlight; ICLR 2025 poster; NeurIPS 2025 poster; ICLR 2026 poster |
| C17 | "RT-Trajectory ICLR 2024 openreview"; "FAST ... RSS 2025 roboticsproceedings" | WebSearch | RSS XXI p012 page fetched, DOI 10.15607/RSS.2025.XXI.012 |
| C18 | Crossref `/works/<doi>` for 8 DOIs (pages, volume, author lists) | curl | All resolved; author lists match arXiv |
| C19 | Full-text reads: arXiv HTML (2605.15480, 2504.12428, 2511.00091, 2412.13630, 2407.16677, 2306.00286, 2501.09747, 2506.06072v3, 2311.01977, 2509.19301); ar5iv (1812.06298, 1812.03201, 2309.15462, 2006.05768, 1803.08287, 2005.05440) | WebFetch | Locators below |
| C20 | PDF text (pdftotext) for 2309.15462, 2006.05768, 1812.06298 to check rate/staleness statements | curl + pdftotext | DTC deployment and optimizer-rate ablation; DDA async branches; RPL hook results |
| C21 | ar5iv for MPD 2312.10008; arXiv HTML 2312.10008 | WebFetch | ar5iv conversion failed; arXiv HTML 404 → MPD not read, excluded |

## 08-predictive-latents-jepa

Search date: 2026-10-30.

| # | Query / action | Tool | Useful hits |
|---|---|---|---|
| 1 | Read `literature/PROTOCOL.md`, Theme 7 (format), eprint list and titles of all `literature/bib/*.bib` | Bash/grep | Format reference; LAPA, UniVLA, Karamzade24, MinD already present |
| 2 | arXiv API id_list (16 IDs): 2301.08243, 2404.08471, 2506.09985, 2307.12698, 2411.04983, 2007.05929, 2203.04955, 2310.16828, 2206.08332, 2502.11831, 2212.03319, 2401.08898, 2501.14622, 2502.14819, 2507.19468, 2412.03572 | Python urllib + arXiv API | All 16 resolved; titles and authors copied from the API |
| 3 | "JEPA robot manipulation policy latent predictor arXiv 2025 2026 vision-language-action" | WebSearch (extended) | VLA-JEPA 2602.10098, V-JEPA Policy 2609.37250, JEPA-WAM 2608.09381, StageWAM 2608.10780, Demo-JEPA 2605.20811, TD-JEPA 2510.00739, JEPA Policy 2609.09630, PSG-JEPA 2608.06799 |
| 4 | "probing V-JEPA versus CLIP SigLIP features physical state dynamics robot policy comparison arXiv" | WebSearch (extended) | JEPA-VLA 2602.11832, ALPS 2610.30120, V-JEPA 2.1 2603.14482, MotionJEPA 2609.23881, Delta-JEPA 2606.31232 |
| 5 | "latent world model compensate observation delay roll forward stale embedding robot policy high-frequency lightweight predictor" | WebSearch (extended) | "LagComp" (ResearchGate only), RoboJEPA (arXiv 2610.10515), ContactGuard 2608.13438, DREAM-Chunk 2606.18589, LeWM (via blog) |
| 6 | arXiv API id_list (15 IDs): 2610.10515, 2603.14482, 2602.11832, 2602.10098, 2609.37250, 2608.06799, 2610.30120, 2608.13438, 2606.18589, 2510.00739, 2609.09630, 2608.09381, 2608.10780, 2609.23881, 2606.31232 | arXiv API | All resolved. 2610.10515 is RoboJEPA, not LagComp. 2608.10780 is now titled "StageWAM" (v3) |
| 7 | "LagComp Compensating Observation-to-Action Latency Through Latent Feature Forecasting arXiv" (3 query variants) | WebSearch | No arXiv / DOI / project page found → Unverified |
| 8 | "LeWorldModel LeWM stable end-to-end JEPA world model from pixels arXiv" | WebSearch | arXiv 2603.19312 |
| 9 | "distilled small latent dynamics model on-robot high frequency world model embedded Jetson latent prediction policy delay" | WebSearch | SkyJEPA 2606.23444, SmolWAM (NeurIPS 2026 virtual page; HF model card), Think-Like-a-WM 2609.24682, LaWAM 2606.15768, Koopman distillation 2605.24924 |
| 10 | arXiv API: 2603.19312, 2606.23444, 2609.24682, 2606.15768 | arXiv API | All resolved |
| 11 | OpenReview API and forum page for LeCun "A Path Towards Autonomous Machine Intelligence" (id BZ5a1r-kVsf) | curl, WebFetch | Blocked by OpenReview browser challenge (403 / verification page) → Unverified |
| 12 | Full-text PDF download + pdftotext for 19 papers (all included arXiv papers) | curl/pdftotext | 19 full texts |
| 13 | arXiv HTML of V-JEPA 2 (Table 2, Table 3, Sec. 3–4) | WebFetch, then cross-checked against pdftotext | V-JEPA 2-AC success rates, 4 fps, 16 s/action confirmed |
| 14 | Venue checks: PMLR v267 index (DINO-WM); ICLR 2024 proceedings index (TD-MPC2, Ni et al.); mlanthology TMLR 2024 (V-JEPA); Crossref bibliographic (I-JEPA); iclr.cc virtual page via WebSearch (SPR) | curl + grep, Crossref API, WebSearch | DINO-WM = ICML 2025, PMLR 267:79115–79135. TD-MPC2, Ni24 = ICLR 2024. V-JEPA = TMLR 2024. **I-JEPA = CVPR 2023** (DOI 10.1109/cvpr52729.2023.01499, pp. 15619–15629); its arXiv comment says "ICCV", which is wrong. SPR = ICLR 2021 (spotlight) |
| 15 | WebSearch for venues of LeWM, Garrido25 (intuitive physics), JEPA-VLA | WebSearch | No venue confirmed for any of them → treated as preprints |
| 16 | "probing vision-language model hidden states for object position physical state robot manipulation VLA" | WebSearch | 2502.04558 (OpenVLA probing); arXiv API + HRILab page (lists "Proceedings of ICAD" 2025, not confirmed on a proceedings site) |
| 17 | "asynchronous VLA latent prediction stale observation forecast latent features ... latency compensation 2026" | WebSearch | FutureRTC, Action ControlNet (already in bib), DEFLECT 2605.19294, Causal World Modeling 2601.21998 (not assessed) |
