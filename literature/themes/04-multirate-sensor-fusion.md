# Theme 4 (RQ4): Multi-rate / asynchronous sensor fusion in learned robot policies

- **RQ4:** How do learned policies fuse sensors that arrive at different rates (IMU, vision, force/torque, proprioception)?
- **Search date:** 2026-10-27 (covers up to the present)
- **Screener:** one AI agent. Every entry below was resolved to an arXiv / DOI / proceedings page in this session.
- **Read-scope labels:** `full_text` = arXiv HTML, ar5iv or PDF text was retrieved and the cited sections were read (through automated extraction, so the locators should be spot-checked). `partial` = only some sections were retrieved. `abstract_only` = only the abstract or metadata page.

## 1. Search log

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

## 2. Inclusion notes

- **Included:** learned policies, or learned estimators feeding a policy, that take at least two modalities at different native rates, or that give a reusable mechanism for irregular or asynchronous time (mTAN, CfC, S5, Chronos). Learned odometry (TLIO, Selective VIO) is included because IMU-rate integration with low-rate learned updates is the same pattern our fast model needs.
- **Excluded:** hierarchical planners whose levels do not fuse sensors at different rates (HiP, MT-ACT, HDP; these belong to RQ1), and papers whose rate handling could not be checked beyond the abstract.
- **Tiers:** Tier 1 = peer-reviewed venue with real-robot or standard-benchmark evidence. Tier 2 = arXiv preprint, or peer-reviewed with narrow evidence. Tier 3 = methodological ML papers that have not been tested on robots.

## 3. Comparison table

| Paper | Year | Modalities & rates | Alignment mechanism | Output rate | Key ablation (locator) |
|---|---|---|---|---|---|
| RDP (Xue et al.) | 2025 | RGB 12 Hz; GelSight 25 fps / MCTac 30 fps; joint torque 120 Hz downsampled to 24 Hz | Two levels: slow latent diffusion (1–2 Hz) emits latent action chunk; fast GRU decoder concatenates latest tactile/force embedding each step; latency matching drops stale first steps | 24 Hz | Chunk size / temporal-ensemble sensitivity (Tab. V); relative trajectory + latency matching (Fig. 10) |
| ImplicitRDP (Chen et al.) | 2025 | Vision 1–2 Hz; force ≥10 Hz | One network: vision tokens (slow) + GRU-encoded force tokens; causal attention mask so action step *s* sees forces f₁…fₛ only | Force rate inside chunk | Full 18/20 vs vision-only DP 0/20 (box flip); w/o SSL 4/20 (results table) |
| LAG-Fusion (He et al.) | 2026 | Vision policy ~2 Hz; F/T 100 Hz input, force policy 25 Hz | Separate per-modality diffusion policies at native rate; delayed guidance rebased into current action frame; staleness-dependent horizon weights | 25 Hz | 95%/60% vs sync composition 100%/25% (Tab. 1); force policy 2 Hz vs 25 Hz gap (Fig. 6a) |
| FoAR (He et al.) | 2025 | RGB-D point cloud; F/T 100 Hz, 200-step (2 s) window | Separate transformer F/T encoder over full-rate window; contact-probability gate φ(t) mixes force feature in | 10 Hz | F/T at 2 / 10 / 100 Hz (Tab. III); force-token & concat baselines worse (Tab. I) |
| ForceVLA (Yu et al.) | 2025 | 2× RGB-D 30 fps; 6-axis F/T; TCP pose | Timestamp sync of streams; single linear force token fused **after** VLM via MoE | Flow-matching action chunks | Late MoE 80% vs early fusion 0–55% (Tab. 3) |
| TA-VLA (Zhang et al.) | 2025 | 3× RGB; joint torque from motor current; joint angles | 10-frame torque history → MLP → one token into the action decoder; rate sync not described | π₀ 50-step / RDT 64-step chunks | Decoder > encoder (Tab. 1); single history token > per-frame (Tab. 3) |
| FACTR (Liu et al.) | 2025 | Images; external joint torque | One MLP force token concatenated with vision tokens; curriculum blurs vision to force use of torque | Chunk k=100; 500 Hz low-level loop | Curriculum operator/scheduler ablation (Tab. II); unseen-object gain (Fig. 5) |
| MResT (Saxena et al.) | 2023 | 3rd-person cam 5 Hz (frozen VLM); wrist cam 20 Hz (ResNet-18); F/T+proprio 75 Hz (linear) | Per-modality encoders at native rate; slow features cached and repeated at 75 Hz; cross-attention fusion | 75 Hz | Single-rate 4.2% vs multi-rate 73.6% on MT-Dynamic (Tab. 3) |
| Chronos Policy (Wang et al.) | 2026 | Real: RGB ~20 Hz, pose 20 Hz, F/T ~90 Hz (perturbed); sim F/T 100 Hz | Physical-time RoPE on tokens (capture time relative to request); temporal randomization of rates/delays | 16 actions at 20 Hz | Time-RoPE vs additive time embedding vs DP under timing perturbation (Fig. 3, Tab. 3) |
| See, Hear, and Feel (Li et al.) | 2022 | Vision 60 Hz; audio 44.1→16 kHz (mel); GelSight 30 Hz | Fixed 3 s window per modality (6 frames stacked; audio spectrogram); self-attention over modality×time tokens | 10 Hz | V+A+T 100% vs V-only 25% (Tab. 1); concat 60% |
| Agarwal et al. | 2022 | Depth ~10 Hz (100±20 ms); proprio 50 Hz | Depth backbone on Jetson sends latent via UDP; GRU at 50 Hz uses latest latent; train-time randomized update period 80–120 ms and latency 10–30 ms | 50 Hz (PD 400 Hz) | Blind vs vision; monolithic vs RMA (main results) |
| Miki et al. | 2022 | Elevation map 20 Hz (height samples); proprio | Policy at 50 Hz samples latest map; recurrent belief encoder with learned gate on exteroception | 50 Hz | GRU gate vs no gate (Supp. S9) |
| Robot Parkour (Zhuang et al.) | 2023 | Depth embedding 10 Hz; proprio 100 Hz ROS | Visual CNN runs asynchronously; GRU policy at 50 Hz fetches latest embedding; depth/proprio latency randomized | 50 Hz | GRU vs MLP, distill vs no-distill (Tab. 2) |
| Extreme Parkour (Cheng et al.) | 2024 | Depth 10±2 Hz; proprio | Depth backbone 10 Hz → latent via UDP; constant enforced 80 ms depth latency, 16 ms proprio latency; GRU base policy | 50 Hz | Direction/scandot variants (Sec. 4) |
| ViNL (Kareer et al.) | 2023 | Depth + proprio; nav 2 Hz | CNN depth + LSTM proprio predict terrain latent; nav policy 2 Hz feeds velocity to locomotor 50 Hz | 50 Hz | LSTM 73.6% vs MLP 66.8% vs blind 1.2% (Tab. I) |
| Selective VIO (Chen et al.) | 2019 | Images 10 Hz; IMU 100 Hz | Bi-LSTM over IMU samples between frames → one feature per frame; soft/hard masks for fusion | 10 Hz pose | Soft/hard vs direct fusion under corruption and misalignment (Tab. 1) |
| TLIO (Liu et al.) | 2020 | IMU 200 Hz | EKF propagates at IMU rate; network on 1 s window outputs displacement + covariance at 20 Hz as measurement update | 200 Hz state | Learned vs fixed covariance (drift −31%) |
| Swift (Kaufmann et al.) | 2023 | Camera 30 Hz; IMU; VIO + learned gate detector (40 ms on TX2) | Kalman filter fuses VIO and gate detections into state; MLP policy takes state | Policy 8 ms per pass on CPU | Sensorimotor latency 40 ms vs 220 ms human (main text) |
| Loquercio et al. | 2021 | Stereo depth; VIO velocity+attitude; goal dir | Two branches (MobileNet-V3 depth; MLP state) concatenated at each frame; latest VIO used | ~24 Hz (41.6 ms on TX2) | Depth vs RGB; multimodal output (Results) |
| Event obstacle avoidance (Bhattacharya et al.) | 2024 | Events binned at 1/FPS (33 ms); VIO 30 Hz + UKF 150 Hz | Events → binary masks; ViT-LSTM velocity head; state from external UKF | ~73 ms latency | Joint vs separate depth/velocity training (Tab. S1, Fig. 4) |
| Chahine et al. (liquid nets) | 2023 | RGB only (gimbal), Jetson TX2 | CfC/LTC recurrent controller; training sequences of random length "to encourage robustness to ... varying update frequencies" | Velocity setpoints | OOD robustness vs LSTM/CNN baselines |
| CfC (Hasani et al.) | 2022 | Generic irregular series | Elapsed time *t* as explicit input to closed-form gate | n/a | Event/PhysioNet/Walker2d (Tabs. 4,5,7) |
| S5 (Smith et al.) | 2023 | Generic | Per-step discretization Āₖ from Δtₖ via parallel scan | n/a | Irregular pendulum MSE vs CRU, 86× faster (Sec. 6.3, Tab. 3) |
| mTAN (Shukla & Marlin) | 2021 | Generic sparse multivariate | Learned continuous-time embedding + time attention to fixed reference points | n/a | not assessed (abstract_only) |

## 4. Annotated entries

### 4.1 Manipulation: force/tactile + vision

**[RDP] Xue, H., Ren, J., Chen, W., Zhang, G., Fang, Y., Gu, G., Xu, H., Lu, C. (2025).** *Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation.* RSS 2025. arXiv:2503.02881. https://arxiv.org/abs/2503.02881
- **Relevance:** the closest structural analogue to our slow/fast split, with the "fast model" closing the loop on a high-rate modality.
- **Key findings:** a slow latent diffusion policy (1–2 Hz) predicts latent action chunks from 12 Hz images. A fast asymmetric tokenizer (GRU decoder) decodes each step at 24 Hz from concat(latent, latest tactile/force embedding). Torque is sampled at 120 Hz and downsampled to 24 Hz. Optical tactile data is PCA-reduced to 15 dims.
- **Method:** imitation learning from TactAR AR-teleop data, 3 contact-rich tasks.
- **Quality tier:** Tier 1 (RSS); small task count.
- **Method weaknesses:** (a) the fast network cannot process high-frequency images; (b) single-task only; (c) TactAR is less intuitive than direct manipulation. All three are author-acknowledged in Sec. VI. (d) The fast-path rate (24 Hz) is set by the downsampled tactile stream, and native-rate torque was not tested (reader-inferred).
- **Relevance to our setup:** a direct template. The cloud VLA plays the slow latent policy, and the on-robot fast decoder concatenates the latest IMU/torque embedding. Latency matching (dropping stale steps) is a cheap staleness fix.
- Read scope: full_text.

**[ImplicitRDP] Chen, W., Xue, H., Wang, Y., Zhou, F., Lv, J., Jin, Y., Tang, S., Wen, C., Lu, C. (2025).** *ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning.* arXiv:2512.10946. https://arxiv.org/abs/2512.10946
- **Relevance:** folds RDP's two levels into one network that takes asynchronous visual and force tokens.
- **Key findings:** vision runs at 1–2 Hz and force at ≥10 Hz. Force is GRU-encoded, and a causal mask lets action step *s* attend only to forces up to *s*. Results table: full 18/20 on both tasks, vision-only DP 0/20 (box flipping), hierarchical RDP 16/20 and 10/20, w/o SSL 4/20 on box flipping. Virtual-target regularization counters modality collapse.
- **Method:** diffusion policy, real robot, 2 tasks.
- **Quality tier:** Tier 2 (preprint).
- **Method weaknesses:** not yet extended to VLAs or tactile (author-acknowledged, Sec. V). Only two tasks with 20 trials each (reader-inferred).
- **Relevance to our setup:** shows that "high-rate tokens with a causal mask inside one chunk" is an alternative to a separate fast network. For us that would need the whole model on the robot, unless only the decoder part runs on-device.
- Read scope: full_text.

**[LAG-Fusion] He, Z., Fang, H., Tang, S., Lu, C., Fang, H.-S. (2026).** *Asynchronous Multimodal Diffusion Policy Composition via Latency-Aware Guidance Fusion.* arXiv:2607.17257. https://arxiv.org/abs/2607.17257
- **Relevance:** the most direct treatment of rate mismatch *and staleness* between a slow and a fast modality policy.
- **Key findings:** the vision policy (~2 Hz) and the force policy (25 Hz, F/T input at 100 Hz) run at their native rates. Delayed vision guidance is rebased into the current action frame (affine transforms preserving the noise distribution, Eqs. 7–10). Weights shift with staleness across the horizon (vision 0.8→0.2). Tab. 1: 95%/60% success on Flip/Polish-Curve, vs synchronous composition 100%/25% and policy consensus 95%/40%. Fig. 6a: the standalone force policy loses about 20 points when run at 2 Hz instead of 25 Hz.
- **Method:** diffusion policy composition, real robot, parallel gripper.
- **Quality tier:** Tier 2.
- **Method weaknesses:** stability is not fully solved, and there is no hybrid force-position prior, no flow-matching variant and no dexterous-hand test (author-acknowledged, §5). The fixed linear staleness schedule is hand-set (reader-inferred).
- **Relevance to our setup:** the rebasing and staleness-weighting idea carries over to conditioning a fast on-robot policy on a stale 5 Hz cloud latent. Fig. 6a is direct evidence that running the high-rate loop slower costs performance.
- Read scope: full_text.

**[FoAR] He, Z., Fang, H., Chen, J., Fang, H.-S., Lu, C. (2025).** *FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation.* IEEE RA-L. arXiv:2411.15753. https://arxiv.org/abs/2411.15753
- **Relevance:** gives an explicit ablation over force sampling rate.
- **Key findings:** F/T at 100 Hz over a 200-step (2 s) window goes through an MLP→transformer encoder with sinusoidal time encoding. A future-contact predictor gates the force feature. Tab. III (wiping): F/T at 2 Hz scores 0.625, at 10 Hz 0.800, at 100 Hz (full) 0.875. Tab. I: naive force-token / concat variants are no better than the vision-only RISE baseline.
- **Method:** built on RISE (point cloud), with position control and reactive thresholds.
- **Quality tier:** Tier 1.
- **Method weaknesses:** static force thresholds and position-only control limit adaptability (author-acknowledged, Sec. V). Actions are output at 10 Hz, so the high-rate input does not mean high-rate output (reader-inferred).
- **Relevance to our setup:** supports keeping the force/IMU history at native rate inside a small temporal encoder rather than downsampling to camera rate. Naive concatenation can fail.
- Read scope: full_text.

**[ForceVLA] Yu, J., Liu, H., Yu, Q., Ren, J., Hao, C., Ding, H., Huang, G., Huang, G., Song, Y., Cai, P., Zhang, W., Lu, C. (2025).** *ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation.* NeurIPS 2025. arXiv:2505.22159. https://arxiv.org/abs/2505.22159
- **Relevance:** shows where force should enter a large VLA (π₀ base).
- **Key findings:** a single linear-projected 6-axis F/T token is fused after the VLM through a sparse MoE (4 experts, top-1). Tab. 3: MoE before VLM 0%, linear before VLM 55%, concat after VLM 60%, ForceVLA 80%. Streams are aligned by timestamp. The paper reports +23.2% average over π₀ baselines (abstract).
- **Method:** fine-tuned π₀, 5 tasks, new dataset.
- **Quality tier:** Tier 1.
- **Method weaknesses:** uses estimated wrench rather than direct high-fidelity sensing, on costly F/T platforms (author-acknowledged, limitations section). Force is a single sample per step, with no history and no high-rate path (reader-inferred).
- **Relevance to our setup:** late fusion, next to the action head, is what our fast on-robot model already does by construction. Injecting force into the cloud VLM backbone would hurt.
- Read scope: full_text.

**[TA-VLA] Zhang, Z., Xu, H., Yang, Z., Yue, C., Lin, Z., Gao, H., Wang, Z., Zhao, H. (2025).** *Elucidating the Design Space of Torque-aware Vision-Language-Action Models.* CoRL 2025 (PMLR 305). arXiv:2509.07962. https://arxiv.org/abs/2509.07962
- **Relevance:** a systematic design-space study of torque tokenization.
- **Key findings:** decoder insertion beats encoder (Tab. 1). Summarizing a 10-frame torque history into one token beats per-frame tokens, 15/20 vs 9/20 (Tab. 3). Predicting future torque as an auxiliary output helps (Fig. 6, Tab. 9). HSIC analysis shows torque is closer to joint angles than to images (Fig. 3).
- **Method:** π₀ and RDT backbones, real-robot tasks.
- **Quality tier:** Tier 1.
- **Method weaknesses:** current-based torque estimation is sensitive to calibration and thermal drift, scaling to more modalities under a token budget is unclear, and evaluation scope is limited (author-acknowledged, Sec. 8). Rate alignment between torque and camera is not described (reader-inferred).
- **Relevance to our setup:** compress high-rate history into one or a few tokens near the action head. Auxiliary future-torque prediction is a cheap regularizer for the fast model.
- Read scope: full_text.

**[FACTR] Liu, J. J., Li, Y., Shaw, K., Tao, T., Salakhutdinov, R., Pathak, D. (2025).** *FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning.* RSS 2025. arXiv:2502.17432. https://arxiv.org/abs/2502.17432
- **Relevance:** addresses modality collapse, where the policy ignores a sparse high-rate signal.
- **Key findings:** joint torque → MLP → one force token concatenated with vision tokens. A curriculum blurs or downsamples vision with decreasing intensity. Generalization to unseen objects improves by 43% (abstract). Attention to force rises during contact (Fig. 8).
- **Quality tier:** Tier 1.
- **Method weaknesses:** limited torque-sensor precision, the need for external joint-torque sensors, and task-dependent curriculum hyperparameters (author-acknowledged, Sec. VI). Camera and force rates are not reported (reader-inferred).
- **Relevance to our setup:** if the cloud latent dominates, the fast model may ignore IMU/torque. Corrupting the latent with a curriculum during training is a direct countermeasure.
- Read scope: full_text.

**[MResT] Saxena, S., Sharma, M., Kroemer, O. (2023).** *Multi-Resolution Sensing for Real-Time Control with Vision-Language Models.* CoRL 2023 (PMLR 229). arXiv:2401.14502. https://arxiv.org/abs/2401.14502
- **Relevance:** the clearest published "big frozen VLM at low rate + small nets at high rate" design with explicit per-modality rates.
- **Key findings:** a frozen CLIP/MDETR on the third-person camera at 5 Hz, a ResNet-18 on the wrist camera at 20 Hz, and a linear layer on F/T+proprio at 75 Hz. Slow features are cached and reused each 75 Hz step, and fused by cross-attention. Tab. 3: single-resolution 4.2% vs multi-resolution 73.6% on dynamic tasks. Tab. 2: dropping the wrist view drops precise tasks from 55.0% to 7.7%. Fig. 6: cross-attention beats concatenation by about 8%.
- **Quality tier:** Tier 1 (much of the evaluation is in sim).
- **Method weaknesses:** no tactile, unclear whether cross-attention scales beyond 2 sensors, and missing-modality handling left open (author-acknowledged, Sec. 6). Feature reuse ignores staleness, since there is no time stamp on cached features (reader-inferred).
- **Relevance to our setup:** almost one-to-one: a 5 Hz VLM feature conditions a 75 Hz small policy. This is the primary baseline to replicate.
- Read scope: full_text.

**[Chronos] Wang, C., Clark, J., Song, S. (2026).** *Chronos Policy: Learning Robot Policies on Physical Time.* arXiv:2610.13929. https://arxiv.org/abs/2610.13929
- **Relevance:** the only included robot-policy paper that encodes timestamps explicitly per token for multi-rate, delayed streams.
- **Key findings:** capture times (relative to the request) enter as time-aware RoPE (16 coordinate pairs, log-spaced periods 0.1–2 s). Training randomizes per-stream rates (e.g. RGB 12.5–33 Hz, F/T 1–100 Hz in sim) and delays (0–100 ms). Time-RoPE is more robust than additive timestamp embeddings, and randomization without explicit time hurts (Fig. 3, Tab. 3). Output is 16 actions at 20 Hz.
- **Quality tier:** Tier 2 (preprint, 3 weeks old).
- **Method weaknesses:** fixed action rate; variable control frequency is future work (author-acknowledged, Sec. 5). Per-condition numbers were extracted automatically and should be re-checked against Fig. 3 before quoting (reader note).
- **Relevance to our setup:** the strongest mechanism candidate for fusing IMU at 100 Hz, torque at 60 Hz, camera at 20 Hz and a cloud latent with variable age into one fast transformer. The cloud latent's age becomes just another timestamp.
- Read scope: full_text.

**[SHF] Li, H., Zhang, Y., Zhu, J., Wang, S., Lee, M. A., Xu, H., Adelson, E., Fei-Fei, L., Gao, R., Wu, J. (2022).** *See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation.* CoRL 2022. arXiv:2212.03858. https://arxiv.org/abs/2212.03858
- **Relevance:** an early three-modality self-attention fusion at very different native rates.
- **Key findings:** vision 60 Hz, GelSight 30 Hz, contact audio 16 kHz. Each is windowed to 3 s (6 stacked frames; mel spectrogram for audio), then self-attention runs over modality×time tokens. Policy at 10 Hz. Tab. 1: V+A+T 100% vs V-only 25%, and concatenation 60%.
- **Quality tier:** Tier 1.
- **Method weaknesses:** relies on BC demonstrations, and how to fuse optimally is left open (author-acknowledged, conclusion). Fixed-window resampling discards native-rate timing (reader-inferred).
- **Relevance to our setup:** a baseline for "resample everything to a common window". Its 10 Hz output is below our 20 Hz target.
- Read scope: full_text.

### 4.2 Legged locomotion: vision at low rate, proprioception at high rate

**[Agarwal22] Agarwal, A., Kumar, A., Malik, J., Pathak, D. (2022).** *Legged Locomotion in Challenging Terrains using Egocentric Vision.* CoRL 2022 (PMLR 205). arXiv:2211.07638. https://arxiv.org/abs/2211.07638
- **Key findings:** depth arrives every 100±20 ms, and the backbone on a Jetson NX sends a compressed latent over UDP (10±10 ms) to a 50 Hz GRU policy on an UPboard (Sec. 3). Training randomizes the height-map update period (80–120 ms) and latency (10–30 ms) (appendix table), and the paper states this lets the student handle "the asynchronous nature of depth".
- **Quality tier:** Tier 1.
- **Method weaknesses:** sim-real visual or terrain mismatch must be engineered back into simulation (author-acknowledged, "Discussion and Limitations").
- **Relevance to our setup:** a split-compute pattern (perception box → latent → fast controller) at small scale. Randomizing rate and latency at train time is the key trick.
- Read scope: full_text.

**[Miki22] Miki, T., Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. (2022).** *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics 7. arXiv:2201.08117. https://arxiv.org/abs/2201.08117
- **Key findings:** the elevation map is built at 20 Hz, and the 50 Hz policy samples heights from the latest map. A recurrent belief encoder (GRU) with a learned gate controls how much exteroception enters the belief. Gate vs no-gate is ablated in Supp. S9.
- **Quality tier:** Tier 1.
- **Method weaknesses:** failures on narrow cliffs or stepping stones under occlusion, and no recovery from stuck legs (author-acknowledged, Discussion).
- **Relevance to our setup:** a gated recurrent belief is a principled way to down-weight a stale or unreliable cloud latent.
- Read scope: full_text.

**[Zhuang23] Zhuang, Z., Fu, Z., Wang, J., Atkeson, C., Schwertfeger, S., Finn, C., Zhao, H. (2023).** *Robot Parkour Learning.* CoRL 2023 (PMLR 229). arXiv:2309.05665. https://arxiv.org/abs/2309.05665
- **Key findings:** the depth embedding is computed asynchronously at 10 Hz, and the 50 Hz GRU policy "asynchronously fetches the latest latent embedding" on a Jetson NX (Sec. 4 / appendix). Depth and proprio latencies are domain-randomized. GRU beats MLP (Tab. 2).
- **Quality tier:** Tier 1.
- **Method weaknesses:** simulation environments must be built by hand (author-acknowledged, "Conclusion, Limitations").
- Read scope: full_text.

**[Cheng24] Cheng, X., Shi, K., Agarwal, A., Pathak, D. (2024).** *Extreme Parkour with Legged Robots.* ICRA 2024. arXiv:2309.14341. https://arxiv.org/abs/2309.14341
- **Key findings:** depth at 10±2 Hz and the base policy at 50 Hz on a Jetson NX over UDP. The code pads perception to a constant 80 ms depth latency and 16 ms proprio latency "to prevent jitter" (Sec. 4.1).
- **Quality tier:** Tier 1.
- **Method weaknesses:** not assessed beyond setup (no explicit limitations section was found in the retrieved text). Reader-inferred: padding to a constant latency trades responsiveness for training–deployment consistency.
- **Relevance to our setup:** constant-latency padding is a simple alternative to time encoding, and could be applied to the cloud latent.
- Read scope: partial.

**[ViNL] Kareer, S., Yokoyama, N., Batra, D., Ha, S., Truong, J. (2023).** *ViNL: Visual Navigation and Locomotion Over Obstacles.* ICRA 2023. arXiv:2210.14791. https://arxiv.org/abs/2210.14791
- **Key findings:** a navigation policy at 2 Hz feeds velocity commands to a 50 Hz visual locomotion policy (CNN depth + LSTM). Tab. I: LSTM 73.6% vs MLP 66.8% vs blind 1.2%.
- **Quality tier:** Tier 1 (mostly sim).
- **Method weaknesses:** no dedicated limitations section was found. Reader-inferred: only small obstacles, and the policies are co-deployed without co-training.
- **Relevance to our setup:** a 2 Hz → 50 Hz command interface between independently trained policies (RQ1/RQ4 crossover).
- Read scope: full_text.

### 4.3 Learned inertial / visual-inertial estimation

**[SelVIO] Chen, C., Rosa, S., Miao, Y., Lu, C. X., Wu, W., Markham, A., Trigoni, N. (2019).** *Selective Sensor Fusion for Neural Visual-Inertial Odometry.* CVPR 2019. arXiv:1903.01534. https://arxiv.org/abs/1903.01534
- **Key findings:** a Bi-LSTM summarizes the 100 Hz IMU samples between consecutive 10 Hz frames into one feature. Soft (sigmoid) or hard (Gumbel) masks select features. Both beat direct fusion under occlusion, blur, missing data and temporal misalignment (Tab. 1).
- **Quality tier:** Tier 1.
- **Method weaknesses:** orientation is worse than classical VIO on clean data (author-acknowledged, results discussion). The output rate is tied to the camera rate (reader-inferred).
- **Relevance to our setup:** "summarize the high-rate stream between low-rate events" is the canonical IMU tokenizer.
- Read scope: full_text.

**[TLIO] Liu, W., Caruso, D., Ilg, E., Dong, J., Mourikis, A. I., Daniilidis, K., Kumar, V., Engel, J. (2020).** *TLIO: Tight Learned Inertial Odometry.* IEEE RA-L 5(4). arXiv:2007.01867. https://arxiv.org/abs/2007.01867
- **Key findings:** the EKF propagates at the 200 Hz IMU rate. The network takes a 1 s window and outputs displacement + covariance at 20 Hz as the measurement update. Learned covariance reduces drift by 31% vs fixed.
- **Quality tier:** Tier 1 (pedestrian headset data, not robots).
- **Method weaknesses:** fails on motions outside the training data (author-acknowledged, results/conclusion).
- **Relevance to our setup:** a hybrid pattern where analytic high-rate propagation is corrected by low-rate learned updates with uncertainty. This is an alternative to making the fast policy itself run at IMU rate.
- Read scope: full_text.

### 4.4 Drones

**[Swift] Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., Scaramuzza, D. (2023).** *Champion-level drone racing using deep reinforcement learning.* Nature 620, 982–987. doi:10.1038/s41586-023-06419-4. https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/
- **Key findings:** a 30 Hz camera feeds a gate detector (40 ms on TX2), and VIO and gate detections are fused by a Kalman filter. The MLP policy takes the state estimate, with 8 ms inference on CPU. Sensorimotor latency is 40 ms (vs 220 ms for humans).
- **Quality tier:** Tier 1.
- **Method weaknesses:** fails under appearance change, and there is no crash recovery (author-acknowledged, Discussion).
- **Relevance to our setup:** an existence proof that a tiny policy on a classical filter-fused state reaches champion level. Rate alignment is delegated to the filter, not learned.
- Read scope: full_text.

**[Loquercio21] Loquercio, A., Kaufmann, E., Ranftl, R., Müller, M., Koltun, V., Scaramuzza, D. (2021).** *Learning high-speed flight in the wild.* Science Robotics 6(59), eabg5810. arXiv:2110.05113. https://arxiv.org/abs/2110.05113
- **Key findings:** stereo depth goes into a MobileNet-V3 branch and VIO velocity/attitude + goal into an MLP branch; the two are concatenated. The system runs at about 24 Hz (41.6 ms total on TX2) and outputs 3 trajectory hypotheses.
- **Quality tier:** Tier 1.
- **Method weaknesses:** expert failures at >10 m/s, dynamics mismatch, and perception latency (author-acknowledged, Discussion).
- Read scope: full_text.

**[EventOA] Bhattacharya, A., Cannici, M., Rao, N., Tao, Y., Kumar, V., Matni, N., Scaramuzza, D. (2024).** *Monocular Event-Based Vision for Obstacle Avoidance with a Quadrotor.* arXiv:2411.03303. https://arxiv.org/abs/2411.03303
- **Key findings:** events are binned at 33 ms into binary masks and passed to a ViT-LSTM velocity head (13.4M parameters total). Outdoors, VIO at 30 Hz is fused with IMU by a UKF at 150 Hz. Total latency is 73 ms on CPU.
- **Quality tier:** Tier 2.
- **Method weaknesses:** model latency is "magnitudes larger" than event-sensor latency, so the latency benefit of events is not realized; calibrated event-depth data is needed (author-acknowledged, limitations).
- **Relevance to our setup:** a warning that a high-rate sensor gives no benefit if the fast model's compute becomes the bottleneck.
- Read scope: full_text.

**[Chahine23] Chahine, M., Hasani, R., Kao, P., Ray, A., Shubert, R., Lechner, M., Amini, A., Rus, D. (2023).** *Robust flight navigation out of distribution with liquid neural networks.* Science Robotics 8, eadc8892. doi:10.1126/scirobotics.adc8892
- **Key findings:** a CfC/LTC recurrent controller on a Jetson TX2 maps images to velocity setpoints. Synthetic sequence lengths are randomized (120–250 frames) "to encourage robustness to ... varying update frequencies" (Methods).
- **Quality tier:** Tier 1.
- **Method weaknesses:** not assessed in full (read scope: partial, Methods only). Reader-inferred: image-only, so it gives no evidence on multi-sensor fusion.
- Read scope: partial.

### 4.5 Continuous-time / irregular-sampling sequence models

**[CfC] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., Rus, D. (2022).** *Closed-form continuous-time neural networks.* Nature Machine Intelligence 4, 992–1003. arXiv:2106.13898. https://arxiv.org/abs/2106.13898
- **Key findings:** per-sample elapsed time *t* is an explicit input. On irregular benchmarks the model is 1–5 orders of magnitude faster than ODE-based nets, and 160× faster training than ODE-RNN on PhysioNet (Tab. 5).
- **Quality tier:** Tier 3 (robot-relevant only through Chahine23).
- **Method weaknesses:** vanishing gradients on long dependencies (author-acknowledged, "Scope, Discussions and Conclusions").
- **Relevance to our setup:** a tiny recurrent cell that updates asynchronously per arriving sample with a Δt input. A candidate fast-model core for an MCU-class budget.
- Read scope: full_text.

**[S5] Smith, J. T. H., Warrington, A., Linderman, S. W. (2023).** *Simplified State Space Layers for Sequence Modeling.* ICLR 2023. arXiv:2208.04933. https://arxiv.org/abs/2208.04933
- **Key findings:** time-domain parallel scan lets each step use its own Āₖ (discretized with Δtₖ). On the irregular pendulum, MSE is 3.41e-3 vs CRU 3.94e-3, and S5 is 86× faster (Sec. 6.3, Tab. 3).
- **Quality tier:** Tier 3.
- **Method weaknesses:** the theory links only constrained S4/S5 variants (author-acknowledged). It has not been tested for control (reader-inferred).
- **Relevance to our setup:** an SSM that can be updated recurrently per arriving sample with a variable Δt, giving O(1) per-step cost on-device.
- Read scope: full_text.

**[mTAN] Shukla, S. N., Marlin, B. M. (2021).** *Multi-Time Attention Networks for Irregularly Sampled Time Series.* ICLR 2021. arXiv:2101.10318. https://arxiv.org/abs/2101.10318
- **Key findings:** a learned continuous-time embedding plus attention maps a variable number of observations to a fixed-length representation.
- **Quality tier:** Tier 3.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** a precursor of Chronos-style learned time embeddings; it resamples irregular IMU/torque into a fixed token set.
- Read scope: abstract_only.

## 5. Taxonomy of rate-alignment approaches

| Approach | How it works | Examples | Pros / cons for our setup |
|---|---|---|---|
| A. Latest-sample hold (zero-order hold of slow features) | The slow encoder runs at native rate; the fast loop reuses its newest output | MResT, Agarwal22, Zhuang23, Miki22, Loquercio21 | Simplest and proven at 50–75 Hz. Unaware of staleness unless latency is randomized at train time |
| B. Hold + train-time rate/latency randomization or constant-latency padding | As in A, with the update period and delay randomized in sim, or latency padded to a constant | Agarwal22, Zhuang23, Cheng24 (padding), Chahine23 (sequence length) | Cheap robustness to jitter. Padding adds fixed delay |
| C. Window resampling / stacking | Each modality is cut to a common time window and stacked or spectrogrammed | SHF, FoAR (2 s window), TA-VLA (10-frame history) | Easy. Loses exact timing, and the output rate is set by the slowest loop |
| D. Summarize the high-rate stream between low-rate ticks | An RNN/MLP compresses the samples between frames into one token | SelVIO (Bi-LSTM), TA-VLA (one history token), RDP (tactile embedding) | Compact tokens. Output rate is still tied to the tick unless decoding is per-sample |
| E. Two-level slow-fast decoding | The slow model emits a latent/chunk; a fast decoder adds the latest high-rate signal each step | RDP, ViNL (command interface), MResT | Matches our cloud/edge split. Needs staleness handling (latency matching in RDP) |
| F. Causal token interleaving in one network | Asynchronous tokens with a causal mask, so actions see only past high-rate samples | ImplicitRDP | End-to-end. The whole model must run at the high rate |
| G. Explicit physical-time encoding | Timestamps enter as RoPE or embeddings; training randomizes rates and delays | Chronos, mTAN | Handles variable latency and arbitrary rates natively. Recent; little real-world evidence yet |
| H. Per-modality policies composed with staleness weights | Each modality gets its own policy; guidance is rebased and weighted by age | LAG-Fusion | Modular and explicit about staleness. More compute (several policies) |
| I. Continuous-time recurrent / SSM state | Recurrent state is updated per sample with Δt input | CfC, S5, Chahine23 | O(1) per update, suits MCUs. Not yet shown fusing several robot modalities |
| J. Filter-based (learned measurement + analytic propagation) | An EKF/UKF propagates at IMU rate, and learned modules give low-rate updates | TLIO, Swift, EventOA | Robust and well understood. Rate alignment is not learned, and the policy only sees the fused state |
| K. Gating of a modality's contribution | A learned gate or contact predictor scales a modality's feature | Miki22 (belief gate), FoAR (contact φ), SelVIO (masks), ForceVLA (MoE) | Down-weights unreliable or stale inputs. Usually not time-aware |

## 6. Theme gaps

1. **No paper fuses IMU + camera + joint torque + a remote latent in one learned fast policy.** Legged work fuses depth + proprio, manipulation fuses vision + force, drones push IMU into classical filters. The exact combination in our setup has no direct precedent.
2. **Few rate ablations.** Only FoAR (Tab. III), MResT (Tab. 3) and LAG-Fusion (Fig. 6) vary the rate of the high-frequency modality. None sweeps IMU rate.
3. **Staleness of the slow modality is mostly handled implicitly** (hold + randomization). Only LAG-Fusion (staleness weights) and Chronos (time-RoPE) model age explicitly, and both are 2026 preprints.
4. **Output rate is often below input rate** (FoAR 10 Hz, SHF 10 Hz). True ≥50 Hz action output from learned multi-rate fusion is shown mainly in locomotion (50 Hz) and MResT (75 Hz).
5. **Continuous-time models (CfC, S5, mTAN) lack multi-modal robot evaluation.** Chahine23 is image-only.
6. **Modality collapse** (FACTR, ImplicitRDP) is likely to affect a fast model conditioned on a strong cloud latent. No paper studies it in a split cloud/edge setting.
7. **Compute vs sensor latency:** EventOA shows that high-rate sensors give no benefit when model latency dominates. No paper co-designs fast-model size against sensor rate on battery-limited hardware.

## 7. Unverified / excluded

| Item | Status | Reason |
|---|---|---|
| HiP (Ajay et al., NeurIPS 2023, arXiv:2309.08587) | Verified, excluded | Hierarchical planning, no multi-rate sensor fusion (belongs to RQ1) |
| MT-ACT / RoboAgent (Bharadhwaj et al., arXiv:2309.01918) | Verified, excluded | Action chunking, single-rate inputs (RQ1/RQ2) |
| Hierarchical Diffusion Policy (Ma et al., CVPR 2024, arXiv:2403.03890) | Verified, excluded | Kinematic hierarchy, no multi-rate fusion (RQ1) |
| DeMUSE (Sun et al., arXiv:2602.19764) | Verified (abstract), excluded | RGB-D + force "serialized stream", no rate handling in the abstract |
| TaF-VLA (arXiv:2601.20321), AT-VLA (arXiv:2605.07308) | Surfaced, not assessed | Tactile-VLA papers seen in search only; candidates for a follow-up pass |
| FACTR 2 (arXiv:2606.12406) | Surfaced, not assessed | Follow-up on learned external force sensing |
| Event-RL UAV tracking (arXiv:2410.14685) | Surfaced, not assessed | Event-only RL, no fusion evidence checked |
| Time-LLM, Mamba-for-irregular-control | Not verified | No robot-control source located in this pass |
