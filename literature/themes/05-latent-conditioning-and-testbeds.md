# Theme 5 (RQ5): Latent conditioning, policy shrinking, and experimental testbeds

- **Search date:** 2026-10-27 (searches cover up to the present)
- **Scope:** RQ5 from `../PROTOCOL.md`. Part A asks how much a conditioning latent from a large or privileged model lets a downstream policy shrink. Part B lists testbeds for a slow ~27B VLM (~5 Hz, networked) conditioning a fast small policy with multi-rate sensors, possibly on a drone.
- **Verification rule:** every included entry was resolved to an arXiv abstract page, a proceedings page, or an official repo/docs page during this session. Numbers come only from those pages and carry a locator. "Read scope" is the deepest text actually read: `abstract_only` (arXiv abs page), `full_text` (ar5iv/arXiv HTML read with targeted questions), or `docs` (repo README/docs).
- **Quality tiers:** Tier 1 = peer-reviewed venue named on the verified page; Tier 2 = arXiv preprint, or workshop paper; Tier 3 = documentation/README only.

## 1. Search log

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

## 2. Inclusion notes

- I included classic privileged-teacher work (LbC, RMA, asymmetric actor-critic, ANYmal, high-speed flight) even though its "large model" is a privileged policy rather than a VLM. It is the most mature evidence that a compact latent or teacher signal lets a deployable student run on cheap sensors. RMA is also the closest structural analogue to our setup: a slow latent estimator and a fast base policy running asynchronously.
- I included dual-system VLA papers (HiRT, RoboDual, UniFS, DuoCore-FS) only because they report student/fast-module size, rate, or update-interval evidence. Their architecture side belongs to Theme 1 (RQ1/RQ2) and should be cross-referenced there.
- I included hypernetwork papers as the "generate the small controller's weights" alternative to "condition a fixed small controller on a latent". None of them uses a VLM teacher.
- I included communication-compression papers (Nakanoya 2020, ComVLA 2026) because they quantify what is sent over the link. ComVLA's direction is robot to cloud (visual tokens). Our setup's downlink (cloud latent to robot) is not studied by any included source; see Gaps.
- Most 2025–2026 entries are Tier 2 preprints, read at abstract scope only.

## 3. Part A: comparison table

"n.s." = not stated in the text read. "abs" = abstract page.

| Paper | Year | Teacher / large model | What is passed (dim, rate) | Student / fast-policy size | Size-vs-performance evidence |
|---|---|---|---|---|---|
| Rusu et al., Policy Distillation | 2016 | DQN teacher (Atari) | Teacher Q-values/policy as distillation targets (offline) | 427,874 / 113,346 / 61,954 params (Sec. 4.3) | Yes. A sweep over three student sizes: the 61,954-param student reaches a geometric mean of 84% of the teacher across 10 games (Sec. 4.3, Fig. 3) |
| Chen et al., Learning by Cheating | 2019 | Privileged driving agent (GT map + actors) | Teacher actions/supervision; no latent at test time | n.s. (abs) | No sweep; vision-only student beats prior SOTA on CARLA/NoCrash (abs) |
| Pinto et al., Asymmetric Actor-Critic | 2017 | Critic with full state | Value signal at train time only | n.s. (abs) | No sweep |
| Lee et al., ANYmal challenging terrain | 2020 | Privileged teacher policy | Teacher supervision; student on proprioception | n.s. (abs) | No sweep |
| Kumar et al., RMA | 2021 | Privileged env-factor encoder (sim) | Extrinsics z_t ∈ R^8, estimated at 10 Hz; base policy at 100 Hz, asynchronous (Sec. III-A, III-C) | n.s. in read scope | No sweep; an 8-D latent at 1/10 the base rate is enough for real A1 deployment |
| Loquercio et al., High-speed flight in the wild | 2021 | Privileged expert planner | Expert trajectories (imitation) | n.s. (abs) | No sweep |
| Hu et al., Scaffolder | 2024 | Privileged sensors in critic/world model/reward | Training-only signals | n.s. (abs) | Student often matches policies with test-time privileged sensors (abs) |
| Jülg et al., RPD | 2025 | Octo / OpenVLA (fine-tuned) | Teacher VLA actions guide RL exploration | "compact" RL expert; size n.s. (abs) | Student outperforms the VLA teacher on ManiSkill3 (abs) |
| Xu et al., RLDG | 2024 | Task-specific RL experts | RL-generated data to fine-tune a generalist | n.s. (abs) | Up to 40% higher success than human-demo data (abs). Direction is the reverse: small to big |
| Ye et al., ActDistill | 2025 | Existing VLA | Action-guided self-distillation | n.s. | >50% less computation, up to 1.67x speedup, "comparable or superior" (abs) |
| Shao et al., X-Distill | 2026 | Frozen DINOv2 | Offline visual-feature distillation | ResNet-18 encoder + diffusion head | Beats scratch ResNet and fine-tuned ViT on 34 sim + 5 real tasks (abs) |
| Lynch et al., Play-LMP | 2019 | Learned plan encoder (from play) | Latent plan z | n.s. (abs) | No sweep |
| Shentu et al., LCB | 2024 | LLM (multimodal) | Learnable latent code (dim n.s.) | n.s. (abs) | No sweep |
| Ye et al., LAPA | 2025 | VQ-VAE latent-action model over videos | Discrete latent actions | n.s. (abs) | Outperforms OpenVLA on real tasks (abs) |
| Bu et al., UniVLA | 2025 | Latent action model in DINO feature space | Task-centric latent actions | n.s. (abs) | <1/20 pretraining compute and 1/10 downstream data vs OpenVLA (abs) |
| Jang et al., BC-Z | 2021 | Pretrained language/video encoders | 512-D task embedding via FiLM, 10 Hz policy (Sec. 5.3; abs) | ResNet-18 backbone | No sweep; 44% average on 24 unseen tasks (abs) |
| Brohan et al., RT-1 | 2022 | Universal Sentence Encoder | Instruction embedding via FiLM into EfficientNet; 3 Hz; <100 ms budget (Sec. 5.1) | 35M total (16M tokenizer + 19M Transformer) (Sec. 5.1) | Paper has data/model-size ablations; not extracted here |
| Zhang et al., HiRT | 2024 | InstructBLIP 7B (Sec. 3.2.1) | MAP-pooled fused VLM latent, updated slowly, asynchronous; latest latent reused (Sec. 3.1, 3.3) | ~35M (sim: EfficientNet-B3+FiLM) / 150M (real: ViT-B/16) (App. A.1) | 9.8 Hz vs 4.1 Hz baseline (Tab. 1); dynamic tasks 48%→75% (abs) |
| Bu et al., RoboDual | 2024 | OpenVLA generalist | Generalist outputs/latents to a DiT specialist | 20M trainable params (abs) | +26.7% real, +12% CALVIN vs OpenVLA; 3.8x control rate; strong with 5% of data (abs) |
| Wang et al., VLA-Adapter | 2025 | 0.5B VLM backbone | Bridge-attention conditions into a policy module | 0.5B backbone + lightweight policy | "SOTA-level" with a tiny backbone (abs); no sweep in read scope |
| Sun et al., DEM | 2026 | DINOv3 (fine-tuned) + frozen NeoBERT | Encoder representations to a MeanFlow action head | n.s.; 6.1 ms/forward, 2.07 J/call (abs) | 55.6% vs 56.9% (GR00T N1.7) on 18 RoboCasa tasks; 66% vs 68% real (abs) |
| Sun et al., UniFS | 2026 | VLM, layer groups at decreasing update rates | Multi-rate latent features | n.s. | 98.3% LIBERO (+2.5 pts vs VLA-Adapter); 36.5→17.8 ms (abs) |
| Zou et al., DuoCore-FS | 2025 | 3B VLM (slow path) | Latent representation buffer | Fast path at 30 Hz; size n.s. (abs) | ~3x faster than synchronous VLA (abs) |
| Subedi et al., VL embeddings for nav | 2025 | Frozen VLM embeddings | Embedding as the only policy input | BC policy, size n.s. (abs) | 74% vs 100% expert success, ~3x more steps (abs) |
| Nakanoya et al., Task-relevant representations | 2020 | Task model (server-side) | Task-tailored compressed sensor encoding (uplink) | n.a. | Up to 11x more compression than competing methods (abs) |
| Liu et al., ComVLA | 2026 | OpenVLA-OFT (cloud) | 32 of 512 visual tokens, language-guided (uplink) | Edge: encoder + selector | 95.4% vs 96.9% LIBERO; −74% compute; −22% latency (abs) |
| Hegde et al., HyperPPO | 2023 | Graph hypernetwork | Generated weights for many architectures | "much smaller than common-use networks"; Crazyflie 2.1 deploy (abs) | Searches architecture size jointly (abs); numbers n.s. |
| Rezaei-Shoshtari et al., Hypernetworks zero-shot | 2023 | Hypernetwork on MDP params | Generated policy/value weights | n.s. | No sweep |
| Ren et al., HyPoGen | 2025 | Optimization-biased hypernetwork | Task spec → policy params | n.s. | No sweep |
| Liang et al., Make-An-Agent | 2024 | Conditional diffusion over parameters | Behavior embedding → latent parameters → policy | n.s. | No sweep |

**Bottom line:** only Rusu et al. (2016) gives an explicit student-size sweep under a fixed teacher, and it is not robotic. None of the 30 sources sweeps fast-policy size against the dimension or rate of a VLM latent.

## 4. Annotated entries (Part A)

### A1. Privileged teacher–student and asymmetric training

**Rusu, A. A., Colmenarejo, S. G., Gulcehre, C., Desjardins, G., Kirkpatrick, J., Pascanu, R., Mnih, V., Kavukcuoglu, K., & Hadsell, R. (2016). Policy Distillation. ICLR 2016. arXiv:1511.06295. https://arxiv.org/abs/1511.06295**
- *Relevance:* the only direct student-size sweep under a fixed teacher in this set.
- *Key findings:* students at 427,874, 113,346 and 61,954 parameters; the smallest reaches a geometric mean of 84% of teacher score over 10 games (Sec. 4.3, Fig. 3). The authors suggest DQN may benefit from reduced capacity or regularization (Sec. 4.3).
- *Method:* supervised distillation of DQN Q-values into smaller networks (KL loss variant).
- *Quality:* Tier 1. Read scope: full_text (targeted).
- *Weaknesses:* (reader-inferred) Atari only, discrete actions, no latent channel, no robotics. The teacher is the same size or larger and task-specific.
- *Relevance to our setup:* a prior that large compression ratios (≥10x) can keep most performance when the teacher signal is clean. It does not transfer the rate or staleness question.

**Chen, D., Zhou, B., Koltun, V., & Krähenbühl, P. (2019). Learning by Cheating. CoRL 2019. arXiv:1912.12294. https://arxiv.org/abs/1912.12294**
- *Relevance:* canonical two-stage privileged teacher → sensorimotor student.
- *Key findings:* 100% success on all tasks in the original CARLA benchmark; new NoCrash record; infractions cut by an order of magnitude (abs).
- *Method:* the teacher sees GT layout and traffic, then supervises a vision-only student.
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* template for training a small on-robot student with a cloud VLM acting as a "privileged" teacher.

**Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., & Abbeel, P. (2017). Asymmetric Actor Critic for Image-Based Robot Learning. arXiv:1710.06542. https://arxiv.org/abs/1710.06542**
- *Relevance:* privileged information in the critic only; the deployed actor stays small and image-based.
- *Key findings:* asymmetric inputs "significantly improve" sim performance; with domain randomization the policy transfers to a real robot without real data (abs).
- *Method:* critic on full state, actor on RGBD.
- *Quality:* Tier 2 as verified (venue not shown on the abs page). *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* a cheap way to use cloud-side information during training without sending it at run time.

**Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020). Learning Quadrupedal Locomotion over Challenging Terrain. Science Robotics 5(47). arXiv:2010.11251. https://arxiv.org/abs/2010.11251**
- *Relevance:* teacher–student with a proprioceptive student; zero-shot sim-to-real.
- *Key findings:* traverses mud, snow, rubble, vegetation and water not seen in training, on two ANYmal generations (abs).
- *Method:* privileged teacher policy, then a student on proprioceptive history.
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* shows that a high-rate proprioceptive (IMU/joint) student can absorb privileged knowledge, which matches our fast model's sensors.

**Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021). RMA: Rapid Motor Adaptation for Legged Robots. RSS 2021. arXiv:2107.04034. https://arxiv.org/abs/2107.04034**
- *Relevance:* the closest structural analogue in the theme. A slow module emits a low-dimensional latent; a fast policy consumes the most recent one asynchronously.
- *Key findings:* extrinsics z_t ∈ R^8 (Sec. III-A). The adaptation module updates ẑ_t about every 0.1 s (10 Hz) from 50 steps of state-action history; the base policy runs at 100 Hz on the most recent ẑ_t; "the two subsystems … run asynchronously and at substantially different frequencies" (Sec. III-C). Deployed on an A1 with no fine-tuning (abs).
- *Method:* phase 1 learns the base policy plus an env-factor encoder with privileged sim info; phase 2 regresses ẑ from history.
- *Quality:* Tier 1. Read scope: full_text (targeted).
- *Weaknesses:* (reader-inferred) the latent encodes slowly varying physical parameters, not task semantics. I did not find an ablation over latent rate or dimension in the read scope.
- *Relevance:* a direct design precedent: a low-dimensional latent at 10x lower rate than control. Our case is about 5 Hz vs ≥20 Hz, plus network jitter.

**Loquercio, A., Kaufmann, E., Ranftl, R., Müller, M., Koltun, V., & Scaramuzza, D. (2021). Learning High-Speed Flight in the Wild. Science Robotics 6(59). arXiv:2110.05113. https://arxiv.org/abs/2110.05113**
- *Relevance:* privileged-expert distillation for a drone, with sensor noise modelled in sim.
- *Key findings:* zero-shot transfer to forests, snow and collapsed buildings; beats conventional obstacle avoidance (abs).
- *Method:* a CNN imitates a privileged expert in simulation and outputs receding-horizon trajectories.
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* evidence that a small onboard drone network can absorb a privileged planner.

**Hu, E. S., Springer, J., Rybkin, O., & Jayaraman, D. (2024). Privileged Sensing Scaffolds Reinforcement Learning. ICLR 2024 (spotlight). arXiv:2405.14853. https://arxiv.org/abs/2405.14853**
- *Relevance:* uses training-only sensors in every auxiliary component; introduces the S3 suite of ten tasks.
- *Key findings:* often matches policies that have test-time privileged sensors (abs).
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the S3 tasks (blind locomotion, touch) are candidate testbeds for "what if the cloud latent is missing".

### A2. Distilling or compressing large VLAs and VLMs into small policies

**Jülg, T., Burgard, W., & Walter, F. (2025). Refined Policy Distillation: From VLA Generalists to RL Experts. IROS 2026 (per arXiv). arXiv:2503.05833. https://arxiv.org/abs/2503.05833**
- *Key findings:* RL students guided by Octo/OpenVLA actions outperform the teacher VLA (dense and sparse reward), converge faster than RL baselines, are robust to camera shifts, and solve variations the VLA cannot (abs).
- *Method:* on-policy RL + BC toward teacher actions, on ManiSkill3.
- *Quality:* Tier 1 (venue listed on arXiv). *Weaknesses:* not assessed (read scope: abstract_only). Student size not given in the abstract.
- *Relevance:* a recipe for a VLA-supervised small student, and a ManiSkill3 precedent.

**Xu, C., Li, Q., Luo, J., & Levine, S. (2024). RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning. arXiv:2412.09858. https://arxiv.org/abs/2412.09858**
- *Key findings:* generalists fine-tuned on RL-generated data beat human-demo fine-tuning by up to 40% success on connector insertion and assembly (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the reverse direction (small specialist → generalist). Useful for producing high-quality data for the fast policy.

**Ye, W., Wang, T., Zhu, L., Li, F., Yang, G., & Shen, H. (2025). ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models. arXiv:2511.18082. https://arxiv.org/abs/2511.18082**
- *Key findings:* >50% computation reduction, up to 1.67x speedup, comparable or superior performance (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* a modest compression ratio. It speeds up a VLA but does not reach a drone-sized student.

**Shao, M., Zhang, F., Zhang, G., Cheng, B., Xue, Z., & Xu, H. (2026). X-Distill: Cross-Architecture Vision Distillation for Visuomotor Learning. arXiv:2601.11269. https://arxiv.org/abs/2601.11269**
- *Key findings:* a DINOv2 → ResNet-18 distilled encoder with a diffusion head beats scratch ResNet and fine-tuned ViT across 34 sim and 5 real tasks (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* an onboard encoder option that keeps foundation-model features without running the large model.

**Subedi, N., Haroon, A., Ganguly, S., Tetteh, S. T. K., Koirala, P., Fleming, C., & Sarkar, S. (2025). Can Pretrained Vision-Language Embeddings Alone Guide Robot Navigation? RSS 2025 Workshop. arXiv:2506.14507. https://arxiv.org/abs/2506.14507**
- *Key findings:* BC on frozen VLM embeddings reaches 74% vs 100% for a state-aware expert, with about 3x more steps (abs).
- *Quality:* Tier 2 (workshop). *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* a lower-bound baseline: an embedding alone, with no fast onboard state, loses about a quarter of successes in this setting.

### A3. Latent plans and latent actions

**Lynch, C., Khansari, M., Xiao, T., Kumar, V., Tompson, J., Levine, S., & Sermanet, P. (2019). Learning Latent Plans from Play. CoRL 2019. arXiv:1903.01973. https://arxiv.org/abs/1903.01973**
- *Key findings:* Play-LMP beats expert-trained policies on 18 visual manipulation tasks and is more robust to perturbations; the latent space self-organizes by task (abs).
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* precedent for a learned plan latent as the slow-to-fast interface.

**Shentu, Y., Wu, P., Rajeswaran, A., & Abbeel, P. (2024). From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control. arXiv:2405.04798. https://arxiv.org/abs/2405.04798**
- *Key findings:* a learnable latent code between the LLM and the low-level policy avoids the limits of language, allows end-to-end fine-tuning, and is evaluated on Language Table and CALVIN (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the most direct precedent for "latent token from a large model conditions a small policy". CALVIN is a shared testbed.

**Ye, S., et al. (2025). Latent Action Pretraining from Videos (LAPA). ICLR 2025. arXiv:2410.11758. https://arxiv.org/abs/2410.11758**
- *Key findings:* VQ-VAE latent actions learned from unlabeled video; the latent-pretrained VLA beats OpenVLA trained on robot labels on real language-conditioned tasks (abs).
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* discrete latent actions are a compact candidate for the downlink (few bits per step).

**Bu, Q., Yang, Y., Cai, J., Gao, S., Ren, G., Yao, M., Luo, P., & Li, H. (2025). UniVLA: Learning to Act Anywhere with Task-centric Latent Actions. RSS 2025. arXiv:2505.06111. https://arxiv.org/abs/2505.06111**
- *Key findings:* <1/20 pretraining compute and 1/10 downstream data vs OpenVLA (abs); latent actions in DINO feature space.
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* task-centric latent actions are a candidate interface; the paper offers an efficiency benchmark against OpenVLA.

### A4. Embedding-conditioned small policies (FiLM and others)

**Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S., & Finn, C. (2021). BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning. CoRL 2021. arXiv:2202.02005. https://arxiv.org/abs/2202.02005**
- *Key findings:* a 512-D language/video task embedding conditions a ResNet-18 policy via FiLM (Sec. 5.3); asynchronous inference and control at 10 Hz; 44% average on 24 unseen tasks (abs).
- *Quality:* Tier 1. Read scope: full_text (targeted). *Weaknesses:* (reader-inferred) the embedding is fixed per episode, so staleness is never tested.
- *Relevance:* a baseline design: a 512-D vector plus FiLM into a small CNN.

**Brohan, A., Brown, N., Carbajal, J., et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817. https://arxiv.org/abs/2212.06817**
- *Key findings:* 35M parameters (16M image tokenizer + 19M Transformer); Universal Sentence Encoder embedding into identity-initialized FiLM layers in EfficientNet; 3 Hz with a <100 ms latency budget (Sec. 5.1).
- *Quality:* Tier 2 as verified. Read scope: full_text (targeted). *Weaknesses:* (reader-inferred) 3 Hz is below our ≥20 Hz target.
- *Relevance:* the FiLM-conditioned small-policy reference point used by HiRT.

**Zhang, J., Guo, Y., Chen, X., Wang, Y.-J., Hu, Y., Shi, C., & Chen, J. (2024). HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers. CoRL 2024. arXiv:2410.05273. https://arxiv.org/abs/2410.05273**
- *Key findings:* InstructBLIP 7B produces a MAP-pooled latent (Sec. 3.1, 3.2.1). The fast policy (~35M sim: EfficientNet-B3+FiLM; 150M real: ViT-B/16 + cross-attention; App. A.1) uses the most recent latent asynchronously (Sec. 3.3). 9.8 Hz vs 4.1 Hz (Tab. 1); dynamic tasks 48%→75% (abs).
- *Quality:* Tier 1. Read scope: full_text (targeted). *Weaknesses:* (reader-inferred) the VLM is 7B on local hardware, with no network jitter; the update-interval sweep was not extracted.
- *Relevance:* closest VLM analogue to RMA's asynchronous latent. Gives a concrete fast-policy size (35M).

**Bu, Q., Li, H., Chen, L., Cai, J., Zeng, J., Cui, H., Yao, M., & Qiao, Y. (2024). Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation (RoboDual). arXiv:2410.08001. https://arxiv.org/abs/2410.08001**
- *Key findings:* a 20M-parameter DiT specialist conditioned by OpenVLA: +26.7% real, +12% CALVIN, 3.8x control frequency, strong with 5% of demos (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the strongest "small specialist + large generalist" size datapoint (20M).

**Wang, Y., Ding, P., Li, L., Cui, C., Ge, Z., et al. (2025). VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model. arXiv:2509.09372. https://arxiv.org/abs/2509.09372**
- *Key findings:* a 0.5B backbone with Bridge Attention reaches SOTA-level performance, trains in 8 h on one consumer GPU, and needs no robot-data pretraining (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* suggests the conditioning source may not need 27B for LIBERO-class tasks. This is a confound to control for.

**Sun, X., Liang, C., Zeng, Z., Wang, Q., Zhang, H., Sun, Y., Li, Q., & Rakita, D. (2026). Decoupling Vision, Language, and Action for Efficient Multi-Task Robot Policies (DEM). arXiv:2609.18374. https://arxiv.org/abs/2609.18374**
- *Key findings:* DINOv3 + frozen NeoBERT + MeanFlow head: 55.6% vs 56.9% (GR00T N1.7) on 18 RoboCasa tasks; 66% vs 68% real; 6.1 ms/forward; 2.07 J/call, 6–15x more efficient (abs).
- *Quality:* Tier 2 (very recent). *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* a "no big VLM in the loop" baseline. Our cloud-latent design must beat it to justify the link.

**Sun, L., Guan, Z., Wang, C., Chen, Z., Yu, J., Li, Z., He, B., Sun, T., Cao, J., & Liu, L. (2026). UniFS: Unified Fast-to-Slow Hierarchical Architecture for Vision-Language-Action Models. arXiv:2606.22794. https://arxiv.org/abs/2606.22794**
- *Key findings:* VLM layer groups updated at progressively lower rates; 98.3% LIBERO (+2.5 pts vs VLA-Adapter); latency 36.5→17.8 ms (abs). The abstract frames the update gap as a staleness vs compute trade-off.
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* multi-rate latent features, a possible scheme for sending a coarse and a fine latent at different rates.

**Zou, T., Zeng, H., Nong, Y., Li, Y., Liu, K., Yang, H., Ling, X., Li, X., & Ma, L. (2025). Asynchronous Fast-Slow Vision-Language-Action Policies for Whole-Body Robotic Manipulation (DuoCore-FS). arXiv:2512.20188. https://arxiv.org/abs/2512.20188**
- *Key findings:* a 3B VLM slow path and a fast path at 30 Hz, linked by a latent representation buffer; ~3x faster than synchronous VLAs (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the latent-buffer design matches our cloud-to-robot buffer.

### A5. Compressing what goes over the link

**Nakanoya, M., Chinchali, S., Anemogiannis, A., Datta, A., Katti, S., & Pavone, M. (2020). Task-relevant Representation Learning for Networked Robotic Perception. arXiv:2011.03216. https://arxiv.org/abs/2011.03216**
- *Key findings:* compressing robot sensor data with a task-aware objective gives up to 11x more compression than competing methods with maintained accuracy (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the uplink (robot → cloud) half of our link.

**Liu, B., Poe, W. Y., Di, J., Trivisonno, R., & Caire, G. (2026). ComVLA: Communication-Aware Split Inference for VLA Models in 6G-Connected Robotics. IEEE GLOBECOM 2026 (per arXiv). arXiv:2609.07838. https://arxiv.org/abs/2609.07838**
- *Key findings:* sends 32 instead of 512 visual tokens on LIBERO: −74% inference compute, −22% latency vs OpenVLA-OFT, 95.4% vs 96.9% success, within capacity under fading (abs).
- *Quality:* Tier 1 (venue per arXiv; very recent). *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* a 16x uplink token reduction for −1.5 points on LIBERO. A natural companion to a downlink-latent study.

### A6. Hypernetworks that generate small controllers

**Hegde, S., Huang, Z., & Sukhatme, G. S. (2023). HyperPPO: A Scalable Method for Finding Small Policies for Robotic Control. arXiv:2309.16663. https://arxiv.org/abs/2309.16663**
- *Key findings:* a graph hypernetwork estimates weights for many architectures at once; finds networks "much smaller than … common-use networks"; deployed on a Crazyflie 2.1 (abs).
- *Quality:* Tier 2. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance:* the only hypernetwork source with a drone deployment. A route to architecture-size sweeps.

**Rezaei-Shoshtari, S., Morissette, C., Hogan, F. R., Dudek, G., & Meger, D. (2023). Hypernetworks for Zero-shot Transfer in Reinforcement Learning. AAAI 2023. arXiv:2211.15457. https://arxiv.org/abs/2211.15457**
- *Key findings:* TD-based hypernetwork training maps MDP parameters to policies; beats multitask and meta-RL baselines on DM Control (abs).
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).

**Ren, H., Sun, L., Wang, X., Zhou, P., Wu, Z., Dong, S., Zou, D., Zheng, Y., & Yang, Y. (2025). HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe52e5ddaa88badea593b306df483d9-Abstract-Conference.html**
- *Key findings:* generates policy parameters from task specifications only, by approximating optimization steps in a compressed policy space; beats SOTA on locomotion and manipulation for unseen tasks (abs).
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).

**Liang, Y., Xu, T., Hu, K., Jiang, G., Huang, F., & Xu, H. (2024). Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion. NeurIPS 2024. arXiv:2407.10973. https://arxiv.org/abs/2407.10973**
- *Key findings:* diffusion over latent parameter representations, conditioned on behavior embeddings; generalizes to unseen tasks from few-shot demos; generated policies deployed on real locomotion robots (search result + NeurIPS page).
- *Quality:* Tier 1. *Weaknesses:* not assessed (read scope: abstract_only).
- *Relevance to A6 as a group:* hypernetworks put the large model's output into the weights of the small controller, so it is sent only once per task instead of at every step. That minimizes link traffic but cannot track fast scene changes.

## 5. Part B: Experimental testbeds

"n.s." = not stated in what I read. "Latency injection" means a built-in mechanism found in docs or paper. Every simulator allows manual buffering in user code, which I do not count.

| Testbed | Robot type | Sensors & native rates (as verified) | GPU-parallel? | Latency injection built in? | License | URL |
|---|---|---|---|---|---|---|
| Isaac Lab | Arms, legged, humanoids, mobile, multi-embodiment | Camera, ray caster, contact, frame transformer, IMU, PVA, joint wrench. Per-sensor `update_period` (docs); "flexible sensor update frequencies … avoids the typical unrealistic assumption of synchronized sensing and control" (paper Sec. 3.3); IMU noise and drift (Sec. 3.3.1) | Yes | Yes: Delayed PD actuator buffers actions by a configurable number of sim steps (Sec. 3.2.2) | BSD-3 (core), Apache-2.0 (mimic); Isaac Sim terms apply | https://arxiv.org/abs/2511.04831 · https://github.com/isaac-sim/IsaacLab |
| robosuite (base of RoboCasa) | Arms | F/T sensor in every gripper wrist by default; joint pos/vel; cameras; per-observable `sampling_rate` (docs) | No (CPU MuJoCo) | Yes: per-observable `delayer` and `corrupter` (docs) | n.s. | https://robosuite.ai/docs/modules/sensors.html |
| RoboCasa / RoboCasa365 | Mobile manipulators (kitchen) | Inherits robosuite observables | No | Via robosuite observables | MIT code; CC BY 4.0 assets | https://arxiv.org/abs/2406.02523 · https://github.com/robocasa/robocasa |
| LIBERO | Arm (130 tasks in 4 suites) | Camera + proprio (README); rates n.s. | No | n.s. | MIT code; CC BY 4.0 data | https://arxiv.org/abs/2306.03310 · https://github.com/Lifelong-Robot-Learning/LIBERO |
| CALVIN | Franka arm (PyBullet) | Static RGB-D 200×200; gripper RGB-D 84×84; tactile 120×160×6; proprio; control 30 Hz (README) | EGL rendering only | n.s. | MIT | https://arxiv.org/abs/2112.03227 · https://github.com/mees/calvin |
| SimplerEnv | Google Robot, WidowX (real-to-sim) | Camera (visual matching); rates n.s. | Yes (ManiSkill3 backend, "10–15x faster") | n.s. | n.s. | https://arxiv.org/abs/2405.05941 · https://github.com/simpler-env/SimplerEnv |
| ManiSkill3 | Arms, mobile manipulators, humanoids | RGB-D, segmentation, point cloud, contact forces; 30,000+ FPS RGBD+seg on a 4090 (README) | Yes | n.s. | Apache-2.0 code; CC BY-NC 4.0 assets | https://arxiv.org/abs/2410.00425 · https://github.com/haosulab/ManiSkill |
| Meta-World | Sawyer arm, 50 tasks | State vector (+ one-hot task ID for MT10/50) | Vectorized env (CPU) | n.s. | n.s. | https://arxiv.org/abs/1910.10897 · https://github.com/Farama-Foundation/Metaworld |
| LeRobot (SO-100 etc.) | Low-cost real arms (SO-100, ALOHA-2) + datasets | LeRobotDataset: Parquet + MP4 for high-rate video and sensors (paper) | n.a. (real/datasets) | Ships an asynchronous inference stack (paper) | n.s. (paper CC BY 4.0) | https://arxiv.org/abs/2602.22818 |
| gym-pybullet-drones | Crazyflie-class quadrotors, multi-agent | Kinematic state, RGB 64×48, depth, seg; physics demonstrated at 240 Hz; control frequency configurable (paper) | No | n.s.; Betaflight SITL integration (README) | n.s. | https://arxiv.org/abs/2103.02142 · https://github.com/learnsyslab/gym-pybullet-drones |
| Aerial Gym Simulator | Multirotors (under/fully/over-actuated) | Ray-cast depth/seg camera and LiDAR; IMU with VN-100 bias random walk + noise (docs); rates n.s. | Yes (Isaac Gym) | n.s. | BSD-3 | https://arxiv.org/abs/2503.01471 · https://github.com/ntnu-arl/aerial_gym_simulator |
| OmniDrones | 4 drone models | 5 sensor modalities incl. IMU and camera (abs); rates n.s. | Yes (Isaac Sim) | n.s. | MIT; maintainer says it is "hard to maintain" | https://arxiv.org/abs/2309.12825 · https://github.com/btx0424/OmniDrones |
| Flightmare | Quadrotor | Multi-modal sensor suite + point cloud; physics up to 200,000 Hz, rendering up to 230 Hz on a laptop (abs) | Parallel quadrotor RL API | n.s. | MIT; last news 2020 | https://arxiv.org/abs/2009.00563 · https://github.com/uzh-rpg/flightmare |
| AirSim | Quadrotor, car (Unreal) | Physics 1000 Hz; IMU, barometer, magnetometer, GPS (GPS 50 Hz with 200 ms latency) (paper) | No | Partly: GPS latency is modelled (paper) | MIT; README points to the successor "Project AirSim" | https://arxiv.org/abs/1705.05065 · https://github.com/microsoft/AirSim |
| PX4 SITL | Multirotor, VTOL, plane, rover | SIH simulates IMU, GPS, baro, mag, airspeed; rates n.s. in the page read; lockstep with simulator (docs) | No | n.s. (lockstep timing) | BSD-3 | https://docs.px4.io/main/en/simulation/ |
| AerialVLN | UAV, VLN (25 city scenes) | Vision + language; continuous navigation (abs) | n.s. | n.s. | n.s. | https://arxiv.org/abs/2308.06735 |
| OpenUAV / UAV-Need-Help | UAV, VLN (~12k trajectories) | "Realistic flight control" (abs); simulator and sensors n.s. | n.s. | n.s. | n.s. | https://arxiv.org/abs/2410.07087 |
| CityNav | Aerial VLN dataset (real cities) | Aerial imagery + geographic semantic maps; 32,637 trajectories (abs) | n.a. | n.a. | n.s. | https://arxiv.org/abs/2406.14240 |

**Notes on Part B**
- **Native IMU at a specified rate:** none of the pages I read states a default IMU sampling rate. Isaac Lab (per-sensor `update_period`) and robosuite (per-observable `sampling_rate`) let you *set* sensor rates independently of control. That is what our 100 Hz IMU / 20 Hz camera / 60 Hz torque mix needs. AirSim gives a 1000 Hz physics rate, so a 100 Hz IMU is reachable.
- **Force/torque or joint torque:** robosuite (wrist F/T by default), Isaac Lab (joint wrench + contact sensors), CALVIN (tactile), ManiSkill3 (contact forces). None of the drone simulators has it, and drones do not need it.
- **Built-in latency injection:** Isaac Lab (Delayed PD actuator, action side) and robosuite (`delayer`, observation side) are the only two I confirmed. Injecting the cloud-latent delay (≈200 ms at 5 Hz plus jitter) must be done in user code everywhere: hold the latent for k steps or sample a delay distribution.
- **VLA baselines:** LIBERO is where ComVLA, UniFS and VLA-Adapter report results. CALVIN is used by LCB and RoboDual. SimplerEnv targets RT-1/Octo-style real-to-sim. These are the testbeds where a 27B-VLM latent can be compared with published numbers.
- **UAV VLN benchmarks** (AerialVLN, OpenUAV, CityNav) test the slow semantic layer, but none confirms a dynamics-level flight model with IMU in the text I read. They suit the "what latent does a 27B VLM produce" half, not the fast-control half.

## 6. Theme gaps

1. **No size-vs-latent sweep:** none of the 30 sources varies fast-policy size against conditioning-latent dimension or rate with a VLM teacher. Rusu et al. (2016) is the only explicit student-size sweep, and it is Atari only.
2. **Downlink compression is unstudied:** ComVLA and Nakanoya et al. compress the robot→cloud uplink. I found no source quantizing or rate-limiting the cloud→robot conditioning latent and measuring the effect on control.
3. **Staleness under network jitter:** RMA (10 Hz vs 100 Hz) and HiRT and DuoCore-FS (asynchronous latest-latent reuse) assume local, deterministic timing. No source in this theme injects realistic network jitter or drops into the latent channel. Cross-check Theme 2.
4. **Goal dropout / classifier-free-guidance robustness to a missing latent:** searches returned no robot source testing policy behavior when the conditioning latent is dropped at run time (CFG-DP uses CFG for timestep phase, not latent dropout).
5. **Drone + VLM latent:** the drone sources (Loquercio, HyperPPO, testbeds) involve no VLM conditioning. The UAV VLN benchmarks lack verified IMU-rate dynamics.
6. **Teacher size confound:** VLA-Adapter (0.5B) and DEM (no VLM) reach near-SOTA results on LIBERO/RoboCasa. An experiment must show that the 27B latent adds value beyond these.

## 7. Recommended first experimental testbed

**Recommendation: robosuite-based LIBERO (sim, arm) first, then Isaac Lab (+ Aerial Gym for the drone variant).**

Reasoning:
1. **Comparability:** LIBERO has the densest set of recent fast/slow and compression baselines (ComVLA 96.9% OpenVLA-OFT reference, UniFS 98.3%, VLA-Adapter). A "27B-VLM latent → small policy" result can be placed against published numbers without re-implementing them.
2. **Multi-rate sensors and delay:** robosuite observables give per-sensor `sampling_rate` and `delayer`, plus a wrist F/T sensor by default. That covers camera ~20 Hz and F/T/proprio at higher rates, with observation-side delay, without patching the simulator. The cloud-latent delay (hold for k control steps, sampled jitter, random drops) is a thin wrapper.
3. **Cheap ablations:** the core RQ5 sweep (fast-policy size {≈1M, 5M, 20M, 35M} × latent dimension {8, 64, 512, full} × latent period {1, 4, 10, 20 steps} × drop rate) is CPU-parallelizable at LIBERO scale. The 27B VLM embeddings can be precomputed per frame offline to decouple VLM cost from the sweep. RMA (8-D), BC-Z (512-D), RoboDual (20M) and HiRT (35M) anchor the grid.
4. **Upgrade path:** move to Isaac Lab for an IMU-equipped, GPU-parallel, multi-rate setup with action-side Delayed PD. Aerial Gym (BSD-3, IMU with VN-100 noise model, Isaac Gym based) or gym-pybullet-drones (Crazyflie, Betaflight SITL, a HyperPPO-style small-policy precedent) handles the drone variant. OmniDrones is not recommended as a first choice given the maintainer's maintenance statement.

Caveat: LIBERO's control rate and observation defaults were not confirmed in the text I read, so check them before fixing the ≥20 Hz target.

## 8. Unverified / excluded

| Item | Reason |
|---|---|
| Genie (Bruce et al., 2024) | Seed item; not fetched this session → unverified |
| Octo goal/language masking as "goal dropout" evidence | Not fetched this session → unverified |
| Fast-in-Slow (FiS-VLA), NeurIPS 2025 | Seen only in search snippets (117.7 Hz claim); not fetched → unverified; belongs to Theme 1 |
| LiMA, arXiv:2609.28431 | Search snippet only; not fetched → unverified |
| DEFLECT, arXiv:2605.19294; FutureRTC, arXiv:2607.24008 | Search snippets only; Theme 2 topic → unverified here |
| ResVLA / "Characterizing VLA across XPUs" arXiv:2604.24447 | Snippet only; not fetched → unverified |
| OpenFly, arXiv:2502.18041 | Snippet only; not fetched → unverified |
| Colosseum (AirSim fork) | Not fetched → unverified |
| LVLM2P, arXiv:2505.11221 (ICASSP 2025) | Verified, but excluded: the abstract does not show an embodied robot task |
| CFG-DP, arXiv:2510.09786 | Verified, but excluded: uses CFG for task-phase tracking, not robustness to a dropped or stale conditioning latent |
| Teacher-Assistant KD, arXiv:1902.03393 | Excluded: image classification, no embodiment |
