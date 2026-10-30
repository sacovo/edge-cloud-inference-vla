# Theme 8 (RQ1, RQ5): Predictive latent representations (JEPA) and small latent predictors

- **Questions:** (1) what the JEPA family (I-JEPA, V-JEPA, V-JEPA 2 / 2-AC) offers robot control; (2) latent-predictive representations for control (self-predictive RL, decoder-free latent MPC); (3) small, fast latent predictors and latent prediction across delay; (4) whether predictive features carry physical scene state better than VLM / CLIP-style features (our H0 concern).
- **Why:** the author asked whether a small JEPA-style model could serve as the fast model, the slow model, or both. Two concrete roles: a small on-robot predictor that rolls a stale remote latent forward to "now" (the Kalman predict step in latent space, Theme 7), and V-JEPA 2 as a slow-side encoder whose latent may carry more dynamic state than a VLM's.
- **Search date:** 2026-10-30. One AI search agent; merged by the main session. **Verification:** all arXiv IDs pass `scripts/verify_bib.py` (2026-10-30). LeCun's JEPA position paper could not be verified and is listed in Section 6.
- **Read scope and tiers:** as in Theme 4 / Theme 7. Most 2026 sources are preprints with 5–20 real-robot trials per cell and no seed variance; re-check table values against the PDFs before quoting (CLAUDE.md gotcha).

## 1. What exists and what is missing

| Role | What exists | Closest precedents | Missing (our opening) |
|---|---|---|---|
| Small on-robot latent predictor | ~9K-parameter GRU JEPA predictor at 100 Hz on a Jetson Orin NX with real flights [SkyJEPA] (Sec. V-C, Fig. 5); ≤15M latent world model under 10 ms next to a >100 ms VLA, including a remote run with ~1 s delay [DREAM-Chunk] (Tab. 2); predictors from 22M to 8B all with non-zero real-Franka success [RoboJEPA] | [SkyJEPA], [DREAM-Chunk] | No source rolls a delayed *remote* embedding forward to "now" with local actions and fast sensors; every predictor encodes the current local observation |
| Latent innovation as trust | Latent prediction error spikes on physical violations, not on colour changes [LeWM] (Fig. 8), [IntPhys-VJEPA] (Fig. 1); predicted-vs-observed latent mismatch picks which slow-model chunk to run [DREAM-Chunk]; abort decision read off a predicted latent [ContactGuard] | [DREAM-Chunk] | Nobody gates a newly arrived remote z by ‖z_new − ẑ‖, calibrates the threshold, or combines it with the force/torque innovation (Theme 7) |
| JEPA latent as the slow model's z | Action-conditioned V-JEPA 2-AC (~300M predictor on a frozen encoder) plans pick-and-place zero-shot on Franka arms in two labs, 65–80% vs 10–15% for Octo, 16 s per action on one RTX 4090 [VJEPA2] (Sec. 3.1, 4; Tab. 2–3) | [VJEPA2] | Not used as a latent for a separate fast policy |
| Does a predictive latent carry state? (H0) | On LIBERO, frozen V-JEPA 2 recovers object/robot state and 10-step state change better than SigLIP and DINOv2 [JEPA-VLA] (Sec. 2.1, Fig. 2c); CLIP and SigLIP 2 have the highest state-probe error of 15 encoders, V-JEPA 2-L among the best [ALPS] (Tab. 1); OpenVLA hidden layers decode symbolic LIBERO predicates at >0.90 [ProbeOpenVLA] | [ALPS] probe protocol (readout, +1 s, +3 s on LIBERO-Object/Spatial) | All encoder comparisons probe *vision towers*, not a VLA's post-fusion latent; nobody probes a VLA latent for continuous or dynamic state |

## 2. Synthesis

- **The fast-side predictor is feasible at our size and rate, but its use as a delay bridge for a remote latent is untested.** [SkyJEPA] shows a JEPA predictor of our 10k class running at 100 Hz on our onboard hardware class; [DREAM-Chunk] shows a small latent model running next to a slow VLA, including over ~1 s of delay. Advancing the *slow model's* latent with local actions and fast sensors is the open slot (EXP queue: late arrivals).
- **Training is simpler for us than for JEPA.** Our targets are a frozen teacher's latents, which removes the collapse problem ([Ni24] Prop. 3); collapse regularizers (SIGReg [LeWM], [SkyJEPA]; SimNorm [TD-MPC2]) are needed only when the latent is learned end-to-end. Multi-step rollout with teacher forcing ([VJEPA2] Fig. 6; [SPR]) and a proprio/torque grounding head ([PSG-JEPA]) are cheap additions.
- **H0 is weaker than "task ID only", and predictive encoders look better on state.** Symbolic task state is decodable inside OpenVLA [ProbeOpenVLA]; predictive vision encoders beat CLIP-style ones on continuous and future state [JEPA-VLA], [ALPS]. Caveats: forward-only JEPA latents can miss robot-centric state [PSG-JEPA], and V-JEPA is near chance on collisions and solidity [IntPhys-VJEPA] (Fig. 2).
- **A direct, cheap H0 test exists.** Apply the [ALPS] probe protocol to our candidate z's (π0.5 R1/R2, V-JEPA 2, the A1 teacher's hidden state) before any closed-loop runs.

## 3. Implications for the experiment queue

1. **Late-arrivals experiment:** the latent predictor variant (rolls z forward) is supported by [SkyJEPA] (size/rate) and [Ni24] (no collapse with a frozen target); test under jittered delay (Theme 7, Wang24 caution). Add a proprio/torque grounding head [PSG-JEPA].
2. **Innovation-trust experiment:** the latent innovation ‖z_new − ẑ‖ is a second trust signal next to the force/torque innovation; [LeWM] Fig. 8 suggests it responds to physical events, not appearance.
3. **VLM-latent stage (A2) / H0:** add V-JEPA 2 as a z-variant and the [ALPS]-style state probe on every z before closed-loop evaluation.

## 4. Search log

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

## 5. Annotated entries

### (1) JEPA family

**[I-JEPA] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., Ballas, N. (2023).** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023. arXiv:2301.08243. https://arxiv.org/abs/2301.08243
- **Key findings:**
  - The JEPA recipe: from one context block, a predictor predicts the *representations* (not pixels) of several target blocks, with targets from an EMA target encoder (abstract; Sec. 3). Large-scale target blocks and a spatially spread context block are what make the features semantic (abstract).
  - With linear probes it beats view-invariance methods on low-level tasks (Clevr object counting and depth), i.e. it keeps some geometric information (Tab. 4).
  - ViT-H/14 trains on ImageNet in under 72 h on 16 A100s (abstract).
- **Quality tier:** Tier 3 (CVPR 2023, venue via Crossref DOI 10.1109/cvpr52729.2023.01499; the arXiv comment wrongly says ICCV). No control or robot evaluation.
- **Method weaknesses:** images only, no temporal or action conditioning (reader-inferred). Collapse is avoided by EMA plus stop-gradient heuristics, with no guarantee (reader-inferred; LeWM later argues this point, [LeWM] Sec. 1).
- **Relevance to our setup:** the seminal definition of "predict the embedding, not the pixels". Our on-robot predictor that rolls z forward is a JEPA predictor whose target encoder is the frozen slow model.
- Read scope: partial (abstract, Sec. 1, Tab. 4).

**[V-JEPA] Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., Ballas, N. (2024).** *Revisiting Feature Prediction for Learning Visual Representations from Video.* Transactions on Machine Learning Research (TMLR) 2024. arXiv:2404.08471. https://arxiv.org/abs/2404.08471
- **Key findings:**
  - Video JEPA trained only with feature prediction (no pretrained image encoder, text, negatives or reconstruction) on 2M videos (abstract).
  - The frozen ViT-H/16 gets 81.9% on Kinetics-400, 72.2% on Something-Something-v2 (motion-heavy) and 77.9% on ImageNet-1K (abstract). The same frozen backbone works for motion and appearance tasks (Sec. 1).
- **Quality tier:** Tier 3 (TMLR 2024, venue via mlanthology.org TMLR 2024 record). No control evaluation.
- **Method weaknesses:** not assessed beyond the abstract and introduction (read scope: partial).
- **Relevance to our setup:** the strong Something-Something-v2 score is the first sign that video JEPA features encode motion, which a VLM's frame-level SigLIP features may not. This is the basis of H0's alternative slow-side encoder.
- Read scope: partial (abstract, Sec. 1).

**[VJEPA2] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., Ballas, N. (2025).** *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv preprint. arXiv:2506.09985. https://arxiv.org/abs/2506.09985
- **Key findings:**
  - Encoder: V-JEPA 2 scaled from 300M (ViT-L) to about 1B (ViT-g) parameters on over 1M hours of video (Sec. 1; Sec. 2).
  - **V-JEPA 2-AC (confirmed):** an action-conditioned predictor on top of the *frozen* V-JEPA 2 encoder. It is a ~300M-parameter transformer (24 layers, 16 heads, width 1024) with block-causal attention over patch features, end-effector state and action tokens (Sec. 3.1, "Architecture"). It is trained with a teacher-forcing loss plus a rollout loss to reduce error accumulation (Fig. 6), on under 62 h of unlabelled Droid video at 256×256 and 4 fps (Sec. 3.1).
  - Planning: CEM minimises the L1 distance between the predicted latent and a goal-image latent (Eq. 5, Sec. 3.2) and runs as receding-horizon MPC.
  - **Real robot (confirmed):** zero-shot on Franka arms with RobotiQ grippers in two labs not in Droid, 10 trials per cell (Sec. 4, Tab. 2). Averages over the two labs: reach 100%, grasp cup 65% / box 25%, reach-with-object 75% / 75%, pick-&-place cup 80% / box 65%. Octo trained on Droid gets 15% / 10% on pick-&-place (Tab. 2; values read from pdftotext, re-check).
  - Speed: 16 s per action on one RTX 4090, vs 4 min per action for the Cosmos video world model, which also does worse (Tab. 3, Sec. 4).
- **Quality tier:** Tier 2 (arXiv preprint; no peer-reviewed venue found). The real-robot evidence is strong but small (10 trials per cell).
- **Method weaknesses:**
  - Sensitive to camera position: the action frame must be inferred from a monocular view, and the authors hand-picked a working camera pose (author-acknowledged, Sec. 4 "Limitations"; App. B.4).
  - Autoregressive error accumulation limits long-horizon planning; tasks need sub-goal images (author-acknowledged, Sec. 4 "Limitations"; Fig. 10).
  - 16 s per action and 300M + 1B parameters are far from on-robot use (reader-inferred).
- **Relevance to our setup:** the strongest evidence that an action-conditioned predictor on a frozen video-JEPA latent captures manipulation-relevant dynamics on real arms. It is also the template for the slow-side alternative encoder. Our predictor would be the same idea at 10³–10⁴× fewer parameters, run as a filter and not a planner.
- Read scope: full_text (Secs. 1–4, Limitations, App. on AC architecture; via pdftotext + arXiv HTML).

**[RoboJEPA] Zholus, A., Beltran-Velez, N., Yuan, J., Chandar, S., Nagarajan, T., Severo, D., Sinha, K., Drozdzal, M., Romero Soriano, A., Bohg, J., Ballas, N., Assran, M. (2026).** *RoboJEPA: Scaling Robotic Latent World Models.* arXiv preprint. arXiv:2610.10515. https://arxiv.org/abs/2610.10515
- **Key findings:**
  - JEPA predictors from **22M to 8B** parameters on a frozen V-JEPA 2.1 encoder, trained on data from 12 embodiments with action and proprioceptive tokens (abstract; Sec. 2).
  - The imagination error (L1 of latent rollouts) follows a power law in compute and extrapolates to 4B / 8B (Fig. 5; Sec. 3.1). It correlates with downstream planning success in RoboCasa (Fig. 6).
  - Real Franka (DROID platform, image goal, zero-shot): every model size, **including the smallest**, has non-zero success on Grasp, Object Lift and Pick-and-Place, and success grows with size (Sec. 3.4, Fig. 10).
  - RoboJEPA-8B: Grasp 67%, Object Lift 50%, Pick-and-Place 27% success. As contextual references (text goals, DROID-fine-tuned), π0.5 gets 5% / 0% / 53% and π0-FAST 22% / 12% / 40% (Tab. 3; columns de-interleaved from pdftotext, re-check).
  - Planning is "not real-time": several seconds per multi-step plan, run on a GPU cluster wired to the robot (Sec. 2.2).
- **Quality tier:** Tier 2 (arXiv preprint, October 2026).
- **Method weaknesses:**
  - The scaling laws cover the predictor on a fixed frozen encoder, not the encoder and predictor jointly (author-acknowledged, Sec. 4).
  - No text conditioning and no policy proposal; actions are sampled uniformly (author-acknowledged, Sec. 4).
  - The VLA comparison is not like-for-like (author-acknowledged, Sec. 3.4).
- **Relevance to our setup:** the only source with a size sweep down to 22M for a JEPA predictor on real-robot data. It suggests the predictor's quality degrades smoothly with size rather than collapsing. It does not test our regime (10k–100k, filtering rather than planning). The "imagination error" is the open-loop version of our latent innovation.
- Read scope: partial (abstract, Secs. 1–2.3, 3.4, Tab. 3, Conclusion).

**[DINO-WM] Zhou, G., Pan, H., LeCun, Y., Pinto, L. (2025).** *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning.* ICML 2025 (PMLR 267:79115–79135). arXiv:2411.04983. https://arxiv.org/abs/2411.04983
- **Key findings:**
  - A ViT predictor (depth 6, 16 heads, about **19M parameters**) predicts future *frozen* DINOv2 patch features from actions. Planning is CEM to a goal-image feature; no reconstruction is needed (Sec. 3; App. A.9).
  - Patch features beat global vectors: PushT success 0.90 for DINO patch vs 0.44 DINO CLS, 0.42 R3M, 0.20 ResNet (Tab. 2). The authors say single-vector encoders lose spatial detail needed for manipulation (Sec. 4.4).
  - One forward pass takes 0.014 s (batch 32, A6000); CEM planning takes 53 s (Tab. 10, App. A.8).
  - Without reward, TD-MPC2 struggles in this offline, goal-image setting (Sec. 4.2).
- **Quality tier:** Tier 2 (ICML 2025, venue via PMLR v267 index). Six sim environments only, no real robot.
- **Method weaknesses:**
  - Needs offline data with good state–action coverage and ground-truth actions (author-acknowledged, Sec. 6 "Limitations").
  - Plans only in action space, with no hierarchy for fine control (author-acknowledged, Sec. 6).
- **Relevance to our setup:** (a) a frozen pretrained encoder plus a small predictor is enough for useful latent dynamics. This is exactly our structure, with the slow model as the frozen encoder. (b) A pooled single-vector latent loses manipulation-relevant spatial state (Tab. 2). That is a warning for our compressed z: if z is a pooled VLM vector, the predictor may have nothing dynamic to roll forward (H0).
- Read scope: full_text (Secs. 3–4, 6, App. A.8–A.9).

**[LeWM] Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. (2026).** *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.* arXiv preprint. arXiv:2603.19312. https://arxiv.org/abs/2603.19312
- **Key findings:**
  - End-to-end JEPA from pixels with two losses: next-embedding prediction plus SIGReg, which pushes embeddings toward an isotropic Gaussian. That leaves one effective hyperparameter and gives provable anti-collapse without EMA or stop-gradient (abstract; Sec. 3, Eq. for L_LeWM).
  - Size: ~5M-parameter ViT-tiny encoder plus ~10M predictor, about 15M in total (Sec. 3).
  - Full planning in 0.98 s vs 47 s for DINO-WM (about 48× faster) at competitive success on Push-T, Reacher and OGBench-Cube (Fig. 3; Sec. 4.2).
  - Linear and MLP probes recover physical quantities (Tab. 1, Push-T; Tabs. 3–4 in App. F.2).
  - **Surprise (prediction error in latent space) spikes on physical violations** (teleported objects, p < 0.01 in all three environments). It rises less, and not significantly, for colour changes (Fig. 8; Sec. 5.2).
- **Quality tier:** Tier 2 (arXiv preprint; no venue confirmed; sim only). An independent reproduction questions part of its evaluation (arXiv 2608.10145, title only, not assessed).
- **Method weaknesses:**
  - Underperforms on the simplest low-dimensional environment (Two-Room); SIGReg struggles when intrinsic dimension is low (author-acknowledged, Sec. 4.2 and Sec. 6).
  - Short-horizon only, and needs offline data with coverage and action labels (author-acknowledged, Sec. 6).
- **Relevance to our setup:** two direct supports. (1) The anti-collapse recipe for training a small latent predictor end-to-end; we may not need it when the target is a frozen teacher z (see [Ni24]). (2) **Latent prediction error works as a calibrated physical "surprise" signal that ignores appearance changes.** That is close to the latent innovation ‖z_new − ẑ‖ we want to use as a trust signal.
- Read scope: full_text (Secs. 1–6, App. F).

### (2) Latent-predictive representations for control

**[SPR] Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., Bachman, P. (2021).** *Data-Efficient Reinforcement Learning with Self-Predictive Representations.* ICLR 2021. arXiv:2007.05929. https://arxiv.org/abs/2007.05929
- **Key findings:**
  - An action-conditioned transition model predicts the agent's own latent K steps ahead. Targets come from an EMA target encoder plus projection, with a cosine-similarity loss and no reconstruction (Sec. 2, Fig. 2).
  - Median human-normalised score 0.415 on Atari-100k, state of the art at the time; it beats prior methods even without augmentation (abstract; Tab. 1).
- **Quality tier:** Tier 1 (ICLR 2021 spotlight, venue via iclr.cc virtual page; standard benchmark, Atari-100k, 10 seeds).
- **Method weaknesses:** collapse avoidance relies on EMA and projection heuristics (reader-inferred). Discrete-action games only, no robots (reader-inferred).
- **Relevance to our setup:** the canonical K-step latent rollout loss. Our on-robot predictor can be trained with the same multi-step loss, but against the *teacher's* future z rather than an EMA of itself.
- Read scope: partial (abstract, Secs. 1–2).

**[TD-MPC2] Hansen, N., Su, H., Wang, X. (2024).** *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. arXiv:2310.16828. https://arxiv.org/abs/2310.16828
- **Key findings:**
  - A decoder-free ("implicit") latent world model trained with joint-embedding prediction, reward and value losses, and MPPI planning in latent space (Sec. 3).
  - SimNorm (a simplex-normalised latent) is essential for training stability (Sec. 3).
  - One hyperparameter set covers 104 continuous-control tasks. The agents range from **1M to 317M** parameters; a single 317M agent handles 80 tasks (abstract; Fig. 1; Sec. 4).
- **Quality tier:** Tier 1 (ICLR 2024, venue via proceedings.iclr.cc 2024 index; standard benchmarks DMControl, Meta-World, ManiSkill2, MyoSuite).
- **Method weaknesses:**
  - Needs rewards. In reward-free offline goal-reaching it underperforms ([DINO-WM] Sec. 4.2; reader-inferred for our case).
  - Inference rate on embedded hardware is not reported (reader-inferred).
- **Relevance to our setup:** shows that a 1M-scale latent model with joint-embedding prediction is a working control substrate. SimNorm is a cheap stabiliser to try for our small predictor's latent.
- Read scope: partial (abstract, Secs. 1, 3, 4 headline results).

**[Ni24] Ni, T., Eysenbach, B., Seyedsalehi, E., Ma, M., Gehring, C., Mahajan, A., Bacon, P.-L. (2024).** *Bridging State and History Representations: Understanding Self-Predictive RL.* ICLR 2024. arXiv:2401.08898. https://arxiv.org/abs/2401.08898
- **Key findings:**
  - Many state and history abstractions (bisimulation, model-based, SPR-style) share one self-predictive condition (ZP: the latent predicts the next latent) (Sec. 3, Theorem 1).
  - ZP plus Q* implies reward prediction (Theorem 2).
  - The ℓ2 self-prediction objective with stop-gradient on the target has stationary points that avoid collapse; online (no stop-gradient) targets can collapse (Sec. 4, Proposition 3).
  - A minimalist algorithm with one auxiliary loss works in MDPs, distractor MDPs and sparse-reward POMDPs (abstract; Sec. 5).
- **Quality tier:** Tier 1 (ICLR 2024, venue via proceedings.iclr.cc 2024 index; standard MuJoCo/MiniGrid benchmarks).
- **Method weaknesses:** the theory assumes idealised optimisation and deterministic or known stochasticity; the practical ℓ2 objective is only an upper bound under stochastic dynamics (author-acknowledged, Propositions 1–2).
- **Relevance to our setup:** the theory for why predicting a *frozen* target cannot collapse. Our predictor targets the slow model's z, which the student cannot move, so the stop-gradient condition holds trivially. Collapse is then not the risk; the risk is a z that carries no dynamic information (H0). Their POMDP history view (Sec. 3) also fits our GRU belief.
- Read scope: partial (abstract, Secs. 1, 3–4 propositions).

**[VLA-JEPA] Sun, J., Zhang, W., Qi, Z., Ren, S., Liu, Z., Zhu, H., Sun, G., Jin, X., Chen, Z. (2026).** *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model.* arXiv preprint. arXiv:2602.10098. https://arxiv.org/abs/2602.10098
- **Key findings:**
  - JEPA pretraining for a VLA (Qwen3-VL backbone). A V-JEPA 2-trained target encoder encodes future frames as targets only. The VLM sees only the current observation, and a predictor maps history latents plus latent actions to future latents (abstract; Sec. 3, Fig. 2).
  - The authors argue pixel-difference latent actions encode nuisance motion and leak future information (Sec. 1).
  - Reported gains on LIBERO, LIBERO-Plus, SimplerEnv and a real Franka (Tabs. 1–2, Sec. 4).
- **Quality tier:** Tier 2 (arXiv preprint).
- **Method weaknesses:** not assessed beyond the method and headline results (read scope: partial). The benefit is shown for a large VLA, not for a small downstream policy (reader-inferred).
- **Relevance to our setup:** adds a JEPA future-latent objective *inside* the slow model. This would make the z we transmit more predictive of the future, so our on-robot predictor has more to work with. It is a slow-side remedy for H0.
- Read scope: partial (abstract, Secs. 1, 3, Tab. 1 caption).

**[DistillWM] Dao, T., Yamsani, S., Park, J., Kim, J., Lee, Y. J. (2026).** *Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies.* arXiv preprint. arXiv:2609.24682. https://arxiv.org/abs/2609.24682
- **Key findings:**
  - A cosine representation-alignment loss between the student VLA's visual features and cached features of a frozen world model. Training needs no teacher forward pass, and the deployed graph is unchanged (Sec. III; Sec. I).
  - The 0.8B student reaches 97.9% on LIBERO, 2.6 points above the undistilled student (Sec. I). It matches a 4B π-style policy at 28/30 on a real pick-and-place task (Sec. I).
  - The teacher choice matters less than having one: V-JEPA2-AC (+1.2), Fast-WAM (+1.6) and Cosmos3-Nano (+2.6) all improve over the 95.3 baseline (Tab. VI, Sec. IV).
  - Inference: 32 ms for the 0.8B student vs 65 ms for π0.5 on an RTX 5090 (Sec. I, Fig. 1).
- **Quality tier:** Tier 2 (arXiv preprint; LIBERO, RoboCasa-GR1 and real robots).
- **Method weaknesses:** the gains are "modest but consistent" (author-acknowledged, Conclusion). The student is still 0.8B, three orders of magnitude above our budget (reader-inferred).
- **Relevance to our setup:** evidence that world-model *features* (including V-JEPA2-AC predictor states) transfer to a smaller policy through feature alignment alone. This is the teacher→student route of H4, using a world-model teacher instead of a VLM.
- Read scope: partial (Secs. I, III, Tab. VI text, Conclusion).

### (3) Small / fast latent predictors and delay

**[SkyJEPA] Rao, P., Zhang, W., Balestriero, R., LeCun, Y., Loianno, G. (2026).** *SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors.* arXiv preprint (under review). arXiv:2606.23444. https://arxiv.org/abs/2606.23444
- **Key findings:**
  - A JEPA latent dynamics model over a state-and-action history: TCN encoders (channels [8,8,16] and [4,4,8]) and a **single-layer GRU predictor with hidden size 24**, about **9K parameters** in total. It is trained with recursive 20-step unrolling (1 s at 20 Hz), history H = 10, and SIGReg against collapse (Sec. V-C).
  - A physics-inspired prober maps frozen latents to interpretable state, which keeps long rollouts physically grounded (Sec. IV; Fig. 8).
  - Runs inside MPPI with TensorRT on a **Jetson Orin NX, fully on board, within the 10 ms (100 Hz) control budget** at U = 20, S = 512 (Sec. VI, Fig. 5).
  - Zero-shot sim-to-real outdoor flights: position RMSE 26–38% lower than an MPPI with a direct predictive state-space model, e.g. circle 0.39 → 0.24 m (Sec. VI, Tab. IV).
  - About 55% lower pose RMSE under observation corruption than the predictive baseline (Sec. VI, Fig. 9).
- **Quality tier:** Tier 2 (arXiv preprint, under review; real outdoor flights, 5 runs per trajectory).
- **Method weaknesses:**
  - The input is an estimated full state, not images or a remote embedding; the "latent" is a learned encoding of low-dimensional state (reader-inferred).
  - 5 runs per trajectory with no seed variation reported (reader-inferred).
- **Relevance to our setup:** **the closest match to our on-robot predictor's size and rate.** A ~9K-parameter GRU JEPA predictor, trained with multi-step latent rollout and SIGReg, runs at 100 Hz on embedded hardware with real-flight evidence. It shows the predict step fits our 10k–100k budget. Missing: conditioning on, or rolling forward, a *remote* latent.
- Read scope: partial (abstract, Secs. I, IV, V-C, VI tracking results).

**[DREAM-Chunk] Chen, W., Zhang, K., Lin, C., Zhang, Z., She, Y., Liu, Y., Yeh, R. A., Mou, S., Gu, Y. (2026).** *DREAM-Chunk: Reactive Action Chunking with Latent World Model.* arXiv preprint. arXiv:2606.18589. https://arxiv.org/abs/2606.18589
- **Key findings:**
  - At test time, sample N candidate chunks from a VLA and roll out each one's latent future with a light auxiliary world model (JEPA/LeWM or RSSM). During execution, keep executing from the chunk whose "dreamed" latent best matches the *observed* latent (abstract; Sec. 4, Fig. 3). This is latent-innovation-based selection.
  - The world model is ≤15M parameters, and encoding plus prediction take <10 ms, vs >100 ms for VLA inference (Sec. 4; Tab. A2).
  - Hardware with SmolVLA on SO-101 (20 rollouts): USB unplug/insert 75 → 95%, pick moving object 60 → 80%, pick-and-insert 35 → 45% (Tab. 1).
  - π0.5 on Franka can insertion: open loop 10% → 65% with local N = 5 (Tab. 2).
  - In the **remote** setup (~1 s communication delay), success is 40% at N = 10 and falls to 30–35% as N grows, because the added delay offsets the gain (Sec. 5.2, Tab. 2).
- **Quality tier:** Tier 2 (arXiv preprint; Kinetix sim plus two real arms).
- **Method weaknesses:**
  - Needs extra test-time compute for many candidates (author-acknowledged, App. "Limitation").
  - Under remote inference, larger N stops helping (author-acknowledged, Sec. 5.2).
  - Only 20 rollouts per cell (reader-inferred).
- **Relevance to our setup:** the nearest precedent for using latent prediction-vs-observation mismatch *online* to decide which slow-model output to trust, with a small (<10 ms) world model next to a >100 ms VLA. It also contains the only remote-inference experiment in this slice. Difference from our idea: it selects among candidate chunks; it does not roll a stale latent forward or gate a newly arrived z.
- Read scope: partial (abstract, Secs. 1, 4, 5.2, Tabs. 1–2).

**[ContactGuard] Zheng, G., Johnson-Roberson, M., Zhi, W. (2026).** *ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models.* arXiv preprint. arXiv:2608.13438. https://arxiv.org/abs/2608.13438
- **Key findings:**
  - A frozen JEPA-style action-conditioned latent world model (multi-view encoder, SIGReg) rolls the visual latent forward under the policy's planned chunk. A logistic-regression probe on the *predicted* post-contact latent triggers an abort before contact (Sec. 3, Fig. 2).
  - Across four real grasp tasks it beats a matched current-latent monitor, a direct MLP and external detectors such as SAFE on ROC AUC (Tab. 2, Sec. 4).
  - Swapping only the proposed action changes the predicted failure probability, so the monitor responds to the action and not only to the scene (Sec. 5).
- **Quality tier:** Tier 2 (arXiv preprint; real robot, small labelled sets).
- **Method weaknesses:**
  - It abstains and cannot recover; it only covers imminent contact events within one chunk (author-acknowledged, Sec. 5 "Limitations").
  - Vision-only; no force/torque (reader-inferred).
- **Relevance to our setup:** shows that a latent predictor plus a cheap linear readout on the predicted latent is enough for an online decision signal on a real robot. For us, the same structure would read the trust in z (or a failure risk) off ẑ, and could add F/T innovation, which it lacks.
- Read scope: partial (abstract, Secs. 1, 3, 5).

### (4) Do predictive features carry physical state?

**[IntPhys-VJEPA] Garrido, Q., Ballas, N., Assran, M., Bardes, A., Najman, L., Rabbat, M., Dupoux, E., LeCun, Y. (2025).** *Intuitive physics understanding emerges from self-supervised pretraining on natural videos.* arXiv preprint. arXiv:2502.11831. https://arxiv.org/abs/2502.11831
- **Key findings:**
  - Surprise is the latent prediction error between predicted and observed future representations (Fig. 1.C). With it, V-JEPA reaches 98% zero-shot pairwise accuracy on IntPhys and 62% on InfLevel (Sec. 1, Fig. 1.A).
  - Pixel-space predictors (VideoMAEv2) and multimodal LLMs (text reasoning) are close to chance (abstract; Fig. 1.A). Even a 115M model, or one trained on one week of video, is above chance (Sec. 1).
  - By property: strong on object permanence, continuity and shape constancy; **close to chance on solidity and collision** (Fig. 2; Sec. 4 "limitations").
- **Quality tier:** Tier 3 (arXiv preprint; synthetic VoE benchmarks, no control).
- **Method weaknesses:** object interactions (collisions, solidity) are poorly captured, possibly because they are rare in the training data or need hierarchy or interaction (author-acknowledged, Discussion).
- **Relevance to our setup:** the clearest evidence that JEPA latents carry dynamic physical state that MLLMs do not, and that latent prediction error works as a physical surprise signal. The contact caveat matters: collision is exactly where our fast side needs F/T rather than z.
- Read scope: partial (abstract, Secs. 1–4, Discussion).

**[JEPA-VLA] Miao, S., Feng, N., Wu, J., Lin, Y., He, X., Li, D., Long, M. (2026).** *JEPA-VLA: Video Predictive Embedding is Needed for VLA Models.* arXiv preprint. arXiv:2602.11832. https://arxiv.org/abs/2602.11832
- **Key findings:**
  - Frozen encoders with a lightweight regression head on **LIBERO-10**: V-JEPA 2 has lower MSE than DINOv2 and SigLIP for (a) current robot and task-object state from the last two frames and (b) the state residual 10 steps ahead (Sec. 2.1–2.2, Fig. 2c, "Finding 1").
  - V-JEPA 2 has *higher* error on task-irrelevant lighting/background regression on LIBERO-plus, i.e. it discards nuisances (Sec. 2.2, Fig. 2c right).
  - Fusing V-JEPA 2 into VLAs (including OpenVLA-OFT) improves LIBERO, LIBERO-plus, RoboTwin2.0 and a real Piper arm (Sec. 4.3, Tabs. 1–4).
- **Quality tier:** Tier 2 (arXiv preprint; a web listing suggests ICML 2026, not confirmed on a proceedings page).
- **Method weaknesses:**
  - The probes compare *vision encoders* (SigLIP, DINOv2, V-JEPA 2), not the internal latents of a full VLM/VLA after language fusion (reader-inferred).
  - Probe MSE is shown relative to V-JEPA 2 in a figure; absolute values are not in the text (reader-inferred).
- **Relevance to our setup:** direct H0 evidence on our benchmark (LIBERO-10). A SigLIP-based VLM vision tower holds less task-relevant object/robot state, and less future-state information, than V-JEPA 2. Running the same probe on π0.5's latent z would directly test H0.
- Read scope: partial (abstract, Secs. 1–2, 4.3 headline).

**[ALPS] Liu, Y., Zhang, L., Liu, C., Wang, Z., Mei, J., Fei, H., Zhou, Y., Xie, C. (2026).** *ALPS: Visual State Encoders for World Models.* arXiv preprint. arXiv:2610.30120. https://arxiv.org/abs/2610.30120
- **Key findings:**
  - A LIBERO state probe: a 2.5M-parameter attentive probe reads frozen tokens of the last 16 frames to estimate movable objects' 3D positions, orientations and predicates. It does readout and action-conditioned prediction at +1 s and +3 s (Sec. 4, Fig. 3).
  - Across 15 public encoders, image-text encoders (CLIP-B/16, SigLIP 2-B/16) have the highest relative state error (>1.1 in the readout column). V-JEPA 2-L reaches an aggregate .829, equal to the best ViT-B (ALPS) with 3.5× the parameters (Tab. 1, Sec. 5.1; columns scrambled in pdftotext, re-check).
  - V-JEPA 2-L and ALPS show complementary strengths across settings (Sec. 5.1).
- **Quality tier:** Tier 2 (arXiv preprint, October 2026).
- **Method weaknesses:** not assessed beyond the probe protocol and Tab. 1 (read scope: partial). The authors propose ALPS, so the comparison may favour it (reader-inferred).
- **Relevance to our setup:** the most systematic LIBERO state-probe comparison found, with a ready-made protocol (readout, +1 s, +3 s) that we could reuse to probe z. It independently confirms that CLIP/SigLIP-style features (the vision towers of most VLMs) carry the least physical state.
- Read scope: partial (abstract, Secs. 1, 4, 5.1, Tab. 1).

**[PSG-JEPA] Yan, H., Zhu, J., Jia, M., Yin, R., He, J., Zhong, Z., Li, J., Lu, J., Li, H., Zhang, T., Chen, J., Song, W., Chen, W., Gao, Y., Li, H. (2026).** *Is Forward Prediction Enough? Physical State Grounding for JEPA World Models.* arXiv preprint. arXiv:2608.06799. https://arxiv.org/abs/2608.06799
- **Key findings:**
  - Forward-prediction-only JEPA world models do not reliably encode robot-centric physical state. Single-latent probes miss some proprioceptive variables, and pairwise probes recover joint/gripper velocity only partly (Sec. 1, Fig. 1c; Tabs. 1–2 on OGBench-Cube).
  - Adding training-only grounding losses (latent → proprioception, latent pair → multi-horizon joint-angle change) improves probes, planning and policy learning, including on a real robot, at unchanged inference cost (abstract; Tab. 3).
- **Quality tier:** Tier 2 (arXiv preprint).
- **Method weaknesses:** not assessed beyond the abstract, Sec. 1 and table captions (read scope: partial).
- **Relevance to our setup:** a counterweight to [JEPA-VLA] and [IntPhys-VJEPA]. JEPA latents are not automatically good state; grounding them in proprioception helps. For our on-robot predictor this suggests an auxiliary loss from ẑ to proprio and torque, which is cheap because the fast side measures both.
- Read scope: partial (abstract, Sec. 1, Sec. 4 probe setup, table captions).

**[ProbeOpenVLA] Lu, H., Li, H., Shahani, P. S., Herbers, S., Scheutz, M. (2025).** *Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architecture.* arXiv preprint (lab page lists Proceedings of ICAD 2025, not confirmed on a proceedings site). arXiv:2502.04558. https://arxiv.org/abs/2502.04558
- **Key findings:**
  - Linear probes on all 33 hidden layers of OpenVLA's Llama-2 7B backbone, on LIBERO-Spatial pick-and-place. They predict 224 binary object-state atoms (properties, relations) and 12 action-state atoms with >0.90 accuracy across most layers (abstract; Secs. IV–V).
  - Object states are not encoded earlier than action states, contrary to the authors' hypothesis (abstract; Sec. V).
- **Quality tier:** Tier 2 (preprint or small venue; one LIBERO suite).
- **Method weaknesses:**
  - Binary symbolic predicates only, with no class balancing (author-acknowledged design choice, Sec. IV "No Class Balancing").
  - No continuous positions, velocities or future states; high accuracy on imbalanced predicates may overstate content (reader-inferred).
- **Relevance to our setup:** the counter-evidence to a strong H0. A VLA's internal latent does linearly encode *symbolic* scene state on LIBERO. Together with [JEPA-VLA] and [ALPS], the open question narrows to whether VLM latents carry *continuous, dynamic* state (positions, velocities, contact), which no included source measures on a VLA latent.
- Read scope: partial (abstract, Secs. I, IV–V).

## 6. Unverified / excluded

| Item | Status | Reason |
|---|---|---|
| LeCun, "A Path Towards Autonomous Machine Intelligence" (OpenReview id BZ5a1r-kVsf, v0.9.2, 2022-06-27 per search snippet) | **Unverified** | OpenReview API and forum returned a browser-verification challenge (403); only search snippets and third-party course mirrors were reachable. Not in the bib. Verify manually in a browser before citing |
| "LagComp: Compensating Observation-to-Action Latency Through Latent Feature Forecasting" (ResearchGate 408480948) | **Unverified** | Found only as a ResearchGate listing; no arXiv ID, DOI or project page after three searches. Highly relevant on topic (forecasts a frozen policy's latent by the measured lag), so re-search later |
| SmolWAM (NeurIPS 2026 virtual page; HF model card) | Unverified | No arXiv ID or proceedings entry fetched; 1.4B parameters, far above our budget. Candidate for Theme 3 |
| MC-JEPA (Bardes, Ponce, LeCun; arXiv:2307.12698) | Verified (arXiv API), excluded | Motion + content JEPA (optical flow); no control evidence. Superseded for us by V-JEPA / V-JEPA 2 |
| TD-MPC (Hansen, Wang, Su; arXiv:2203.04955) | Verified (arXiv API), excluded | Superseded by TD-MPC2 for this slice |
| BYOL-Explore (Guo et al.; arXiv:2206.08332) | Verified (arXiv API), excluded | Latent prediction error as an exploration bonus; adjacent to "innovation", but RL exploration in games. Candidate if the synthesis needs a "prediction error as signal" lineage |
| Understanding Self-Predictive Learning for RL (Tang et al.; arXiv:2212.03319) | Verified (arXiv API), excluded | Theory of non-collapse in BYOL-style RL; Ni24 covers it for our purposes |
| PLDM: Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models (Sobal et al.; arXiv:2502.14819) | Verified (arXiv API), not assessed | End-to-end JEPA planning baseline used in LeWM; navigation-only |
| Back to the Features: DINO as a Foundation for Video World Models (Baldassarre et al.; arXiv:2507.19468) | Verified (arXiv API), not assessed | Video world model on DINO features; lower priority than DINO-WM |
| Navigation World Models (Bar et al.; CVPR 2025 per arXiv comment; arXiv:2412.03572) | Verified (arXiv API), excluded | Navigation, 1B-scale diffusion world model; out of scope |
| ACT-JEPA (Vujinovic, Kovacevic; IEEE Access 14, 2026; arXiv:2501.14622, DOI 10.1109/ACCESS.2026.3696039) | Verified (arXiv API with journal_ref), not assessed | JEPA objective in an imitation policy; candidate for the (2) list |
| V-JEPA 2.1 (Mur-Labadia et al.; arXiv:2603.14482) | Verified (arXiv API), not assessed | Dense-feature successor, used as RoboJEPA's frozen encoder; cite with RoboJEPA if needed |
| V-JEPA Policy (Zhang et al.; arXiv:2609.37250) | Verified (arXiv API), downloaded, not assessed | World-action model on frozen V-JEPA 2.1 latents with a dual-arm real robot; compares DINOv2/v3, WAN VAE, InternVideo3, V-JEPA 2/2.1 foundations. **High-priority follow-up for (4)** |
| JEPA-WAM (Lin et al.; arXiv:2608.09381), StageWAM (Liu et al.; arXiv:2608.10780, v1 titled "JEPA-WAM: Stage-Level ..."), Demo-JEPA (2605.20811), JEPA Policy (2609.09630), Delta-JEPA (2606.31232), MotionJEPA (2609.23881), TD-JEPA (2510.00739), LaWAM (CoRL 2026 per arXiv comment; 2606.15768) | Verified (arXiv API) except Demo-JEPA (search hit only), not assessed | 2026 JEPA-for-VLA wave; all large-model policies, none on delay or small predictors. Screen in a second pass |
| DEFLECT (2605.19294), Causal World Modeling for Robot Control (2601.21998) | Search hits only, not assessed | Async-VLA latency papers; check overlap with Theme 2 before adding |
| "The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom" (arXiv:2608.10145) | Search hit only, not verified | Relevant caveat for [LeWM]; verify and read before citing LeWM numbers |
