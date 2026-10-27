# Theme 2 (RQ2): Latency, asynchrony and staleness; edge–cloud split robot inference

- **Agent run date:** 2026-10-27 (search window: no lower bound for seminal work, upper bound = 2026-10-27)
- **Protocol:** `../PROTOCOL.md` (rapid scoping review, single AI-assisted screener)
- **Included:** 28 verified sources. One verified source deferred to Theme 3; 11 excluded or unverified (listed at the end).

## 1. Search log

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

## 2. Inclusion notes

- **Included:** work that addresses inference latency or staleness in learned robot policies (chunking, async execution, delay-conditioned training, correction heads, delayed-MDP theory), or that offloads or splits robot or VLA inference over a network.
- **Kept although indirect:** speculative decoding (Spec-VLA) and early exit (DeeR-VLA). They lower latency on the slow model rather than compensating for it.
- **Kept as theory anchors:** two delayed-MDP RL papers.
- **Deferred:** pure hardware benchmarking (Cross-platform scaling, 2509.11480) goes to Theme 3 (RQ3).
- **Excluded:** "Robots that ask for help", per the brief.
- **Count:** 28 included, slightly over the 12–25 target. The 2025–2026 async-VLA literature is dense, and several of these papers are direct baselines for each other.

## 3. Comparison table

"Link" means what crosses the boundary between slow and fast, or between cloud and edge. For pure on-board async methods, that boundary is the inference thread versus the control thread.

| Paper | Year | What is offloaded / what crosses the link | Latency handled (stated) | Technique | Reported robustness |
|---|---|---|---|---|---|
| ACT (Zhao et al.) | 2023 | Nothing offloaded; chunk of k target joint positions | Inference ~0.01 s, 50 Hz control (Sec. on ACT impl.; hardware section) | Action chunking + temporal ensembling (exp. weights) | Not evaluated vs. delay |
| Diffusion Policy (Chi et al.) | 2023 | Nothing offloaded; action sequence | 0.1 s inference (DDIM, 10 steps, RTX 3080); 10 Hz commands interpolated to 125 Hz | Receding-horizon position control, action horizon 8 | Peak performance kept up to 4 steps of simulated latency (Fig. 5) |
| BID (Liu et al.) | 2024/25 | Nothing offloaded | Not stated (abstract) | Test-time sampling with backward coherence + forward contrast | Gains on 7 sim + 2 real tasks (abstract) |
| RTC (Black et al.) | 2025 | Remote GPU over LAN: observations up, chunks (H=50) down | 50 Hz; π0.5 76 ms → 97 ms with RTC; +10–20 ms LAN; injected +100/+200 ms (d≈6/11/16) | Async chunking as inpainting: freeze committed actions, soft-masked guidance | No degradation up to d=16 (real); TE baselines triggered safety stops at +100/+200 ms |
| Training-time RTC (Black et al.) | 2025 | Remote H100: obs up, chunks down | 50 Hz; 108 ms (TT) vs 135 ms (IT-RTC) end-to-end | Simulate delay in training; condition on ground-truth action prefix | Beats IT-RTC at d≥2 in sim; parity in real |
| SmolVLA async (Shukor et al.) | 2025 | PolicyServer (GPU) ↔ RobotClient: obs up, action chunk down | 30 fps (Δt=33 ms); latency assumed dominated by server inference | Queue threshold g, overlap aggregation, joint-space near-duplicate filter | Async 9.7 s vs sync 13.75 s per task; 73.3% vs 78.3% success (Table 5) |
| VLASH (Tang et al.) | 2025/26 | On-board/local GPU | 50 Hz; 30.4 ms (RTX 5090) to 103 ms (2 imgs, 4090) | Condition on rolled-forward future state; train with random offsets | LIBERO 93.1% at d=4 vs 96.8% at d=0; reaction-latency gain 17.4× (v1) / 11.8× (v2 abstract) |
| A2C2 (Sendai et al.) | 2025 | Slow VLA chunk + features → fast per-step head | Head 4.7 ms vs SmolVLA 101 ms (RTX 5080) | Residual correction head on latest obs + chunk index + base features | +23 pp (Kinetix), +7 pp (LIBERO-Spatial) vs RTC; sim only |
| Async comparison (Agouzoul) | 2026 | n/a (benchmark) | Delays up to d=20 steps | Controlled comparison of IT-RTC, TT-RTC, VLASH, A2C2 | A2C2 >90% to d=8 (Kinetix); TT-RTC most robust training-based; IT-RTC collapses at high d |
| REMAC (Wang et al.) | 2026 | On-board | 15 Hz; 122–140 ms (d≈2–3) | Masked action chunking (LoRA) + prefix-preserved sampling | Beats RTC/BID d=0–4; ±1 delay jitter tested |
| Legato (Liu et al.) | 2026 | On-board (RTX 4090) | ~30 Hz; ~170 ms/forward; d∈{6,8,10} (≈200–333 ms) | Train from a schedule-shaped action/noise mixture | ~10% faster completion vs RTC; smoother than TT-RTC |
| Action ControlNet (Guo & Guo) | 2026 | On-board | MT50: 91 ms vs RTC 159 ms; d up to 15 | Adapter conditioned on executed "delay action" suffix | ≈TT-RTC success with ~20% trainable params |
| FutureRTC (Jiang et al.) | 2026 | On-board | 30 Hz; 170 ms and 270 ms (real); d up to 20 (sim) | Predict future state residual + latent future observation via motion-aware feature transport | 88.5% vs VLASH 74.5% at d=20 (LIBERO) |
| Real-Time EXPO-FT (Dong et al.) | 2026 | Server VLA proposes chunks; fast edit policy acts on latest obs | 30 Hz; VLA ~67 ms + 100 ms injected; d=3/5 | Bounded RL edits + Q-selection of chunks | 42% → 97% on 4 dynamic real tasks (abstract) |
| Streaming Flow Policy (Jiang et al.) | 2025 | Nothing offloaded; actions streamed during flow integration | Not stated (abstract) | Flow from narrow Gaussian around last action | Not stated for delay |
| Derman et al. | 2021 | n/a (theory) | m-step execution delay | Non-stationary Markov policies; model-based Q-learning without augmentation | Converges under substantial delays (abstract) |
| DCAC (Ramstedt et al.) | 2020/21 | n/a (RL) | Random action and observation delays | Hindsight resampling of trajectory fragments; SAC-based | Better than SAC on delay-augmented MuJoCo (abstract) |
| FogROS2 (Ichnowski et al.) | 2023 | Whole ROS 2 nodes to cloud; compressed video up | Network RTT for images −97% vs FogROS1 (abstract) | Automated cloud provisioning, video compression | Not evaluated for control robustness |
| FogROS2-PLR (Chen et al.) | 2024/25 | ROS 2 traffic duplicated over multiple NICs/servers | P99 latency up to 3.7× better (dual 5G) | Redundant links + replicated servers | Tail latency cut 33% indoors |
| SPO (Nguyen et al.) | 2026 | Cloud world model + policy → K=2–10 future state-action tuples (~624 B/step) | 50 Hz; 150 ms RTT ± 30 ms jitter (emulated) | Speculative waypoint buffer + edge verifier + adaptive horizon | Idle time 10.7 s → 4.8 s; sim only |
| AsyncShield (Yang et al.) | 2026 | Cloud VLA → 55 ego-frame waypoints (~3 Hz) | 150–250 ms + 10% spikes 0.5–1.5 s, 15% loss, 2 s outages | Pose-buffer re-projection + PPO-Lagrangian safety adapter | 76.7% vs A2C2 43.3% / RTC 30.0% (mixed degradation); Go2 real 80–90% |
| CloudEdgeVLA (Peng et al.) | 2026 | Cloud OpenVLA-7B → action-token hidden states h∈R^(L×D); edge SigLIP-B + MLP head | Uniform delay up to 40 steps (sim); +0/400/1000 ms RTT (real) | Paired current/delayed-frame training → representational specialisation | 63.8–78.0% at d_max=40 vs VLASH ≤6.4%; 80/70% at +1000 ms (real) |
| RAPID→ActECC (Zheng et al.) | 2026 | Edge obs → cloud; action chunks back; replanning scheduled | Per-action latency −27.4% vs fixed ECC | Kinematics-guided usable-range estimate + delay-aware replanning | Execution error −6.5% to −7.9% |
| RoboECC (Zheng et al.) | 2026 | Split-layer activations (e.g. [1,17,3072] = 102 KB vs [1,17,768] = 25.5 KB) | Orin edge-only 1119 ms → 354 ms | Model-hardware-aware split point + LSTM bandwidth predictor | Success unaffected (not reported) |
| Beyond LLM Serving (Jung et al.) | 2026 | Characterisation: edge server vs on-board SoC | Not stated in abstract | 43,200 closed-loop episodes profiling | No single config dominates SLOs |
| CALF (Purves & Liò) | 2026 | RL policy units as networked services | Injected latency/jitter/loss/bandwidth | Network-aware training via NetworkShim | Sim-to-real gap cut ~4× (CartPole) / ~3× (MiniGrid); jitter and loss worse than constant latency |
| Spec-VLA (Wang et al.) | 2025 | n/a (slow-model acceleration) | 1.42× speed-up vs OpenVLA | Speculative decoding with relaxed acceptance | n/a |
| DeeR-VLA (Yue et al.) | 2024 | n/a (on-board early exit) | Compute 5.2–6.5× lower (CALVIN) | Multi-exit MLLM with budgeted exit criteria | n/a |

## 4. Annotated entries

### 4.1 Chunking foundations

**[ZhaoACT2023] Zhao, T. Z., Kumar, V., Levine, S., Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS 2023.** arXiv:2304.13705. https://arxiv.org/abs/2304.13705 · https://www.roboticsproceedings.org/rss19/p016.html
- **Relevance:** Origin of action chunking and temporal ensembling. These are the default baselines for every async method below.
- **Key findings:**
  - Data and control run at 50 Hz.
  - The model has about 80M parameters, and inference takes about 0.01 s on an RTX 2080 Ti (implementation paragraph).
  - Success rises from 1% at k=1 to 44% at k=100, then dips at k=200 and k=400, which approach open-loop control (ablation, Fig. 8).
  - Temporal ensembling adds about 3.3% for ACT (Fig. 8b).
- **Method:** CVAE transformer that predicts k target joint positions. Temporal ensembling averages overlapping predictions with w_i = exp(−m·i).
- **Quality tier:** Tier 1.
- **Read scope:** partial_full_text.
- **Method weaknesses:**
  - Author-acknowledged (ablation text): long chunks lack reactivity.
  - Reader-inferred: temporal ensembling needs a query every step, which assumes inference is faster than the control period. That fails at our ~5 Hz slow model (see the RTC finding on TE oscillations).
- **Relevance to our setup:** Defines the chunk interface the fast model could execute. Temporal ensembling is unsuitable at 200 ms+ latency.

**[ChiDP2023] Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S. (2023/2024). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023; extended IJRR version.** arXiv:2303.04137. https://arxiv.org/abs/2303.04137
- **Relevance:** Receding-horizon chunked generative policy. Contains an explicit latency ablation.
- **Key findings:**
  - DDIM with 10 inference steps gives 0.1 s latency on an RTX 3080.
  - Real UR5 commands run at 10 Hz and are interpolated to 125 Hz.
  - An action horizon of 8 is the best trade-off.
  - Peak performance holds up to 4 steps of simulated latency, and velocity control suffers more than position control (Sec. "Robustness against latency", Fig. 5).
- **Method:** Conditional denoising diffusion over action sequences, with an observation horizon, prediction horizon and execution horizon.
- **Quality tier:** Tier 1.
- **Read scope:** partial_full_text.
- **Method weaknesses:**
  - Author-acknowledged: an overly long action horizon slows reaction (Fig. 5 left text).
  - Reader-inferred: latency is tested only as a fixed simulated step delay, with no jitter.
- **Relevance to our setup:** Position-control chunks tolerate a few steps of staleness. Prefer position targets over velocity targets on the fast side.

**[LiuBID2025] Liu, Y., Hamid, J. I., Xie, A., Lee, Y., Du, M., Finn, C. (2024/2025). Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling (v1 title: "...via Closed-Loop Resampling"). ICLR 2025.** arXiv:2408.17355. https://arxiv.org/abs/2408.17355
- **Relevance:** Test-time method that restores closed-loop reactivity to chunked policies. It is a baseline in RTC and REMAC.
- **Key findings:** Samples several candidate chunks and picks one by backward coherence and forward contrast. Improves two generative policies on 7 sim benchmarks and 2 real tasks (abstract). RTC reports that BID uses 2.3× more compute than RTC and does worse in the Kinetix sim (RTC paper).
- **Method:** Batch sampling plus a selection criterion at every step.
- **Quality tier:** Tier 1. The venue comes from the iclr.cc listing; the arXiv comment field was not checked.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Multi-sample compute on the slow side is expensive at 27B. Low priority.

**[JiangSFP2025] Jiang, S., Fang, X., Roy, N., Lozano-Pérez, T., Kaelbling, L. P., Ancha, S. (2025). Streaming Flow Policy: Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories. CoRL 2025, PMLR 305:238–257.** arXiv:2505.21851. https://arxiv.org/abs/2505.21851 · https://proceedings.mlr.press/v305/jiang25a.html
- **Relevance:** Actions are emitted during sampling rather than after it, which tightens the loop for a small fast policy.
- **Key findings:** Flow starts from a narrow Gaussian around the last action. Actions stream on the fly and remain multimodal (abstract).
- **Method:** Flow matching over the action trajectory itself.
- **Quality tier:** Tier 1.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Candidate formulation for the on-robot fast model. It produces actions incrementally at ≥20 Hz.

### 4.2 Asynchronous execution of chunked VLAs

**[BlackRTC2025] Black, K., Galliker, M. Y., Levine, S. (2025). Real-Time Execution of Action Chunking Flow Policies. NeurIPS 2025.** arXiv:2506.07339. https://arxiv.org/abs/2506.07339. Companion blog (Tier 3): https://www.pi.website/research/real_time_chunking
- **Relevance:** The reference method for asynchronous chunk execution under inference and network latency with a remote GPU.
- **Key findings:**
  - 50 Hz control (Δt = 20 ms), H=50, s=25, 5 denoising steps.
  - π0.5 latency is 76 ms, or 97 ms with RTC guidance. LAN adds 10–20 ms.
  - Injected delays of +100 ms and +200 ms give d≈11 and d≈16. RTC shows no degradation up to d=16, while synchronous inference degrades linearly.
  - Temporal-ensembling variants oscillated and triggered safety stops at the injected delays (real-world experiment section).
  - The blog breaks down total latency as 108 ms (static) and 139 ms (mobile, 21 ms network), and claims success with delays above 300 ms.
- **Method:** Freezes the actions that will execute during inference and inpaints the rest with soft-masked guidance (exponentially decaying weights, β clip = 5).
- **Quality tier:** Tier 1 (blog: Tier 3).
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (limitations): diffusion/flow policies only; significant compute overhead from backprop through denoising; real evaluation limited to bimanual manipulation, with no legged or locomotion hardware.
  - Reader-inferred: delay d is assumed known or bounded per chunk, and jitter is not studied separately.
- **Relevance to our setup:** Directly applicable if the slow model emits action chunks. The ~20% latency overhead and per-call guidance land on the slow side.

**[BlackTTRTC2025] Black, K., Ren, A. Z., Equi, M., Levine, S. (2025). Training-Time Action Conditioning for Efficient Real-Time Chunking. arXiv preprint.** arXiv:2512.05964. https://arxiv.org/abs/2512.05964
- **Relevance:** Removes the inference-time overhead of RTC by simulating delay during training. This is the cleanest "delay-aware training" recipe.
- **Key findings:**
  - Real-world training samples delays uniformly from 0 to 10 steps, which supports ≤200 ms at 50 Hz. Sim uses {0..4} with exponentially decreasing weights.
  - End-to-end latency on a remote H100 is 108 ms for training-time RTC versus 135 ms for inference-time RTC.
  - Training-time RTC beats inference-time RTC in sim at d≥2 (Fig. 3) and matches it on box building and espresso making with π0.6.
- **Method:**
  - Per-token flow timesteps.
  - Prefix tokens set to ground truth at τ=1.
  - Loss masked to the postfix.
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (Sec. VI): supports only a hard prefix rather than soft masking, and the delay distribution must be chosen carefully.
  - Reader-inferred: behaviour beyond the trained maximum delay is unreported in the paper itself. Agouzoul (2026) reports that it generalises beyond the training delay.
- **Relevance to our setup:** Cheapest way to make a chunk-emitting slow model robust to a 5 Hz cadence plus jitter. Train with our measured delay distribution.

**[ShukorSmolVLA2025] Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., Alibert, S., Cord, M., Wolf, T., Cadene, R. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv preprint.** arXiv:2506.01844. https://arxiv.org/abs/2506.01844
- **Relevance:** Open-source client/server async inference stack (LeRobot) in which observations cross the network and action chunks come back.
- **Key findings:**
  - RobotClient sends o_t and the PolicyServer returns chunk A_t.
  - A new observation is sent when the queue fraction drops below g, with g=0.7 as the balanced setting.
  - Overlapping chunks are aggregated by a function f.
  - Near-duplicate observations, judged by joint-space distance below ε, are dropped.
  - Runs at 30 fps (Δt = 33 ms). Async averages 9.7 s per task versus 13.75 s for sync, completes 19 versus 9 tasks in 60 s, and succeeds 73.3% versus 78.3% (Sec. 3.3, Sec. 4.6, Table 5).
- **Method:** Queue-based async scheduling, with latency modelled as E[ℓ] ≈ E[ℓ_S].
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (Sec. 5.1): limited data diversity, short-horizon tasks.
  - Reader-inferred: network latency is assumed negligible relative to inference, and there is no staleness compensation beyond overlap aggregation. Success drops by 5 pp under async.
- **Relevance to our setup:** Ready-made baseline transport and scheduler for the stationary-box ↔ robot link.

**[TangVLASH2025] Tang, J., Sun, Y., Zhao, Y., Yang, S., Lin, Y., Zhang, Z., Hou, J., Lu, Y., Liu, Z., Han, S. (2025/2026). VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference. arXiv preprint.** arXiv:2512.01031. https://arxiv.org/abs/2512.01031 · https://hanlab.mit.edu/projects/vlash
- **Relevance:** Compensates for staleness by predicting the execution-time proprioceptive state.
- **Key findings:**
  - The state is rolled forward with the already-committed actions; the image is not predicted.
  - Fine-tuning uses random offsets δ ∈ {0..Δ_max}.
  - Latency at 50 Hz with K=25 is 30.4 ms (RTX 5090), 36.1 ms (4090), 64.1 ms (5070), and 103 ms with two images on a 4090.
  - LIBERO success is 93.1% at d=4 versus 96.8% at d=0.
  - The reaction-latency improvement is up to 17.4× in v1 (530.4 → 30.4 ms). The v2 abstract (revised 2026-07-26) says 11.8×. Cite the version used.
- **Method:** Future-state conditioning plus action quantisation for speed-up.
- **Quality tier:** Tier 2.
- **Read scope:** full_text* (v1).
- **Method weaknesses:**
  - Author-acknowledged: accuracy drops at higher delays.
  - Reader-inferred: the visual input stays stale. CloudEdgeVLA (2026) reports VLASH at ≤6.4% under a 40-step uniform delay.
- **Relevance to our setup:** Cheap trick for the fast side: feed the rolled-forward state. Insufficient alone at multi-hundred-ms delays.

**[SendaiA2C22025] Sendai, K., Alvarez, M., Matsushima, T., Matsuo, Y., Iwasawa, Y. (2025). Leave No Observation Behind: Real-time Correction for VLA Action Chunks. arXiv preprint.** arXiv:2509.23224. https://arxiv.org/abs/2509.23224
- **Relevance:** Closest structural analogue to our slow/fast split. A small per-step head fuses the latest observation with the slow model's chunk and latent features.
- **Key findings:**
  - Head inputs: latest observation, base action, sinusoidal chunk-index feature, base latent z_t, and language.
  - Head size is 0.31M parameters (MLP, Kinetix) or 32M (transformer, LIBERO). It runs in 4.7 ms versus 101 ms for SmolVLA on an RTX 5080.
  - Gains over RTC are +23 pp on Kinetix and +7 pp on LIBERO-Spatial, with delays d≤4 (Kinetix) and d up to 10 (LIBERO).
- **Method:** Residual correction trained on top of a frozen VLA.
- **Quality tier:** Tier 2. An IJCAI 2026 PDF appeared in search but was not verified.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (conclusion): needs validation on richer instructions, out-of-distribution settings and more dynamic tasks.
  - Reader-inferred: no real robot, and fixed integer delays.
- **Relevance to our setup:** A blueprint for the fast model: slow latent plus chunk position plus fresh sensors produce the corrected action.

**[AgouzoulAsync2026] Agouzoul, A. (2026). Understanding Asynchronous Inference Methods for Vision-Language-Action Models. arXiv preprint.** arXiv:2605.08168. https://arxiv.org/abs/2605.08168
- **Relevance:** The only controlled head-to-head comparison of IT-RTC, TT-RTC, VLASH and A2C2.
- **Key findings:**
  - Kinetix uses a ~310K MLP-Mixer and d up to 15. LIBERO uses SmolVLA with the action expert only and d up to 20.
  - A2C2 stays above 90% up to d=8 on Kinetix and leads LIBERO from d≥4 by about 10 points.
  - IT-RTC leads at d=0 (~75%) but falls to ~20% at d=20.
  - TT-RTC is the most robust training-based method and generalises beyond its training delay.
  - Inference overhead: IT-RTC 3.0× / 1.20×, TT-RTC and VLASH 1.0×, A2C2 ~1.02× / 1.75×.
- **Method:** Unified codebase and protocol.
- **Quality tier:** Tier 2 (single author).
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (limitations): fixed integer delays only; hyperparameters not retuned; VLASH on LIBERO uses oracle future states; sim only.
  - Reader-inferred: small base policies, which may not transfer to 27B models.
- **Relevance to our setup:** Strongest evidence for choosing between correction-head (A2C2-style) and training-time (TT-RTC) compensation.

**[WangREMAC2026] Wang, H., Zhang, G., Yan, Y., Shang, Y., Kompella, R. R., Liu, G. (2026). Real-Time Robot Execution with Masked Action Chunking. ICLR 2026.** arXiv:2601.20130. https://arxiv.org/abs/2601.20130
- **Relevance:** Delay-aware fine-tuning that handles intra-chunk inconsistency.
- **Key findings:**
  - LoRA fine-tuning adds ≤1.5% parameters on π0.
  - Real Franka runs at 15 Hz with d≈2–3 (122–140 ms).
  - Success goes from 0.888 at d=0 to 0.779 at d=4 (sim). It beats BID and RTC and composes with them, and ±1 delay fluctuation was tested.
- **Method:**
  - Prefix masking.
  - Self-conditioned curriculum.
  - Residual alignment.
  - Prefix-preserved sampling.
- **Quality tier:** Tier 1.
- **Read scope:** full_text*.
- **Method weaknesses:** Author-acknowledged (limitations): the maximum delay must be specified in advance, and exceeding it may fail; needs substantial fine-tuning data.
- **Relevance to our setup:** Shows that bounded-delay training breaks beyond the bound, so the bound must be set from measured p99 jitter.

**[LiuLegato2026] Liu, Y., Yu, H., Zhao, J., Li, B., Zhang, D., Li, M., Wu, W., Hu, Y., Xie, J., Guo, J., Wang, D., Gao, Y. (2026). Learning Native Continuation for Action Chunking Flow Policies. RSS 2026.** arXiv:2602.12978. https://arxiv.org/abs/2602.12978
- **Relevance:** Training-time continuation alternative to RTC and TT-RTC.
- **Key findings:**
  - Tested with π0.5, π0 and GR00T N1.6. Chunks are 60 steps (2 s, ~30 Hz) and a forward pass takes about 170 ms on an RTX 4090.
  - Delays d ∈ {6, 8, 10} correspond to about 200–333 ms.
  - Completion is about 10% faster than RTC on 5 dual-arm tasks; for example, pouring takes 75.73 s versus 95.07 s.
  - On the pour task, NSPARC smoothness is 1.65 versus 2.46 for TT-RTC.
- **Method:** Denoising starts from a schedule-shaped mixture of known actions and noise, with reshaped flow dynamics.
- **Quality tier:** Tier 1.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (limitations): the number of denoising steps is fixed at training time.
  - Reader-inferred: the TT-RTC comparison covers one task only.
- **Relevance to our setup:** Evaluated at 200–333 ms delays, close to our ~5 Hz plus network regime.

**[GuoACNet2026] Guo, T., Guo, M. (2026). Action ControlNet: A Lightweight Delay-Aware Adapter for Smooth Asynchronous Control in Vision-Language-Action Models. arXiv preprint.** arXiv:2606.25985. https://arxiv.org/abs/2606.25985
- **Relevance:** Explicit delay conditioning through the executed action suffix, with a mostly frozen backbone.
- **Key findings:**
  - Trains about 20% of parameters (Kinetix).
  - On MT50, latency is 91 ms (11.0 Hz) versus 159 ms (6.28 Hz) for RTC and 134 ms for TT-RTC.
  - Average success is 0.79 / 0.74 (Kinetix / MT50) versus 0.80 / 0.74 for TT-RTC.
  - SO-ARM101: 20/20 versus 17/20 for naive async.
- **Method:** Transformer encoder over the "delay action" with residual injection into the action head.
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (future work): larger and time-varying delays remain open.
  - Reader-inferred: small real-robot sample (10 trials per task).
- **Relevance to our setup:** Shows that "what was executed while the slow model was thinking" is a compact, effective conditioning signal for the fast side.

**[JiangFutureRTC2026] Jiang, H., Zou, Y., Liang, B., Liu, B., Meng, F., Liu, S. (2026). FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking. arXiv preprint.** arXiv:2607.24008. https://arxiv.org/abs/2607.24008
- **Relevance:** Predicts the future observation in latent space as well as the future state, going beyond VLASH.
- **Key findings:**
  - Base policies are frozen π0.5 and SmolVLA-450M.
  - Control runs at 30 Hz. Real latency is about 170 ms (d≈5), or 270 ms with injected delay (d≈10).
  - On LIBERO at d=20, success is 88.5% versus 74.5% for VLASH, with fewer execution steps (175 versus 197–200).
- **Method:**
  - State-correction residual.
  - Motion-aware feature transport (bilinear warping) plus a synthesis gate.
  - Policy-consistency loss.
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:** Author-acknowledged (discussion): warping struggles with contact deformation and disocclusion.
- **Relevance to our setup:** Feature-space forecasting could run on the slow side, so the embedding is sent "pre-aged".

**[DongEXPOFT2026] Dong, P., Hung, K.-H., Sadigh, D., Finn, C. (2026). Reinforcement Learning for Real-Time Vision-Language-Action Policies. arXiv preprint.** arXiv:2609.18207. https://arxiv.org/abs/2609.18207
- **Relevance:** Slow VLA proposes and a fast lightweight edit policy reacts. It is trained with online RL.
- **Key findings:**
  - Base model π0.5, real control at 30 Hz.
  - VLA latency is about 67 ms, plus 100 ms injected; tested delays are d=3 and d=5.
  - Best on 10 of 10 delayed Kinetix environments.
  - Real success goes from 42% to 97% on 4 dynamic tasks with 10 minutes of online data (abstract).
- **Method:** Bounded edits in [−β, β] plus Q-function chunk selection (EXPO, maximum-entropy RL).
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:** Author-acknowledged (limitations): human resets and task-specific success detectors.
- **Relevance to our setup:** RL fine-tuning route for the fast model once a sim with realistic delay exists.

### 4.3 Delayed-MDP theory

**[DermanDelay2021] Derman, E., Dalal, G., Mannor, S. (2021). Acting in Delayed Environments with Non-Stationary Markov Policies. ICLR 2021.** arXiv:2101.11992. https://arxiv.org/abs/2101.11992
- **Relevance:** Theory for m-step execution delay.
- **Key findings:** State augmentation is exponential in m. Non-stationary Markov policies in the original state space are sufficient, while stationary ones are generally suboptimal (abstract).
- **Method:** Model-based non-stationary Q-learning.
- **Quality tier:** Tier 1.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Justifies conditioning the fast policy on the time index within the chunk, as A2C2 does with τ_k.

**[RamstedtDCAC2021] Ramstedt, S., Bouteiller, Y., Beltrame, G., Pal, C., Binas, J. (2021). Reinforcement Learning with Random Delays. ICLR 2021.** arXiv:2010.02966. https://arxiv.org/abs/2010.02966
- **Relevance:** Handles random (jittery) action and observation delays, as found in remote control.
- **Key findings:** DCAC uses hindsight resampling of trajectory fragments for off-policy multi-step value estimation and beats baselines on delay-augmented MuJoCo (abstract).
- **Method:** SAC-based, with the delay included in the state.
- **Quality tier:** Tier 1.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** RL recipe if the fast model is trained by RL with random link delays.

### 4.4 Cloud / edge offloading and network robustness

**[IchnowskiFogROS22023] Ichnowski, J., Chen, K., Dharmarajan, K., Adebola, S., Danielczuk, M., Mayoral-Vilches, V., Jha, N., Zhan, H., LLontop, E., Xu, D., Buscaron, C., Kubiatowicz, J., Stoica, I., Gonzalez, J., Goldberg, K. (2023). FogROS2: An Adaptive Platform for Cloud and Fog Robotics Using ROS 2. ICRA 2023, pp. 5493–5500.** DOI 10.1109/ICRA48891.2023.10161307. arXiv:2205.09778. https://arxiv.org/abs/2205.09778
- **Relevance:** Standard ROS 2 offloading platform.
- **Key findings:** SLAM latency −50%, grasp planning 14 s → 1.2 s, motion planning 45× faster. Image round-trip latency −97% versus FogROS1, using video compression (abstract).
- **Method:** Automated cloud provisioning, secure networking, compressed transport.
- **Quality tier:** Tier 1. The venue and DOI come from search results citing IEEE Xplore; the DOI page was not fetched.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Infrastructure option. It offloads whole nodes and does not compensate for staleness.

**[ChenFogROS2PLR2025] Chen, K., Tian, N., Juette, C., Qiu, T., Ren, L., Kubiatowicz, J., Goldberg, K. (2024/2025). FogROS2-PLR: Probabilistic Latency-Reliability For Cloud Robotics. ICRA 2025.** arXiv:2410.05562. https://arxiv.org/abs/2410.05562
- **Relevance:** Targets tail latency (jitter) rather than mean latency.
- **Key findings:** Proposes an "impossibility triangle" theorem. P99 is up to 3.7× better with dual 5G; indoors, mean latency drops 36% and P99 drops 33% (abstract).
- **Method:** Redundant network interfaces plus replicated servers.
- **Quality tier:** Tier 1.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Systems-level jitter mitigation that complements learned compensation.

**[NguyenSPO2026] Nguyen, C., Jin, S., Pokorny, F. T., Elmroth, E. (2026). Speculative Policy Orchestration: A Latency-Resilient Framework for Cloud-Robotic Manipulation. arXiv preprint.** arXiv:2603.19418. https://arxiv.org/abs/2603.19418
- **Relevance:** The cloud speculates future waypoints, and the edge verifies them against the measured state.
- **Key findings:**
  - Each stream step is a state-action tuple of 4(d_s+d_a) bytes, for example 624 B in Stack Blocks. The horizon K varies from 2 to 10.
  - Emulated network: 150 ms RTT ± 30 ms jitter, 50 Hz control.
  - Idle time falls from 10.7 s to 4.8 s, and about 60% fewer predictions are discarded than with static caching.
  - Verifier rule: the error e_t is a weighted norm with tolerance 20. On a miss, the robot holds zero velocity and replans.
- **Method:** World-model rollout, edge buffer, adaptive horizon scaling.
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:** Author-acknowledged (limitations): offline calibration of W; learned models only modestly accurate; zero-velocity hold must become a jerk-limited stop on hardware; sim only (RLBench).
- **Relevance to our setup:** Shows that verified speculation with a "hold" fallback works. It sends kinematics rather than latents.

**[YangAsyncShield2026] Yang, K., Chu, Z., Guo, Y., Wang, Z., Xie, S., Shen, Y., Wu, X., Li, X., Xu, M. (2026). AsyncShield: A Plug-and-Play Edge Adapter for Asynchronous Cloud-based VLA Navigation. arXiv preprint.** arXiv:2604.24086. https://arxiv.org/abs/2604.24086
- **Relevance:** Cloud VLA with an edge adapter on a mobile quadruped, tested under realistic heavy-tail jitter, packet loss and outages.
- **Key findings:**
  - The cloud sends 55 ego-frame waypoints at 20 cm spacing, at about 3 Hz.
  - Mixed degradation: 150–250 ms delay 90% of the time, 0.5–1.5 s spikes 10% of the time, 15% packet loss, outages up to 2 s.
  - Success is 76.7% versus 43.3% for A2C2, 30.0% for RTC and 16.7% naive.
  - Unitree Go2 reaches 80–90% across 3 VLAs versus 25–40% with direct execution.
- **Method:** Pose-buffer kinematic re-projection of stale intents, plus a PPO-Lagrangian constrained MDP with LiDAR safety and collision-radius inflation.
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (conclusion): 2D navigation only; 3D unstructured environments are future work.
  - Reader-inferred: geometric re-projection relies on odometry and does not extend to contact-rich manipulation.
- **Relevance to our setup:** Best jitter, loss and outage stress protocol found. Re-projecting stale spatial intents into the current frame is directly usable for a drone.

**[PengCloudEdgeVLA2026] Peng, D., Ma, F., Wang, B., Wang, S., Ma, J. (2026). Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization. arXiv preprint.** arXiv:2608.00569. https://arxiv.org/abs/2608.00569
- **Relevance:** Closest match to our system: a cloud VLA sends a latent, and a light edge head fuses it with current local vision under large delays.
- **Key findings:**
  - The cloud runs OpenVLA-7B with LoRA and sends action-token hidden states h∈R^(L×D); D is not stated in the main text.
  - The edge runs a frozen SigLIP-B and a residual-MLP head on an RTX 5080 workstation. It uses the most recent cloud feature, with no delay or age input.
  - Sim training uses d_max=20, and tests go up to 40.
  - At d_max=40, LIBERO success is 76.4 / 75.6 / 78.0 / 63.8% (Spatial / Object / Goal / Long). VLASH reaches ≤6.4% and OpenVLA-OFT 0% (Table 1).
  - Real Franka at +1000 ms RTT: 80% / 70% (static / dynamic) versus 0% for VLASH (Table 2b).
- **Method:** Paired current and delayed frame training toward the same action target, which induces "slow task feature / fast state correction" specialisation.
- **Quality tier:** Tier 2.
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged (limitations): robustness only shown for the tested delay distributions; long disconnection breaks it; two backbone passes in training; a frozen RGB encoder misses depth, force and contact. Timestamped features and a local fallback are needed.
  - Reader-inferred: the edge is a desktop GPU rather than embedded hardware; the edge control rate is not stated; the link payload size is not reported.
- **Relevance to our setup:** Near-direct prior art. Its age-agnostic design is a key ablation for us, comparing it against explicit delay embedding.

**[ZhengActECC2026] Zheng, Z., Mao, Z., Huang, H., Tian, S., Cao, H., Zhang, Z., Li, C., Chen, J., Li, M., Zou, H., Luo, G., Chen, X. (2026). ActECC: Action Chunk Temporal Harnessing for Edge-Cloud Collaborative VLA Inference System. arXiv preprint (v1, 2026-03-09, titled "RAPID: Redundancy-Aware and Compatibility-Optimal Edge-Cloud Partitioned Inference for Diverse VLA models"; retitled in the revision of 2026-10-26).** arXiv:2603.07949. https://arxiv.org/abs/2603.07949
- **Relevance:** Edge-cloud VLA scheduling: when to request a new chunk given network delay.
- **Key findings:**
  - Evaluated with π0.5 and GR00T N1.5, on a Jetson AGX Orin or Thor edge and an A100 cloud.
  - Per-action latency is up to 27.4% lower than a fixed edge-cloud baseline, with up to 45.1% fewer cloud calls.
  - Execution error is 6.5–7.9% lower, and overhead is sub-millisecond.
  - v1 (RAPID) reported up to 1.73× speed-up with 5–7% overhead.
- **Method:** Kinematics-guided usable-range test B_t + Σ(‖v‖Δ + λ_a‖a‖Δ²) ≤ δ, plus delay-aware replanning.
- **Quality tier:** Tier 2.
- **Read scope:** full_text* (current version).
- **Method weaknesses:**
  - Author-acknowledged: the δ and λ_a calibration is task-specific.
  - Reader-inferred: network settings are not specified; the metric is millimetre error rather than task success; the substantial content change between versions makes citation version-sensitive.
- **Relevance to our setup:** Its "how many committed actions remain valid" estimate could gate when the robot needs a fresh slow-model embedding.

**[ZhengRoboECC2026] Zheng, Z., Cao, H., Chen, J., Tian, S., Li, C., Li, M., Sun, X., Luo, G., Chen, X. (2026). RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models. IJCNN 2026 (accepted).** arXiv:2603.20711. https://arxiv.org/abs/2603.20711
- **Relevance:** Layer-split (split computing) of VLAs between Jetson and A100, adapted to bandwidth.
- **Key findings:**
  - The link carries split-layer activations. A [1,17,3072] split is 102 KB versus 25.5 KB for [1,17,768] (Fig. 3).
  - Latency on Orin+A100 falls from 1119.4 ms (edge-only) to 354.4 ms, and on Thor+A100 from 628.9 ms to 300.1 ms. On the real arm it falls from 1274 ms to 362 ms (Orin).
  - An LSTM bandwidth predictor (20.1 MB) drives the choice of split point.
  - Success rate is described as unaffected and is not reported.
- **Method:** Model-hardware co-aware segmentation, a parameter-sharing pool and threshold switching.
- **Quality tier:** Tier 1 (accepted to IJCNN 2026, per the arXiv comment).
- **Read scope:** full_text*.
- **Method weaknesses:**
  - Author-acknowledged: GPU-only hardware model.
  - Reader-inferred: bandwidth is shown only as illustrative 1–10 MB/s values; no staleness compensation; success is not measured.
- **Relevance to our setup:** Gives realistic activation payload sizes. A ~100 KB tensor at 5 Hz is about 4 Mbit/s, a useful upper bound for our latent link.

**[JungBeyondLLM2027] Jung, S., Moon, S., Jeong, J., Lee, J., Huh, J. (2026). Beyond LLM Serving: Characterizing Vision-Language-Action Workloads for Embodied AI System Design. ASPLOS '27 (to appear).** arXiv:2610.05062. https://arxiv.org/abs/2610.05062
- **Relevance:** VLA serving characterisation in the batch-1, deadline-bound regime, on edge servers versus on-board SoCs.
- **Key findings:**
  - 4 VLAs and 43,200 closed-loop episodes.
  - Output tensor size determines whether a model is memory-bound or compute-bound.
  - GPU frequency scaling gives a platform-dependent energy-latency sweet spot.
  - Overlapping inference with execution trades accuracy against speed and energy, and no single configuration dominates (abstract).
- **Quality tier:** Tier 1 (accepted).
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Methodology template for measuring our own latency and energy budget.

**[PurvesCALF2026] Purves, C., Liò, P. (2026). CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning. arXiv preprint.** arXiv:2603.12543. https://arxiv.org/abs/2603.12543
- **Relevance:** Treats network impairments as a dimension of sim-to-real randomisation.
- **Key findings:**
  - Baselines degrade 40–80% over realistic networks.
  - Network-aware training cuts the gap about 4× (CartPole) and 3× (MiniGrid).
  - Jitter and packet loss hurt more than constant latency (abstract and search snippet).
- **Method:** NetworkShim middleware injects latency, jitter, loss and bandwidth limits.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Not assessed (read scope: abstract_only). Reader-inferred from the abstract: toy tasks only.
- **Relevance to our setup:** Supports training with jitter and loss rather than a fixed delay. The NetworkShim idea is reusable for our sim.

### 4.5 Reducing slow-model latency

**[WangSpecVLA2025] Wang, S., Yu, R., Yuan, Z., Yu, C., Gao, F., Wang, Y., Wong, D. F. (2025). Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance. EMNLP 2025.** arXiv:2507.22424. https://arxiv.org/abs/2507.22424
- **Key findings:** 1.42× speed-up over OpenVLA, with acceptance length +44% (abstract). Relaxed acceptance uses the distance between action tokens.
- **Quality tier:** Tier 1.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Only relevant if the slow model emits autoregressive action tokens. The gain is modest.

**[YueDeeR2024] Yue, Y., Wang, Y., Kang, B., Han, Y., Wang, S., Song, S., Feng, J., Huang, G. (2024). DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution. NeurIPS 2024.** arXiv:2411.02359. https://arxiv.org/abs/2411.02359
- **Key findings:** A multi-exit MLLM with exit criteria tied to average and peak compute and memory budgets. On CALVIN, compute is 5.2–6.5× lower and GPU memory 2–6× lower without a performance loss (abstract).
- **Quality tier:** Tier 1.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Early exit is the main split-computing or adaptive-depth example for robots. It could let the slow model trade depth for freshness.

## 5. Theme gaps

1. **Explicit age conditioning is under-tested.** No included paper conditions a fast policy on a measured, varying delay embedding fed with timestamped features. CloudEdgeVLA deliberately avoids it, and the others use bounded integer delays. Agouzoul and REMAC note fixed or bounded delays as a limitation.
2. **Latent links lack basic measurements.** Only CloudEdgeVLA and A2C2 (base latent z_t) pass a slow-model latent to a fast model under delay. Neither reports the link payload size or a sweep of latent size against robustness.
3. **Embedded and aerial edges are missing.** Edge compute is a desktop GPU (CloudEdgeVLA, A2C2), with Jetson only in the ECC systems papers, which do no staleness compensation. AsyncShield (Go2) is the only mobile-robot result. No drone was found.
4. **Realistic network models are rare.** Only AsyncShield (heavy tail, loss, outages), SPO (±30 ms jitter) and CALF (toy RL) go beyond fixed delay. Most async-VLA work tests integer delays in Kinetix or LIBERO.
5. **Disconnection fallback is unevaluated.** CloudEdgeVLA lists this explicitly as missing. SPO's zero-velocity hold is the only implemented fallback, and only in sim.
6. **Multi-rate sensing on the fast side is unstudied.** No delay paper fuses IMU or torque at a different rate from vision on the fast side. FutureRTC flags contact and deformation as a failure mode of warping-based forecasting. This overlaps with RQ4.

## 6. The three most important papers for us

1. **CloudEdgeVLA (Peng et al., 2026, arXiv:2608.00569).** It is the same architecture as ours: a 7B cloud VLA latent conditions a light edge head with fresh vision. It is robust to delays of 40 steps and +1000 ms RTT, and it is the baseline to beat or extend with age embedding and multi-rate sensors.
2. **A2C2 (Sendai et al., 2025, arXiv:2509.23224), read together with Agouzoul (2026, arXiv:2605.08168).** It is the per-step fast correction head over a slow chunk and latent, using chunk-index time features. The independent comparison ranks it best at mid-to-high delay.
3. **Training-time RTC (Black et al., 2025, arXiv:2512.05964).** It is the zero-overhead delay-aware training recipe for the slow chunk generator, with measured remote-H100 latencies of 108 versus 135 ms at 50 Hz. Agouzoul finds it generalises beyond its training delay.

## 7. Verified but deferred, and unverified or excluded

**Verified, deferred to Theme 3 (RQ3):**
- Taherin et al. (2025), Cross-Platform Scaling of Vision-Language-Action Models from Edge to Cloud GPUs, Asilomar 2025. arXiv:2509.11480 (abs page fetched). Deferred because it is hardware benchmarking without staleness handling.

**Unverified or excluded (not fetched or not checked; do not cite from this file):**
1. HeiSD, hybrid speculative decoding for VLA (arXiv:2603.17573 from search). Abs page not fetched.
2. DEFLECT, delay-robust execution via flow matching (search snippet only). No ID verified.
3. "Latency-aware VLAs with system identification" (Chef Robotics blog, search snippet). Not fetched, and Tier 3 at best.
4. StreamingVLA (arXiv:2603.28565, search result). Not fetched.
5. Tube Diffusion Policy (arXiv:2604.23609, search result). Not fetched, and off-topic (tactile).
6. FogROS2-Config (2311.05600), FogROS2-SGC (2306.17157), FogROS2-FT and FogROS2-Sky. Not fetched; infrastructure variants.
7. "RL via Conservative Agent for Environments with Random Delays" (arXiv:2507.18992). Not fetched.
8. "Vision-Language Models on the Edge for Real-Time Robotic Perception" (arXiv:2601.14921). Not fetched; perception rather than control.
9. A possible IJCAI 2026 version of A2C2 (ijcai.org/proceedings/2026/0228.pdf). Not verified to be the same paper.
10. "Robots that ask for help". Excluded by the brief.
11. Katsikopoulos & Engelbrecht (2003), the classic delayed-MDP paper. Not searched in this run.
