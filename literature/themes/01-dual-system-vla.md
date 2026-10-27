# Theme 1 (RQ1): Dual-system / hierarchical "slow-fast" robot policies

- **Date of search:** 2026-10-27 (coverage up to present)
- **Screener:** single AI agent (rapid review shortcut, see PROTOCOL.md)
- **Read-scope convention:** "full text (HTML)" means the arXiv HTML full text was fetched and targeted facts were extracted by an automated reader. Numbers carry the section/table locator the extraction reported. "Official page" means a lab blog/project page. "abstract_only" means only the arXiv abs page was read. Numbers from automated extraction should be spot-checked against the PDF before they are quoted in the synthesis.

## 1. Search log

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

## 2. Inclusion notes

- Included: systems where (a) a large VLM/VLA and a smaller policy run at **different rates or on different compute**, or (b) the paper is a key reference for the **interface** between a reasoning level and a control level (language, latent token, waypoint/path, action candidates), or (c) the paper studies the cloud/edge split of a VLA directly.
- Single-rate unified VLAs (π0, π0.5, knowledge insulation) are included because their VLM backbone plus small action-expert split is the **intra-model** version of the slow/fast idea, and because π0 reports off-board WiFi inference latency.
- SayCan and Code-as-Policies are kept as short background entries only.
- Off-focus exclusions (verified, not included): DexVLA (2502.05855; plug-in 1B diffusion expert, no rate split described in abstract), TriVLA (2507.01424; triple system, ~36 Hz, world-model emphasis), LoHoVLA (2506.00411; unified planner/controller in Ravens sim, no rate split), StreamVLA (2602.01100; gating of reasoning in a single model, more RQ2), Fast ECoT (2506.07639; reasoning-reuse scheduler, RQ2), Agile-VLA (2603.22899; 10 Hz perception / 50 Hz control on Jetson Orin Nano, but primitives rather than a VLM latent, more RQ3), MiniVLA (only seen second-hand inside OpenHelix, not fetched).
- Not verified (dropped): (i) the claim that RT-2 ran at 1-3 Hz (55B) / ~5 Hz (5B) on a multi-TPU cloud service. This is widely repeated, but the PDF exceeded the fetch limit and the abstract does not state it. (ii) Any Helix successor or "Helix 02" architecture details; no official source was fetched.

## 3. Comparison table

"n.s." = not stated in the read scope. Rates are as reported by the source; "effective" rates usually come from action chunking.

| Paper | Year | Slow model: size & rate | Fast model: size & rate | Interface | Slow off-board? | Latency / staleness handling |
|---|---|---|---|---|---|---|
| Figure Helix | 2025 | 7B open-weight VLM; 7-9 Hz | 80M cross-attn enc-dec transformer; 200 Hz | Single continuous latent vector, projected into S1 token space | No: both on "dual low-power-consumption embedded GPUs" onboard | Temporal offset between S1/S2 inputs in training, calibrated to deployed latency gap; S2 runs as async background process |
| GR00T N1 | 2025 | Eagle-2 VLM 1.34B (12th LLM layer used); 10 Hz on L40 | DiT (2.2B total model); 120 Hz, H=16 chunk, 63.9 ms/chunk on L40 | VLM mid-layer embeddings via cross-attention | n.s. (inference on L40 GPU) | Jointly trained end-to-end; no explicit staleness method stated |
| GR00T N1.5 | 2025 | VLM 2.1B (Eagle 2.5 based), **frozen** | n.s. | VLM embeddings + FLARE future-latent alignment | n.s. | n.s. |
| π0 | 2024 | PaliGemma 3B | ~300M action expert (3.3B total); up to 50 Hz, H=50 | Shared transformer (VLM prefix -> action expert attention) | Mobile robots: **off-board over WiFi** (86 ms vs 73 ms onboard, RTX 4090) | Open-loop chunk execution (re-infer every 16 or 25 actions) |
| π0.5 | 2025 | Same model emits text subtask, at lower frequency than actions | Action expert ("significantly smaller"); 50 Hz with chunking (extraction) | Self-generated **language subtask** + shared weights | n.s. | n.s. |
| Knowledge insulation (π0.5+KI) | 2025 | PaliGemma 2B LM backbone | 300M action expert | Stop-gradient from action expert to VLM; FAST tokens + flow | n.s. | Faster than autoregressive π0-FAST (~750 ms per 1 s chunk, cited) |
| Hi Robot | 2025 | PaliGemma-3B high level; runs every 1 s or on user input | π0 (3.3B); ~10 Hz, 50 Hz with chunking | **Natural-language** atomic commands | Uses 1-2 RTX 4090; low level reported both onboard (73 ms) and off-board WiFi (86 ms) | Re-plans on a 1 s timer or user interrupt |
| RoboDual | 2024 | OpenVLA 7B (LoRA) | DiT 20M trainable; 15 Hz real-world | Discretized generalist actions + latent features (cross-attn KV) | n.s. (A5000 Ada; also tested on RTX 4060 laptop) | Shifted-window reuse of generalist actions + **latency-aware augmentation** τ∈[0,k] |
| HiRT | 2024 | InstructBLIP 7B; ~1.97 Hz in async mode | ~35M (sim, EfficientNet-B3+FiLM) / ~150M (real, ViT-B/16); ~9.8 Hz | MAP-pooled VLM latent (FiLM / cross-attn / prefix) | n.s. | Asynchronous VLM update interval (1 and 6 steps tested) |
| LCB | 2024 | LLaVA (size n.s.); updates every 40 env steps | LangTable policy / 3D Diffuser Actor (size n.s.) | Learnable **<ACT> token** last-layer embedding, linear projection | n.s. | Async: low-level runs on held latent |
| DP-VLA | 2024 | OpenVLA (called only on new instruction / big change) | BC-Transformer; 0.03 s/inference | 4096-d latent vector | n.s. | Event-triggered slow calls |
| OpenHelix | 2025 | LLaVA-7B, frozen, prompt-tuned | 3D Diffuser Actor | Learnable <ACT> token embedding, 4096 -> 512 projection | n.s. | MLLM every 10 or 60 steps: **performance roughly unchanged**; latent found insensitive to visual change |
| FiS-VLA | 2025 | LLaMA2-7B based VLA (System 2) | Last ~2 LLM blocks shared as System 1; 21.9 Hz (chunk 1), 117.7 Hz (chunk 8) | Shared parameters + System-2 latent conditioning | n.s. | Slow:fast ratio ablated 1:1 to 1:8; 1:4 best |
| Hume | 2025 | VLM-based System 2 with value-query head; 4 Hz | Lightweight (DINOv2-small + small transformer); 90 Hz control | **Action candidates** (best-of-5 chunks, H=30) via shared queue | n.s. | Async queue; S1 refines sub-chunks (h=15) |
| MinD | 2025 | LoDiff video generator (low freq.) | HiDiff diffusion policy; overall 11.3 FPS | Single-step low-res video latents (DiffMatcher) | n.s. | Uses partially denoised latents to avoid waiting |
| HAMSTER | 2025 | Fine-tuned VLM (size n.s. in abstract) | 3D-aware low-level policy | **2D end-effector path** drawn on image | n.s. | n.s. |
| RT-H | 2024 | Same VLA predicts "language motions" | Same model predicts actions | **Fine-grained language motions** | n.s. | n.s. |
| RT-2 | 2023 | Large VLM, actions as tokens | (single model) | Actions as text tokens | Cloud deployment widely reported, **not verified** here | n.s. |
| Gemini Robotics | 2025 | Gemini Robotics backbone, **cloud**, <160 ms query-response | Local action decoder on robot; 50 Hz effective, ~250 ms end-to-end | Backbone output -> local decoder (content n.s.) | **Yes** (cloud backbone) | Action chunking hides ~250 ms latency |
| Gemini Robotics On-Device | 2025 | (single on-device VLA, size n.s.) | — | — | No (runs "independent of a data network") | Slightly lower success than the cloud model |
| Gemini Robotics 1.5 / ER 1.5 | 2025 | ER 1.5 orchestrator | GR 1.5 VLA as a "tool" | **Open-vocabulary natural language** | n.s. | Success detection at 5 Hz |
| AsyncVLA | 2026 | OmniVLA 8.26B on RTX 4090 workstation; ≤5 Hz | Edge Adapter 76M on Jetson Orin (30 W); 8 Hz | **Projected action-token embeddings** (8×1024) over WiFi + timestamp | **Yes** (WiFi, delays up to 6 s tested) | Timestamped image buffer, random delayed-context sampling in training, re-weighting dynamic samples |
| VLA-ULAP | 2026 | Frozen cloud VLAs (GR00T N1.7 3B, Cosmos-Policy 2B, π0.5, others); 212 ms/call on A6000 | ULAP ~7.4M (incl. frozen Theia-Tiny) on Jetson Orin Nano; 20.7 ms, 0.122 J | **None** (local predictor sees only obs + action history; replaces whole chunks) | Yes (by design; network delay not modelled in main evals) | Accumulator scheduler alternates cloud/local chunks |
| DuoCore-FS | 2025 | PaliGemma-3B (also Qwen2.5-VL 3B/7B); 1-3 Hz | π0-small-style diffusion decoder; 25-30 Hz | **Latent buffer** of instruction + fusion-query embeddings | n.s. (RTX 4090) | Training with delayed buffer B_{t-Δ}, Δ~U[0,Δmax]; fast path reuses last buffer |
| Libra-VLA | 2026 | Semantic planner (discrete coarse action tokens) | Action refiner (continuous) | **Explicit coarse actions** (macro-direction) | n.s. | Asynchronous execution; granularity inverted-U |
| SayCan (bg) | 2022 | LLM (PaLM in paper) | Pre-trained skills + value functions | Skill selection (language) | n.s. | — |
| Code as Policies (bg) | 2022 | Code LLM | Perception/control APIs | Python code | n.s. | — |

## 4. Annotated entries

### 4.1 Figure AI (2025). Helix: A Vision-Language-Action Model for Generalist Humanoid Control
- **Citation:** Figure AI, 2025, "Helix: A Vision-Language-Action Model for Generalist Humanoid Control", company tech blog. https://www.figure.ai/news/helix
- **Relevance:** The closest industrial analogue to our system: a 7B VLM emitting one latent that conditions a small fast policy.
- **Key findings:** S2 is a 7B open-weight VLM at 7-9 Hz. S1 is an 80M cross-attention encoder-decoder transformer at 200 Hz. They communicate through a "single continuous latent vector" projected into S1's token space and concatenated with visual features. About 500 h of teleoperation data.
- **Method:** S2 runs as an async background process. In training, a temporal offset between S1 and S2 inputs is "calibrated to match the gap between S1 and S2's deployed inference latency". Both models run onboard on dual embedded GPUs.
- **Quality tier:** Tier 3 (blog/tech report).
- **Method weaknesses:** Reader-inferred: no quantitative ablations, success rates, or latent dimension given; no off-board setting; S1 size not ablated.
- **Relevance to our setup:** Direct template (latent + delay-matched training offset), but both levels are onboard, so it does not test network jitter.

### 4.2 Bjorck, Castañeda, Cherniadev, et al. (NVIDIA) (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- **Citation:** NVIDIA (J. Bjorck et al.; Linxi Fan and Yuke Zhu leads), 2025, arXiv:2503.14734. https://arxiv.org/abs/2503.14734
- **Relevance:** Open dual-system VLA with stated rates and sizes.
- **Key findings:** 2.2B parameters total, 1.34B in the VLM (Eagle-2: SmolLM2 + SigLIP-2), using 12th-layer LLM embeddings (§2.1). System 2 runs at 10 Hz and System 1 (DiT, flow matching, H=16, K=4 steps) at 120 Hz. A 16-action chunk takes 63.9 ms on an L40 (§2).
- **Method:** DiT cross-attends to VLM mid-layer tokens; jointly trained end-to-end on real, human-video and synthetic data; deployed on Fourier GR-1.
- **Quality tier:** Tier 2 (preprint / tech report).
- **Method weaknesses:** Reader-inferred: rates measured on a datacenter GPU (L40), not embedded; the two systems are jointly trained, so their "asynchrony" is mostly nominal; there is no stale-latent experiment.
- **Relevance to our setup:** Shows that a mid-layer VLM embedding works as the conditioning interface, and gives a size baseline (~0.9B action side).

### 4.3 NVIDIA GEAR (2025). GR00T N1.5
- **Citation:** NVIDIA GEAR Lab, 2025-06-11, "GR00T N1.5", research page. https://research.nvidia.com/labs/gear/gr00t-n1_5/
- **Relevance:** Shows the frozen-VLM variant of the dual system.
- **Key findings:** The VLM (2.1B, Eagle 2.5-based) is "frozen during both pretraining and finetuning". The adapter MLP is simplified. Adds FLARE (Future LAtent Representation Alignment).
- **Method:** Flow matching + FLARE alignment to future embeddings.
- **Quality tier:** Tier 3.
- **Method weaknesses:** Not assessed (read scope: official page summary). Inference rates not stated.
- **Relevance to our setup:** A frozen slow model means the slow model can be served unchanged off-board while only the fast side is trained.

### 4.4 Black, Brown, Driess, et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control
- **Citation:** K. Black et al. (Physical Intelligence), 2024, arXiv:2410.24164; RSS 2025. https://arxiv.org/abs/2410.24164
- **Relevance:** Baseline VLM + action expert design; one of the few papers with measured off-board WiFi latency.
- **Key findings:** PaliGemma 3B + ~300M action expert = 3.3B (§IV); H=50 chunks; up to 50 Hz control. On an RTX 4090: 73 ms onboard vs 86 ms off-board over WiFi (Table I, App. A-D). Mobile robots ran inference off-board.
- **Method:** Flow matching on an action expert that attends to the VLM prefix; open-loop chunk execution.
- **Quality tier:** Tier 1 (RSS 2025).
- **Method weaknesses:** Reader-inferred: the slow and fast parts run at the same rate (no true rate split); WiFi jitter is not characterised; chunks run open-loop between inferences.
- **Relevance to our setup:** Gives the latency budget of a ~3B off-board VLA over WiFi; 27B would be considerably slower.

### 4.5 Physical Intelligence, Black, et al. (2025). π0.5: a VLA Model with Open-World Generalization
- **Citation:** Physical Intelligence et al., 2025, arXiv:2504.16054. https://arxiv.org/abs/2504.16054
- **Relevance:** Hierarchical inference inside one model: text subtask (slow) then actions (fast).
- **Key findings:** Predicts a tokenized text subtask, and "the high-level inference process still runs at a lower frequency than low-level action inference". The action expert is "significantly smaller than the rest of the LLM backbone". FAST tokens in pre-training, flow matching in post-training.
- **Method:** Co-training on heterogeneous data (multi-robot, web, verbal instructions).
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged (Limitations section): mistakes on unfamiliar hardware interfaces, partial observability, high-level "distraction". Reader-inferred: rates and parameter counts are not reported precisely; off-board deployment is not stated.
- **Relevance to our setup:** Language as the slow-to-fast interface inside one model; it does not split compute.

### 4.6 Driess, Springenberg, Ichter, et al. (2025). Knowledge Insulating Vision-Language-Action Models
- **Citation:** D. Driess et al., 2025, arXiv:2505.23705. https://arxiv.org/abs/2505.23705
- **Relevance:** How to train a big backbone with a small fast action module without degrading the backbone.
- **Key findings:** Naively adding a flow-matching expert "significantly harms both training speed and knowledge transfer". Stop-gradient from the expert to the VLM, joint FAST-token + flow training, and VLM co-training fix this. Backbone is the PaliGemma 2B LM; expert is 300M (width 1024). Cites ≈750 ms per 1 s chunk for π0-FAST on an RTX 4090.
- **Method:** Stop-gradient insulation; ablations on language following.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged: ~20% training overhead; imperfect language following due to "correlations in the training data". Reader-inferred: no asynchronous or off-board evaluation.
- **Relevance to our setup:** Supports a frozen/insulated slow model, with the fast policy trained on its features.

### 4.7 Shi, Ichter, Equi, et al. (2025). Hi Robot: Open-Ended Instruction Following with Hierarchical VLA Models
- **Citation:** L. X. Shi et al., 2025, arXiv:2502.19417; ICML 2025. https://arxiv.org/abs/2502.19417
- **Relevance:** Canonical language-interface hierarchy with timing details.
- **Key findings:** The high level (PaliGemma-3B) runs "when one second has elapsed, or when a new interaction with the user takes place". The low level is π0. High-level prefill/decode takes 47 ms / 13.2 ms on an RTX 4090. Low level runs at ~10 Hz, 50 Hz with chunking. Inference uses 1-2 RTX 4090s.
- **Method:** Synthetic + labelled data for the high-level VLM; language commands to the low level.
- **Quality tier:** Tier 1 (ICML 2025).
- **Method weaknesses:** Author-acknowledged (App. C.4): no memory, so it struggles with long-context reasoning; low-level instruction-following failures (proximity bias). Reader-inferred: the language bottleneck limits bandwidth; no staleness study.
- **Relevance to our setup:** Language is a low-bandwidth, link-friendly interface, but it is the coarse end of the spectrum next to latent conditioning.

### 4.8 Bu, Li, Chen, et al. (2024). Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation (RoboDual)
- **Citation:** Q. Bu et al., 2024, arXiv:2410.08001. https://arxiv.org/abs/2410.08001
- **Relevance:** Small specialist (20M) conditioned on a 7B generalist, with explicit latency-aware training.
- **Key findings:** OpenVLA-7B generalist + DiT specialist with 20M trainable params (§3.3); 15 Hz real-world (§4.4), "3.8 times higher control frequency" than OpenVLA. Conditioning comes from discretized generalist actions plus latent features through cross-attention (§3.2). Runs on an RTX 4060 laptop GPU (App. C).
- **Method:** Shifted-window reuse of the generalist's action chunk. Training augmentation with offset τ∈[0,k_m] simulates deployment asynchrony (§3.2).
- **Quality tier:** Tier 2.
- **Method weaknesses:** Reader-inferred (from extraction of §4.5): no ablation of specialist size or generalist frequency; generalist and specialist share one machine.
- **Relevance to our setup:** Strong evidence that a ~20M fast policy is enough when conditioned on a big model's latents, plus a ready-made delay-augmentation recipe.

### 4.9 Zhang, Guo, Chen, et al. (2024). HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers
- **Citation:** J. Zhang et al., 2024, arXiv:2410.05273; CoRL 2024. https://arxiv.org/abs/2410.05273
- **Relevance:** Explicit low-rate VLM latent feeding a high-rate vision policy, with conditioning ablations.
- **Key findings:** InstructBLIP-7B (LoRA) produces MAP-pooled latents. The fast policy is ~35M (sim) or ~150M (real). The policy runs at ~9.8 Hz vs 4.1 Hz for a vanilla VLA, with the VLM at ~1.97 Hz. Dynamic-task success rises from 48% to 75%. Removing conditioning drops Franka-Kitchen success from 80.8% to 64.2% (Table 3).
- **Method:** FiLM, cross-attention and prefix conditioning are compared; asynchronous update intervals of 1 and 6 steps.
- **Quality tier:** Tier 1 (CoRL 2024).
- **Method weaknesses:** Reader-inferred: low absolute rates (~10 Hz); no off-board test; latent age is not given to the policy.
- **Relevance to our setup:** Direct evidence for latent conditioning plus a fast policy, and that the latent is worth about 16 pp of success.

### 4.10 Shentu, Wu, Rajeswaran, Abbeel (2024). From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control (LCB)
- **Citation:** Y. Shentu et al., 2024, arXiv:2405.04798 (v3 2025). https://arxiv.org/abs/2405.04798
- **Relevance:** Origin of the learnable <ACT> latent-token interface.
- **Key findings:** The LLaVA last-layer embedding at the <ACT> token is linearly projected to condition the low-level policy (§III-A). The LLM updates "every 40 environment steps" and runs asynchronously (§IV-A). Outperforms GPT-4V-based baselines on Language Table and CALVIN.
- **Method:** Learnable latent codes; avoids catastrophic forgetting compared with full fine-tuning.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged motivation: language cannot express all behaviours. Reader-inferred: sim only, sizes not stated, no latency study.
- **Relevance to our setup:** Template for a learned, compact latent sent over the link.

### 4.11 Han, Kim, Jang (2024). A Dual Process VLA: Efficient Robotic Manipulation Leveraging VLM (DP-VLA)
- **Citation:** B. Han, J. Kim, J. Jang (ETRI), 2024, arXiv:2410.15549. https://arxiv.org/abs/2410.15549
- **Relevance:** Event-triggered large model + small BC policy.
- **Key findings:** OpenVLA as L-Sys2 emits a 4096-d latent; a BC-Transformer is S-Sys1. Inference takes 0.03 s vs 0.253 s for fine-tuned OpenVLA (Table 2). Success on RoboCasa is 57.3% vs 9.8% (OpenVLA-ft) and 47.6% (BC-T).
- **Method:** L-Sys2 is invoked only when instructions change or the scene changes a lot.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged (§5): dynamic scheduling and multi-level decomposition are left to future work. Reader-inferred: sim only; parameter counts not given; the gain over BC alone is modest.
- **Relevance to our setup:** Shows that very infrequent latent updates can still help, and supports event-triggered transmission to save link bandwidth.

### 4.12 Chen, Liu, Gu, et al. (2025). Fast-in-Slow: A Dual-System VLA Model Unifying Fast Manipulation within Slow Reasoning (FiS-VLA)
- **Citation:** H. Chen et al., 2025, arXiv:2506.01953; NeurIPS 2025 (poster, per neurips.cc listing). https://arxiv.org/abs/2506.01953
- **Relevance:** One of the few papers that ablates the slow:fast rate ratio and the fast-module size.
- **Key findings:** System 2 is LLaMA2-7B based; System 1 reuses the final LLM blocks, with performance saturating at 2 shared blocks (§4.2, Fig. 3). Runs at 21.9 Hz with chunk 1 and 117.7 Hz with chunk 8 (Table 1). Slow:fast ratio ablated from 1:1 to 1:8, with 1:4 best. System 1 also takes 3D point clouds and proprioception.
- **Method:** Dual-aware co-training (diffusion + next-token losses).
- **Quality tier:** Tier 1 (NeurIPS 2025; venue seen on neurips.cc search result).
- **Method weaknesses:** Reader-inferred: System 1 shares weights with the 7B model, so it cannot be split across devices; Hz measured on a desktop GPU.
- **Relevance to our setup:** Shows that rate ratio matters and that an intermediate 1:4 beats both extremes. Our ~5 Hz : ≥20 Hz is in the same range.

### 4.13 Cui, Ding, Song, et al. (2025). OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model
- **Citation:** C. Cui et al., 2025, arXiv:2505.03912. https://arxiv.org/abs/2505.03912
- **Relevance:** Systematic empirical comparison of dual-system design choices, including asynchrony.
- **Key findings:** Frozen, prompt-tuned LLaVA-7B <ACT> token (4096 -> 512) + 3D Diffuser Actor. Calling the MLLM every 10 or every 60 steps gives "quite similar" performance (Table 4, Fig. 4). The latent "primarily encodes the target object, spatial relations, and action semantics from the instruction" and is "largely insensitive to changes in visual information" (§2.4). Projector pre-alignment is essential (Table 6). CALVIN ABC-D 5-task success is 46.0% (Table 8).
- **Method:** Survey + controlled ablations on CALVIN.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged (§2.4): the MLLM latent ignores visual change, which is "counterintuitive". Reader-inferred: CALVIN only; that staleness is harmless may follow from the latent carrying little visual information.
- **Relevance to our setup:** Key caution: stale latents may look harmless only because the latent carries instruction-level information. We should measure the information content of the latent.

### 4.14 Song, Qu, et al. (2025). Hume: Introducing System-2 Thinking in Visual-Language-Action Model
- **Citation:** H. Song et al. (12 authors incl. X. Li), 2025, arXiv:2505.21432. https://arxiv.org/abs/2505.21432
- **Relevance:** Async dual system where the interface is a set of action candidates.
- **Key findings:** System 2 runs at 4 Hz with a value-query head and best-of-N=5 chunk selection (H=30). System 1 (DINOv2-small + light transformer) does cascaded denoising, with 90 Hz robot control. The two communicate through a shared queue.
- **Method:** Value-guided candidate selection; asynchronous queue.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Reader-inferred: backbone size not stated in the extraction; compute location not stated.
- **Relevance to our setup:** Sending action priors instead of latents is an alternative interface, and degrades more gracefully if the link drops.

### 4.15 Chi, Ge, Liu, et al. (2025). MinD: Learning A Dual-System World Model for Real-Time Planning and Implicit Risk Analysis
- **Citation:** X. Chi et al., 2025, arXiv:2506.18897. https://arxiv.org/abs/2506.18897
- **Relevance:** Slow video-generation level conditioning a fast diffusion policy through single-step latents.
- **Key findings:** Low-frequency LoDiff video generator + high-frequency HiDiff policy; uses "low-resolution latents generated in a single denoising step"; 11.3 FPS; 63% RLBench, 60% real Franka; flags 74% of failures in advance.
- **Method:** DiffMatcher co-training.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Partially computed slow-model latents may be enough, which suggests streaming early or partial latents.

### 4.16 Li, Deng, Zhang, et al. (2025). HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation
- **Citation:** Y. Li et al. (NVIDIA/UW), 2025, arXiv:2502.05485. https://arxiv.org/abs/2502.05485
- **Relevance:** Waypoint/path interface.
- **Key findings:** The high-level VLM outputs a "coarse 2D path" of the end effector, which conditions a 3D-aware low-level policy. Average +20% success over OpenVLA across seven generalization axes.
- **Method:** The VLM is fine-tuned on off-domain data (videos, sketches, sim).
- **Quality tier:** Tier 2 (venue not stated on abs page).
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** A compact, interpretable geometric interface, attractive for drones (2D/3D path) and bandwidth-light.

### 4.17 Belkhale, Ding, Xiao, et al. (2024). RT-H: Action Hierarchies Using Language
- **Citation:** S. Belkhale et al., 2024, arXiv:2403.01823. https://arxiv.org/abs/2403.01823
- **Relevance:** Fine-grained language ("move arm forward") as an intermediate layer.
- **Key findings:** Language motions enable human correction and outperform teleoperated interventions.
- **Method:** A single VLA predicts the language motion, then the action.
- **Quality tier:** Tier 2 (as verified; venue not on abs page).
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Mid-granularity language interface; not rate-split.

### 4.18 Brohan, Brown, Carbajal, et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **Citation:** A. Brohan et al., 2023, arXiv:2307.15818. https://arxiv.org/abs/2307.15818
- **Relevance:** Origin of VLAs (actions as tokens). The usual motivation for why large VLAs are slow.
- **Key findings:** Co-fine-tuning VLMs on robot data and web VQA; emergent generalization and chain-of-thought.
- **Method:** Action tokenization.
- **Quality tier:** Tier 2 (preprint; often cited as CoRL 2023, not verified here).
- **Method weaknesses:** Not assessed (read scope: abstract_only). The multi-TPU cloud deployment and Hz figures were **not verified**.
- **Relevance to our setup:** Background and motivation only.

### 4.19 Gemini Robotics Team (2025). Gemini Robotics: Bringing AI into the Physical World
- **Citation:** Gemini Robotics Team, Google DeepMind, 2025, arXiv:2503.20020. https://arxiv.org/abs/2503.20020
- **Relevance:** **Cloud backbone + onboard decoder**, the topology closest to ours.
- **Key findings:** "VLA backbone hosted in the cloud ... and a local action decoder running on the robot's onboard computer". Backbone query-response is "under 160ms". End-to-end latency is "approximately 250ms" and the effective control rate 50 Hz via chunks (§3.1).
- **Method:** Gemini 2.0-based backbone; chunked actions.
- **Quality tier:** Tier 2 (tech report).
- **Method weaknesses:** Reader-inferred: what is sent from backbone to decoder is not specified in the extracted section; decoder size not given; no jitter/dropout analysis.
- **Relevance to our setup:** Existence proof that cloud slow + local fast works at ~250 ms, but the interface is opaque.

### 4.20 Parada / Google DeepMind (2025). Gemini Robotics On-Device brings AI to local robotic devices
- **Citation:** C. Parada, Google DeepMind blog, 2025-06-24. https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/
- **Relevance:** Counterpoint: fully on-device VLA with no network.
- **Key findings:** "engineered to require minimal computational resources"; runs "independent of a data network"; slightly lower generalization scores than cloud Gemini Robotics.
- **Method:** Not described.
- **Quality tier:** Tier 3.
- **Method weaknesses:** Reader-inferred: no size, Hz, or hardware disclosed.
- **Relevance to our setup:** Baseline to beat: a split system must outperform a compressed all-onboard model.

### 4.21 Gemini Robotics Team (2025). Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots ...
- **Citation:** Gemini Robotics Team, 2025, arXiv:2510.03342. https://arxiv.org/abs/2510.03342
- **Relevance:** Three levels: ER orchestrator -> VLA ("tool") via natural language.
- **Key findings:** "The action model ... receives instructions via open-vocabulary natural language" (§2.1); success detection at 5 Hz.
- **Method:** Thinking VLA + Motion Transfer.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Reader-inferred: deployment topology and latencies not stated.
- **Relevance to our setup:** Suggests a three-level design (planner in language, VLA, fast controller); our slow model may sit at the middle level.

### 4.22 Hirose, Glossop, Shah, Levine (2026). AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge
- **Citation:** N. Hirose et al., 2026, arXiv:2602.13476. https://arxiv.org/abs/2602.13476
- **Relevance:** **Most direct match**: remote 8B VLA + onboard 76M adapter over WiFi on a mobile robot.
- **Key findings:** OmniVLA 8.26B on an RTX 4090 workstation at ≤5 Hz. Edge Adapter 76M ViT on Jetson Orin (30 W) at 8 Hz (Table I). Interface is 8×1024 projected action-token embeddings plus current-observation tokens (§III-A). Tolerates delays up to 6 s. Success is 40% higher than baselines.
- **Method:** Timestamp-matched image buffer (Alg. 1); training samples delayed VLA observations from a long context; re-weights samples with large chunk discrepancies; two-stage then end-to-end fine-tuning (§III-B/C).
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged (§VII): needs open-weight VLAs that can be fine-tuned end to end; few dynamic interactions in the data. Reader-inferred: navigation only (low-dimensional actions); edge rate only 8 Hz.
- **Relevance to our setup:** Nearly our architecture. Its delay-sampling recipe should be the first baseline.

### 4.23 Cao, Oi, Matsushima, et al. (2026). VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge
- **Citation:** D. Cao et al., 2026, arXiv:2609.18663. https://arxiv.org/abs/2609.18663
- **Relevance:** Cloud/edge VLA split with energy measurements; a lower bound on fast-model size.
- **Key findings:** ULAP has ~7.4M params including a frozen Theia-Tiny encoder. On a Jetson Orin Nano: 20.7 ms and 0.122 J per inference, vs 212.2 ms and 27.3 J for GR00T on an A6000 (App. B.2). It removes 45.3-77.3% of VLA calls while keeping 95.0-98.5% of success (sim) and 70-72% of calls with no observed loss on SO-101.
- **Method:** Accumulator scheduler interleaves cloud chunks with local chunks. The local model gets no VLA latent, only observations and action history.
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged (§5): bounded tabletop pick-and-place only; not a safety controller; no online verification. Reader-inferred: network latency not modelled in the main evaluation; fixed schedule rather than adaptive.
- **Relevance to our setup:** Baseline without a latent. Our latent-conditioned fast model should beat it at equal energy.

### 4.24 Zou, Zeng, Nong, et al. (2025). Asynchronous Fast-Slow VLA Policies for Whole-Body Robotic Manipulation (DuoCore-FS)
- **Citation:** T. Zou et al., 2025, arXiv:2512.20188. https://arxiv.org/abs/2512.20188
- **Relevance:** "Truly asynchronous" latent buffer with explicit delay training.
- **Key findings:** PaliGemma-3B slow path at 1-3 Hz; π0-small-style diffusion fast path at 25-30 Hz. 32.3 Hz vs 12.5 Hz for π0, with 90% vs 85% success (Table 1). An RVQ-VAE action tokenizer beats FAST (83.3% vs 0%, Table 5).
- **Method:** The buffer stores instruction + fusion-query embeddings. Training uses B_{t-Δ} with Δ~U[0,Δmax] (§3.3). At inference the fast path reuses the last buffer (§3.4).
- **Quality tier:** Tier 2.
- **Method weaknesses:** Author-acknowledged (§5): broader real-world evaluation needed; flow steps could be reduced. Reader-inferred: single 4090 (no network); fast-path size not stated.
- **Relevance to our setup:** Directly reusable recipe for uniform delay sampling with a latent buffer.

### 4.25 Wei, Zhong, Liu, et al. (2026). Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System
- **Citation:** Y. Wei et al., 2026, arXiv:2604.24921; accepted ACL 2026 (per arXiv comment). https://arxiv.org/abs/2604.24921
- **Relevance:** Argues for explicit coarse actions instead of implicit latents.
- **Key findings:** Semantic Planner -> discrete coarse direction tokens -> continuous Action Refiner. "performance follows an inverted-U curve relative to action decomposition granularity, peaking exactly when the learning difficulty is balanced".
- **Method:** Asynchronous coarse-to-fine execution.
- **Quality tier:** Tier 1 (ACL 2026, per arXiv comment).
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Suggests splitting the learning burden between the levels, which is a design axis for how much the fast model must do.

### 4.26 Ahn, Brohan, Brown, et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan), background
- **Citation:** M. Ahn et al., 2022, arXiv:2204.01691. https://arxiv.org/abs/2204.01691
- **Relevance / key finding:** An LLM picks skills, weighted by affordance value functions. The early language-interface hierarchy.
- **Quality tier:** Tier 2 as verified (commonly cited as CoRL 2022; not verified on the abs page).
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Historical: a discrete skill interface, the coarsest option.

### 4.27 Liang, Huang, Xia, et al. (2022). Code as Policies: Language Model Programs for Embodied Control, background
- **Citation:** J. Liang et al., 2022, arXiv:2209.07753. https://arxiv.org/abs/2209.07753
- **Relevance / key finding:** An LLM writes Python policy code calling perception and control APIs.
- **Quality tier:** Tier 2 as verified (commonly cited as ICRA 2023; not verified on the abs page).
- **Method weaknesses:** Not assessed (read scope: abstract_only).
- **Relevance to our setup:** Historical: code as the interface, so the slow model runs once per task.

## 5. Theme-level gaps

1. **Almost no work at our size gap and topology.** Slow models in the literature are 1-8B. Only AsyncVLA (8B remote, WiFi), Gemini Robotics (cloud, size undisclosed) and VLA-ULAP (cloud, no latent) actually put the slow model off-board. None use a ~27B slow model over a jittery link.
2. **Staleness is handled only by delay augmentation** (Helix offset, RoboDual τ-window, DuoCore-FS Δ~U, AsyncVLA delayed-context sampling). No paper gives the latent's age to the fast policy as an explicit input, models jitter distributions rather than fixed delays, or reports success versus delay curves for latent interfaces (AsyncVLA comes closest, up to 6 s).
3. **Few ablations of fast-model size.** FiS-VLA ablates shared blocks and the rate ratio. RoboDual (20M), HiRT (35M/150M), Helix (80M), AsyncVLA (76M) and ULAP (7.4M) each report a single point, so no scaling curve of fast-policy size versus latent quality exists.
4. **Do the latents carry visual information?** OpenHelix finds the MLLM latent is nearly invariant to visual change. It is unclear whether the slow latent in other systems carries scene state or only task semantics. This matters for how often we must transmit.
5. **Multi-rate sensor fusion in the fast model is barely addressed.** FiS adds point clouds and proprioception; RoboDual adds depth and tactile. None fuse IMU at ~100 Hz or torque at 60 Hz. No drone or other highly dynamic platform uses a latent-conditioned dual system (search 8 found none).
6. **Energy and weight are rarely reported.** VLA-ULAP (Jetson Orin Nano energy per inference) and AsyncVLA (Orin at 30 W) are the exceptions.

## 6. The 3 most important papers for our question

1. **AsyncVLA (Hirose et al., 2026, arXiv:2602.13476)**: remote 8B VLA, 76M onboard adapter on Jetson, embedding interface over WiFi, delay-robust training. Nearly our exact architecture.
2. **Figure Helix (2025, blog)**: 7B to 80M through a single latent, 7-9 Hz to 200 Hz, with a training offset matched to the deployed latency. The clearest industrial template for latent conditioning plus staleness training.
3. **RoboDual (Bu et al., 2024, arXiv:2410.08001)** (with DuoCore-FS as a close runner-up): a 20M specialist conditioned on 7B generalist latents and actions, with latency-aware augmentation. The best evidence on how small the fast model can be.
