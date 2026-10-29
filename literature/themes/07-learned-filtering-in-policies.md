# Theme 7 (RQ2, RQ4): Learned filtering and Kalman-inspired mechanisms in learned policies

- **Questions:** which Kalman-filter and control mechanisms can make a tiny fast policy use a slow, delayed latent z well? Sub-questions: (a) learned / differentiable filters with a predict/update structure; (b) delayed and out-of-sequence measurements; (c) innovation / prediction-error gating, forward models; (d) feedforward plan + learned feedback under plan staleness; (e) event-triggered querying of a slow model.
- **Why now:** the A1-teacher results (EXP-0001) show a 100k student reaching its 10M teacher's success when the teacher's plan is fresh (k ≤ 4), and collapsing once the plan is older than its 8-step chunk. This theme backs the draft amendment `experiments/EXP-0001-latent-taskid-vs-state/DRAFT-amendment-A1-teacher-v2.md`.
- **Search date:** 2026-10-29. Three AI search agents, one per slice (a+b, c+e, d); merged and re-graded by the main session.
- **Verification:** every included arXiv ID passes `scripts/verify_bib.py`; every DOI-only entry was checked against Crossref and the two proceedings-URL entries against their proceedings pages (2026-10-29).
- **Read-scope labels:** as in Theme 4. Full texts were read through automated extraction (pdftotext / arXiv HTML); table columns were sometimes scrambled, so **re-check every number against the PDF before quoting it** (CLAUDE.md gotcha).
- **Tiers:** the Theme 4 rule. Tier 1 = peer-reviewed with real-robot or standard-benchmark evidence. Tier 2 = preprint, or peer-reviewed with narrow evidence. Tier 3 = methodological or classical work not tested on robots (classical estimation and control references are included as the seminal definitions of a mechanism, per the protocol's baseline-reference exception).

## 1. Mechanisms, what exists, and what is missing

The rows follow the mechanisms in the draft amendment (L, A, R, P, G, E). Citations use the entry labels of Section 3.

| Mechanism (draft ID) | Kalman / control anchor | What exists in learned systems | Closest precedents | Missing (our opening) |
|---|---|---|---|---|
| Learned predict/update cell as the fast core | Kalman filter | Learned gain [KalmanNet]; learned measurement + covariance [BKF], [Kloss21]; factorized latent KF cells with uncertainty-gated updates that skip missing observations [RKN], [acRKN]; Δt-aware continuous-time prediction [CRU]; RSSM world models [PlaNet]. Small enough for us: BKF 7,493 parameters ([BKF] Tab. 1), KalmanNet architecture #2 about 2.5·10⁴ ([KalmanNet] Sec. III-C). | [MTS3]: a slow latent conditions a fast Kalman belief, and the slow uncertainty becomes fast process noise (Eq. 6) | Robot evidence is almost all offline estimation; no Kalman-structured *policy* closes the loop under a delayed remote latent |
| L: plan that outlives the delay; A: index the plan by its age | Prediction over the delay; Smith predictor | Plan indexed by time since retrieval, trained with variable update periods, success vs planner rate 1–50 Hz [jenelten2024dtc] (Fig. 7A); continuous-time chunks evaluable at any time [zhou2025beast], DCT chunks [pertsch2025fast] | [jenelten2024dtc] (model-based planner, legged) | No learned-planner chunk evaluated at a shifted time to absorb latency |
| R: residual feedback around the plan | Feedforward + feedback; tube MPC | a = a_base + a_res with zero-initialised, bounded correction [silver2018residual], [johannink2019residual]; on chunked / VLA bases [yuan2025policy], [ankile2025imitation], [ankile2026residual], [xiao2026selfimproving]; per-step beats per-chunk correction ([ankile2025imitation] Sec. IV-B3); learned nominal plan + ancillary feedback with off-nominal augmentation [tagliabue2024efficient]; tube widening with horizon [koller2018learning] | [deng2026residual]: LSTM predictor + residual under 40–290 ms stochastic delay (teleoperation, human leader) | In every residual paper the base action is fresh; nobody trains the residual while a learned plan ages |
| P: rewind-and-replay of late z | Out-of-sequence measurement update: reprocessing, retrodiction [BarShalom02], [BarShalom04]; extrapolate the measurement to now [Larsen98] | Learned delay handling conditions on the stale observation + actions since [Wang24], [Karamzade24], rolls a model forward [Walsh08], [Karamzade24], or feeds the delay/age [TSRL], [LAA] | [LAA] beats a re-filtering UKF with a small learned estimator + age input | No learned re-filtering of a policy's recurrent state; learned work excludes out-of-order arrival and jitter by design; **caution:** explicit state prediction helps under fixed delay but hurts under random delay ([Wang24] Sec. 5) |
| G: trust in z set by the innovation | χ² test on normalized innovations, with false-alarm rate and window trade-off ([Willsky1976] Sec. VII); internal forward models / efference copy [Wolpert1995] | Learned failure monitors from the policy's own outputs [Sentinel], [FAIL-Detect], [TUD], [AAC]; reconstruction-based anomaly detection from robot signals [LSTM-VAE], [Grambow-AD]; wrench-prediction "surprise gate" for online learning [IMPACT]; slow model shipping an expected wrench, residual used for compliance [CW-WAM] | [CW-WAM] + [IMPACT] together | Nobody forms the residual between the force/torque a plan implies and what is measured, and uses it to down-weight a stale plan; [FAIL-Detect] lists missing force inputs as a limitation (Sec. VII) |
| E: re-query the slow model on an innovation spike | Event-triggered control ([Tabuada2007], min. inter-event time Thm. 3.1); innovation-thresholded transmission with error bound ([Trimpe-EBSE] Eq. 13, 19); event-triggered learning [ETL]; learned triggers with a communication penalty [DRL-ETC] | Learned "when to replan / call the big model": [WhenToReplan] (Table I), [TUD] (75% fewer generalist calls at higher success, Table 5), [AAC]; uncertainty-triggered help [KnowNo] | [TUD] | No trigger uses fast-sensor innovation; none models network latency between trigger and response |

## 2. Synthesis

- **The pieces of the draft amendment each have precedents; the combination has none.** Age-indexed plans (DTC), zero-initialised residuals on large-model bases (ResiP, Policy Decorator), Kalman-structured learned cells (RKN, MTS3), innovation tests (Willsky) and innovation-triggered communication (Trimpe) all exist. A tiny policy that tracks an ageing plan from a learned slow model with a residual, re-filters when the plan arrives late, and trusts the plan by the force/torque innovation does not appear in any included source. This is the gap EXP-0001 A1-teacher v2 targets.
- **The author's innovation idea is the clearest open slot.** Every learned trust signal found is computed from the policy's own outputs (chunk consistency, action density or entropy, prediction dispersion). The plan-implied-wrench residual is split across two 2026 preprints that use it for compliance [CW-WAM] or to gate learning [IMPACT], not to gate trust in a slow model. The classical χ² gate gives a calibrated threshold and a known trade-off between window length and detection delay [Willsky1976]; conformal thresholds from successful rollouts only [Sentinel], [FAIL-Detect] give a learned-system recipe for setting it.
- **Replay needs a guard.** The only direct evidence on predicting forward under *random* delay is negative ([Wang24] Sec. 5). Rewind-and-replay is different (it re-runs the filter on recorded inputs rather than predicting them), but the experiment should compare it against the delay-trained student (V2) under jittered delay, not only fixed delay.
- **Event-triggered re-query can lose at low rates.** Innovation triggering can be worse than periodic updates at very low communication rates ([Trimpe-EBSE] Fig. 7), and better models can trigger more communication under partial observability ([ETL] Sec. 7.2). Keep a max-interval fallback (as [TUD] does) and compare against fixed-rate refresh.
- **A plan alone is not enough; feedback closes the loop.** Plan-only students fail without their own sensing ([kaufmann2020deep] Tab. I, "Only Ref"; [gu2024rttrajectory]). This matches our A1 observation that the small student fails in closed loop while matching the large model's validation error.

## 3. Implications for the draft amendment (A1-teacher v2)

1. **Keep L, A, R as planned;** consider a B-spline plan [zhou2025beast] instead of a 32-step discrete chunk: one teacher serves any k, and age-indexing becomes evaluation at a shifted time. Cost: a new teacher head; decision for the author.
2. **R: bound the residual** ([yuan2025policy] Fig. 9) and apply it per step ([ankile2025imitation] Sec. IV-B3).
3. **P: test under jittered delay**, not only fixed (Section 2, [Wang24]).
4. **G-inn: set the threshold conformally** on successful validation rollouts ([Sentinel] App. D; [FAIL-Detect] Sec. IV-B) rather than a fixed 99th percentile, and report the false-alarm rate on unperturbed episodes ([Willsky1976] Sec. VII trade-off).
5. **E: add a max-interval fallback** and a communication cost, and compare against fixed-rate refresh at equal query count ([Trimpe-EBSE] Fig. 7; [TUD]).
6. **Optional core variant:** an RKN-style cell [RKN], [acRKN] with z as an observation and age-inflated process noise [MTS3] (Eq. 6) instead of a GRU. Not in the draft yet.

## 4. Second pass (verified but not read, or found late)

Heemels et al. 2012 (event-triggered overview), Trimpe & D'Andrea 2014, Haddadin et al. 2017 (momentum-observer collision detection, the classical torque innovation), Mehra & Peschon 1971, the Bar-Shalom estimation textbook (NIS / validation gate), FIPER (NeurIPS 2025), Liotet et al. 2022 "Delayed RL by Imitation" (arXiv:2205.05569, close to our teacher→student setup), and delay-aware model-based RL (Chen et al. 2020). Also search: learned slip/drop detection from wrist F/T (not tactile), and delay-aware event-triggered estimation.

## 5. Search log

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

## 6. Annotated entries

### (a) Learned / differentiable filters with explicit predict/update structure

**[KalmanNet] Revach, G., Shlezinger, N., Ni, X., Lopez Escoriza, A., van Sloun, R. J. G., Eldar, Y. C. (2022).** *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics.* IEEE Transactions on Signal Processing 70. arXiv:2107.10043. https://arxiv.org/abs/2107.10043
- **Key findings:**
  - What it keeps and learns: it keeps the EKF predict step with known (possibly mismatched) f, h, and the update x̂ = x̂⁻ + K·Δy (Eqs. 5a, 8). Only the Kalman gain is learned, by a GRU fed difference features (innovation, observation difference, state-update difference) (Sec. III-A/B, Figs. 2–4). Q, R and Σ are never estimated explicitly; noise statistics live implicitly in the GRU state (Sec. III-A).
  - Linear model: it matches the optimal KF (−21.90 vs −21.91 dB, Tab. I).
  - Lorenz with sampling mismatch: −11.28 dB vs EKF −6.43 dB, at lower run-time than the PF (Tab. IX).
  - Size: architecture #2 (3 GRUs mirroring the KF moments) needs about 2.5·10⁴ parameters vs about 5·10⁵ for architecture #1 (Sec. III-C).
- **Quality tier:** Tier 2 (IEEE TSP, venue via Crossref DOI 10.1109/TSP.2022.3158588). Real-data evidence is limited to the NCLT odometry dataset (Tab. X).
- **Method weaknesses:**
  - It cannot export an explicit state-space model or covariance (author-acknowledged, Sec. III-E).
  - Full BPTT may be costly and unstable (author-acknowledged, Sec. III-D).
  - It has no mechanism for delayed, missing or out-of-sequence measurements, and assumes f and h are known (reader-inferred).
- **Relevance to our setup:** the closest template for an "innovation-driven trust in z". Our gain network would take Δy = (observed torque/F-T − torque/F-T expected given z) and output how much z moves the fast policy's state. Architecture #2's 2.5·10⁴ parameters already fit our 10k–100k budget.
- Read scope: full_text.

**[BKF] Haarnoja, T., Ajay, A., Levine, S., Abbeel, P. (2016).** *Backprop KF: Learning Discriminative Deterministic State Estimators.* NeurIPS 2016. arXiv:1605.07148. https://arxiv.org/abs/1605.07148
- **Key findings:**
  - A CNN maps each raw observation to a low-dimensional pseudo-measurement z_t and a per-step covariance R_t (Cholesky). The KF predict/update stays fixed, and everything is trained end-to-end by backprop through the filter (Secs. 4, 5.1, Fig. 3).
  - Disc tracking: RMS 0.0537 with 7,493 parameters, vs 0.1407 for LSTM-64 (33,506 parameters) and 0.1160 for a piecewise KF with constant R (Tab. 1).
  - With constant R the model learns a very large R and relies on dynamics. Input-conditioned R_t balances the two, for example under occlusion (Sec. 5.3, Fig. 5).
- **Quality tier:** Tier 1 (NeurIPS 2016, venue via papers.nips.cc index; KITTI odometry, Tab. 2).
- **Method weaknesses:**
  - A large LSTM can match BKF accuracy, so the gain is mainly in data and parameter efficiency (author-acknowledged, Sec. 6).
  - Linear(ised) dynamics only; no delayed measurements (reader-inferred).
- **Relevance to our setup:** the direct analogue of our measurement head. The slow model's z becomes a pseudo-measurement with a learned, input-dependent R(z, age). This is a cheap way to make the fast policy discount stale or uncertain latents, and the 7.5k-parameter scale matches our budget.
- Read scope: full_text.

**[DPF] Jonschkowski, R., Rastogi, D., Brock, O. (2018).** *Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors.* RSS 2018. arXiv:1805.11122. https://arxiv.org/abs/1805.11122
- **Key findings:**
  - It keeps the PF predict/weight/resample structure. Learned parts: an action-noise sampler, an optional dynamics model, an observation encoder, a particle proposer and a likelihood estimator (Sec. IV, Eqs. 5–9, Fig. 3).
  - Under train/test policy swap the LSTM error rate exceeds 80%, while DPF stays at 5–26% (Sec. V-A4, Fig. 9).
  - It reaches LSTM-level accuracy with 1/8 of the data (Sec. V-A3).
  - KITTI translation error 0.1467 vs BKF 0.2062 (Tab. II).
- **Quality tier:** Tier 1 (RSS 2018, venue via roboticsproceedings.org index; simulated mazes + KITTI).
- **Method weaknesses:**
  - Resampling is not differentiable, so gradients stop after one step, which limits training to supervised learning with state labels (author-acknowledged, Sec. IV-F).
  - A particle set is heavy for a 10k–100k-parameter, 20 Hz MCU policy (reader-inferred).
- **Relevance to our setup:** evidence that algorithmic priors (predict/update) help most under distribution shift in the policy that generates data. That matters because our fast policy is trained on teacher rollouts but deployed closed-loop. A particle belief is likely too costly on-robot.
- Read scope: full_text.

**[Kloss21] Kloss, A., Martius, G., Bohg, J. (2021).** *How to Train Your Differentiable Filter.* Autonomous Robots 45(4). arXiv:2012.14313. https://arxiv.org/abs/2012.14313
- **Key findings:**
  - Systematic comparison of differentiable EKF, UKF, MCUKF and PF with learned sensor, process and noise models (Sec. IV).
  - Heteroscedastic R roughly halves dEKF RMSE on disc tracking (16.2 → 8.8, Tab. I). Learned σ_r rises from 0.9 (visible) to 29.3 (occluded) (Sec. VI-E).
  - Differentiable filters match or beat LSTMs with fewer weights; an LSTM needs 2×512 units to compete (Tab. III, Sec. IX).
  - Returns from longer training sequences diminish after about 10 steps (Sec. VI-D, Fig. 3).
- **Quality tier:** Tier 1 (Autonomous Robots, venue via Crossref DOI 10.1007/s10514-021-09990-9; KITTI + real planar-pushing data).
- **Method weaknesses:** all author-acknowledged in Sec. IX:
  - training stability and many hyper-parameters;
  - all filters fail on an outlier KITTI trajectory, so OOD detection is needed;
  - end-to-end training helps only the noise models, not the sensor/process models.
  - Separately, Sec. VII-B notes that heteroscedastic R can overfit on KITTI.
- **Relevance to our setup:** a practical recipe. Start with a dEKF-like cell, learn heteroscedastic R(z, age) and Q(torque, F/T), pretrain the sub-models, and train with about 10-step sequences. It also warns that OOD inputs (e.g. a wrong z) need an explicit innovation check, which supports our innovation-gated trust.
- Read scope: full_text.

**[RKN] Becker, P., Pandya, H., Gebhardt, G., Zhao, C., Taylor, J., Neumann, G. (2019).** *Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces.* ICML 2019 (PMLR 97). arXiv:1905.07357. https://arxiv.org/abs/1905.07357
- **Key findings:**
  - An encoder emits a latent observation and a per-dimension variance σ_obs.
  - Locally linear transitions are mixed from banded basis matrices.
  - A factorized covariance makes the Kalman gain elementwise, with no matrix inversion (Secs. 2.1–2.4). The gain acts as an uncertainty-driven input gate (Sec. 2.6).
  - Missing inputs are handled by skipping the update (posterior = prior) (Sec. 2.6).
  - Real hydraulic arm, 2 s-ahead prediction: log-likelihood 4.93 vs LSTM 0.95 and GRU 1.19 (Sec. 3.5).
  - Parameter count grows linearly in state size (Sec. 2.6).
- **Quality tier:** Tier 1 (ICML 2019, venue via PMLR v97 index; real hydraulic arm data, pendulum, KITTI).
- **Method weaknesses:**
  - The factorization "might be a crude approximation" (author-acknowledged, Sec. 2.3).
  - Prediction/filtering only, no control (author-acknowledged as future work, Sec. 4).
- **Relevance to our setup:** probably the best cell for our fast policy's belief. It is a GRU-sized recurrent cell with an explicit predict (driven by torque/F-T/proprio) and update (driven by z with learned σ_obs). It degrades gracefully to "predict only" when z has not arrived yet.
- Read scope: full_text.

**[acRKN] Shaj, V., Becker, P., Buchler, D., Pandya, H., van Duijkeren, N., Taylor, C. J., Hanheide, M., Neumann, G. (2020).** *Action-Conditional Recurrent Kalman Networks For Forward and Inverse Dynamics Learning.* CoRL 2020 (PMLR 155). arXiv:2010.10201. https://arxiv.org/abs/2010.10201
- **Key findings:**
  - Adds the action to the RKN predict step (z⁻ = A z⁺ + b(a), with b a nonlinear MLP being best). Actions do not change the covariance or the gain (Sec. 4.1, Fig. 5a).
  - An inverse-dynamics variant decodes joint torques from the posterior (Sec. 4.3).
  - Trained with three-quarters of observations randomly removed, for multi-step prediction (App. B).
  - Hardware data: real hydraulic BROKK-40, a pneumatic-muscle arm, Franka Panda (torque actions) and Barrett WAM (Sec. 5, App. B).
  - Hydraulic 200-step prediction NLL 6.25 vs RKN 4.93 (Fig. 4a inset; extraction garbled, check before quoting).
- **Quality tier:** Tier 1 (CoRL 2020, venue via PMLR v155 index; real-robot datasets, offline only).
- **Method weaknesses:**
  - No closed-loop or real-time control (author-acknowledged as future work, Sec. 6).
  - Small hyper-parameter-specific latent sizes (App. D); no runtime or parameter totals (reader-inferred).
- **Relevance to our setup:** shows the RKN predict step can be driven by torque-level actions and trained with heavy observation dropout. That training regime is the one we need for "z missing most steps, fast sensors always present".
- Read scope: full_text.

**[MTS3] Shaj, V., Gholam Zadeh, S., Demir, O., Douat, L. R., Neumann, G. (2023).** *Multi Time Scale World Models.* NeurIPS 2023. arXiv:2310.18534. https://arxiv.org/abs/2310.18534
- **Key findings:** two coupled Kalman-style SSMs.
  - **Slow task latent l_k:** updated once per window of H fast steps by a batch precision update over the window's observations (Eqs. 7–9).
  - **Fast state:** conditioned on the slow prior N(μ_l, Σ_l). The slow mean shifts the fast transition (C μ_l), and the slow covariance inflates fast process noise (Q + C Σ_l Cᵀ) (Eq. 6).
  - **Window length:** H·Δt fixed at 0.3 s (H=30 at 100 Hz). H ≤ 10 is much worse and H=75 degrades (Sec. 6.6, Fig. 4a, App. E).
  - Beats a multi-scale Transformer on long-horizon NLL for Kitchen, Panda and Hydraulic (Tab. 1), but not on Medium Maze. RKN and HiP-RSSM diverge at these horizons.
- **Quality tier:** Tier 1 (NeurIPS 2023, venue via proceedings index; real excavator and Panda torque data, prediction only).
- **Method weaknesses:**
  - Only two levels; not used for control; proprioception only (author-acknowledged, App. F).
  - Periodic jumps at window boundaries (Sec. 6.6, Fig. 4b).
  - The slow latent can only linearly modify fast dynamics (author-acknowledged, Sec. 3.2).
- **Relevance to our setup:** the closest structural match to "slow z conditions a fast Kalman-like belief", with an explicit, principled rule. z's uncertainty becomes extra process noise for the fast state, so staleness can be encoded as Σ_l growing with age. The window-boundary jumps warn about discontinuities when a new z arrives.
- Read scope: full_text.

**[PlaNet] Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., Davidson, J. (2019).** *Learning Latent Dynamics for Planning from Pixels.* ICML 2019 (PMLR 97). arXiv:1811.04551. https://arxiv.org/abs/1811.04551
- **Key findings:**
  - The RSSM splits the state into a deterministic RNN path h_t and a stochastic s_t. The prior p(s_t|h_t) is the predict step; the posterior q(s_t|h_t, o_t) is the filtering update, described as "a non-linear Kalman filter" (Sec. 3, Eq. 4, Fig. 2c).
  - Without the stochastic part the agent does not learn (Sec. 5, Fig. 4).
  - The planner rolls the prior forward from the current belief (Sec. 2).
  - PlaNet at 1,000 episodes ≈ or > D4PG at 100,000 on DMC (Tab. 1).
- **Quality tier:** Tier 1 (ICML 2019, venue via PMLR v97 index; DMC standard benchmark, simulation only).
- **Method weaknesses:**
  - Fixed action repeat, no value function beyond the horizon (author-acknowledged, Sec. 7).
  - Uncertainty is not used to weight late or unreliable observations explicitly (reader-inferred).
- **Relevance to our setup:** RSSM is the standard learned predict/update cell, and rolling the prior forward is exactly the machinery for "replay fast-sensor history after a late z". Karamzade24 uses this cell for delays.
- Read scope: full_text.

**[CRU] Schirmer, M., Eltayeb, M., Lessmann, S., Rudolph, M. (2022).** *Modeling Irregular Time Series with Continuous Recurrent Units.* ICML 2022 (PMLR 162). arXiv:2111.11344. https://arxiv.org/abs/2111.11344
- **Key findings:**
  - The latent follows a linear SDE. A continuous-discrete Kalman filter predicts over the actual gap Δt via the matrix exponential, then applies a standard update weighted by encoder σ_obs (Secs. 3.2–3.3, Eqs. 8–14).
  - The gain norm tracks observation noise (Fig. 4) and drops for sparse inputs (Fig. 5).
  - Pendulum interpolation MSE 0.996e-3 vs RKN 5.20e-3 and RKN-Δt 1.90e-3 (Tab. 2).
  - The f-CRU variant halves training time (Sec. 4.3, Tab. 1).
- **Quality tier:** Tier 3 (ICML 2022, venue via PMLR v162 index; no robot data).
- **Method weaknesses:**
  - Linear latent dynamics limit expressiveness, and the latent size is limited by matrix inversion/exponential (author-acknowledged, Sec. 3.4).
  - Discrete RNNs are still faster (author-acknowledged, Sec. 4.3).
- **Relevance to our setup:** a principled way to handle variable z latency and jitter. The predict step integrates over the true elapsed time, so a z stamped at t−d can be fused against the state at t−d (with stored history) or have its covariance grown by the gap. It overlaps with Theme 4's continuous-time cells (CfC, S5).
- Read scope: full_text.

**[DVRL] Igl, M., Zintgraf, L., Le, T. A., Wood, F., Whiteson, S. (2018).** *Deep Variational Reinforcement Learning for POMDPs.* ICML 2018 (PMLR 80). arXiv:1806.02426. https://arxiv.org/abs/1806.02426
- **Key findings:**
  - The policy's internal state is a set of K weighted particles, updated by a learned model through resample → propose → reweight (Sec. 3.3, Eqs. 15–18).
  - A second RNN summarizes the particle belief for actor and critic (Fig. 2).
  - Model (SMC-ELBO) and A2C losses are trained jointly (Eqs. 19–20).
  - Beats an RNN policy on 5 of 10 flickering-Atari games, e.g. Pong 18.2 vs 6.3 (Tab. 1).
  - One particle or no ELBO is worse (Sec. 5.3, Fig. 5).
- **Quality tier:** Tier 3 (ICML 2018, venue via PMLR v80 index; Atari/toy tasks, no robots).
- **Method weaknesses:** no limitations stated; future work mentions stronger architectures (Sec. 6). The K-particle cost and the variance of joint training are likely too heavy for an MCU-class policy (reader-inferred).
- **Relevance to our setup:** the clearest evidence that a learned filter inside a policy, trained jointly with an auxiliary model loss, beats a plain RNN when observations drop out. "Flickering" frames are analogous to z being absent most steps. An auxiliary "predict next torque/F-T" loss is the cheap version for us.
- Read scope: full_text.

### (b) Delayed and out-of-sequence measurements

**[BarShalom02] Bar-Shalom, Y. (2002).** *Update with out-of-sequence measurements in tracking: exact solution.* IEEE Transactions on Aerospace and Electronic Systems 38(3), 769–777. DOI:10.1109/TAES.2002.1039398. https://doi.org/10.1109/TAES.2002.1039398
- **Key findings:**
  - This is the seminal definition of the OOSM update problem. In multisensor tracking, different transmission delays make measurements arrive out of sequence. The paper derives the exact update of the current estimate with an older measurement whose lag is under one sampling interval: retrodiction to the measurement time, accounting for process noise, without full re-filtering. Three algorithms are compared on realistic examples, including GMTI radar (abstract only, via the SPIE 2000 conference version and search snippets).
  - Context: MathWorks' implementation guide reports retrodiction covariance traces only about 2–3% above full reprocessing at much lower memory and compute. This is a vendor doc, not a peer-reviewed source.
- **Quality tier:** Tier 3 (IEEE TAES, venue via Crossref). A classical non-learned method, included as the seminal reference for out-of-sequence fusion.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** defines the two options we will compare:
  - (i) **rewind and replay:** buffer fast-sensor history, apply the late z at its timestamp, re-predict forward;
  - (ii) **retrodiction:** a single correction from cross-covariances, without replay.
  For a GRU belief only (i) is directly available unless we learn the retrodiction.
- Read scope: abstract_only.

**[BarShalom04] Bar-Shalom, Y., Chen, H., Mallick, M. (2004).** *One-step solution for the multistep out-of-sequence-measurement problem in tracking.* IEEE Transactions on Aerospace and Electronic Systems 40(1), 27–37. DOI:10.1109/TAES.2004.1292140. https://doi.org/10.1109/TAES.2004.1292140
- **Key findings:**
  - Extends the OOSM update to arbitrary (multistep) lag.
  - Earlier multistep extensions needed large storage; this one-step solution updates the current estimate directly with a lag-l measurement (abstract only, via search snippets).
  - MathWorks' retrodiction example implements its "algorithm Bl1" (vendor doc).
- **Quality tier:** Tier 3 (IEEE TAES, venue via Crossref). Classical non-learned method; seminal reference.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** our z is multistep-late, with 100–500 ms latency spanning 2–10 fast steps at 20 Hz. This is the relevant classical baseline for "apply late z once, without replaying the buffer", against which learned rewind-and-replay can be compared for cost.
- Read scope: abstract_only.

**[Larsen98] Larsen, T. D., Andersen, N. A., Ravn, O., Poulsen, N. K. (1998).** *Incorporation of time delayed measurements in a discrete-time Kalman filter.* Proc. 37th IEEE Conference on Decision and Control (CDC), vol. 4, 3972–3977. DOI:10.1109/CDC.1998.761918. https://doi.org/10.1109/CDC.1998.761918
- **Key findings:**
  - Targets sensors with long processing delay, such as vision.
  - It compares methods from the literature and proposes a new one: "extrapolate" the delayed measurement to the present using past and present filter estimates, with an optimal gain for the extrapolated measurement. This trades optimality against computation versus re-filtering (abstract only, via the DTU Orbit record).
- **Quality tier:** Tier 3 (IEEE CDC proceedings, venue via Crossref). Classical non-learned method; seminal reference. The motivating case is a slow vision sensor, close to our setting.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** a third mechanism besides rewind-and-replay and retrodiction. Extrapolate z forward to "now" (as RTC/VLASH-style methods do for action chunks) and fuse it with a gain matched to the extrapolation. This is cheap enough for an MCU.
- Read scope: abstract_only.

**[Walsh08] Walsh, T. J., Nouri, A., Li, L., Littman, M. L. (2008).** *Learning and planning in environments with delayed feedback.* Autonomous Agents and Multi-Agent Systems 18(1), 83–105. DOI:10.1007/s10458-008-9056-7. https://doi.org/10.1007/s10458-008-9056-7
- **Key findings:**
  - Markovian environments with constant observation and reward delays.
  - "Model Based Simulation" plans by using a learned model to simulate forward from the last observed state through the pending actions.
  - Hardness result for general planning; positive results for deterministic or constrained dynamics; empirical advantages over alternatives (abstract only, via indexer summary; publisher page behind login).
- **Quality tier:** Tier 3 (peer-reviewed journal, venue via Crossref; small/toy domains per abstract).
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** the classical root of "roll a model forward over the delay". On our fast side this is predicting the current state from the late z plus the buffered fast actions and sensors. Hardness for stochastic dynamics argues for keeping a belief rather than a point roll-forward.
- Read scope: abstract_only.

**[Karamzade24] Karamzade, A., Kim, K., Kalsi, M., Fox, R. (2024).** *Reinforcement Learning from Delayed Observations via World Models.* Reinforcement Learning Journal 5 (RLC 2024). arXiv:2403.12309. https://arxiv.org/abs/2403.12309
- **Key findings:**
  - A world model congruent with the POMDP turns a delayed POMDP into a delayed MDP over the latent state (Prop. 3, Sec. 4.1).
  - Built on DreamerV3, with three actors:
    - **Extended:** latent at t−d plus actions a_{t−d:t−1};
    - **Memoryless:** latent at t−d only;
    - **Latent:** rolls the RSSM prior d steps forward and acts on the prediction (Secs. 4.3–4.4, Fig. 1).
  - Extended improves about 250% on average over a delay-agnostic Dreamer on DMC vision tasks (Sec. 5.2, Fig. 3f).
  - Latent roll-forward degrades at long delays from compounding errors (Sec. 5.2).
  - 5 seeds, delays 2–20 steps (Fig. 3).
- **Quality tier:** Tier 1 (RLJ/RLC 2024, venue via rlj.org volume page; DMC simulation).
- **Method weaknesses:**
  - Extended is sensitive to hyper-parameters (author-acknowledged, Sec. 6).
  - Only fixed delays were tested, despite claims of generality (author-acknowledged, Sec. 5.1).
  - No out-of-order arrival (reader-inferred).
- **Relevance to our setup:** direct evidence on "roll forward over the delay" vs "condition on the stale latent plus actions since". For our fast policy, the "Extended" pattern means feeding z with the actions and fast sensors since z's timestamp. It is cheap and beats pure roll-forward at long delays.
- Read scope: full_text.

**[Wang24] Wang, W., Han, D., Luo, X., Li, D. (2024).** *Addressing Signal Delay in Deep Reinforcement Learning.* ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/45cc967d7899616f51993b7b363d35b5-Abstract-Conference.html
- **Key findings:**
  - Formalizes the delayed-observation MDP. Inference, action and observation delays are equivalent; only their sum matters (Theorem 2.1, Sec. 2).
  - Fixes:
    - a delay-reconciled critic trained on time-realigned true states (Sec. 4.1);
    - actor state augmentation with the actions since the observation (Sec. 4.2);
    - optional auxiliary prediction of the true current state (Sec. 4.3).
  - Normalized MuJoCo performance under fixed delay: SAC 1.2% → +critic 50.8% → +actor augmentation 75.9%; random delay 8.4% → 48.1% → 77.0% (Tab. 1).
  - Explicit state prediction helps under fixed delay but hurts under random delay (Sec. 5, "Fixed v.s. Unfixed").
  - Random delays are generated in order, so out-of-sequence arrival is excluded by design (App. E).
- **Quality tier:** Tier 1 (ICLR 2024, venue via ICLR proceedings page; MuJoCo simulation only).
- **Method weaknesses:**
  - Simulation only (author-acknowledged, Sec. 6).
  - Timestamp input left to future work (App. C.1).
  - No out-of-order arrival (reader-inferred).
- **Relevance to our setup:**
  - The time-realigned critic maps onto training the fast policy with teacher targets aligned to true time.
  - "Prediction helps under fixed delay but hurts under random delay" is a warning for explicit roll-forward of z under network jitter.
  - The delay-equivalence theorem justifies lumping cloud compute and network latency.
- Read scope: full_text.

**[TSRL] Sandha, S. S., Garcia, L., Balaji, B., Anwar, F., Srivastava, M. (2021).** *Sim2Real Transfer for Deep Reinforcement Learning with Stochastic State Transition Delays.* CoRL 2020 (PMLR 155, 1066–1083). https://proceedings.mlr.press/v155/sandha21a.html
- **Key findings:**
  - Time-in-State RL appends the measured execution latency and sampling interval to the observation (Sec. 3).
  - Compared against domain-randomizing delays (PPO; random per-episode delays with per-step jitter) (Secs. 3.1–3.2).
  - Real 1/18-scale car, laps completed out of 24: TSRL vs DR 17 vs 11 at 60 ms and 13 vs 7 at 100 ms (Tab. 1a).
  - Camera interval measured at 25–45 ms nominally at 30 Hz (Sec. 4); 3 seeds per policy.
- **Quality tier:** Tier 1 (CoRL 2020, venue via PMLR page; real small car + PyBullet).
- **Method weaknesses:**
  - Degrades past a task-specific latency (author-acknowledged, Sec. 6).
  - Communication delay folded into execution latency; no roll-forward or out-of-order handling (author-stated assumption, Sec. 2.1).
- **Relevance to our setup:** the minimal baseline for z staleness. Give the fast policy z's age (and update interval) as inputs. Real-robot evidence shows this beats plain delay randomization, which is the default in Theme 4's legged-robot works.
- Read scope: full_text.

**[LAA] Agarwal, S., Kaul, S. K., Anand, S., Sujit, P. B. (2022).** *A Deep Learning Approach To Estimation Using Measurements Received Over a Network.* arXiv:2201.08020. https://arxiv.org/abs/2201.08020
- **Key findings:**
  - An LSTM estimator takes the previous estimate, the freshest received measurement and a per-element age-of-information vector, with no history window (Sec. IV, Eq. 2, Fig. 1).
  - Network delay is random (FCFS queue with retransmissions, Sec. III).
  - It matches the optimal time-varying KF when controls are known (Fig. 3).
  - It beats the time-varying KF and a UKF that re-filters aged measurements at their timestamp when controls are also delayed (Sec. VII-B, Fig. 4).
  - It is robust to noisy age estimates (Sec. VII-C, Fig. 5). Results are given only as figures.
- **Quality tier:** Tier 3 (arXiv preprint; synthetic linear/nonlinear systems).
- **Method weaknesses:**
  - No limitations stated.
  - Ground truth is needed for training (footnote 2, Sec. V).
  - FCFS means no out-of-order packets (reader-inferred).
- **Relevance to our setup:** the closest learned counterpart to "z as a delayed measurement". A small recurrent estimator with an explicit age input matched an optimal KF and beat a re-filtering UKF under random network delay. It is the learned alternative to rewind-and-replay that we should include as a baseline.
- Read scope: full_text.

### (c) Innovation / prediction-error gating, forward models, failure and anomaly detection

**[Willsky1976] Willsky, A. S. (1976).** *A survey of design methods for failure detection in dynamic systems.* Automatica 12(6), 601–611. DOI 10.1016/0005-1098(76)90041-8. https://doi.org/10.1016/0005-1098(76)90041-8
- **Key findings:** surveys failure-sensitive filters, voting, multiple-hypothesis filter banks, jump processes and innovations-based detection (Abstract; Secs. IV–VII). Sec. VII: under normal operation the Kalman innovations are zero-mean and white with known covariance. The windowed sum of normalized squared innovations over N steps is chi-squared with Np degrees of freedom, which gives a threshold rule with a computable false-alarm probability. Willsky credits this test to Mehra and Peschon. A larger N lowers false alarms but makes the detector slow to respond to new data. The chi-squared test is "basically an alarm method": dramatic failures are detected, subtle ones are hard, and it does not isolate failures (Sec. VII). The GLR approach (Sec. VII, after the chi-squared test) adds failure signatures for isolation, at higher cost.
- **Quality tier:** Tier 3 (Automatica; Crossref). Read via the NASA NTRS tech-report version (ESL-P-633, Nov 1975). Page and section numbering may differ from the journal version.
- **Method weaknesses:** linear-Gaussian scope (author-acknowledged, Abstract). The chi-squared test has mixed performance, detects only large failures and has no isolation (author-acknowledged, Sec. VII).
- **Relevance to our setup:** the canonical form of "trust depends on innovation, not age". A windowed NIS of the fast sensors (torque, F/T) against what z predicts, with a chi-squared threshold, is a non-learned baseline gate for the latent. The window-length trade-off (N) is exactly the latency vs false-alarm knob we must tune.
- Read scope: partial (Abstract and Sec. VII of the tech-report version).

**[Wolpert1995] Wolpert, D. M., Ghahramani, Z., Jordan, M. I. (1995).** *An Internal Model for Sensorimotor Integration.* Science 269(5232), 1880–1882. DOI 10.1126/science.7569931. https://doi.org/10.1126/science.7569931
- **Key findings:** participants estimated hand location after movements in the dark under imposed forces. The temporal propagation of their errors fits an optimal state-estimation framework, which the authors take as direct support for an internal (forward) model (abstract only).
- **Quality tier:** Tier 3 (Science; Crossref).
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** the seminal neuroscience anchor for "forward model plus sensory prediction error" (efference-copy style estimation). It motivates a fast-policy head that predicts the expected proprio/F/T from the commanded action and z.
- Read scope: abstract_only.

**[LSTM-VAE] Park, D., Hoshi, Y., Kemp, C. C. (2018).** *A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder.* IEEE RA-L 3(3), 1544–1551. arXiv:1711.00614. https://arxiv.org/abs/1711.00614
- **Key findings:** 17 signals: sound, 3-D end-effector F/T, 7 joint torques, spoon and mouth positions, all resampled to 20 Hz, the robot's anomaly-check rate (Sec. on data). The anomaly score is the negative log-likelihood of the observation under the reconstructed distribution. The threshold is **state-dependent**: an SVR maps the latent z to the expected score, plus a constant c. AUC 0.8710 on raw signals vs 0.8075 for EncDec-AD, and 0.8564 vs 0.8121 (HMM-GP) on hand-engineered features (Table I). Data: 1,555 executions with 12 anomaly types (abstract).
- **Quality tier:** Tier 1 (IEEE RA-L; Crossref).
- **Method weaknesses:** no dedicated limitations section. Detection delay is not reported (reader-inferred, from the full text). Reconstruction-based, so it does not condition on a plan or commanded action (reader-inferred).
- **Relevance to our setup:** the state-based threshold is a learned analogue of an innovation covariance: how surprising a torque/F/T reading is depends on the phase. The same idea would make our trust gate context-dependent rather than fixed.
- Read scope: full_text.

**[Grambow-AD] Grambow, N., Fenner, L.-M., Kempkes, F., Hotz, P., Wan, D., Krüger, J., Haninger, K. (2026).** *Anomaly detection for generic failure monitoring in robotic assembly, screwing and manipulation.* IEEE RA-L 11(4), 4737–4744. arXiv:2509.26308. https://arxiv.org/abs/2509.26308
- **Key findings:** reconstruction-based detectors (VAE, AE, UAE, SWCVAE) on F/T, EE pose/velocity and gripper signals (Table III), with feature-wise thresholds from nominal reconstruction errors (Sec. III-D). Three tasks: diffusion-policy cabling, position-controlled screwing, force-controlled polishing. Cabling AUROC > 0.96 for four of five failure types. Screwing "missing screw" scores only 0.52 and polishing "surface roughness" 0.27 (Table IV). Detection latencies are 0–182 ms (Table IV).
- **Quality tier:** Tier 1 (IEEE RA-L; Crossref).
- **Method weaknesses:** subtle low-force failures are hard to detect, task variability hurts, threshold selection is difficult, and detected features do not map to failure causes (author-acknowledged, Sec. VI / Discussion). No forward model of expected force given the action (reader-inferred).
- **Relevance to our setup:** evidence that F/T-based detectors catch abrupt events within about 0.2 s but miss subtle ones. Slip and drop events are abrupt, which favours our use case. It also warns that a plain reconstruction score without plan conditioning fails on high-variance tasks.
- Read scope: full_text.

**[Sentinel] Agia, C., Sinha, R., Yang, J., Cao, Z., Antonova, R., Pavone, M., Bohg, J. (2024).** *Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress.* CoRL 2024 (PMLR 270: 689–723). arXiv:2410.04640. https://arxiv.org/abs/2410.04640
- **Key findings:** STAC compares the action distributions that two consecutive chunk predictions give for the overlapping h−k steps, using MMD or KDE-based forward/reverse KL. The score accumulates over time, and its threshold is the (1−δ) quantile on successful rollouts (conformal, false-positive rate ≤ δ) (Sec. 4.1, Eq. 1; App. A.1, App. D Prop. 1). A VLM progress monitor covers slow task-progress failures (Sec. 4.2). Close Box: STAC 99% TPR / 89% TNR, Sentinel 100% / 87% (Table 1). Real push-chair: 100% TPR / 90% TNR with 10 calibration demos (Table 2).
- **Quality tier:** Tier 1 (CoRL, PMLR 270).
- **Method weaknesses:** failure categories are not exhaustive, there are no formal detection guarantees without failure data, and it detects failures but does not predict them (author-acknowledged, Conclusion). STAC needs multiple samples per step, which is costly (reader-inferred; confirmed as "too slow for real-time" by [FAIL-Detect], Fig. 5 discussion).
- **Relevance to our setup:** STAC is an "innovation between successive plans": if a fresh z disagrees with the stale z on the overlapping horizon, trust in the stale plan should drop. This is a slow-side analogue to our fast-sensor innovation.
- Read scope: full_text.

**[FAIL-Detect] Xu, C., Nguyen, T. K., Dixon, E., Rodriguez, C., Miller, P., Lee, R., Shah, P., Ambrus, R., Nishimura, H., Itkina, M. (2025).** *Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies.* RSS 2025. arXiv:2503.08558. https://arxiv.org/abs/2503.08558
- **Key findings:** distils policy inputs (visual features plus robot state) and outputs into scalar scores. It compares flow-density (logpZO, new), second-order (NatPN, DER), one-class (RND, CFM) and post-hoc (SPARC, STAC, PCA-kmeans) scores (Table I). The threshold is a time-varying conformal band, upper_t = μ_t + h_t·s_cal(t) (Sec. IV-B). Sim: average best balanced accuracy about 78%, with logpZO top-1 in 10/16 cases (Fig. 4). Hardware: about 72%, top-1 in 8/12 (Fig. 5). logpZO runs 36–44× faster than STAC.
- **Quality tier:** Tier 1 (RSS XXI; Crossref DOI 10.15607/RSS.2025.XXI.073).
- **Method weaknesses:** scores over-rely on low-dimensional robot state, false positives occur in OOD settings, there is no long temporal context, and there is no tactile/force input (author-acknowledged, Sec. VII).
- **Relevance to our setup:** the time-varying conformal band is a ready recipe for calibrating our innovation gate's threshold from successful rollouts only. Their call for force inputs is exactly the gap our F/T innovation would fill.
- Read scope: full_text.

**[IMPACT] Gao, J., Liu, C., Wu, P., Chen, H., Du, Y. (2026).** *IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation.* arXiv:2606.10818. https://arxiv.org/abs/2606.10818
- **Key findings:** an internal model predicts slowly varying external wrench from a 32-step joint-state history, and the prediction is fed forward through Jᵀ (Sec. 2.3). A **surprise gate** turns online learning on when prediction error exceeds a threshold and off once it falls below (Sec. 3.3, Fig. 7). The policy runs at 30 Hz and control at 1 kHz (App. A.1). Franka FR3 at a 5 kg test mass: 21/25 vs vanilla DP 0/25 and augmented DP 5/25 (Table 1).
- **Quality tier:** Tier 2 (preprint).
- **Method weaknesses:** limited to rigid-body force properties (gravity, stiffness, friction). It cannot handle deformables, fluids or granular media (author-acknowledged, Sec. 5). The network architecture is not specified, and the threshold is hand-set (reader-inferred).
- **Relevance to our setup:** a direct learned instance of a prediction-error gate on the wrench channel in a slow-policy/fast-loop stack. Our variant would gate trust in z instead of gating learning, but the predict-wrench → residual → threshold pipeline is the same.
- Read scope: full_text.

**[CW-WAM] Ding, H., Qi, X., Wang, L., Yang, M. (2026).** *CW-WAM: A Structured Contact-Wrench World Action Model for Force-Sensitive Manipulation.* arXiv:2610.30117. https://arxiv.org/abs/2610.30117
- **Key findings:** a 5B world-action model (Wan2.2-TI2V-5B backbone) jointly predicts 24 actions and 24 TCP wrenches over 1.6 s, taking 0.5–0.6 s per call on an RTX 5090. At runtime a 20 Hz admittance loop drives the TCP offset from e_w = ŵ − w_meas. Joint tracking runs at 1 kHz. Real-robot square-peg insertion improves from 25% to 70% and adapter insertion from 40% to 95% vs a baseline WAM. Admittance lowers force RMSE from 4.8 to 2.7 N on peg insertion (Fig. 4).
- **Quality tier:** Tier 2 (preprint).
- **Method weaknesses:** four in-distribution tasks only, with manually tuned gains and predefined compliant directions (author-acknowledged, limitations). The wrench residual is used only for compliance, never to gate trust in the plan or to re-query (reader-inferred).
- **Relevance to our setup:** the slow model already ships an *expected wrench* with its plan, which is exactly the quantity we need to form an F/T innovation on the robot. It stops short of the step we propose: using a large ‖e_w‖ to distrust the plan and re-query.
- Read scope: full_text.

### (e) Event-triggered / self-triggered control and estimation, and learned "when to query"

**[Tabuada2007] Tabuada, P. (2007).** *Event-Triggered Real-Time Scheduling of Stabilizing Control Tasks.* IEEE Trans. Automatic Control 52(9), 1680–1685. DOI 10.1109/TAC.2007.904277. https://doi.org/10.1109/TAC.2007.904277
- **Key findings:** treats the scheduler as a feedback controller. The control task executes when the measurement error e, relative to the state norm, crosses a threshold, γ(|e|) ≥ σα(|x|), with σ < 1 preserving ISS-Lyapunov decrease (Sec. 2.2). Thm. 3.1: inter-execution times are lower-bounded by τ > 0 (no Zeno behaviour) when the execution delay Δ is small enough. Linear case: the rule |e| ≥ σ|x| has an explicit bound (Cor. 4.1, Eqs. 4.3–4.6).
- **Quality tier:** Tier 3 (IEEE TAC; Crossref). Read from the author's preprint PDF.
- **Method weaknesses:** a deterministic state-feedback setting with full state and no noise (reader-inferred). The guarantees need a small Δ (author-stated condition, Thm. 3.1).
- **Relevance to our setup:** the classical form of "act or communicate only when the error is large relative to the state". The minimum inter-event time result is the guarantee we would want for re-query storms (no unbounded re-query rate).
- Read scope: partial (Abstract, Secs. 1–4).

**[Trimpe-EBSE] Trimpe, S. (2017).** *Distributed Event-based State Estimation* (published as *Event-based state estimation: an emulation-based approach*, IET Control Theory & Applications 11(11), 1684–1693, DOI 10.1049/iet-cta.2016.1021). arXiv:1511.05223. https://arxiv.org/abs/1511.05223
- **Key findings:** an agent broadcasts measurement yᵢ(k) iff ‖yᵢ(k) − Cᵢx̂ᵢ(k|k−1)‖ ≥ δᵢ, a threshold on the **measurement innovation** (Eq. 13). Estimation error stays bounded by a term proportional to ‖δ‖ (Eq. 19). Sec. 1 classifies the triggers: send-on-delta, measurement- or innovation-based [Trimpe & D'Andrea 2011a; Sijs et al.], decorrelated innovation [Wu et al. 2013], **variance-based** (innovation variance, offline-computable) [Trimpe & D'Andrea 2014], relevant sampling (KL gain) and stochastic triggers. Balancing-cube simulation: communication rises after an impulsive disturbance at 10 s (Fig. 5). About 80% of communication can be saved before estimation error grows, but slow periodic sampling wins at very low rates (Fig. 7).
- **Quality tier:** Tier 3 (IET CTA; Crossref). The arXiv v2 states the published DOI. The published title differs.
- **Method weaknesses:** linear models with a shared broadcast bus. Needs synchronous averaging to stop inter-agent estimate drift (Fig. 6, Sec. 6.1.3). Thresholds are hand-chosen (reader-inferred).
- **Relevance to our setup:** the exact classical mechanism behind the author's idea. Both sides run the same model prediction, and a message is sent only when the innovation exceeds δ. With roles swapped, the robot asks the slow model for a new z when its F/T/torque innovation against z's prediction exceeds δ. Fig. 7's crossover warns that innovation triggering is not always better than periodic updates at very low rates.
- Read scope: full_text.

**[ETL] Solowjow, F., Trimpe, S. (2020).** *Event-triggered Learning.* Automatica 117, 109009. arXiv:1904.03042. https://arxiv.org/abs/1904.03042
- **Key findings:** in event-triggered state estimation, the sender transmits when ‖X(t) − X̌(t)‖ ≥ δ (Sec. 2.1, Eq. 3). Inter-communication times are stopping times whose distribution depends on the model (Sec. 3). Learning triggers fire when observed inter-event times deviate from the model's prediction: an expectation trigger via Hoeffding (Thm. 4) and a density trigger via two-sample KS / DKW (Thm. 7). Sec. 6: learning triggered at k = 4961 on a first-order system with a perturbed model (Fig. 3). The expectation trigger can miss mismatches that the density trigger catches (Figs. 4–6). Sec. 7.2: a better model can *increase* communication under partial observation (Figs. 7–8).
- **Quality tier:** Tier 3 (Automatica; Crossref).
- **Method weaknesses:** linear-Gaussian only, the choice of trigger signal is problem-dependent, and the learning step is abstracted away (author-acknowledged, Sec. 8).
- **Relevance to our setup:** a second-level trigger. If the fast policy's innovation-triggered re-query rate drifts from its calibrated rate, that is a statistical sign that the fast model (or its expectation of z) is miscalibrated. It also gives a principled monitor for "too many re-queries".
- Read scope: full_text.

**[DRL-ETC] Baumann, D., Zhu, J.-J., Martius, G., Trimpe, S. (2018).** *Deep Reinforcement Learning for Event-Triggered Control.* IEEE CDC 2018, 943–950. arXiv:1809.05152. https://arxiv.org/abs/1809.05152
- **Key findings:** jointly learns a control policy and a binary communicate/skip policy with reward rₖ = −xᵀQx − uᵀRu − λγₖ (Eq. 15). It has an end-to-end DDPG variant (Sec. III-A) and a policy-gradient trigger on top of a fixed controller (Sec. III-B). Savings: about 90% on inverted pendulum (Fig. 3), about 80% on swing-up (Fig. 4) and about 60% on Ant (Fig. 6). Model-based ETC wins at moderate savings but goes unstable beyond about 70% savings, where DRL still works (Sec. IV-B).
- **Quality tier:** Tier 3 (IEEE CDC; Crossref).
- **Method weaknesses:** no stability guarantees, zero-order hold instead of model-based prediction between events, and only moderate success on locomotion (author-acknowledged, Secs. IV-A, IV-D). Simulation only (reader-inferred).
- **Relevance to our setup:** the learned template for a fast policy that outputs both an action and a "query the slow model" bit with a per-query cost λ. λ maps directly to network/energy cost per cloud call.
- Read scope: full_text.

**[WhenToReplan] Honda, K., Yonetani, R., Nishimura, M., Kozuno, T. (2024).** *When to Replan? An Adaptive Replanning Strategy for Autonomous Navigation using Deep Reinforcement Learning.* IEEE ICRA 2024, 6650–6656. arXiv:2304.12046. https://arxiv.org/abs/2304.12046
- **Key findings:** a DRL replanner decides at each step whether to call the global planner. It observes a 62-D input: downsampled scans, reference path, past trajectory and goal (Sec. IV-B). Baselines are distance-, stuck-, time- and time-with-patience triggers (Sec. III-C). 16-pillar env: 77% success with 2,577 replans vs time-based 70% with 3,076 (Table I). The gain holds across planner pairs except with stochastic RRT* (Table II). Q-difference prioritized replay gives 77% vs 69% without (Table III).
- **Quality tier:** Tier 1 (ICRA 2024; Crossref).
- **Method weaknesses:** real-world transfer is open because of layout diversity and observation uncertainty, and maps are simple (author-acknowledged, Sec. VI). Navigation only, with no contact sensing (reader-inferred).
- **Relevance to our setup:** shows a learned replanning trigger beats fixed-interval and heuristic triggers on both success and call count. The same structure applies to re-querying the slow VLM. Its observation does not include an explicit innovation signal, which is our proposed addition.
- Read scope: full_text.

**[KnowNo] Ren, A. Z., Dixit, A., Bodrova, A., Singh, S., Tu, S., Brown, N., Xu, P., Takayama, L., Xia, F., Varley, J., Xu, Z., Sadigh, D., Zeng, A., Majumdar, A. (2023).** *Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners.* CoRL 2023 (PMLR 229: 661–682). arXiv:2307.01928. https://arxiv.org/abs/2307.01928
- **Key findings:** uses conformal prediction over LLM planner options to decide when to ask a human for help. It gives statistical task-completion guarantees while minimizing help requests, in sim and real robot setups (abstract only).
- **Quality tier:** Tier 1 (CoRL, PMLR 229).
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relevance to our setup:** the standard "ask when uncertain" result with a calibrated trigger. In our stack the "help" is a fresh slow-model query, and the uncertainty would come from the fast-sensor innovation instead of LLM token scores.
- Read scope: abstract_only.

**[TUD] Yang, H., Jeong, W., Lee, G., Yoon, K.-J. (2026).** *Triggering Generalist Reasoning via Predictive Uncertainty for Dual-System VLA.* arXiv:2610.05025 (arXiv comment: accepted to NeurIPS 2026). https://arxiv.org/abs/2610.05025
- **Key findings:** System 2 is a PaliGemma-based generalist that caches KV tokens. System 1 is a Q-Former plus flow-matching action expert (Sec. 3.1, Eqs. 2–3), running at about 73.9 Hz vs about 9.4 Hz for the generalist (13.5 vs 105.9 ms per call, Table 3). The trigger is the **dispersion** (std) of the specialist's repeated predictions for the same future chunk slots over a sliding buffer (Eq. 5). It calls the generalist when the running mean exceeds a percentile threshold θ_p or a maximum interval elapses (Sec. 3.3). Prop. 1 bounds the action deviation by a dispersion term plus a drift term (App. B). Real robot: 83.3% vs 46.7% success with 12.1 vs 48.5 calls against a fixed Δ = 9 interval (Table 5). Signal comparison: 86.7% at 17.8 calls/episode (Table 4).
- **Quality tier:** Tier 2 (venue only from the arXiv comment; not confirmed on a proceedings page).
- **Method weaknesses:** the percentile p sets the trade-off but cannot target a latency or success budget (author-acknowledged, Sec. 5). The trigger uses only plan self-consistency and no proprio/force residual (reader-inferred).
- **Relevance to our setup:** the closest existing "fast policy decides when to call the slow VLM" result, with 75% fewer calls at higher success. Its signal is internal plan dispersion. Our innovation gate would add an *external* check (do sensors agree with the plan?), which could be combined with a maximum-interval fallback like theirs.
- Read scope: full_text.

**[AAC] Liang, Y., Wang, X., Wang, K., Wang, S., Peng, X., Chen, H., Chua, D. K. H., Vadakkepat, P. (2026).** *Adaptive Action Chunking at Inference-time for Vision-Language-Action Models.* arXiv:2604.04161 (arXiv comment: accepted by CVPR 2026). https://arxiv.org/abs/2604.04161
- **Key findings:** chooses chunk length, and so the re-inference point, from action entropy. It uses Gaussian differential entropy for translation/rotation and Shannon entropy for the gripper, averaged over candidate horizons, with h* at the maximum-difference point plus a magnitude floor ξ (Sec. 4, Eqs. 4–5; App. A Eq. 6). GR00T+AAC vs GR00T: RoboCasa 62.0% vs 59.7%, LIBERO 95.0% vs 94.1% (Table 1), real-world 82.0% vs 67.0% (Table 5).
- **Quality tier:** Tier 2 (venue only from the arXiv comment).
- **Method weaknesses:** the chunk choice is not yet optimal per task (author-acknowledged, Table 6 note). Strong OOD degradation, down to 0.0% on LIBERO-Pro ×0.4 (Table 3). 20 samples add about 20 ms per step (Table 4). Small simulation gains (reader-inferred).
- **Relevance to our setup:** an uncertainty-based "when to replan" rule for VLAs that runs entirely on the slow side. It is a baseline the fast-side innovation trigger should beat when surprises come from contact (slip, drop) rather than from the policy's own uncertainty.
- Read scope: full_text.

### (d1) Residual policies on top of a base controller or base policy

**[silver2018residual] Silver, T., Allen, K., Tenenbaum, J., Kaelbling, L. (2018).** *Residual Policy Learning.* arXiv preprint. arXiv:1812.06298. https://arxiv.org/abs/1812.06298
- **Key findings:** the policy is π_θ(s) = π(s) + f_θ(s); the gradient does not depend on the base policy π, so π may be non-differentiable (hand-designed or MPC), and the last layer of f_θ is initialised to zero (Sec. IV). Base policies include reactive controllers, a discrete MPC and a cached PETS model-based controller (Sec. V-A). With sensor noise (NoisyHook) the base policy succeeds ~0.15 of the time, DDPG+HER from scratch never succeeds, and RPL converges to ~0.8 (Fig. 4a, Sec. V). Under model mismatch (ComplexHook) the base drops to 0.55 and RPL again reaches ~0.8 (Fig. 4b).
- **Quality tier:** Tier 2 (arXiv preprint; no venue found on Crossref).
- **Method weaknesses:** the method targets tasks "where good but imperfect controllers are available" (author framing, abstract); it is not shown to help without a reasonable base (reader-inferred). Simulation only (Fetch, MuJoCo), sparse-reward RL rather than imitation (reader-inferred).
- **Relevance to our setup:** the canonical form of our residual head: action = teacher-planned action + learned correction, with the correction initialised to zero so a fresh plan is never made worse. Their noisy/mismatched base is our stale plan.
- Read scope: full_text.

**[johannink2019residual] Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Loskyll, M., Aparicio Ojea, J., Solowjow, E., Levine, S. (2019).** *Residual Reinforcement Learning for Robot Control.* ICRA 2019, pp. 6023–6029. arXiv:1812.03201. https://arxiv.org/abs/1812.03201
- **Key findings:** u = π_H(s_m) + π_θ(s_m, s_o): a hand-engineered feedback controller (impedance control on the real robot) plus an RL residual (Sec. III-A). Real block assembly with misaligned blocks: 15/20 residual RL vs 2/20 for the hand-engineered controller, after ~3 h / ~8,000 samples (Fig. 3). In simulation the residual compensates control noise σ ∈ [0.01, 0.1] and control bias μ ∈ [0, 0.2] (Fig. 4).
- **Quality tier:** Tier 1 (ICRA 2019, DOI 10.1109/ICRA.2019.8794127).
- **Method weaknesses:** needs a carefully crafted vision setup to estimate positions and angles, which limits novel scenes (author-acknowledged, Sec. VIII). One task family (reader-inferred).
- **Relevance to our setup:** shows the residual can absorb a systematic bias in the base command, which is what a time-shifted (stale) plan looks like to the fast loop.
- Read scope: full_text.

**[yuan2025policy] Yuan, X., Mu, T., Tao, S., Fang, Y., Zhang, M., Su, H. (2025).** *Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model.* ICLR 2025. arXiv:2412.13630. https://arxiv.org/abs/2412.13630
- **Key findings:** executed action = π_base(s) + π_res(s) on top of a frozen Behavior Transformer or Diffusion Policy (16-action chunks with receding horizon) (Sec. 4.1, 5.1.2). The residual is bounded to (−α, α) after the tanh (Sec. 4.2), and a progressive schedule raises the probability of using the residual from 0 to 1 over H steps (Eq. 1). All three components matter (Fig. 9). Residual input: observation alone usually works better than observation + base action (Sec. 4.3, Fig. 19). α works over a wide range (0.1–0.5 PushChair, 0.03–0.1 StackCube).
- **Quality tier:** Tier 1 (ICLR 2025, OpenReview poster).
- **Method weaknesses:** online RL remains resource-intensive, especially for slow-inference base models like diffusion policies (author-acknowledged, Sec. 6). Simulation only (ManiSkill, Adroit) (reader-inferred).
- **Relevance to our setup:** the bounded residual (α) is a direct design knob for our head: it caps how far the small policy can depart from the teacher plan. Fig. 19 is a useful prior for whether to feed the planned action into the residual.
- Read scope: full_text.

**[ankile2025imitation] Ankile, L., Simeonov, A., Shenfeld, I., Torne, M., Agrawal, P. (2025).** *From Imitation to Refinement -- Residual RL for Precise Assembly.* ICRA 2025. arXiv:2407.16677. https://arxiv.org/abs/2407.16677
- **Key findings:** a frozen chunked Diffusion Policy supplies a_base; a small MLP residual runs at every control step on [s_t, a_t^base] and outputs a_t = a_t^base + a_t^res, with near-zero output initialisation (Sec. II-C). Success: one_leg 98% vs DP 54%, round_table 94% vs 12%, peg-in-hole 99% vs 5% (Tab. I). Chunk-level corrections (ResiP-C) saturate at 92% vs 98% for per-step (Sec. IV-B3). Under perturbations ResiP drops 12% vs 19–26% for chunk-based methods (Fig. 8).
- **Quality tier:** Tier 1 (ICRA 2025, DOI 10.1109/ICRA55743.2025.11127442).
- **Method weaknesses:** struggles under very high initial scene randomness because it relies on the base for macro behaviour; teacher–student and sim-to-real gaps remain (author-acknowledged, Sec. VI, App. G).
- **Relevance to our setup:** the closest evidence that a *per-step* residual on an open-loop chunk beats correcting at chunk rate, i.e. the fast loop should correct inside the plan, not only at plan arrival. Their base chunk is fresh, not stale.
- Read scope: full_text.

**[ankile2026residual] Ankile, L., Jiang, Z., Duan, R., Shi, G., Abbeel, P., Nagabandi, A. (2026).** *Residual Off-Policy RL for Finetuning Behavior Cloning Policies.* ICRA 2026, pp. 5789–5796. arXiv:2509.19301. https://arxiv.org/abs/2509.19301
- **Key findings:** per-step residual a_t = a_t^base + a_t^res on top of a frozen action-chunked BC policy; the residual sees observation + base action and is agnostic to chunk size (Sec. III-B). Off-policy residual RL converges in ~200k steps vs ~40M for PPO-based residual RL on BoxCleanup (Fig. 3). Real 29-DoF wheeled humanoid with dexterous hands: 14%→64% (WoollyBallPnP, 134 rollouts) and 23%→64% (PackageHandover, 343 episodes) (Sec. IV-B).
- **Quality tier:** Tier 1 (ICRA 2026, DOI 10.1109/ICRA57385.2026.11695959).
- **Method weaknesses:** learned behaviour stays close to the base policy; needs human resets and reward labels (author-acknowledged, Sec. VI).
- **Relevance to our setup:** confirms ResiP's per-step residual on a high-DoF real platform; the chunk-size-agnostic residual is the property we need when chunk age varies.
- Read scope: full_text.

**[xiao2026selfimproving] Xiao, W., Lin, H., Peng, A., Xue, H., He, T., Xie, Y., Hu, F., Wu, J., Luo, Z., Fan, L. "J.", Shi, G., Zhu, Y. (2026).** *Self-Improving Vision-Language-Action Models with Data Generation via Residual RL.* ICLR 2026. arXiv:2511.00091. https://arxiv.org/abs/2511.00091
- **Key findings:** ā = a_b + a_δ with a frozen VLA base (π₀ or OpenVLA) and a 3-layer, 256-unit MLP Gaussian residual conditioned on (s, a_b) (Eq. 2); a_δ is bounded to [−ξ, ξ], ξ = 0.5 (LIBERO), 0.1 (SimplerEnv) (App. B.2). LIBERO: π₀ 93.4→97.2%, OpenVLA 91.8→99.2% (Tab. 1); SimplerEnv average 71.8→96.6% (Tab. 2). The residual is then used to collect data that is distilled back into the VLA ("Probe, Learn, Distill").
- **Quality tier:** Tier 1 (ICLR 2026, OpenReview poster).
- **Method weaknesses:** no dedicated limitations section found. Real-world training needs 2–8 h per subtask and human perturbation of the setup (App. D.1); only two VLA types (reader-inferred from the reported setup).
- **Relevance to our setup:** shows a tiny MLP residual on a large VLA's actions is enough to close most of the gap on LIBERO, our benchmark. The VLA runs synchronously here, so staleness is not tested.
- Read scope: full_text.

### (d2) Delay compensation with learned models (Smith-predictor-like)

**[deng2026residual] Deng, K., Yang, Z. (2026).** *Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays.* Accepted at IFAC World Congress 2026 (per arXiv comment). arXiv:2605.15480. https://arxiv.org/abs/2605.15480
- **Key findings:** computed-torque control with PD feedback on *LSTM-predicted* leader states (a learned predictor running at 250 Hz between intermittent packets, Eq. 7), plus a SAC residual torque (Eq. 4) (Sec. 3). Observation delays drawn from U(120,160), U(200,240) or U(40,240) ms plus 50 ms constant (Sec. 3). Simulated mean tracking error 0.036–0.050 m vs 0.099–0.119 m for the model-based PMDC baseline and 0.62–0.67 m for plain PD under high delay (Tab. 3, Fig. 2), i.e. ~58–64% lower than PMDC. Real Franka Panda: 0.037 m (sim) → 0.045 m (real) with no retraining (Tab. 1).
- **Quality tier:** Tier 2 (venue only stated in the arXiv comment; not confirmed on a proceedings page).
- **Method weaknesses:** the authors criticise predictor-based baselines for step discontinuities under high-variance delay (Sec. 1.1) but do not report an ablation of predictor alone vs residual alone in the extracted text (reader-inferred). Single robot, tracking task only (reader-inferred).
- **Relevance to our setup:** the clearest example of our proposed combination: a learned forward predictor (time-aligns a delayed reference) plus a residual RL correction, evaluated against stochastic delay. Their reference is a human leader, not a learned planner.
- Read scope: full_text.

**[mompoalepuz2025learning] Mompó Alepuz, A., Papageorgiou, D., Tolu, S. (2025).** *Learning-based Delay Compensation for Enhanced Control of Assistive Soft Robots.* arXiv preprint. arXiv:2504.12428. https://arxiv.org/abs/2504.12428
- **Key findings:** a learned nonlinear Smith predictor: a Kernel Recursive Least Squares Tracker learns {x(t), u[t−d,t]} → Δx(t) online, and a Legendre Delay Network compresses the input history to p = 3 states per channel (Sec. IV-A–C). Real two-module cable-driven soft arm with 0.14 s input–output delay (Sec. III-C). RMS tracking error 9.9 mm vs 27.6 mm baseline at high gains (−64%), 8.3 vs 14.4 mm at medium gains (Tab. I).
- **Quality tier:** Tier 2 (arXiv preprint, "submitted to the IEEE").
- **Method weaknesses:** first-order model omits acceleration terms (Sec. III-B); human-interaction tests only preliminary (Sec. VI-C) (author-acknowledged). Fixed, known delay (reader-inferred).
- **Relevance to our setup:** the Legendre Delay Network is a cheap, fixed-size encoding of "actions issued since the plan was computed", which is exactly the extra input a delay-compensating student needs; fits a 10k–100k parameter budget.
- Read scope: full_text.

**[chen2021delay] Chen, B., Xu, M., Li, L., Zhao, D. (2021).** *Delay-aware model-based reinforcement learning for continuous control.* Neurocomputing 450, pp. 119–128. arXiv:2005.05440. https://arxiv.org/abs/2005.05440
- **Key findings:** delay-aware MDP with augmented state 𝒮 × 𝒜ⁿ (Def. 2); only the undelayed dynamics are learned, and the buffered actions are rolled through the learned model during planning, Smith-predictor style (Sec. IV). DATS stays stable for n = 1…16 delay steps where RTAC degrades (Tab. I); models transfer across delay lengths without retraining (Tab. I).
- **Quality tier:** Tier 2 (Neurocomputing, DOI 10.1016/j.neucom.2021.04.015).
- **Method weaknesses:** delay must be an integer number of steps; observation delay not tested empirically (author-acknowledged, Sec. II-B). Gym/MuJoCo only (reader-inferred).
- **Relevance to our setup:** argues that the delay should be handled by forward-rolling a model, not by re-learning the policy per delay, so a time-aligned plan can generalise across plan ages. Complements [DermanDelay2021] and [RamstedtDCAC2021] in Theme 2.
- Read scope: full_text.

### (d3) Nominal plan + ancillary feedback (tube / learning-based MPC)

**[tagliabue2024efficient] Tagliabue, A., How, J. P. (2024).** *Efficient Deep Learning of Robust Policies From MPC Using Imitation and Tube-Guided Data Augmentation.* IEEE T-RO 40, pp. 4301–4321. arXiv:2306.00286. https://arxiv.org/abs/2306.00286
- **Key findings:** robust tube MPC = nominal plan (x̄, ū) plus ancillary feedback u = ū* + K(x − x̄*) that keeps the state in a disturbance-invariant tube (Eq. 11, Sec. III-A, Fig. 3). Tube-guided augmentation samples states inside the tube and labels them with the ancillary controller, so no extra MPC solves are needed (Eq. 13). Resulting DNN policy: ~15 μs inference on a Jetson TX2 CPU at 500 Hz; policies from one simulated demo transfer to a real quadrotor, two demos for a 360° flip, robust to unseen wind (Sec. I-A, VI–VII).
- **Quality tier:** Tier 1 (IEEE T-RO, DOI 10.1109/TRO.2024.3431988). Builds on the ICRA 2022 version (arXiv:2109.09910, see exclusions).
- **Method weaknesses:** linearisation errors on aggressive trajectories are treated as extra process uncertainty (Sec. III-A); computing tubes analytically for nonlinear systems is hard and needs Monte-Carlo or learned approximations (Sec. IV-A) (author-acknowledged). Requires an MPC expert with a known model (reader-inferred).
- **Relevance to our setup:** the textbook decomposition we want: teacher plan = nominal trajectory, student = ancillary feedback. The tube suggests training the student on states *off* the stale plan, labelled with a feedback law, rather than only on teacher rollouts.
- Read scope: full_text.

**[koller2018learning] Koller, T., Berkenkamp, F., Turchetta, M., Krause, A. (2018).** *Learning-based Model Predictive Control for Safe Exploration.* CDC 2018, pp. 6059–6066. arXiv:1803.08287. https://arxiv.org/abs/1803.08287
- **Key findings:** GP model uncertainty is propagated as a sequence of ellipsoids around the planned trajectory (Eqs. 18–22, Corollary 1); affine feedback π_t(x) = K_t(x − p_t) + u_t contracts the tube (Eqs. 25–26), and the terminal ellipsoid must reach a safe set, giving δ-safety (Theorem 2). Inverted pendulum: horizon T = 4 balances conservatism and exploration (Figs. 3–5).
- **Quality tier:** Tier 2 (IEEE CDC 2018, DOI 10.1109/CDC.2018.8619572); narrow evidence (one simulated system).
- **Method weaknesses:** ellipsoidal over-approximation adds conservatism; generalized eigenvalue problem per step adds compute; bounded-RKHS-norm assumption; no high-dimensional experiments (reader-inferred from Sec. III–V).
- **Relevance to our setup:** gives the formal template for "uncertainty grows with plan age": the tube widens with prediction horizon, so trust in a stale plan should decay with its age. A guarantee is out of reach for a learned 100k student, but the age-dependent widening is a usable design prior.
- Read scope: full_text.

### (d4) Learned trackers on references from a slower planner

**[jenelten2024dtc] Jenelten, F., He, J., Farshidian, F., Hutter, M. (2024).** *DTC: Deep Tracking Control.* Science Robotics 9(86), eadh5401. arXiv:2309.15462. https://arxiv.org/abs/2309.15462
- **Key findings:** a trajectory optimizer (TAMOLS) supplies footholds, touch-down joint positions, contact state and time left in phase; an RL policy at 50 Hz tracks them (Materials and Methods, "Policy Observations"). In training the plan is recomputed mainly after each touch-down (~0.465 s) and after falls/pushes, so the policy sees variable plan ages, and the tracking reward uses t₀, "the time elapsed since the solution was retrieved", to index the plan (Methods). On deployment the optimizer runs in a separate thread (~400 Hz for TAMOLS) and the policy reads the most recent solution Δt = 0.02 s ahead of its time index (Methods, "Deployment"). Raising optimizer rate from 1 Hz to 50 Hz lowers failure rate by 7.11% and raises success by 4.25% (Fig. 7A, Results). Foothold tracking error 2.3 cm (Fig. 3B); learned recovery when plan and execution diverge (invisible gap, Fig. 4).
- **Quality tier:** Tier 1 (Science Robotics, DOI 10.1126/scirobotics.adh5401).
- **Method weaknesses:** not more sample-efficient than comparable RL (two weeks of training); planner may produce foothold patterns unseen in training, degrading tracking (author-acknowledged, Discussion).
- **Relevance to our setup:** the strongest prior evidence for our time-alignment idea: the tracker indexes the plan by its age and is trained with variable update periods, and Fig. 7A gives a plan-rate vs success curve (1–50 Hz) we can compare against our update-period sweep.
- Read scope: full_text.

**[kaufmann2020deep] Kaufmann, E., Loquercio, A., Ranftl, R., Müller, M., Koltun, V., Scaramuzza, D. (2020).** *Deep Drone Acrobatics.* RSS 2020. arXiv:2006.05768. https://arxiv.org/abs/2006.05768
- **Key findings:** a privileged MPC teacher with ground-truth state is distilled into a sensorimotor student that sees feature tracks, IMU (100 Hz, temporal convolutions) and the reference trajectory (sampled at 50 Hz); asynchronous branches feed an MLP at 100 Hz that outputs thrust and body rates (Sec. III-D, Fig. 4). Up to 45% lower tracking error than VIO-MPC (Tab. I). A student with *only* the reference and no sensing ("Ours (Only Ref)", effectively open loop) "perform[s] poorly across all maneuvers" (Tab. I, Sec. V). 100% success on Power Loop and Barrel Roll on hardware (Tab. III).
- **Quality tier:** Tier 1 (RSS 2020, DOI 10.15607/RSS.2020.XVI.040).
- **Method weaknesses:** reference trajectories are fixed and known in advance, not replanned or delayed (reader-inferred). No explicit limitations section found in the extracted text.
- **Relevance to our setup:** a teacher–student split with a small on-board student that tracks a reference while fusing IMU at native rate. The "Only Ref" ablation is the drone analogue of our blind-student ablation: the plan alone is not enough without feedback.
- Read scope: full_text.

**[gu2024rttrajectory] Gu, J., Kirmani, S., Wohlhart, P., Lu, Y., Gonzalez Arenas, M., Rao, K., Yu, W., Fu, C., Gopalakrishnan, K., Xu, Z., Sundaresan, P., Xu, P., Su, H., Hausman, K., Finn, C., Vuong, Q., Xiao, T. (2024).** *RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches.* ICLR 2024. arXiv:2311.01977. https://arxiv.org/abs/2311.01977
- **Key findings:** an RT-1 policy is conditioned on a 2D (or 2.5D) end-effector path sketch drawn into the camera image, with time encoded in colour and gripper events as markers (Sec. 3.2–3.3). Sketches at test time come from humans, human videos, LLM code generation or image generation (Sec. 3, Tab. 1). Unseen tasks: 67% (2.5D) / 50% (2D) vs RT-1 16.7%, RT-2 11.1%, RT-1-Goal 26% (Tab. 4).
- **Quality tier:** Tier 1 (ICLR 2024 spotlight, OpenReview).
- **Method weaknesses:** assumes a stationary robot that manipulates with the end effector only; cannot express strict spatial constraints (author-acknowledged, Sec. 5). The sketch is given once per episode, so staleness is not studied (reader-inferred).
- **Relevance to our setup:** evidence that a coarse geometric plan from a large model is a strong conditioning interface. The sketch encodes time along the path, which is the representation we would need to time-align a plan by its age.
- Read scope: full_text.

### (d5) Continuous-time / parametric action representations

**[pertsch2025fast] Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., Levine, S. (2025).** *FAST: Efficient Action Tokenization for Vision-Language-Action Models.* RSS 2025. arXiv:2501.09747. https://arxiv.org/abs/2501.09747
- **Key findings:** per-dimension DCT of a 1 s chunk, scaling/rounding of coefficients, then BPE (Sec. V-B, Alg. 1); invertible decoding. Compression vs naive binning: 1.75× at 5 Hz up to 13.2× at 50 Hz, ~30 tokens per chunk per arm (Tab. I). Naive binning degrades as sampling rate rises (Fig. 3) and fails on 20 Hz and 50 Hz real tasks where FAST works (Fig. 6). π₀-FAST needs ~750 ms per chunk vs ~100 ms for diffusion π₀ on a 4090 (Sec. VI-E).
- **Quality tier:** Tier 1 (RSS 2025, DOI 10.15607/RSS.2025.XXI.012).
- **Method weaknesses:** tested on static manipulators only; slower autoregressive inference may hurt dynamic tasks (author-acknowledged, Sec. VI-E, VII). Resampling the DCT at arbitrary times is not discussed (reader-inferred).
- **Relevance to our setup:** a DCT chunk is a continuous-time basis, so a fast student could evaluate the plan at t = age + k·Δt instead of indexing discrete steps. FAST's compression (~30 tokens/s/arm) also bounds the link payload. The 750 ms decode is itself a staleness source.
- Read scope: full_text.

**[zhou2025beast] Zhou, H., Liao, W., Huang, X., Tang, Y., Otto, F., Jia, X., Jiang, X., Hilber, S., Li, G., Wang, Q., Yağmurlu, Ö. E., Blank, N., Reuss, M., Lioutikov, R. (2025).** *BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning.* NeurIPS 2025. arXiv:2506.06072. https://arxiv.org/abs/2506.06072
- **Key findings:** N B-spline control points fitted by ridge regression (Eq. 4), quantised for discrete tokens (Fig. 2). Clamped splines set the first control point to the last action of the previous chunk, guaranteeing continuity across chunks (Sec. 4.2). The trajectory y(u) = Φ(u)c is defined for any u ∈ [0,1], so any control rate can be decoded (Sec. 4.1). 4–8× fewer tokens than binning; 617 Hz throughput, 19 ms per chunk (Tab. 3). CALVIN ABC chain success 99.8% (1 task) down to 74.4% (5 tasks) (Tab. 1); LIBERO 92.5% avg (Tab. 2).
- **Quality tier:** Tier 1 (NeurIPS 2025, OpenReview poster).
- **Method weaknesses:** sensitive to the number of basis functions (author-acknowledged, Sec. 6, App. A).
- **Relevance to our setup:** the most directly usable plan format: a fixed-size control-point vector (small link payload) that the on-robot student can evaluate at the plan's current age, with continuity clamped to the robot's current command when a new plan arrives.
- Read scope: full_text.

## 7. Unverified / excluded

### From the slice on learned filters and delays

| Item | Status | Reason |
|---|---|---|
| Deep Kalman Filters (Krishnan, Shalit, Sontag; arXiv:1511.05121) | Verified (arXiv API), excluded | Foundational deep SSM. For our mechanisms it is superseded by RKN/RSSM (explicit gain, filtering posterior); not read beyond metadata |
| Deep Variational Bayes Filters (Karl, Soelch, Bayer, van der Smagt; ICLR 2017; arXiv:1605.06432) | Verified (arXiv API), excluded | Same reason as DKF; candidate if the synthesis needs a full deep-SSM lineage |
| Particle Filter Networks (Karkus, Hsu, Lee; CoRL 2018; arXiv:1805.08975) | Verified (arXiv API), excluded | Overlaps DPF (differentiable PF for localization); particle cost is out of our budget |
| Delay-Aware Model-Based RL for Continuous Control (Chen, Xu, Li, Zhao; Neurocomputing 450, 2021; arXiv:2005.05440) | Verified (arXiv API), not assessed | Roll-forward over delay with a learned model; covered by Karamzade24 and Walsh08 for this slice. Candidate for a follow-up |
| At Human Speed: Deep RL with Action Delay (Firoiu, Ju, Tenenbaum; arXiv:1810.07286) | Verified (arXiv API), not assessed | Predictive model over action delay (games); lower priority |
| Delayed Reinforcement Learning by Imitation (Liotet, Maran, Bisi, Restelli; arXiv:2205.05569) | Verified (arXiv API), not assessed | Imitating an undelayed expert from delayed observations. Close to our teacher→student setting, so a high-priority follow-up for Theme 5/H4 |
| Optimal update with out-of-sequence measurements (Zhang, Li, Zhu; IEEE TSP 53(6), 2005; DOI 10.1109/tsp.2005.847830) | Verified (Crossref), excluded | Further classical OOSM optimality; Bar-Shalom02/04 suffice as seminal references |
| Markov decision processes with delays and asynchronous cost collection (Katsikopoulos, Engelbrecht; IEEE TAC 48(4), 2003; DOI 10.1109/tac.2003.809799) | Verified (Crossref), excluded | Augmented-state formulation for delayed MDPs; already represented via Derman21 / Ramstedt21 in Theme 2 and Wang24 here |
| State-Action Inpainting Diffuser for Continuous Control with Delay (Han, Wang, Zhang, Li; arXiv:2603.01553) | Surfaced (arXiv API), not assessed | 2026 follow-up of Wang24; diffusion-based delay handling |
| The Role of Time Delay in Sim2real Transfer of RL for Cyber-Physical Systems (Chehadeh, Boiko, Zweiri; arXiv:2209.15216) | Surfaced, not assessed | Seen in search only |
| Learning-based Delay Compensation for Assistive Soft Robots (arXiv:2504.12428); Delay-aware Diffusion Policy (DA-DP) | Surfaced via WebSearch, not verified | Not fetched from arXiv in this pass |
| MathWorks "Handle out-of-sequence measurements with filter retrodiction" | Grey literature, excluded from bib | Vendor documentation; used only as context for the retrodiction-vs-reprocessing trade-off |
| Bar-Shalom 2000 SPIE conference version (DOI 10.1117/12.392006) | Verified (Crossref), not cited separately | Conference version of Bar-Shalom02 |

### From the slice on innovation gating and event triggers

| Item | Status | Reason |
|---|---|---|
| Mehra & Peschon (1971), *An innovations approach to fault detection and diagnosis in dynamic systems*, Automatica 7(5):637–640, DOI 10.1016/0005-1098(71)90028-8 | Verified (Crossref), not read | Origin of the innovations chi-squared test. No abstract or full text was accessible. Its content is cited only through [Willsky1976] Sec. VII. Add it once the author reads it |
| Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, DOI 10.1002/0471221279 | Verified (Crossref), not read | Standard reference for NIS, filter-consistency tests and validation gates. Contents were not accessible, so no locator could be given. Cite only after the author checks the chapter |
| Heemels, Johansson & Tabuada (2012), *An introduction to event-triggered and self-triggered control*, IEEE CDC, pp. 3270–3285, DOI 10.1109/CDC.2012.6425820 | Verified (Crossref), not read | Requested overview. IEEE abstract elided, and the open-access copy (KTH DiVA) sat behind a bot wall. Good candidate once read |
| Trimpe & D'Andrea (2014), *Event-Based State Estimation With Variance-Based Triggering*, IEEE TAC 59(12):3266–3281, DOI 10.1109/TAC.2014.2351951 | Verified (Crossref), not read | Paywalled. Its variance-based vs innovation-based distinction is cited only through [Trimpe-EBSE] Sec. 1 |
| Haddadin, De Luca & Albu-Schäffer (2017), *Robot Collisions: A Survey on Detection, Isolation, and Identification*, IEEE T-RO 33(6):1292–1312, DOI 10.1109/TRO.2017.2723903 | Verified (Crossref), not read | Momentum-observer residuals are the classical "torque innovation". The postprint link returned HTML. High priority to read |
| Heo et al. (2019), *Collision Detection for Industrial Collaborative Robots: A Deep Learning Approach*, IEEE RA-L 4(2):740–746, DOI 10.1109/LRA.2019.2893400 | Verified (Crossref), not read | Learned collision detection from joint signals. Not read in this pass |
| FINO-Net (Inceoglu et al., IROS 2021, arXiv:2011.05817, DOI 10.1109/IROS51168.2021.9636455) | Verified, not assessed | Multimodal failure detection. Overlaps with [LSTM-VAE] and [Grambow-AD] |
| FIPER, Römer et al., *Failure Prediction at Runtime for Generative Robot Policies* (arXiv:2510.09459, NeurIPS 2025 per Crossref DOI 10.52202/085713-0260) | Verified, not assessed | Action-chunk entropy + RND with conformal calibration. Overlaps with Sentinel/FAIL-Detect. Strong candidate for a follow-up pass |
| INSIGHT (Karli et al., arXiv:2510.01389, ICRA 2026 DOI 10.1109/ICRA57385.2026.11697385) | Verified (arXiv), not assessed | Token-level uncertainty → help triggers in VLAs. Only the search snippet was read |
| TOWN-VLA, *Think Only When Needed* (Zhou et al., arXiv:2608.23224) | Verified, excluded | Read in full. The slow-path trigger is oracle/task-label based ("Task-Prior Admission", Eq. 9). The learned gate authorizes only 2/36 held-out cells (Table 6) |
| *Fast and Accurate: An Adaptive VLA Inference Framework...* (Sun et al., arXiv:2608.06434) | Verified (arXiv), not assessed | Model selection rather than query timing |
| *Slow Brain, Fast Planner* (Peng et al., arXiv:2606.20458) | Verified (arXiv), not assessed | Navigation. Search snippet mentions querying the VLM only when the planner is uncertain. Candidate |
| *When in Doubt, Plan It Out* (Gavenski et al., arXiv:2606.16995, ICML 2026 workshop) | Verified (arXiv), excluded | Workshop paper on small-LM deliberation in RL. No embodiment check |
| Speculative Verification for VLA (Wang et al., arXiv:2604.02965) | Verified (arXiv), not assessed | Open-loop planning with closed-loop verification. Possibly relevant to plan-vs-observation checks |
| Proactive slip control (Nazari et al., arXiv:2209.06019) | Verified (arXiv), not assessed | Learned action-conditioned slip predictor (tactile) |
| FACTR 2 (arXiv:2606.12406) | Surfaced, not assessed | Learned free-space torque model, so that external torque is a residual. Same idea as a torque innovation |
| Force/Torque-Based Kinematic Adaptation (arXiv:2608.21592) | Surfaced, not assessed | Uses F/T prediction error for adaptation |
| ML in Event-Triggered Control survey (Sedghi et al., arXiv:2009.12783) | Verified (arXiv), not assessed | Survey of learned ETC. Read it before adding |
| Solowjow et al., *Event-Triggered Learning for Resource-Efficient Networked Control*, ACC 2018, DOI 10.23919/ACC.2018.8431102 | Verified (Crossref), excluded | Earlier conference version of [ETL] |
| Recurrent Kalman Networks; Backprop KF | Not searched | Learned Kalman filters probably belong to the sibling slice on Kalman-inspired architectures |

### From the slice on plan + learned feedback

| Item | Status | Reason |
|---|---|---|
| Tagliabue, Kim, Everett, How, *Demonstration-Efficient Guided Policy Search via Imitation of Robust Tube MPC* (arXiv:2109.09910; ICRA 2022 per arXiv comment "submitted to") | Verified (arXiv API), excluded | Superseded by the T-RO journal version [tagliabue2024efficient]; ICRA acceptance not confirmed in this session |
| Tagliabue et al., output-feedback tube MPC data augmentation (arXiv:2210.10127) | Surfaced, not assessed | Seen in search only; same line as above |
| Scheikl et al., *Movement Primitive Diffusion* (RA-L 9, 2024, DOI 10.1109/LRA.2024.3382529; arXiv:2312.10008) | Verified (arXiv API + Crossref), excluded | ar5iv conversion failed and arXiv HTML returned 404, so not read; the continuous-time/boundary-condition claim could not be checked. Candidate for a follow-up pass |
| Li et al., *ProDMPs* (arXiv:2210.01531) | Verified (arXiv API), not read | Representation paper; superseded for our purpose by BEAST; IEEE RA-L venue only from a lab web page (not confirmed) |
| ChunkFlow (arXiv:2607.12992) | Surfaced, not assessed | Chunk-seam continuity; belongs with RTC/Legato in Theme 2 |
| J-PARC (arXiv:2607.01804) | Surfaced, not assessed | Residual calibrator on a frozen VLA for joint faults; seen via aggregator only, not fetched |
| VLA-Corrector | Surfaced, not assessed | Detect-and-truncate stale chunks; no arXiv ID resolved in this session |
| HiWET (arXiv:2602.06341), LaCoLoco, Hierarchical VL planning for humanoids (arXiv:2506.22827) | Surfaced, not assessed | Hierarchical planner→whole-body tracker; search snippets only |
| *Neural Network-based Smith predictor* (RBF teleoperation, koreascience.or.kr) | Unverified, excluded | No DOI/arXiv resolved; pre-learning-era bilateral control |
| Koller et al. extensions / Hewing et al. learning-based MPC review | Not searched | Out of time budget; Koller covers the tube mechanism |
| PMDC baseline cited in [deng2026residual] | Not resolved | Name only, from the paper text; no ID fetched |
