# Cross-embodiment training data for a VLA: how heterogeneous actions are handled (quick brief)

2026-11-02. Question (author): should the FAST fine-tune of `pi05_base` use several embodiments and action formats with a robot-type condition, rather than converting everything to LIBERO's format?

Method: quick brief (ARS deep-research, quick mode). A subagent fetched each paper from arXiv (abs + HTML full text) and confirmed the IDs; the main session re-checked π0.5 §IV-C and X-VLA Table 1 against the fetched text. Sources are not yet in `literature/bib/`; add them there (fetched metadata) before citing any of this in a paper. AI-assisted: retrieval, extraction and drafting by Claude; claims carry locators for checking.

## Sources (arXiv, fetched)

Octo 2405.12213 · OpenVLA 2406.09246 · π0 2410.24164 · π0.5 2504.16054 · FAST 2501.09747 · CrossFormer 2408.11812 · HPT 2409.20537 · RDT-1B 2410.07864 · GR00T N1 2503.14734 · Open X-Embodiment / RT-X 2310.08864 · X-VLA 2510.10274 · RoboVLMs 2412.14058.

## Four ways the field handles heterogeneous actions

1. **Restrict or standardize to one format.** Octo trains only on OXE datasets with delta end-effector control and aligns the gripper (+1 open, 0 closed) (2405.12213 §III-B). OpenVLA keeps only single-arm EEF data "to ensure a coherent input and output space" (2406.09246 §3.3). X-VLA converts everything to EEF xyz + absolute 6D rotation + binary gripper (2510.10274 §4.2.2). GR00T N1 standardizes as far as possible (axis-angle actions, min-max normalization, fixed field order) (2503.14734 App. E.2).
2. **Keep native formats in one padded vector, normalized per dataset.** RT-X aligns only coarsely to a 7-d vector; frames are not aligned and values may be absolute or relative positions or velocities, de-normalized per embodiment (2310.08864 §IV-A). π0 zero-pads state and action to the largest robot (2410.24164 §V-A). **π0.5** predicts joint *or* end-effector targets, normalizes "using the 1% and 99% quantile of each action dimension of the individual dataset", pads to the largest action space, and marks the format in the prompt (2504.16054 §IV-C). FAST+ was trained on ~1M one-second chunks spanning joint and EEF control at various rates, padded to 32 dims, with no embodiment conditioning in the tokenizer; it matches per-dataset tokenizers (2501.09747 §V-C, App. A, §VI-C Fig. 6).
3. **Per-embodiment heads, stems or encoders.** CrossFormer: one readout/head per action type (single-arm delta Cartesian, navigation, bimanual joints, quadruped joints) (2408.11812 §3.3). HPT: per-embodiment stems and heads around a shared trunk, actions normalized to [−1, 1] per dataset (2409.20537 §3.1, App. A.2). GR00T N1: per-embodiment state/action encoders and decoders (2503.14734 §2.1).
4. **Put the embodiment or format into the input.** π0.5: "‹control_mode› joint/end effector ‹control_mode›" in the text prompt (2504.16054 §IV-C). RDT-1B: control frequency as an input, a 128-d unified physical action space with an availability mask, unified units instead of [−1, 1] scaling (2410.07864 App. C, D, F). X-VLA: learned per-source soft prompts (2510.10274 §3).

## Evidence

- **Native heterogeneous formats can work.** RT-X shows positive transfer despite unaligned, mixed absolute/relative actions (2310.08864 §V-A Fig. 3, §V-B Table II). CrossFormer reports no negative transfer with native formats per head, and beats a manually aligned baseline by ~3× (2408.11812 §4.1–4.2, Figs. 5–6). π0 / π0.5 / FAST+ train on mixed joint and EEF data (above).
- **…but naive mixing can hurt.** X-VLA's cumulative ablation (2510.10274 Table 1): heterogeneous pretraining without further measures lowers adaptation accuracy from 39.6 to 25.0; adding action alignment, intention abstraction and balanced sampling (grouped in one row) brings it to 50.0.
- **Conditioning on the source/embodiment: suggestive, not cleanly tested.** No source ablates the same model with vs without a robot-type condition. X-VLA's soft prompts add +9.2 (64.6 → 73.8) as one step of a cumulative ablation (Table 1), and give more stable training than per-embodiment heads, HPT-style projections or language descriptions of the hardware (Fig. 4, training curves, qualitative). π0.5 adds the control mode to the prompt without ablating it (§IV-C).
- **Other embodiments help the target robot**, with caveats: π0.5 loses performance without its multi-/cross-embodiment data (2504.16054 §V-C Fig. 10); Octo's 25-dataset mix beats Bridge-only on WidowX (2405.12213 Table II); OpenVLA (2406.09246 App. D.1 Table 9); RDT pretraining vs scratch (2410.07864 Table 2). RT-X underfits large domains (2310.08864 Table I). RoboVLMs: OXE co-training helps mainly when followed by target post-training, and in-domain data from the same robot beats OXE (2412.14058 §II-E).
- **Common practice regardless of strategy:** per-dataset normalization to [−1, 1] (OXE §IV-A, OpenVLA §3.2, FAST §V-B, π0.5 §IV-C, HPT App. A.2, GR00T App. E.2; RDT is the exception); padding to a max action dim (π0 §V-A, π0.5 §IV-C, FAST+ App. A); size-dependent mixture weights (π0 n^0.43 §V-A; HPT √n App. A.3).

## Recommendation for EXP-0005's fine-tune

The author's proposal matches how `pi05_base` itself was trained (strategy 2 + 4), so fine-tuning the same way changes the least:

1. **Keep each dataset's native action format**; normalize per dataset to [−1, 1] by the 1%/99% quantiles (π0.5 §IV-C; FAST+ card); pad to the backbone's action dimension. This also makes absolute-EEF and joint-space datasets usable without conversion (e.g. DROID's joint actions, `iamlab`), widening the mixture beyond the six delta-EEF datasets.
2. **Condition in the text prompt**, in π0.5's style, which the LeRobot port already supports (the prompt is text): robot, control mode, and rate, e.g. `robot: franka; control: delta end effector; 20 Hz`. Since `robot_type` is "unknown" in the LeRobot OXE ports, fill these from the OXE spreadsheet (`oxe_action_formats.md`). Rate as a condition follows RDT-1B (App. D).
3. **Still align what is cheap and unambiguous**: gripper sign (Octo §III-B), and exclude datasets with conflicting documentation (`taco_play` gripper, `toto` action space) rather than guess.
4. **Weight LIBERO up** (in-domain data beats OXE, RoboVLMs §II-E) and the rest by a size-dependent rule (π0 n^0.43).
5. **Expect no clean prior answer** on whether the condition matters; X-VLA's +9.2 is the closest evidence. If it matters for us, it shows in `slow-only` on LIBERO; it is not this experiment's question.

## Limitations

Quick brief, 12 sources, no systematic search. Several numbers (π0.5 Fig. 10, X-VLA Fig. 4) were read from charts qualitatively. Not checked: how the openpi / LeRobot `pi05_base` code pads actions and formats the prompt (the 18-d figure is π0's); Octo's and CrossFormer's normalization (not stated in the papers).
