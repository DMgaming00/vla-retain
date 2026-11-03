# Decisions

One line per decision, with the date and the reason. Lines tagged `[spec-conflict]` differ from the original project brief because the code at the pinned commit, or the brief's own numbers, required it.

- 2026-11-02: Pin LeRobot `v0.6.1` (`7e241bd6`) because it is the latest tagged release and carries the SmolVLA fixes we rely on.
- 2026-11-02: [spec-conflict] Support Python 3.12 only, not 3.10-3.12, because LeRobot `v0.6.1` declares `requires-python >=3.12`.
- 2026-11-02: [spec-conflict] Under M1, LM layer 15, `text_model.norm`, and `lm_head` stay frozen, because LeRobot's `set_requires_grad` freezes them whenever `train_expert_only=False`.
- 2026-11-02: Upcast every parameter to float32 on load, because the base checkpoint stores the LM and expert in bfloat16, where AdamW updates at our learning rates round to zero.
- 2026-11-02: Use `lerobot/libero` at a pinned revision rather than `HuggingFaceVLA/libero`, because it holds the same demonstrations in a 1.9 GB video format instead of 69.9 GB.
- 2026-11-02: Use `HuggingFaceVLA/smolvla_libero` as the smoke-test checkpoint, because its features match LIBERO keys and it ships processors.
- 2026-11-02: Use our own perturbation wrapper, not LIBERO-Plus, because LIBERO-Plus replaces `hf-libero` in the environment, needs system packages we cannot assume on the cluster, and would force a second environment for robustness evals.
- 2026-11-02: Add an MIT `LICENSE` because the repository had none.
- 2026-11-02: Commit `splits/episodes.json` because it derives only from the pinned dataset revision and removes a cluster step.
- 2026-11-02: A 2% replay pool has fewer episodes than Stage-1 has tasks, so some tasks get none; we keep the exact 2% from the brief rather than inflating it to one episode per task.
- 2026-11-02: Replay batches use error diffusion (6 or 7 of 32), because 20% of 32 is not an integer and this keeps the long-run share at exactly 20%.
- 2026-11-02: The batch sampler and per-step RNG are pure functions of (seed, step), because this makes resume exact without saving sampler or RNG state.
- 2026-11-02: LM groups are thirds (early 0-4, mid 5-10, late 11-15); `embed_tokens` joins early, and `text_model.norm` and `lm_head` join late; `state_proj` joins the expert because M0 trains it.
- 2026-11-02: In per-module merges the connector follows the LM, because it feeds the LM and the brief's grid has only LM and expert axes.
