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
- 2026-11-02: Build the M2 adapter with `peft.inject_adapter_in_model` rather than `wrap_with_peft`, because the policy keeps its class and every helper in `models/load.py` works unchanged.
- 2026-11-02: M2 checkpoints hold only the adapter, and the evaluator bakes it onto S1 in memory, because a full copy per checkpoint costs 1.8 GB and the baked result is tested against PEFT's own merge.
- 2026-11-02: Rolling resume state lives in `checkpoints/latest`, swapped in by directory rename, because a job killed mid-save must not corrupt the only resume point.
- 2026-11-02: Stage-2 reuses S1's saved normalization stats, because any change in action normalization would shift every comparison against S1 and would make weight merges inconsistent.
- 2026-11-02: Stage-1 stats are dataset-level (all 40 tasks), because they ship with the pinned revision; this leaks summary statistics of libero_10 into S1 but no demonstrations.
- 2026-11-02: [spec-conflict] M5 tunes lambda over {1e-2, 1e-1, 1} and drops 1e-3 from tuning, because the brief lists 4 values but allows 3 trials per method, and with a mean-squared penalty 1e-3 is closest to plain M1.
- 2026-11-02: M3's 3 trials are (2%, 3e-5), (10%, 3e-5), (10%, 1e-4), because the budget cannot cover the full pool-by-learning-rate grid.
- 2026-11-02: M0, M1, M3, and M5 share the M1 trainable set where they train the LM, and use learning rate 3e-5 unless tuned, because it is the middle of the brief's grid.
- 2026-11-02: Stage-1 trains expert-only at learning rate 1e-4, because that is LeRobot's default and the setting of the public `smolvla_libero` checkpoint; the first LM movement then happens in Stage-2, which gives M1 a clean contrast with M0.
- 2026-11-02: Eval replans every 10 steps (`n_action_steps=10`), because replanning on every step multiplies denoising cost and a full chunk of 50 reacts too slowly; the value is a flag and the smoke test reports its cost.
- 2026-11-02: Envs run at 256x256 rather than the env-config default of 360, to match the dataset's frames.
- 2026-11-02: libero_goal success is read from the retention rows, because those are the same episodes and rerunning them would cost 100 rollouts.
- 2026-11-02: The instruction swap gives each goal task the next task's instruction and scores against the swapped goal, because all 10 goal tasks share one scene, so the result measures whether the policy follows language.
- 2026-11-02: A checkpoint's eval totals 750 rollouts (300 + 200 + 200 + 50), slightly above the brief's approximate 700, because the swap test adds 50.
- 2026-11-02: Validation uses init states 40-44 on every task (200 rollouts), because a merge sweep repeats it for each alpha and the 20-episode budget would cost too much; test sets use indices below 20, so the two are disjoint.
- 2026-11-02: Robustness tasks are 3 spatial, 4 object, 3 goal at evenly spaced ids, because the brief asks for 10 fixed Stage-1 tasks and even spacing avoids hand-picking.
- 2026-11-02: Camera perturbation moves only the agentview camera, because the wrist camera is rigidly mounted on the robot.
