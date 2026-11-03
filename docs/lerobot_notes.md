# LeRobot API notes

We pin LeRobot to release `v0.6.1`, commit `7e241bd630a3719a56157a497ce5d08f244784f1`. Every path below is
relative to `src/lerobot/` at that commit. We read the source rather than the documentation wherever they could disagree.

## Environment pins

| Item | Value | Where |
| --- | --- | --- |
| Python | `>=3.12` | `pyproject.toml:32` |
| torch | `>=2.7,<2.12` (resolved locally to 2.11.0) | `pyproject.toml:62` |
| transformers | `>=5.4.0,<5.6.0` (resolved to 5.5.4) | `pyproject.toml:152` |
| peft | `>=0.18,<1` (resolved to 0.21.2) | `pyproject.toml:156` |
| LIBERO | `hf-libero>=0.1.4,<0.2.0`, Linux only | `pyproject.toml:271` |

## SmolVLA configuration (`policies/smolvla/configuration_smolvla.py`)

| Field | Default | Line | Note |
| --- | --- | --- | --- |
| `freeze_vision_encoder` | `True` | 69 | Freezes `vlm.model.vision_model`. |
| `train_expert_only` | `True` | 70 | Freezes the whole VLM. |
| `train_state_proj` | `True` | 71 | `state_proj` stays trainable under M0. |
| `num_steps` | 10 | 63 | Euler denoising steps. |
| `chunk_size`, `n_action_steps` | 50, 50 | 29-30 | The queue length follows `n_action_steps`. |
| `max_state_dim`, `max_action_dim` | 32, 32 | 41-42 | Inputs are padded, so LIBERO (8, 7) loads the base weights without shape changes. |
| `num_vlm_layers` | 16 | 96 | The VLM is truncated to its first 16 layers. |
| `load_vlm_weights` | `False` (base config: `True`) | 85 | If true, SmolVLM2 weights are downloaded before the checkpoint overwrites them. We set it to false when loading a full checkpoint. |
| `pad_language_to`, `tokenizer_max_length` | `"longest"`, 48 | 93, 60 | Base processor config pads to `max_length`. |

## Freezing logic (`policies/smolvla/smolvlm_with_expert.py`)

- `set_requires_grad` is at lines 152-191.
- With `train_expert_only=False`, LeRobot still freezes `lm_head`, `text_model.norm.weight`, and **all of LM layer 15**, the
  last LM layer (lines 163-181). Hence "LM trains" means layers 0-14 plus `embed_tokens` (see `docs/decisions.md`).
- `train()` puts frozen sub-modules back in eval mode (lines 193-200).

## Loss and forward signature (`policies/smolvla/modeling_smolvla.py`)

- `SmolVLAPolicy.forward(batch, noise=None, time=None, reduction="mean")` returns `(loss, loss_dict)` (lines 277-332).
- Flow-matching convention (lines 699-701): `x_t = t*noise + (1-t)*actions`, target `u_t = noise - actions`. Time `t=1` is pure noise.
- Time sampling: `Beta(1.5, 1.0)*0.999 + 0.001` (line 550).
- `noise` must have the padded shape `(B, chunk_size, max_action_dim)`, because `prepare_action` pads actions before `model.forward`.

## Action sampling with caller-supplied noise

- `predict_action_chunk(batch, noise=None, **kwargs)` and `select_action(batch, noise=None, **kwargs)` both accept noise (lines 232-269). No subclass is needed.
- `VLAFlowMatching.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None)` is at lines 726-773.
- Integration is forward Euler from `t=1` to `t=0` with `config.num_steps` steps (`policies/common/flow_matching.py:61-122`).
- Output is in **normalized** action space, unpadded to `action_feature.shape[0]` (lines 216-218). All our divergences are measured there.

## Velocity at a given t

- Not exposed as a single call. `model.forward` computes it internally (lines 702-722) and only returns losses.
- We rebuild it from public pieces in `src/vlaretain/models/load.py::predict_velocity`: `embed_prefix`, `embed_suffix`,
  `make_att_2d_masks`, `vlm_with_expert.forward`, `action_out_proj`. No LeRobot file is edited.

## Per-layer hidden states

- `SmolVLMWithExpertModel.forward` (lines 410-509) loops over layers manually and returns only final embeddings. It never calls
  `layer.forward`, so a hook on the layer module does not fire.
- The residual stream entering layer `i` is the input of `layer.input_layernorm` (line 239); the stream leaving the last layer
  is the input of `text_model.norm` (line 505). We capture hidden states with forward pre-hooks on these modules.
- A prefix-only forward must use `use_cache=True`, as in `sample_actions` (lines 750-756). With `use_cache=False` and
  `inputs_embeds=[prefix, None]`, `forward_cross_attn_layer` dereferences the missing expert embeddings (line 349).

## Parameters of `lerobot/smolvla_base`

Revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, measured on CPU:

| Module | Parameters |
| --- | --- |
| `vlm.model.vision_model` | 86,433,024 |
| `vlm.model.connector` | 11,796,480 |
| `vlm.model.text_model` (16 layers, embeddings, norm) | 204,626,880 |
| `vlm.lm_head` | 47,308,800 |
| `lm_expert` + projections | 99,880,992 |
| **Total** | **450,046,176** |

- Stored dtypes are mixed: 474 tensors are `BF16`, 26 are `F32`. After loading, the text model and the expert are in `bfloat16`.
- The base checkpoint's features are `camera1..3`, state dim 6, action dim 6. LIBERO uses `image`, `image2`, state dim 8, action dim 7.
  `make_policy` keeps pretrained `input_features` if already set (`policies/factory.py`, inside `make_policy`), so we set features explicitly in `models/load.py`.

## Processors

- `make_smolvla_pre_post_processors` (`policies/smolvla/processor_smolvla.py:33-82`): rename, batch, newline, tokenize, device, normalize; output: unnormalize, CPU.
- `make_pre_post_processors(policy_cfg, pretrained_path=None, dataset_stats=...)` builds fresh pipelines (`policies/factory.py:150-237`).
- `normalization_mapping`: identity for images, mean/std for state and action (`configuration_smolvla.py:32-38`).
- Environment observations also pass through `LiberoProcessorStep`, returned by `LiberoEnv.get_env_processors` (`envs/configs.py:447-451`). It flips images 180 degrees and builds the 8-dimensional state.

## LIBERO environment (`envs/libero.py`)

- `LiberoEnv.__init__` is at lines 110-132. Default observation size is 256 in the class but 360 in `LiberoEnv` config (`envs/configs.py:333-334`). The dataset is 256, so we construct the env directly at 256.
- **Init-state selection**: `init_state_id = episode_index`, then `init_states[init_state_id % len]`, advanced by `n_envs` on every reset (lines 173-175, 344-346). We pass explicit indices with `n_envs=1`.
- Each `.pruned_init` file holds 50 states of dimension 92 for `libero_spatial`; the tensor shape was read from the pickle without executing it.
- Max steps: `TASK_SUITE_MAX_STEPS`, lines 98-104 (spatial 280, object 280, goal 300, libero_10 520).
- `create_libero_envs` is at lines 445-532. Its async path uses `_LazyAsyncVectorEnv`, which builds `AsyncVectorEnv(context="forkserver", shared_memory=True, autoreset_mode=NEXT_STEP)` with `freeze_after_episode_end` (`envs/utils.py:285-293`).
- `hard_reset` defaults to `True`. Soft resets are faster but not bit-identical (`docs/source/libero.mdx`).
- Goal evaluation: `parsed_problem["goal_state"]` is a conjunction evaluated by `_eval_predicate` (`libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py:136-144` in `hf-libero` 0.1.4). All 10 `libero_goal` BDDL files have identical fixtures and objects (we hashed those sections), so a goal from one task can be evaluated in another.

## `lerobot-eval` (`scripts/lerobot_eval.py`)

- `rollout(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, seeds, render_callback, ...)` is at lines 167-411. We call it directly.
- Step order (lines 274-302): `preprocess_observation`, add `task`, `env_preprocessor`, `preprocessor`, `policy.select_action`, `postprocessor`, `env_postprocessor`.
- The task string comes from `env.call("task_description")` (lines 279-281). Our instruction-swap test overrides this attribute in a subclass.
- Success accounting (lines 553-566): mask after the first done, then `any` over the episode.

## `lerobot-train` (`scripts/lerobot_train.py`)

- Setup: lines 282-400. `make_policy` then optional `wrap_with_peft` (lines 325-341), processors with dataset stats (lines 349-400).
- We do not call `lerobot-train`: we need method losses, replay mixing, and stateless resume. We reuse `make_pre_post_processors`, `LeRobotDataset`, and `resolve_delta_timestamps` (`datasets/factory.py:34`).
- PEFT defaults target only the expert's q and v projections (`modeling_smolvla.py:420-429`). Our M2 passes its own targets through `peft.inject_adapter_in_model`.

## Dataset

- `lerobot/libero`, revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`: 1,693 episodes, 273,465 frames, 40 tasks, two 256x256 video streams, state 8, action 7, format v3.0, `fps=10`.
- All 40 `meta/tasks.parquet` strings match the language strings derived from `libero_suite_task_map.py`. We store the suite table in `src/vlaretain/data/libero_tasks.json`.
- `meta/episodes` has no `tasks` column in this revision, so the episode-to-task map is read from the `task_index` column of the data parquet files.
- `LeRobotDataset(repo_id, episodes=[...], delta_timestamps=..., revision=...)` is at `datasets/lerobot_dataset.py:47-70`. Items carry `task` from `dataset_reader.py:352`.

## Smoke-test checkpoint

`HuggingFaceVLA/smolvla_libero`, revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`: features `observation.images.image`, `observation.images.image2`, state 8, action 7, with saved processors. Its config sets `n_action_steps=1`.

## LIBERO-Plus

`docs/source/libero_plus.mdx`: installing it **replaces** `hf-libero`, it must be cloned into `PYTHONPATH`, and it needs `libmagickwand-dev` and a separate asset download. See `docs/decisions.md` for our choice.
