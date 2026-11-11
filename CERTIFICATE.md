# Hugging Face Deep RL Certificate checklist

Owner: [panuza007](https://huggingface.co/panuza007)

The official [certificate instructions](https://huggingface.co/learn/deep-rl-course/en/communication/certification) require **9 of 11 assignments (80%)** for completion. The [progress checker](https://huggingface.co/spaces/ThomasSimonini/Check-my-progress-Deep-RL-Course) looks for **public model repositories on the Hugging Face Hub**, with the expected environment and library tags in the model card and the score **mean_reward - std_reward** meeting the threshold. Code committed to GitHub alone does not count.

| Assignment | Checker environment tag | Library tag | Minimum score | Hub verification |
| --- | --- | --- | ---: | --- |
| Unit 1 | `LunarLander-v2` | `stable-baselines3` | 200 | Pending |
| Unit 2 | `Taxi-v3` | `q-learning` | 4 | Pending |
| Unit 3 | `SpaceInvadersNoFrameskip-v4` | `stable-baselines3` | 200 | Pending |
| Unit 4 / CartPole | `CartPole-v1` | `reinforce` | 350 | Pending |
| Unit 4 / Pixelcopter | `Pixelcopter-PLE-v0` | `reinforce` | 5 | Pending |
| Unit 5 / SnowballTarget | `ML-Agents-SnowballTarget` | `ml-agents` | -100 | Pending |
| Unit 5 / Pyramids | `ML-Agents-Pyramids` | `ml-agents` | -100 | Pending |
| Unit 6 | `PandaReachDense-v2` or `-v3` | `stable-baselines3` | -3.5 | Pending |
| Unit 7 / SoccerTwos | `ML-Agents-SoccerTwos` | `ml-agents` | -100 | Pending |
| Unit 8 / Part I | `LunarLander-v2` | `deep-rl-course` | -500 | Pending |
| Unit 8 / Part II | `doom_health_gathering_supreme` | `sample-factory` | 5 | Pending |

**Notes**
- Unit 2's FrozenLake exercise is useful practice but is **not** an assignment tracked by this checker; Taxi-v3 is the tracked model.
- For assignments with a minimum of -100, the course describes them as upload-only tasks, but model-card metadata still needs to be discoverable.
- The official course reports that Unit 7 (AI vs AI) is not functional. It may be necessary to pass 9 of the remaining 10 tasks.
- Unit 1 and Unit 8 Part I share an environment tag but use **different library tags**; one SB3 model does not satisfy both.
- `Pending` means **not verified**, not failed. Training, Hub upload, and certificate issuance have **not** been performed by the GitHub edits.

## Local workflow

1. Keep the existing `ml` environment unchanged. For each incompatible unit, use a separate Conda environment; inspect requirements before installing.
2. Unit 1 requires the older `LunarLander-v2` stack; use its `requirements-unit1.txt` and the notebook's conda-forge Box2D instructions.
3. For Unit 2, use a fresh Python 3.11 Conda environment and run `python -m pip install -r notebooks/unit2/requirements-unit2.txt`; ensure the Jupyter kernel points to that environment.
4. Run the notebook, evaluate the trained agent, and **only then** upload to the Hub with your own credentials. Do not commit Hub tokens.
5. Use the [progress checker](https://huggingface.co/spaces/ThomasSimonini/Check-my-progress-Deep-RL-Course) with username `panuza007` and record the actual passes here. Generate the certificate from the [certificate page](https://huggingface.co/spaces/huggingface-projects/Deep-RL-Course-Certification/) after reaching the threshold.

## Implementation references

- [Official course](https://github.com/huggingface/deep-rl-class)
- [chavicoski/Deep-RL-course](https://github.com/chavicoski/Deep-RL-course) — independent local scripts and dependency isolation by unit
- [chizkidd/huggingface-deep-RL-course](https://github.com/chizkidd/huggingface-deep-RL-course) — completed-unit examples and Hub publishing links

These are references only; the checker and the actual published artifacts are authoritative.
