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
## Dedicated environments for remaining Units

Run these commands **from the repository root**. Existing `hf-rl-unit1`, `hf-rl-unit2`, and `hf-rl-unity` can be reused if their versions and `pip check` pass; do not reinstall into the shared `ml` environment.

### Unit 3 (Atari DQN)

```bash
conda create -n hf-rl-atari python=3.11 -y
conda activate hf-rl-atari
python -m pip install -r notebooks/unit3/requirements-unit3.txt
python -m ipykernel install --user --name hf-rl-atari --display-name "hf-rl-atari"
```

### Unit 4 (REINFORCE)

```bash
conda create -n hf-rl-reinforce python=3.10 -y
conda activate hf-rl-reinforce
python -m pip install -r notebooks/unit4/requirements-unit4.txt
python -m ipykernel install --user --name hf-rl-reinforce --display-name "hf-rl-reinforce"
```

### Unit 5 (Unity ML-Agents)

```bash
conda create -n hf-rl-unity python=3.10 -y
conda activate hf-rl-unity
python -m pip install -r notebooks/unit5/requirements-unit5.txt
python -m ipykernel install --user --name hf-rl-unity --display-name "hf-rl-unity"
```

### Unit 6 (PandaGym A2C)

```bash
conda create -n hf-rl-panda python=3.11 -y
conda activate hf-rl-panda
python -m pip install -r notebooks/unit6/requirements-unit6.txt
python -m ipykernel install --user --name hf-rl-panda --display-name "hf-rl-panda"
```

### Unit 8 Part I (PPO from scratch)

```bash
conda create -n hf-rl-ppo8 python=3.10 -y
conda activate hf-rl-ppo8
python -m pip install -r notebooks/unit8/requirements-unit8-part1.txt
python -m ipykernel install --user --name hf-rl-ppo8 --display-name "hf-rl-ppo8"
```

### Unit 8 Part II (ViZDoom)

```bash
conda create -n hf-rl-vizdoom python=3.10 -y
conda activate hf-rl-vizdoom
python -m pip install -r notebooks/unit8/requirements-unit8-part2.txt
python -m ipykernel install --user --name hf-rl-vizdoom --display-name "hf-rl-vizdoom"
```

- For Unit 1 and Unit 8 Part I: install Box2D and SWIG in the respective Conda environments with `conda install -c conda-forge box2d-py swig` before installing their notebook requirements if building Box2D from pip fails.
- Unit 3: from `notebooks/unit3`, the training cell references the provided `dqn.yml`.
- Unit 5: the notebook downloads Unity executables; the YAML files are in `notebooks/unit5/config/`. These binaries and Linux permissions are not available in the GitHub repository by default.
- Unit 8 Part I: from `notebooks/unit8`, the notebook executes the standalone `ppo.py` and uploads only after training and authentication.
- Unit 7: the original AI-vs-AI feature is currently not functional. Completion requires 9 other passes; an excellence certificate still requires all 11.
- Verify each new interpreter/kernel, `python -m pip check`, environment reset/step, training and upload **on the actual Ubuntu laptop**. These have not been run by the GitHub-only changes.

**Do not upload a model card with invented scores.** Publish genuine evaluated `mean_reward +/- std_reward` values. You must log in to Hugging Face locally, upload the model to your own public profile, and run the official checker to get a real pass.
