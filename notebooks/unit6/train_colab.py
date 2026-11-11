"""Train, evaluate, and upload Unit 6 PandaGym A2C agents from Colab."""
import argparse
import gymnasium as gym
import panda_gym  # Registers Panda* environments.
from huggingface_sb3 import package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", choices=["PandaReachDense-v3", "PandaPickAndPlace-v3"], required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    env_id = args.env_id
    train_env = VecNormalize(make_vec_env(env_id, n_envs=4), norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = A2C("MultiInputPolicy", train_env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model_name = f"a2c-{env_id}"
    model.save(model_name)
    stats = f"{model_name}-vecnormalize.pkl"
    train_env.save(stats)
    train_env.close()

    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    eval_env = VecNormalize.load(stats, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model = A2C.load(model_name, env=eval_env)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"{env_id}: mean_reward={mean:.2f}, std_reward={std:.2f}")
    package_to_hub(
        model=model,
        model_name=model_name,
        model_architecture="A2C",
        env_id=env_id,
        eval_env=eval_env,
        repo_id=f"panuza007/{model_name}",
        commit_message="Train A2C on PandaGym",
    )
    eval_env.close()


if __name__ == "__main__":
    main()
