"""Unit 4 REINFORCE training and Hugging Face submission (Python 3.10)."""
import argparse
import csv
import datetime
import json
import os
import tempfile
from collections import deque
from pathlib import Path

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size) if s_size == 4 else nn.Linear(h_size, h_size * 2)
        self.fc3 = None if s_size == 4 else nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.fc3 is not None:
            x = self.fc3(F.relu(x))
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def save_checkpoint(path, episode, policy, optimizer):
    # Atomic replacement protects the previous checkpoint if Colab disconnects.
    tmp = path.with_suffix(".tmp")
    torch.save({
        "episode": episode,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }, tmp)
    tmp.replace(path)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every,
              checkpoint_path, log_path, checkpoint_every, eval_env,
              best_path, eval_log_path, eval_episodes, start_episode=0,
              train_seed=None, eval_seed=None):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(start_episode + 1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset(seed=train_seed + i_episode) if train_seed is not None else env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t) 
        n_steps = len(rewards) 
        # Compute the discounted returns at each timestep,
        # as 
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity 
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...
        
        # Given this formulation, the returns at each timestep t can be computed 
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order 
        # to avoid computing them multiple times)
        
        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...
        
        
        ## Given the above, we calculate the returns at timestep t as: 
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed 
        ## if we were to do it from first to last.
        
        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )    
            
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is 
        # added to the standard deviation of the returns to avoid numerical instabilities        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8: PyTorch prefers gradient descent 
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            average = float(np.mean(scores_deque))
            print(f"Episode {i_episode}\tAverage Score: {average:.2f}", flush=True)
            with log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["episode", "average_reward_100"])
                writer.writerow([i_episode, average])
        if i_episode % checkpoint_every == 0 or i_episode == n_training_episodes:
            save_checkpoint(checkpoint_path, i_episode, policy, optimizer)
            mean, std = evaluate_agent(eval_env, max_t, eval_episodes, policy, seed=eval_seed)
            score = float(mean - std)
            print(f"Checkpoint {i_episode}: eval={mean:.2f} +/- {std:.2f}, score={score:.2f}", flush=True)
            with eval_log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["episode", "mean_reward", "std_reward", "score"])
                writer.writerow([i_episode, float(mean), float(std), score])
            old_best = float("-inf")
            if best_path.exists():
                old_best = float(torch.load(best_path, map_location="cpu", weights_only=True)["score"])
            if score > old_best:
                tmp = best_path.with_suffix(".tmp")
                torch.save({"episode": i_episode, "score": score, "model": policy.state_dict()}, tmp)
                tmp.replace(best_path)
                print(f"New best checkpoint: score={score:.2f}", flush=True)

    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy, seed=None):
    """Evaluate sampled actions, optionally on the same reproducible rollouts."""
    saved_rng = torch.get_rng_state() if seed is not None else None
    if seed is not None:
        torch.manual_seed(seed)
    episode_rewards = []
    try:
        for episode in range(n_eval_episodes):
            state = env.reset(seed=seed + episode) if seed is not None else env.reset()
            total_reward = 0
            for _ in range(max_steps):
                action, _ = policy.act(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            episode_rewards.append(total_reward)
    finally:
        if saved_rng is not None:
            torch.set_rng_state(saved_rng)
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def record_video(env, policy, out_directory, fps=30, max_steps=10000):
  """
  Generate a replay video of the agent
  :param env
  :param Qtable: Qtable of our agent
  :param out_directory
  :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
  """
  images = []  
  done = False
  state = env.reset()
  img = env.render(mode='rgb_array')
  images.append(img)
  for _ in range(max_steps):
    # Take the action (index) that have the maximum expected future reward given that state
    action, _ = policy.act(state)
    state, reward, done, info = env.step(action) # We directly put next_state = state for recording logic
    img = env.render(mode='rgb_array')
    images.append(img)
    if done:
      break
  imageio.mimsave(out_directory, images, fps=fps)

def push_to_hub(repo_id, 
                model,
                hyperparameters,
                eval_env,
                video_fps=30,
                eval_result=None
                ):
  """
  Evaluate, Generate a video and Upload a model to Hugging Face Hub.
  This method does the complete pipeline:
  - It evaluates the model
  - It generates the model card
  - It generates a replay video of the agent
  - It pushes everything to the Hub

  :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
  :param model: the pytorch model we want to save
  :param hyperparameters: training hyperparameters
  :param eval_env: evaluation environment
  :param video_fps: how many frame per seconds to record our video replay 
  """

  _, repo_name = repo_id.split("/")
  api = HfApi()
  
  # Step 1: Create the repo
  repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
  )

  with tempfile.TemporaryDirectory() as tmpdirname:
    local_directory = Path(tmpdirname)
  
    # Step 2: Save the model
    torch.save(model, local_directory / "model.pt")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
      json.dump(hyperparameters, outfile)
    
    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = eval_result if eval_result is not None else evaluate_agent(
        eval_env, hyperparameters["max_t"], hyperparameters["n_evaluation_episodes"], model
    )
    # Get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
          "env_id": hyperparameters["env_id"], 
          "mean_reward": float(mean_reward),
          "std_reward": float(std_reward),
          "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
          "eval_datetime": eval_form_datetime,
    }

    # Write a JSON file
    with open(local_directory / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = hyperparameters["env_id"]
    
    metadata = {}
    metadata["tags"] = [
          env_name,
          "reinforce",
          "reinforcement-learning",
          "custom-implementation",
          "deep-rl-class"
      ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
      )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Reinforce** Agent playing **{env_name}**
  This is a trained model of a **Reinforce** agent playing **{env_name}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """

    readme_path = local_directory / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
      f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path =  local_directory / "replay.mp4"
    try:
        record_video(eval_env, model, video_path, video_fps, hyperparameters["max_t"])
    except Exception as exc:
        print(f"Replay video skipped: {exc}. Model upload will continue.")

    # Step 7. Push everything to the Hub
    api.upload_folder(
          repo_id=repo_id,
          folder_path=local_directory,
          path_in_repo=".",
    )

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", choices=["CartPole-v1", "Pixelcopter-PLE-v0"], required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-dir", default="/content/unit4_runs")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cartpole-solution", action="store_true")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    global env
    env_id = args.env_id
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if env_id == "Pixelcopter-PLE-v0":
        import gym_pygame  # Registers Pixelcopter with Gym
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    if env_id == "CartPole-v1":
        h_size, lr, gamma, episodes, max_t, every = (
            (128, 1e-2, 0.99, 3000, 500, 100)
            if args.cartpole_solution else (16, 1e-2, 1.0, 1000, 1000, 100)
        )
        repo_id = "panuza007/Reinforce-CartPole-v1"
    else:
        h_size, lr, gamma, episodes, max_t, every = 64, 1e-4, 0.99, 50000, 10000, 1000
        repo_id = "panuza007/Reinforce-Pixelcopter-PLE-v0"
    if args.episodes is not None:
        episodes = args.episodes
    params = dict(h_size=h_size, n_training_episodes=episodes, n_evaluation_episodes=10,
                  max_t=max_t, gamma=gamma, lr=lr, env_id=env_id,
                  state_space=s_size, action_space=a_size)
    model = Policy(s_size, a_size, h_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    run_dir = Path(args.output_dir) / env_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "train_log.csv"
    best_path = run_dir / "best_model.pt"
    eval_log_path = run_dir / "eval_log.csv"
    params["n_evaluation_episodes"] = (
        max(100, args.eval_episodes) if args.cartpole_solution else args.eval_episodes
    )
    if args.eval_episodes < 2:
        raise ValueError("--eval-episodes must be at least 2")

    start_episode = 0
    if args.resume or args.upload_only or args.eval_only:
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = int(checkpoint["episode"])
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            print(f"Restored checkpoint after episode {start_episode}: {checkpoint_path}")
        elif args.upload_only or args.eval_only:
            if not best_path.exists():
                raise FileNotFoundError(f"No saved model at {checkpoint_path} or {best_path}")
            print("Latest checkpoint unavailable; loading best saved model.")

    if not args.upload_only and not args.eval_only and start_episode < episodes:
        if args.checkpoint_every < 1:
            raise ValueError("--checkpoint-every must be at least 1")
        print("Training", env_id, "on", device)
        reinforce(model, optimizer, episodes, max_t, gamma, every,
                  checkpoint_path, log_path, args.checkpoint_every,
                  eval_env, best_path, eval_log_path, args.eval_episodes, start_episode,
                  train_seed=args.seed if env_id == "CartPole-v1" else None,
                  eval_seed=args.seed + 10_000 if env_id == "CartPole-v1" else None)
    else:
        print("Using saved model; no training needed.")

    if best_path.exists():
        best = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
        print(f"Selected best saved model: episode={best['episode']}, checkpoint score={best['score']:.2f}")
        params["n_training_episodes"] = int(best["episode"])
    elif args.eval_only or args.upload_only:
        params["n_training_episodes"] = start_episode

    results = evaluate_agent(
        eval_env, max_t, params["n_evaluation_episodes"], model,
        seed=args.seed + 20_000 if env_id == "CartPole-v1" else None,
    )
    score = float(results[0] - results[1])
    print(f"Evaluation: mean={results[0]:.2f}, std={results[1]:.2f}, score={score:.2f}")
    if not args.eval_only:
        if args.min_score is not None and score < args.min_score:
            print(f"Score {score:.2f} is below {args.min_score:.2f}; Hub upload skipped.")
        else:
            push_to_hub(repo_id, model, params, eval_env, eval_result=results)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
