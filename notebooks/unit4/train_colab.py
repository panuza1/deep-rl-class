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
import gym_pygame
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
    }, tmp)
    tmp.replace(path)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every,
              checkpoint_path, log_path, checkpoint_every, start_episode=0):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(start_episode + 1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
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
        if i_episode % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, i_episode, policy, optimizer)

    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The Reinforce agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      action, _ = policy.act(state)
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward
        
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

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
                video_fps=30
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
    mean_reward, std_reward = evaluate_agent(eval_env, 
                                            hyperparameters["max_t"],
                                            hyperparameters["n_evaluation_episodes"], 
                                            model)
    # Get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
          "env_id": hyperparameters["env_id"], 
          "mean_reward": float(mean_reward),
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
    args = parser.parse_args()

    global env
    env_id = args.env_id
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    if env_id == "CartPole-v1":
        h_size, lr, gamma, episodes, max_t, every = 16, 1e-2, 1.0, 1000, 1000, 100
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

    start_episode = 0
    if args.resume or args.upload_only:
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = int(checkpoint["episode"])
            print(f"Restored checkpoint after episode {start_episode}: {checkpoint_path}")
        elif args.upload_only:
            raise FileNotFoundError(f"No saved model at {checkpoint_path}; cannot upload only.")

    if not args.upload_only and start_episode < episodes:
        if args.checkpoint_every < 1:
            raise ValueError("--checkpoint-every must be at least 1")
        print("Training", env_id, "on", device)
        reinforce(model, optimizer, episodes, max_t, gamma, every,
                  checkpoint_path, log_path, args.checkpoint_every, start_episode)
        save_checkpoint(checkpoint_path, episodes, model, optimizer)
    else:
        print("Using saved model; no training needed.")

    if args.upload_only:
        params["n_training_episodes"] = start_episode
    print("Evaluation:", evaluate_agent(eval_env, max_t, 10, model))
    push_to_hub(repo_id, model, params, eval_env)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
