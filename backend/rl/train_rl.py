import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DQN
from env_learning_path import LearningPathEnv
from agent_dqn import build_agent

topics = [
    "Arrays", "Strings", "Linked Lists", "Stacks", "Queues",
    "Trees", "Graphs", "Hash Tables", "Sorting", "Dynamic Programming"
]

activity_types = ["Video", "Quiz", "Assignment", "Mini Project"]

def train_rl_model(episodes=50000, save_dir="models/rl_dqn"):
    os.makedirs(save_dir, exist_ok=True)
    env = LearningPathEnv(topics=topics, activity_types=activity_types)
    check_env(env)
    env = Monitor(env)
    agent = build_agent(env)
    checkpoint = CheckpointCallback(save_freq=5000, save_path=save_dir, name_prefix="rl_agent")
    agent.learn(total_timesteps=episodes, callback=checkpoint)
    model_path = os.path.join(save_dir, "final_rl_agent")
    agent.save(model_path)
    env.close()
    return model_path

if __name__ == "__main__":
    path = train_rl_model(episodes=10000)
    print(f"Model trained and saved at: {path}")
