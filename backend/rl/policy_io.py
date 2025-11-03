import os
import numpy as np
from stable_baselines3 import DQN
from env_learning_path import LearningPathEnv

topics = [
    "Arrays", "Strings", "Linked Lists", "Stacks", "Queues",
    "Trees", "Graphs", "Hash Tables", "Sorting", "Dynamic Programming"
]

activity_types = ["Video", "Quiz", "Assignment", "Mini Project"]

def load_agent(model_path, env=None):
    if env is None:
        env = LearningPathEnv(topics=topics, activity_types=activity_types)
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    agent = DQN.load(model_path, env=env)
    return agent, env

def recommend_next_step(agent, env, obs=None):
    if obs is None:
        obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    topic = topics[info["topic_idx"]]
    activity = activity_types[info["act_idx"]]
    return {
        "topic": topic,
        "activity": activity,
        "reward": reward,
        "terminated": terminated,
        "next_obs": next_obs
    }

if __name__ == "__main__":
    model_path = "models/rl_dqn/final_rl_agent"
    agent, env = load_agent(model_path)
    rec = recommend_next_step(agent, env)
    print(f"Next Recommendation → Topic: {rec['topic']} | Activity: {rec['activity']}")
