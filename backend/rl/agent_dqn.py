from stable_baselines3 import DQN

def build_agent(env, seed=42):
    return DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        tau=0.02,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        seed=seed,
    )
