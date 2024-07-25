import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.utils.play import play,PlayPlot
from stable_baselines3.common.monitor import Monitor

#Manual game play
#env = gym.make("LunarLander-v2", render_mode="rgb_array")

'''
def callback(obs_t,obs_tplm, action, rew, teriminated, truncated, info):
    return["rew"]
    
plotter = PlayPlot(callback=150,["reward"])
play(env, keys_to_action = {
    "w":2,
    "a":1,
    "d":3,
}, noop = 0, callback=plotter.callback)
'''

#Agent with ppo
log_dir = "ppo_lunarlander_tensorboard/"
timesteps = 100000
env = gym.make("LunarLander-v2", render_mode="human")
env = Monitor(env, log_dir)
model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log=log_dir)
model.learn(total_timesteps=timesteps, progress_bar=True)
model.save("ppo_LunarLander_v2")
del model

model = PPO.load("ppo_LunarLander_v2", env)
mean_reward,std_reward = evaluate_policy(model, model.get_env(),n_eval_episodes=10)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)
    vec_env.render("human")
    if done:
        print(rewards)

