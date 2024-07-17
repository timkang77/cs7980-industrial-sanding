from ray import tune
from env import GridWorldEnv
from ray.rllib.algorithms.algorithm import Algorithm


def run_checkpoint(env,result_path):
    
    best_dqn = Algorithm.from_checkpoint(result_path)

    # Reset the environment to get the initial observation
    obs, info = env.reset()

    # Run the policy in the environment
    done = False
    total_reward = 0

    print("")
    print("Initial variance is", info["variance"])
    print("")

    while not done:
        action = best_dqn.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        env.render()  
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        total_reward += reward
        print("total_reward",total_reward)
        print(info)