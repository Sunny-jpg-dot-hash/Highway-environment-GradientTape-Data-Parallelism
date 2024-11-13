import highway_env  # 確保 highway-env 被正確載入
from tensorflow.keras.models import load_model
import gymnasium as gym
import numpy as np
import os

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))

def main(env_name='roundabout-v0'):
    total_reward = 0

    # 加載模型
    model = load_model(os.path.join(root_path, 'YOURMODEL.h5'))

    # 創建環境
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except gym.error.NameNotFound:
        print(f"Environment '{env_name}' does not exist.")
        return None  # 如果環境不存在則退出
    except AttributeError:
        print("Environment does not support configure method.")
        return None

    for _ in range(10):  # 測試 10 輪
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            env.render()

            # 預測動作
            obs = obs.reshape(1, 25)  # 調整觀測數據的形狀為 (1, 25)
            action = np.argmax(model.predict(obs), axis=1)[0]  # 選擇模型的動作

            # 執行動作
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward

    return int(total_reward)

if __name__ == "__main__":
    rewards = []
    for round in range(10):  # 執行 10 輪測試
        reward = main()
        if reward is not None:
            rewards.append(reward)
    print("Rewards from each test run:", rewards)