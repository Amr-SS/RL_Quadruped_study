import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ppo = pd.read_csv("algo_comparison/logs/ppo/monitor.csv", skiprows=1)
td3 = pd.read_csv("algo_comparison/logs/td3/monitor.csv", skiprows=1)
sac = pd.read_csv("algo_comparison/logs/sac/monitor.csv", skiprows=1)

def smooth(y, w=50):
    return y.rolling(w).mean()

# 1. Smoothed reward
plt.figure()
plt.plot(smooth(ppo['r']), label="PPO")
plt.plot(smooth(td3['r']), label="TD3")
plt.plot(smooth(sac['r']), label="SAC")
plt.title("Smoothed Reward")
plt.legend()
plt.savefig("smooth_reward.png")

# 2. Boxplot
plt.figure()
data = pd.DataFrame({"PPO": ppo['r'], "TD3": td3['r'], "SAC": sac['r']})
sns.boxplot(data=data)
plt.title("Reward Distribution")
plt.savefig("boxplot.png")

# 3. Cumulative reward
plt.figure()
plt.plot(ppo['r'].cumsum(), label="PPO")
plt.plot(td3['r'].cumsum(), label="TD3")
plt.plot(sac['r'].cumsum(), label="SAC")
plt.title("Cumulative Reward")
plt.legend()
plt.savefig("cumulative.png")

# 4. Variance (stability)
plt.figure()
plt.plot(ppo['r'].rolling(100).std(), label="PPO")
plt.plot(td3['r'].rolling(100).std(), label="TD3")
plt.plot(sac['r'].rolling(100).std(), label="SAC")
plt.title("Reward Variance")
plt.legend()
plt.savefig("variance.png")

print("Extra plots generated ✅")
