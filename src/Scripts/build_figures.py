import pandas as pd
import os
import matplotlib.pyplot as plt
basedir = r"C:\studies\IDC_dataScience\thesis\Danfoa_CommonsGame\csv's"

equality = pd.read_csv(os.path.join(basedir, "equality.csv"))
peace = pd.read_csv(os.path.join(basedir, "peace.csv"))
sustainability = pd.read_csv(os.path.join(basedir, "sustainability.csv"))
total_reward = pd.read_csv(os.path.join(basedir, "total reward.csv"))

fig, axs = plt.subplots(2,2)
axs[0, 0].plot(equality["Step"], equality["Value"], 'tab:orange')
axs[0, 0].grid()
axs[0, 0].set(ylabel="equality", title="equality")

axs[0, 1].plot(peace["Step"], peace["Value"], 'tab:green')
axs[0, 1].set(ylabel="peace", title="peace")
axs[0, 1].grid()

axs[1, 0].plot(sustainability["Step"], sustainability["Value"], 'tab:red')
axs[1, 0].set(ylabel="sustainability", title="sustainability")
axs[1, 0].grid()

axs[1, 1].plot(total_reward["Step"], total_reward["Value"])
axs[1, 1].set(ylabel="total_reward", title="total_reward")
axs[1, 1].grid()
plt.tight_layout()
plt.show()

