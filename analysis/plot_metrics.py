import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics_by_checkpoint.csv")

plt.figure(figsize=(8,5))
plt.plot(df["checkpoint"], df["sycophancy_rate"], marker="o")
plt.title("Sycophancy Rate by Model")
plt.ylabel("Rate")
plt.xlabel("Model")
plt.grid(True)
plt.savefig("results/sycophancy_trend.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(df["checkpoint"], df["correction_rate"], marker="o")
plt.title("Correction Rate by Model")
plt.ylabel("Rate")
plt.xlabel("Model")
plt.grid(True)
plt.savefig("results/correction_trend.png")
plt.show()