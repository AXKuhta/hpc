import matplotlib.pyplot as plt
import seaborn as sns

import json

grids = []
times = []

plt.subplot(1,2,1)
plt.title("Convergence")
plt.xlabel("Step")
plt.ylabel("Function (population min)")

with open("results.jsonl") as f:
	text = f.read()
	lines = text.replace("}", "}\n")
	for line in lines.split("\n"):
		if not line:
			continue

		x = json.loads(line)

		plt.semilogy(x["log_a"])
		plt.semilogy(x["log_b"])
		plt.semilogy(x["log_c"])
		plt.semilogy(x["log_d"])

		grids.append(str(x["blockdim"]))
		times.append(x["elapsed"])

plt.subplot(1,2,2)
plt.title("Performance")
plt.xlabel("Block size")
plt.ylabel("Time elapsed (s)")
sns.barplot(x=grids, y=times)
plt.show()
