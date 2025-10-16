import matplotlib.pyplot as plt
import json

with open("python_results.json", "rb") as f:
	python_data = json.load(f)

with open("results_c.json", "rb") as f:
	c_data = json.load(f)

with open("results_cuda.json", "rb") as f:
	cuda_data = json.load(f)

cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.subplots_adjust(right=0.80)

plt.semilogx(python_data["x"], python_data["y"], marker="o", label="Python")
plt.semilogx(c_data["x"], c_data["y"], marker="o", label="gcc -O2")
plt.semilogx(cuda_data["x"], cuda_data["y"], marker="o", label="cuda")

plt.annotate(f"{python_data['y'][-1]:.1f}ms",
	xy=(python_data["x"][-1], python_data["y"][-1]),
	xycoords='data',
	xytext=(20, 0),
	textcoords='offset points',
	color=cycler.pop(0),
	weight="bold"
)

plt.annotate(f"{c_data['y'][-1]:.1f}ms",
	xy=(c_data["x"][-1], c_data["y"][-1]),
	xycoords='data',
	xytext=(20, 0),
	textcoords='offset points',
	color=cycler.pop(0),
	weight="bold"
)

plt.annotate(f"{cuda_data['y'][-1]:.1f}ms",
	xy=(cuda_data["x"][-1], cuda_data["y"][-1]),
	xycoords='data',
	xytext=(20, 0),
	textcoords='offset points',
	color=cycler.pop(0),
	weight="bold"
)

plt.ylabel("time (ms)")
plt.xlabel("n")
plt.legend()
plt.show()
