import matplotlib.pyplot as plt
import numpy as np
import json

with open("conjugate_gradient/results_parallel_cg.json", "rb") as f:
	parallel_cg_data = json.load(f)

with open("conjugate_gradient/results_sequential_cg.json", "rb") as f:
	sequential_cg_data = json.load(f)

with open("jacobi/results_parallel_jacobi.json", "rb") as f:
	parallel_jacobi_data = json.load(f)

with open("jacobi/results_sequential_jacobi.json", "rb") as f:
	sequential_jacobi_data = json.load(f)

x = [str(x) for x in sequential_jacobi_data.get("x")]

np_jacobi = [x*1000 for x in sequential_jacobi_data.get("y")]
cp_jacobi = [x*1000 for x in parallel_jacobi_data.get("y")]

np_cg = [x*1000 for x in sequential_cg_data.get("y")]
cp_cg = [x*1000 for x in parallel_cg_data.get("y")]

def plot_a():
	cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
	plt.subplots_adjust(right=0.80, bottom=0.2)

	plt.plot(x, np_jacobi, marker="o", label="numpy jacobi")
	plt.plot(x, cp_jacobi, marker="o", label="cupy jacobi")
	plt.plot(x, np_cg, marker="o", label="numpy cg")
	plt.plot(x, cp_cg, marker="o", label="cupy cg")
	plt.xticks(x, [f"{z}" for z in x], rotation=30, ha="right")

	plt.annotate(f"{np_jacobi[-1]:.1f}ms",
		xy=(x[-1], np_jacobi[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{cp_jacobi[-1]:.1f}ms",
		xy=(x[-1], cp_jacobi[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{np_cg[-1]:.1f}ms",
		xy=(x[-1], np_cg[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{cp_cg[-1]:.1f}ms",
		xy=(x[-1], cp_cg[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.title("Solving least squares on 4000x4000 system")
	plt.ylabel("time, ms")
	plt.xlabel("eps")
	plt.legend()
	plt.show()

plot_a()
