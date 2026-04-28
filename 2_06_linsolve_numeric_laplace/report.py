import matplotlib.pyplot as plt
import numpy as np
import json

with open("results_sequential.json", "rb") as f:
	sequential = json.load(f)

with open("results_parallel.json", "rb") as f:
	parallel = json.load(f)

x = [str(x) for x in sequential.get("x")]

np_lstsq = [x*1000 for x in sequential.get("y_lstsq")]
np_cg = [x*1000 for x in sequential.get("y_cg")]
cp_cg = [x*1000 for x in parallel.get("y")]

def plot_a():
	cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
	plt.subplots_adjust(right=0.80, bottom=0.2)

	#plt.plot(x, np_lstsq, marker="o", label="numpy lstsq")
	plt.plot(x, np_cg, marker="o", label="numpy cg")
	plt.plot(x, cp_cg, marker="o", label="cupy cg")
	plt.xticks(x, [f"{z}" for z in x], rotation=30, ha="right")

	"""
	plt.annotate(f"{np_lstsq[-1]:.1f}ms",
		xy=(x[-1], np_lstsq[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)
	"""

	color_a = cycler.pop(0)
	color_b = cycler.pop(0)

	plt.annotate(f"{np_cg[-1]:.1f}ms",
		xy=(x[-1], np_cg[-1]),
		xycoords='data',
		xytext=(20, -5),
		textcoords='offset points',
		color=color_a,
		weight="bold"
	)

	plt.annotate(f"{np_cg[-2]:.1f}ms",
		xy=(x[-2], np_cg[-2]),
		xycoords='data',
		xytext=(-60, 5),
		textcoords='offset points',
		color=color_a,
		weight="bold"
	)

	plt.annotate(f"{cp_cg[-1]:.1f}ms",
		xy=(x[-1], cp_cg[-1]),
		xycoords='data',
		xytext=(20, +5),
		textcoords='offset points',
		color=color_b,
		weight="bold"
	)

	plt.annotate(f"{cp_cg[-2]:.1f}ms",
		xy=(x[-2], cp_cg[-2]),
		xycoords='data',
		xytext=(10, -5),
		textcoords='offset points',
		color=color_b,
		weight="bold"
	)

	plt.title("Solving laplace equation")
	plt.ylabel("time, ms")
	plt.xlabel("size")
	plt.legend()
	plt.show()

plot_a()
