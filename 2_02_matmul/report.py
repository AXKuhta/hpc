import matplotlib.pyplot as plt
import numpy as np
import json

with open("results.json", "rb") as f:
	data = json.load(f)

size = data.get("size")
np_native = data.get("native")
np_ijk = data.get("np_ijk")
np_kij = data.get("np_kij")

cp_native = data.get("cp_native")
cp_striped = data.get("cp_striped")
cp_tiled = data.get("cp_tiled")

np_native = [x*1000 for x in np_native]
np_ijk = [x*1000 for x in np_ijk]
np_kij = [x*1000 for x in np_kij]
cp_native = [x*1000 for x in cp_native]
cp_striped = [x*1000 for x in cp_striped]
cp_tiled = [x*1000 for x in cp_tiled]

def plot_a():
	cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
	plt.subplots_adjust(right=0.80, bottom=0.2)

	plt.plot(size, np_native, marker="o", label="numpy native")
	plt.plot(size, np_ijk, marker="o", label="numpy ijk")
	plt.plot(size, np_kij, marker="o", label="numpy kij")
	plt.xticks(size, [f"{x}x{x}" for x in size], rotation=30, ha="right")

	plt.annotate(f"{np_native[-1]:.1f}ms",
		xy=(size[-1], np_native[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{np_ijk[-1]:.1f}ms",
		xy=(size[-1], np_ijk[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{np_kij[-1]:.1f}ms",
		xy=(size[-1], np_kij[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.ylabel("time, ms")
	plt.xlabel("size")
	plt.legend()
	plt.show()

def plot_b():
	cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
	plt.subplots_adjust(right=0.80, bottom=0.2)

	plt.plot(size, np_native, marker="o", label="numpy native")
	plt.plot(size, cp_native, marker="o", label="cupy native")
	plt.plot(size, cp_striped, marker="o", label="cupy striped")
	plt.plot(size, cp_tiled, marker="o", label="cupy tiled")
	plt.xticks(size, [f"{x}x{x}" for x in size], rotation=30, ha="right")

	plt.annotate(f"{np_native[-1]:.1f}ms",
		xy=(size[-1], np_native[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{cp_native[-1]:.1f}ms",
		xy=(size[-1], cp_native[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{cp_striped[-1]:.1f}ms",
		xy=(size[-1], cp_striped[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.annotate(f"{cp_tiled[-1]:.1f}ms",
		xy=(size[-1], cp_tiled[-1]),
		xycoords='data',
		xytext=(20, 0),
		textcoords='offset points',
		color=cycler.pop(0),
		weight="bold"
	)

	plt.ylabel("time, ms")
	plt.xlabel("size")
	plt.legend()
	plt.show()
