from time import perf_counter
from random import random
import json

a, b = 1, 3 # Left, Right
u, v = 0, 3 # Bottom, Top

box_area = (b-a)*(v-u)

# True area: ~3.79176
f = lambda x: x*x/(x+1) + 1/x

N = [100, 1000, 10000, 100000]
runtime_history = []

for n in N:
	print("Running Trials for n =", n)

	n_hits = 0

	start = perf_counter()

	for i in range(n):
		pt_x = random() * (b - a) + a
		pt_y = random() * (v - u) + u

		y = f(pt_x)

		if pt_y < y:
			n_hits += 1

	elapsed = perf_counter() - start
	runtime_history.append(elapsed*1000)

	print(n, n_hits)
	print("area", n_hits/n * box_area)

with open("python_results.json", "w") as f:
	json.dump({
		"x": N,
		"y": runtime_history
	}, f)
