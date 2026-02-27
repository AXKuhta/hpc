import numpy as np
import cupy as cp

from time import perf_counter

np.random.seed(42)
cp.random.seed(42)

n = 4

# В этой матрице есть дешевый путь от 3 к 0:
# 3 -> 2 -> 1 -> 0
ptmatrix = np.array([
	[.0, .9, .9, .9],
	[.1, .0, .9, .9],
	[.9, .1, .0, .9],
	[.9, .9, .6, .0]
])

l2 = ptmatrix

def cost(path):
	return l2[path[:-1], path[1:]].sum()

# Перебор путей...
def path(u, v, oracle):
	pend = [ [u] ]

	while pend:
		pend_ = []

		for path in pend:
			loc = path[-1]

			if loc == v:
				return path

			avail = set(np.argwhere(cost(path) + l2[loc] <= oracle).flatten().tolist()) - {loc, u}

			for hop in avail:
				pend_.append(path + [hop])

		pend = pend_

	assert 0, "No path"

#
# Реализация 1: цикл на чистом питоне
#

def impl1():
	print("================== impl1 ===================")

	dist = l2.copy()

	start = perf_counter()

	for k in range(n):
		for i in range(n):
			for j in range(n):
				dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

	elapsed = perf_counter() - start
	print(f"Elapsed {elapsed:.1f}s")

	return dist.copy()


#
# Реализация 2: на операциях numpy
#

def impl2():
	print("================== impl2 ===================")

	dist = l2.copy()

	start = perf_counter()

	for k in range(n):
		detour_cost = dist.T[k]
		detour_gain = dist[k]

		dist = np.minimum(dist, detour_gain + detour_cost[:, None])

	elapsed = perf_counter() - start
	print(f"Elapsed {elapsed:.1f}s")

	return dist.copy()


#
# Реализация 3: на операциях cupy
#

def impl3():
	print("================== impl3 ===================")

	dist = cp.array(l2.copy()).astype("float32")

	start = perf_counter()

	for k in range(n):
		detour_cost = dist.T[k]
		detour_gain = dist[k]

		dist = cp.minimum(dist, detour_gain + detour_cost[:, None])

	elapsed = perf_counter() - start
	print(f"Elapsed {elapsed:.1f}s")

	return cp.asnumpy(dist)

#res1 = impl1()
res2 = impl2()
res3 = impl3()

x = path(3, 0, res2[3, 0])
print(x)
