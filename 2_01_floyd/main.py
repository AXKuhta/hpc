
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

import numpy as np
import cupy as cp

from time import perf_counter

np.random.seed(42)
cp.random.seed(42)

#
# Делаем случайные точки на плоскости
#
n = 100
x = np.random.randint(1, 10000, n)/100
y = np.random.randint(1, 10000, n)/100

pts = np.dstack([x, y])[0]
vor = Voronoi(pts)

vor_ridges = list(filter(lambda x: -1 not in x, vor.ridge_vertices))

# Находим расстояния
x = vor.vertices.T[0]
y = vor.vertices.T[1]
dx = x - x[:, None]
dy = y - y[:, None]
l2 = np.round(np.sqrt(dx*dx + dy*dy))

v_ = np.array(vor_ridges).T
l2_ = np.ones_like(l2)*99999
l2_[v_[0], v_[1]] = l2[v_[0], v_[1]]
l2_[v_[1], v_[0]] = l2[v_[0], v_[1]]

asd = l2_[v_[0], v_[1]].tolist()
for z in vor.vertices[vor_ridges]: plt.plot(*z.T, c="orange"); plt.annotate(f"{asd.pop(0)}", np.mean(z.T, 1), color="slategray", fontweight="bold")
plt.scatter(*vor.vertices.T, c="orange")
plt.scatter(*pts.T)
plt.xlim(1, 100)
plt.ylim(1, 100)
plt.show()

l2 = l2_

"""
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
"""

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

path(5, 0, res2[5, 0])
print(x)
