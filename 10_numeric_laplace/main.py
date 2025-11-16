
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

#
# Laplace's equation in 2d:
# u''.x + u''.y = 0
#
# It can be evaluated numerically by assuming
# u to be discrete and taking its derivatives
#
# For u''.x:
# u'.x = u(i, j + 1) - u(i, j)
# u''.x = u'.x(i, j) - u'.x(i, j - 1)
#       = u(i, j + 1) - u(i, j) - u(i, j - 1 + 1) + u(i, j - 1)
#       = u(i, j + 1) - 2 u(i, j) + u(i, j - 1)
#
# Similarly for u''.y:
#       u''.y = u(i + 1, j) - 2 u(i, j) + u(i - 1, j)
#
# Substituting that:
#       u(i, j + 1) - 2 u(i, j) + u(i, j - 1) +
#       u(i + 1, j) - 2 u(i, j) + u(i - 1, j) = 0
#
#       4 u(i, j) = u(i, j + 1) + u(i, j - 1) +
#                   u(i + 1, j) + u(i - 1, j)
#
#       u(i, j) = 1/4 ( u(i, j + 1) + u(i, j - 1) +
#                       u(i + 1, j) + u(i - 1, j) )
#

# Heating map
# Initial conditions must be enforced throughout
x_ = np.array([
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
	[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
	[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
	[0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
	[0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
	[0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]) + 0.0

# Set of larger map for full experiment
a = np.zeros([64, 64], dtype="float32")
b = np.zeros([128, 128], dtype="float32")
c = np.zeros([256, 256], dtype="float32")
d = np.zeros([512, 512], dtype="float32")

h, w = x_.shape

a[30:30+h, 30:30+w] = x_
b[60:60+h, 60:60+w] = x_
c[120:120+h, 120:120+w] = x_
d[250:250+h, 250:250+w] = x_

bench_set = [a, b, c, d]

#plt.imshow(x)
#plt.show()

def run_v0(x):
	z = np.zeros_like(x) + x

	start = perf_counter()

	for i in range(100):
		z *= 1 - x # Enforce temperature
		z += x

		acc = np.zeros_like(z)
		h, w = acc.shape

		for k in range(1, h-1):
			for l in range(1, w-1):
				acc[k, l] = z[k-1, l] + z[k+1, l] + z[k, l-1] + z[k, l+1]

		z = acc / 4

	elapsed = perf_counter() - start

	#plt.imshow(z)
	#plt.show()

	return elapsed

def run_v1(x):
	z = np.zeros_like(x) + x

	start = perf_counter()

	for i in range(100):
		z *= 1 - x # Enforce temperature
		z += x

		acc = np.zeros_like(z)
		acc += np.roll(z, shift=(-1, 0), axis=(0, 1))
		acc += np.roll(z, shift=(+1, 0), axis=(0, 1))
		acc += np.roll(z, shift=(0, -1), axis=(0, 1))
		acc += np.roll(z, shift=(0, +1), axis=(0, 1))
		z = acc / 4

	elapsed = perf_counter() - start

	#plt.imshow(z)
	#plt.show()

	return elapsed

def run_v2(x):
	ind = np.indices(x.shape)
	h, w = x.shape

	# u d l r
	u = ind + np.array([-1, 0])[:, None, None]
	d = ind + np.array([+1, 0])[:, None, None]
	l = ind + np.array([0, -1])[:, None, None]
	r = ind + np.array([0, +1])[:, None, None]

	z = np.zeros_like(x) + x

	start = perf_counter()

	for i in range(100):
		z *= 1 - x # Enforce temperature
		z += x

		acc = np.zeros_like(z)

		for a, b in [u, d, l, r]:
			acc += z[a % h, b % w]

		z = acc/4

	elapsed = perf_counter() - start

	#plt.imshow(z)
	#plt.show()

	return elapsed

def run_v3(x):
	z = np.zeros_like(x) + x

	start = perf_counter()

	for i in range(100):
		z *= 1 - x # Enforce temperature
		z += x

		acc = np.zeros_like(x)

		acc[+1:, :] += z[:-1, :]
		acc[:-1, :] += z[+1:, :]
		acc[:, +1:] += z[:, :-1]
		acc[:, :-1] += z[:, +1:]
		z = acc / 4

	elapsed = perf_counter() - start

	#plt.imshow(z)
	#plt.show()

	return elapsed

def run_v4(x):
	x = cp.asarray(x)
	z = cp.zeros_like(x) + x

	start = perf_counter()

	for i in range(100):
		z *= 1 - x # Enforce temperature
		z += x

		acc = cp.zeros_like(x)

		acc[+1:, :] += z[:-1, :]
		acc[:-1, :] += z[+1:, :]
		acc[:, +1:] += z[:, :-1]
		acc[:, :-1] += z[:, +1:]
		z = acc / 4

	elapsed = perf_counter() - start

	#plt.imshow(z.get())
	#plt.show()

	return elapsed

x = ["64x64", "128x128", "256x256", "512x512"]

y_native = np.array([run_v0(x) for x in bench_set]) * 1000
y_numpy_a = np.array([run_v1(x) for x in bench_set]) * 1000
y_numpy_b = np.array([run_v2(x) for x in bench_set]) * 1000
y_numpy_c = np.array([run_v3(x) for x in bench_set]) * 1000
y_cupy = np.array([run_v4(x) for x in bench_set]) * 1000

plt.plot(x, y_native, label="python loop")
plt.plot(x, y_numpy_a, label="numpy roll")
plt.plot(x, y_numpy_b, label="numpy gather")
plt.plot(x, y_numpy_c, label="numpy slices")
plt.plot(x, y_cupy, label="cupy ops")
plt.xlabel("size")
plt.ylabel("ms")
plt.legend()
plt.show()
