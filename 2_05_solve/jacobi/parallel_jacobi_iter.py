from time import perf_counter
import json

import matplotlib.pyplot as plt

import numpy as np
import cupy as cp

np.random.seed(42)
cp.random.seed(42)

A = cp.random.rand(4000, 4000) + cp.eye(4000)*10000
x = cp.random.rand(4000)
b = A@x

#print ("lstsq...")
#z, _, _, _ = cp.linalg.lstsq(A, b)
#print ("validate...")
#assert np.allclose(x, z)

def jacobi_iter(A_, b_, iters=20, eps=0.01):
	A = A_.T @ A_ # Problem must be least squares
	b = A_.T @ b_

	cp.cuda.Stream.null.synchronize()

	#plt.imshow(A)
	#plt.show()

	# A = D + E
	#
	# Ax = b
	# Dx + Ex = b
	# Dx = -Ex + b
	# x = -inv(D) E x + inv(D) b
	#
	# x = Bx + z,	B = -inv(D) E	z = inv(D) b

	start = perf_counter()

	n = len(A)

	D = cp.diag(A)
	E = A - cp.eye(n) * D

	B = -E / D[:, None] # divide in dim 0
	z = b / D

	# Initial guess
	x = cp.zeros(n)

	log = [x]

	for i in range(iters):
		x = B @ x + z
		log.append(x)

		if cp.linalg.norm(log[-2] - log[-1]) < eps:
			print(f"converged in {i} steps")
			break

	elapsed = perf_counter() - start
	print(f"Elapsed: {elapsed*1000:.1f}ms")

	return x, elapsed

c, eps0p1 = jacobi_iter(A, b, eps=0.1)
c, eps0p01 = jacobi_iter(A, b, eps=0.01)
c, eps0p001 = jacobi_iter(A, b, eps=0.001)
c, eps0p0001 = jacobi_iter(A, b, eps=0.0001)
c, eps0p00001 = jacobi_iter(A, b, eps=0.00001)

with open("results_parallel_jacobi.json", "w") as f:
	json.dump(dict(
		x=[0.1, 0.01, 0.001, 0.0001, 0.00001],
		y=[eps0p1, eps0p01, eps0p001, eps0p0001, eps0p00001]
	), f)
