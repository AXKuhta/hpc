from time import perf_counter
import json

import matplotlib.pyplot as plt

import numpy as np

np.random.seed(42)

A = np.random.rand(4000, 4000) + np.eye(4000)*10000
x = np.random.rand(4000)
b = A@x

#print ("lstsq...")

#z, _, _, _ = np.linalg.lstsq(A, b)

#print ("validate...")

#assert np.allclose(x, z)

def jacobi_iter(A_, b_, iters=20000, eps=0.01):
	A = A_.T @ A_ # Problem must be least squares
	b = A_.T @ b_

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

	D = np.diag(A)
	E = A - np.eye(n) * D

	B = -E / D[:, None] # divide in dim 0
	z = b / D

	print(B)
	print(z)

	# Initial guess
	x = np.zeros(n)

	log = [x]

	for i in range(iters):
		x = B @ x + z
		log.append(x)

		if np.linalg.norm(log[-2] - log[-1]) < eps:
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

with open("results_sequential_jacobi.json", "w") as f:
	json.dump(dict(
		x=[0.1, 0.01, 0.001, 0.0001, 0.00001],
		y=[eps0p1, eps0p01, eps0p001, eps0p0001, eps0p00001]
	), f)
