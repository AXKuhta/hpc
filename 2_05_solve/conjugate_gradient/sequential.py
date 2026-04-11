from time import perf_counter
import json

import numpy as np

np.random.seed(42)

# Large diagonal is not strictly needed for conjugate gradient
# but it will converge quicker

A = np.random.rand(4000, 4000) + np.eye(4000)*10000
x = np.random.rand(4000)
b = A@x

#print ("lstsq...")

#z, _, _, _ = np.linalg.lstsq(A, b)

#print ("validate...")

#assert np.allclose(x, z)

def conjugate_gradient_lstsq(A_, b_, eps=0.01):
	A = A_.T @ A_ # Problem needs to be least squares
	b = A_.T @ b_

	start = perf_counter()

	n = b_.shape[0]

	# Initial guess
	x = np.random.rand(n)
	z = np.zeros_like(x)

	r = b - A@x
	d = r

	delta_ = np.dot(r, r)
	delta = delta_

	log = [x]

	for i in range(n):
		q = A@d
		alpha = delta / np.dot(d, q)
		x = x + alpha*d
		log.append(x)

		r = r - alpha*q
		delta_ = delta
		delta = np.dot(r, r)
		beta = delta / delta_
		d = r + beta*d

		if np.linalg.norm(log[-2] - log[-1]) < eps:
			print("early exit: converged on step", i+1)
			break

	elapsed = perf_counter() - start

	print(f"Elapsed {elapsed*1000:.1f}ms")

	return x, elapsed

c, eps0p1 = conjugate_gradient_lstsq(A, b, eps=0.1)
c, eps0p01 = conjugate_gradient_lstsq(A, b, eps=0.01)
c, eps0p001 = conjugate_gradient_lstsq(A, b, eps=0.001)
c, eps0p0001 = conjugate_gradient_lstsq(A, b, eps=0.0001)
c, eps0p00001 = conjugate_gradient_lstsq(A, b, eps=0.00001)

with open("results_sequential_cg.json", "w") as f:
	json.dump(dict(
		x=[0.1, 0.01, 0.001, 0.0001, 0.00001],
		y=[eps0p1, eps0p01, eps0p001, eps0p0001, eps0p00001]
	), f)
