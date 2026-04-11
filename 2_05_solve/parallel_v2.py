from time import perf_counter

import numpy as np
import cupy as cp

np.random.seed(42)
cp.random.seed(42)

print("mk random...")

A = cp.random.rand(4000, 4000)
x = cp.random.rand(4000)
b = A@x

#print ("lstqsq...")

#z, _, _, _ = cp.linalg.lstsq(A, b)

#print ("validate...")

#assert cp.allclose(x, z)

@cp.fuse
def fused_elementwise1(x, alpha, r, d, q):
	x = x + alpha*d
	r = r - alpha*q

	return x, r

def conjugate_gradient_lstsq(A_, b_):
	A = A_.T @ A_ # Needs to be least squares
	b = A_.T @ b_

	n = b_.shape[0]

	# Initial guess
	x = cp.random.rand(n)
	z = cp.zeros_like(x)

	r = b - A@x
	d = r

	delta_ = cp.dot(r, r)
	delta = delta_

	for i in range(n):
		q = A@d
		alpha = delta / cp.dot(d, q)
		x,r = fused_elementwise1(x, alpha, r, d, q)

		delta_ = delta
		delta = cp.dot(r, r)
		beta = delta / delta_
		d = r + beta*d

	return x

start = perf_counter()
c = conjugate_gradient_lstsq(A, b)
c.get()
elapsed = perf_counter() - start

print(f"Elapsed {elapsed*1000:.1f}ms")
