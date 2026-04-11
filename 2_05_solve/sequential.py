from time import perf_counter

import numpy as np

np.random.seed(42)

print("mk random...")

A = np.random.rand(4000, 4000)
x = np.random.rand(4000)
b = A@x

#print ("lstqsq...")

#z, _, _, _ = np.linalg.lstsq(A, b)

#print ("validate...")

#assert np.allclose(x, z)

def conjugate_gradient(A_, b_):
	A = A_.T @ A_ # Needs to be least squares
	b = A_.T @ b_

	# Initial guess
	x = np.random.rand(2)

	residuals = b - A@x
	orthogonal = residuals

	for i in range(3):
		alpha = np.dot(residuals, residuals) / np.dot(residuals, A@residuals)
		x = x + alpha * orthogonal
		oldresiduals = residuals
		residuals = residuals - alpha * np.dot(A, orthogonal)
		beta = np.dot(residuals, residuals) / np.dot(oldresiduals, oldresiduals)
		orthogonal = residuals + beta*orthogonal

	return x


def conjugate_gradient_lstsq(A_, b_):
	A = A_.T @ A_ # Needs to be least squares
	b = A_.T @ b_

	n = b_.shape[0]

	# Initial guess
	x = np.random.rand(n)
	z = np.zeros_like(x)

	r = b - A@x
	d = r

	delta_ = np.dot(r, r)
	delta = delta_

	for i in range(n):
		q = A@d
		alpha = delta / np.dot(d, q)
		x = x + alpha*d

		r = r - alpha*q
		delta_ = delta
		delta = np.dot(r, r)
		beta = delta / delta_
		d = r + beta*d

		if np.allclose(d, z):
			print("early exit: converged on step", i+1)
			return x

	return x

start = perf_counter()
c = conjugate_gradient_lstsq(A, b)
elapsed = perf_counter() - start

print(f"Elapsed {elapsed*1000:.1f}ms")
