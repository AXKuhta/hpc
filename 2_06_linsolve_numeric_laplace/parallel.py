from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

# Heating map
# Initial conditions must be enforced throughout
x_ = cp.array([
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
# We're memory limited pretty bad without sparse matrices
a = cp.zeros([64, 64], dtype="float32")
b = cp.zeros([128, 128], dtype="float32")

h, w = x_.shape

a[30:30+h, 30:30+w] = x_
b[60:60+h, 60:60+w] = x_

board = cp.array(np.loadtxt("board128x128.csv"))

#
# Shape of answer: a 64*64 = 4096 vector
# Shape of A: a 4096x4096 matrix
#
# 1 T(i+1, j) + 1 T(i-1, j) + 1 T(i, j+1) + 1 T(i, j-1) - 5 T(i, j) = -T.old(i, j)
#
# A = (....0,1,0...0,1,-5,1,0...0,1,0....)
# x = T
# b = -T.old
#
def simulate_v1(ics, metal=None, tau=99):
	h, w = ics.shape
	n = w

	assert h == w

	#
	# Use offsets: +1, -1, +n, -n
	#
	A = tau*cp.eye(n*n, k=+1, dtype="float32") \
	  + tau*cp.eye(n*n, k=-1, dtype="float32") \
	  + tau*cp.eye(n*n, k=+n, dtype="float32") \
	  + tau*cp.eye(n*n, k=-n, dtype="float32") \
	  - (1+4*tau)*cp.eye(n*n, dtype="float32")

	# Top boundary
	# Bottom boundary
	A[:n, :] = 0
	A[-n:, :] = 0

	# Left boundary
	# Right boundary
	A[cp.arange(n)*n, :] = 0
	A[cp.arange(n)*n + n - 1, :] = 0

	b = ics.flatten()

	# Metal cutouts if metal zone specified
	if metal is not None:
		ind = cp.argwhere(metal.flatten() == 0)
		A[ind, :] = 0
		A[:, ind] = 0
		b[ind] = 0

	start = perf_counter()

	x, _, _, _ = cp.linalg.lstsq(A, -b)

	elapsed = perf_counter() - start

	print(f"Elapsed: {elapsed*1000:.1f}ms")

	#plt.imshow(metal, alpha=0.5)
	plt.imshow(x.get().reshape(64, 64))
	#plt.imshow(np.log10(np.abs(x.get().reshape(64, 64) + 0.0001)))
	plt.show()

def conjugate_gradient_lstsq(A_, b_, eps=0.01):
	A = A_.T @ A_ # Problem needs to be least squares
	b = A_.T @ b_

	start = perf_counter()

	n = b_.shape[0]

	# Initial guess
	# WARNING: Avoid using random here, it will create junk in dead parameters of ill-conditioned systems
	x = cp.zeros(n)
	z = cp.zeros_like(x)

	r = b - A@x
	d = r

	delta_ = cp.dot(r, r)
	delta = delta_

	log = [x]

	for i in range(n):
		q = A@d
		alpha = delta / cp.dot(d, q)
		x = x + alpha*d
		log.append(x)

		# [optional] refresh residuals vs ground truth
		if i % 50 == 0:
			r = b - A@x
		else:
			r = r - alpha*q

		delta_ = delta
		delta = cp.dot(r, r)
		beta = delta / delta_
		d = r + beta*d

		if cp.linalg.norm(log[-2] - log[-1]) < eps:
			print("early exit: converged on step", i+1)
			break

	elapsed = perf_counter() - start

	print(f"CG Elapsed {elapsed*1000:.1f}ms")

	return x, elapsed

# simulate_v2(a*9999999, board[:64, :64], tau=30, impulse=False)
def simulate_v2(ics, metal=None, impulse=True, tau=99):
	h, w = ics.shape
	n = w

	assert h == w

	#
	# Use offsets: +1, -1, +n, -n
	#
	A = tau*cp.eye(n*n, k=+1, dtype="float32") \
	  + tau*cp.eye(n*n, k=-1, dtype="float32") \
	  + tau*cp.eye(n*n, k=+n, dtype="float32") \
	  + tau*cp.eye(n*n, k=-n, dtype="float32") \
	  - (1+4*tau)*cp.eye(n*n, dtype="float32")

	# Top boundary
	# Bottom boundary
	A[:n, :] = 0
	A[-n:, :] = 0

	# Left boundary
	# Right boundary
	A[cp.arange(n)*n, :] = 0
	A[cp.arange(n)*n + n - 1, :] = 0

	b = ics.flatten()

	if not impulse:
		A[cp.argwhere(b), :] = 0
		A[cp.argwhere(b), cp.argwhere(b)] = -1

	# Metal cutouts if metal zone specified
	if metal is not None:
		ind = cp.argwhere(metal.flatten() == 0)
		A[ind, :] = 0
		A[:, ind] = 0
		b[ind] = 0

	start = perf_counter()

	x, _ = conjugate_gradient_lstsq(A, -b, eps=0.001)

	elapsed = perf_counter() - start

	print(f"Elapsed: {elapsed*1000:.1f}ms")

	#plt.imshow(metal, alpha=0.5)
	#plt.imshow(x.get().reshape(64, 64))
	plt.imshow(np.log10(np.abs(x.get().reshape(64, 64) + 0.0001)))
	plt.show()
