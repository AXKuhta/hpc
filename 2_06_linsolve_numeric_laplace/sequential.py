from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import json

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
# We're memory limited pretty bad without sparse matrices
a = np.zeros([64, 64], dtype="float32")
b = np.zeros([128, 128], dtype="float32")

h, w = x_.shape

a[30:30+h, 30:30+w] = x_
b[10:10+h, 10:10+w] = x_

board = np.loadtxt("board128x128.csv")

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
	A = tau*np.eye(n*n, k=+1) \
	  + tau*np.eye(n*n, k=-1) \
	  + tau*np.eye(n*n, k=+n) \
	  + tau*np.eye(n*n, k=-n) \
	  - (1+4*tau)*np.eye(n*n)

	# Top boundary
	# Bottom boundary
	A[:n, :] = 0
	A[-n:, :] = 0

	# Left boundary
	# Right boundary
	for i in range(n):
		A[i*n, :] = 0
		A[i*n + n - 1, :] = 0

	A = np.float32(A)
	b = np.float32(ics).flatten()

	# Metal cutouts if metal zone specified
	if metal is not None:
		for i in np.argwhere(metal.flatten() == 0):
			A[i, :] = 0
			A[:, i] = 0
			b[i] = 0

	start = perf_counter()

	x, _, _, _ = np.linalg.lstsq(A, -b)

	elapsed = perf_counter() - start

	print(f"Elapsed: {elapsed*1000:.1f}ms")

	#plt.imshow(metal, alpha=0.5)
	plt.imshow(x.reshape(h, w))
	#plt.imshow(np.log10(np.abs(x.reshape(h, w) + 0.0001)))
	plt.show()

	return elapsed

def conjugate_gradient_lstsq(A_, b_, eps=0.01):
	A = A_.T @ A_ # Problem needs to be least squares
	b = A_.T @ b_

	start = perf_counter()

	n = b_.shape[0]

	# Initial guess
	# WARNING: Avoid using random here, it will create junk in dead parameters of ill-conditioned systems
	x = np.zeros(n, dtype="float32")
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

		# [optional] refresh residuals vs ground truth
		if i % 50 == 0:
			r = b - A@x
		else:
			r = r - alpha*q

		delta_ = delta
		delta = np.dot(r, r)
		beta = delta / delta_
		d = r + beta*d

		if np.linalg.norm(log[-2] - log[-1]) < eps:
			print("early exit: converged on step", i+1)
			break

	elapsed = perf_counter() - start

	print(f"CG Elapsed {elapsed*1000:.1f}ms")

	return x, elapsed

def simulate_v2(ics, metal=None, tau=99):
	h, w = ics.shape
	n = w

	assert h == w

	#
	# Use offsets: +1, -1, +n, -n
	#
	A = tau*np.eye(n*n, k=+1) \
	  + tau*np.eye(n*n, k=-1) \
	  + tau*np.eye(n*n, k=+n) \
	  + tau*np.eye(n*n, k=-n) \
	  - (1+4*tau)*np.eye(n*n)

	# Top boundary
	# Bottom boundary
	A[:n, :] = 0
	A[-n:, :] = 0

	# Left boundary
	# Right boundary
	for i in range(n):
		A[i*n, :] = 0
		A[i*n + n - 1, :] = 0

	A = np.float32(A)
	b = np.float32(ics).flatten()

	# Metal cutouts if metal zone specified
	if metal is not None:
		for i in np.argwhere(metal.flatten() == 0):
			A[i, :] = 0
			A[:, i] = 0
			b[i] = 0

	start = perf_counter()

	x, _ = conjugate_gradient_lstsq(A, -b, eps=0.001)

	elapsed = perf_counter() - start

	print(f"Elapsed: {elapsed*1000:.1f}ms")

	#plt.imshow(metal, alpha=0.5)
	plt.imshow(x.reshape(h, w))
	#plt.imshow(np.log10(np.abs(x.reshape(h, w) + 0.0001)))
	plt.show()

	return elapsed


benchmarks = dict(
	x=["32x32", "48x48", "64x64", "96x96"],
	y_lstsq=[
	#	simulate_v1(b[:32, :32], tau=1),
	#	simulate_v1(b[:48, :48], tau=1),
	#	simulate_v1(b[:64, :64], tau=1),
	#	simulate_v1(b[:96, :96], tau=1)
	],
	y_cg=[
		simulate_v2(b[:32, :32], tau=1),
		simulate_v2(b[:48, :48], tau=1),
		simulate_v2(b[:64, :64], tau=1),
		simulate_v2(b[:96, :96], tau=1)
	]
)

with open("results_sequential.json", "w") as f:
	json.dump(benchmarks, f)
