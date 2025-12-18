
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

np.random.seed(42)
cp.random.seed(42)

#
# Target function:
# sum over i(x.i^2 - 10cos(2 pi x.i) + 10) -> min
#
# N = 1000
# x.i between -100, 100
#
# Population size:
# 4096
#
# Fraction means island count
#

#
# Init
#
x = 100*(2*cp.random.rand(4096, 1000)-1)

def objective(x):
	return cp.sum(x*x - 10*cp.cos(2*cp.pi*x) + 10, -1)

print("Mean fitness", cp.mean(objective(x)))

history = []

for i in range(1000):
	#
	# Selection
	#
	a = x[:2048]
	b = x[2048:]

	u = objective(a)
	v = objective(b)

	# Winner should actually have index 0
	# So compare u > v
	winner = 0 + (u > v)

	# For some small fraction, the unfittest actually survives
	winner[cp.random.rand(2048) > 0.99] ^= 1
	x = cp.where(winner[:, None], b, a)

	#
	# Cross polination (probabilistic)
	#

	a = x[:1024]
	b = x[1024:]

	# Do it twice to create more creatures,
	# have the two be opposites of each other
	swap_map = cp.random.rand(1024, 1000) > 0.5
	u = cp.where(swap_map, b, a)
	v = cp.where(swap_map, a, b)

	x = cp.vstack([x, u, v])

	#
	# Shuffling
	#
	cp.random.shuffle(x)


	#
	# Mutations (probabilistic)
	#

	mask = cp.random.rand(4096, 1000) < 0.1
	decay = 1 - i/1000
	x += (2*cp.random.rand(4096, 1000)-1) * mask * decay

	history.append( cp.min(objective(x)).get() )

print("Best fitness", cp.min(objective(x)).get() )

plt.plot(history)
plt.show()
