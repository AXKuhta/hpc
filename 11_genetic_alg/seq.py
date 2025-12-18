
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

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
x = 100*(2*np.random.rand(4096, 100)-1)

def objective(x):
	return np.sum(x*x - 10*np.cos(2*np.pi*x) + 10, -1)

print("Mean fitness", np.mean(objective(x)))

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
	winner[np.random.rand(2048) > 0.99] ^= 1
	x = np.where(winner[:, None], b, a)

	#
	# Cross polination (probabilistic)
	#

	a = x[:1024]
	b = x[1024:]

	# Do it twice to create more creatures,
	# have the two be opposites of each other
	swap_map = np.random.rand(1024, 100) > 0.5
	u = np.where(swap_map, b, a)
	v = np.where(swap_map, a, b)

	x = np.vstack([x, u, v])

	#
	# Shuffling
	#
	np.random.shuffle(x)

	#print("Mean fitness", np.mean(objective(x)))


	#
	# Mutations (probabilistic)
	#

	mask = np.random.rand(4096, 100) < 0.1
	decay = 1 - i/1000
	x += (2*np.random.rand(4096, 100)-1) * mask * decay

	history.append( np.min(objective(x)) )

print("Best fitness", np.min(objective(x)))

plt.plot(history)
plt.show()
