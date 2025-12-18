
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

def objective(x):
	return np.sum(x*x - 10*np.cos(2*np.pi*x) + 10, -1)

def advance_island(x):
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
	np.random.shuffle(x, 1)

	#print("Mean fitness", np.mean(objective(x)))


	#
	# Mutations (probabilistic)
	#

	mask = np.random.rand(4096, 100) < 0.1
	decay = 1 - i/1000
	x += (2*np.random.rand(4096, 100)-1) * mask * decay

	return x

def shuffle_along_axis(a, axis):
	idx = np.random.rand(*a.shape).argsort(axis=axis)
	return np.take_along_axis(a,idx,axis=axis)

def advance_island_b(x):
	#
	# Selection
	#
	a = x[:, :2048]
	b = x[:, 2048:]

	u = objective(a)
	v = objective(b)

	# Winner should actually have index 0
	# So compare u > v
	winner = 0 + (u > v)

	# For some small fraction, the unfittest actually survives
	winner[np.random.rand(4, 2048) > 0.99] ^= 1
	x = np.where(winner[:, :, None], b, a)

	#
	# Cross polination (probabilistic)
	#

	a = x[:, :1024]
	b = x[:, 1024:]

	# Do it twice to create more creatures,
	# have the two be opposites of each other
	swap_map = np.random.rand(4, 1024, 100) > 0.5
	u = np.where(swap_map, b, a)
	v = np.where(swap_map, a, b)

	x = np.concatenate([x, u, v], 1)

	#
	# Shuffling
	#
	x = shuffle_along_axis(x, 1)

	#print("Mean fitness", np.mean(objective(x)))


	#
	# Mutations (probabilistic)
	#

	mask = np.random.rand(4, 4096, 100) < 0.1
	decay = 1 - i/1000
	x += (2*np.random.rand(4, 4096, 100)-1) * mask * decay

	return x


a = 100*(2*np.random.rand(4, 4096, 100)-1)

log_a = []
log_b = []
log_c = []
log_d = []

for i in range(100):
	print(i)

	a = advance_island_b(a)

	log_a.append( np.min(objective(a[0])) )
	log_b.append( np.min(objective(a[1])) )
	log_c.append( np.min(objective(a[2])) )
	log_d.append( np.min(objective(a[3])) )


plt.plot(log_a)
plt.plot(log_b)
plt.plot(log_c)
plt.plot(log_d)
plt.show()


def asd():
	#
	# Init
	#
	a = 100*(2*np.random.rand(4096, 100)-1)
	b = 100*(2*np.random.rand(4096, 100)-1)
	c = 100*(2*np.random.rand(4096, 100)-1)
	d = 100*(2*np.random.rand(4096, 100)-1)

	log_a = []
	log_b = []
	log_c = []
	log_d = []

	# 1000 iterations
	# 4 islands
	for i in range(200):
		a = advance_island(a)
		b = advance_island(b)
		c = advance_island(c)
		d = advance_island(d)

		log_a.append( np.min(objective(a)) )
		log_b.append( np.min(objective(b)) )
		log_c.append( np.min(objective(c)) )
		log_d.append( np.min(objective(d)) )

	print("Best fitness", np.min(objective(a)))
	print("Best fitness", np.min(objective(b)))
	print("Best fitness", np.min(objective(c)))
	print("Best fitness", np.min(objective(d)))

	plt.plot(log_a)
	plt.plot(log_b)
	plt.plot(log_c)
	plt.plot(log_d)
	plt.show()
