
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
	a = x[:512]
	b = x[512:]

	u = objective(a)
	v = objective(b)

	# Winner should actually have index 0
	# So compare u > v
	winner = 0 + (u > v)

	# For some small fraction, the unfittest actually survives
	winner[np.random.rand(512) > 0.99] ^= 1
	x = np.where(winner[:, None], b, a)

	#
	# Cross polination (probabilistic)
	#

	a = x[:256]
	b = x[256:]

	# Do it twice to create more creatures,
	# have the two be opposites of each other
	swap_map = np.random.rand(256, 100) > 0.5
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

	mask = np.random.rand(1024, 100) < 0.1
	decay = 1 - i/1000
	x += (2*np.random.rand(1024, 100)-1) * mask * decay

	return x

#
# Init
#
a = 100*(2*np.random.rand(1024, 100)-1)
b = 100*(2*np.random.rand(1024, 100)-1)
c = 100*(2*np.random.rand(1024, 100)-1)
d = 100*(2*np.random.rand(1024, 100)-1)

log_a = []
log_b = []
log_c = []
log_d = []

# 1000 iterations
# 4 islands
for i in range(1000):
	print(i)

	a = advance_island(a)
	b = advance_island(b)
	c = advance_island(c)
	d = advance_island(d)

	log_a.append( np.min(objective(a)) )
	log_b.append( np.min(objective(b)) )
	log_c.append( np.min(objective(c)) )
	log_d.append( np.min(objective(d)) )

	# Migration event
	if i % 25 == 0:
		u = a[:256]
		v = b[:256]
		w = c[:256]
		x = d[:256]

		pool = np.vstack([u, v, w, x])
		np.random.shuffle(pool)

		a[:256] = pool[    : 256]
		b[:256] = pool[ 256: 512]
		c[:256] = pool[ 512: 768]
		d[:256] = pool[ 768:1024]

print("Best fitness", np.min(objective(a)))
print("Best fitness", np.min(objective(b)))
print("Best fitness", np.min(objective(c)))
print("Best fitness", np.min(objective(d)))

plt.plot(log_a)
plt.plot(log_b)
plt.plot(log_c)
plt.plot(log_d)
plt.show()
