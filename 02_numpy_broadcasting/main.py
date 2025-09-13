import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(-100, 100, 1000)
y = np.random.randint(-100, 100, 1000)

plt.scatter(x, y)
#plt.show()

# sqrt( (u-v)**2 + (k-l)**2 )
# sqrt( (dx)**2 + (dy)**2 )

dx = x - x[:, None]
dy = y - y[:, None]

dm = np.sqrt(dx**2 + dy**2)

#plt.imshow(dm)

# Find neighbors: sort inside rows
dm_sort = np.argsort(dm, 1)

# Clip to two neighbors
nearest = dm_sort[:, :3]

# Draw lines
for pt in nearest:
	i, k, l = pt

	plt.plot(
		[ x[i], x[k] ],
		[ y[i], y[k] ],
		c="red"
	)

	plt.plot(
		[ x[i], x[l] ],
		[ y[i], y[l] ],
		c="red"
	)

plt.show()
