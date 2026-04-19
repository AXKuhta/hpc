import matplotlib.pyplot as plt
import numpy as np

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
a = np.zeros([64, 64], dtype="float32")
b = np.zeros([128, 128], dtype="float32")
c = np.zeros([256, 256], dtype="float32")
d = np.zeros([512, 512], dtype="float32")

h, w = x_.shape

a[30:30+h, 30:30+w] = x_
b[60:60+h, 60:60+w] = x_
c[120:120+h, 120:120+w] = x_
d[250:250+h, 250:250+w] = x_

#
# Aight
#
# u(i, j) = .25 * u(i-1, j) + .25 * u(i+1, j) + .25 * u(i, j-1) + .25 * u(i, j+1) - u.prev(i, j)
# u(i, j) = .25a + .25b + .25c + .25d - u.prev(i, j)
#

#
# Shape of answer: a 64*64 = 4096 vector
# Shape of A: a 4096x4096 matrix
#

#
# Use offsets: +1, -1, +64, -64
#
A = np.eye(4096, k=1) + np.eye(4096, k=-1) + np.eye(4096, k=64) + np.eye(4096, k=-64)

# Forward iteration mode
# A = A/4
#
# plt.imshow((A@A@A@A@A@A@A@A@A@A@a.flatten()).reshape(64,64))
#

# CG mode
A = A - 5*np.eye(4096)

# Top boundary
# Bottom boundary
A[:64, :] = 0
A[-64:, :] = 0

# Left boundary
# Right boundary
for i in range(64):
	A[i*64, :] = 0
	A[i*64 + 63, :] = 0


A = np.float32(A)
a = np.float32(a)

ans, _, _, _ = np.linalg.lstsq(A, -a.flatten())

plt.imshow(ans.reshape(64, 64))
plt.show()
