
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from time import perf_counter

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

objective_k = cp.ReductionKernel(
	"float32 x",				# Input
	"float32 y",				# Output
	"x * x - 10*cos(2*M_PI*x) + 10",	# Map ( B = ... )
	"a + b",				# Reduce ( A = ... )
	"y = a",				# Fin expression of A
	"0",					# Initial value of A
)

selection_k = cp.RawKernel(r"""
extern "C" {
__global__ void my_add(float* z, const float* s, const float* r) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_top = y * 1000 + x;
	int idx_bottom = (y+512) * 1000 + x;

	if (s[y] > s[y+512] && r[y] < 0.99) {
		z[idx_top] = z[idx_bottom];
	}
}}
""", "my_add")


mutation_k = cp.RawKernel(r"""
extern "C" {
__global__ void my_add(float* z, const float* p, const float* q, float decay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * 1000 + x;

	if (p[idx] < 0.1f) {
		z[idx] += (2*q[idx] - 1)*decay;
	}
}}
""", "my_add")

def objective(x):
	return cp.sum(x*x - 10*cp.cos(2*cp.pi*x) + 10, -1)

def advance_island(x):
	#
	# Selection
	#
	z = objective_k(x, axis=-1)

	#
	# We run a kernel for half of creatures
	# Then we run a kernel for quarter of creatures
	#
	r = cp.random.rand(512, dtype="float32")
	selection_k((50, 16), (20, 32), (x, z, r))

	#
	# Cross polination (probabilistic)
	#

	x = x[:512]
	a = x[:256]
	b = x[256:]

	# Do it twice to create more creatures,
	# have the two be opposites of each other
	swap_map = cp.random.rand(256, 1000, dtype="float32") > 0.5
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

	decay = 1 - i/1000

	p = cp.random.rand(1024, 1000, dtype="float32")
	q = cp.random.rand(1024, 1000, dtype="float32")

	mutation_k((10, 128), (100, 8), (x, p, q, cp.float32(decay)))

	return x

#
# Init
#
a = 100*(2*cp.random.rand(1024, 1000, dtype="float32")-1)
b = 100*(2*cp.random.rand(1024, 1000, dtype="float32")-1)
c = 100*(2*cp.random.rand(1024, 1000, dtype="float32")-1)
d = 100*(2*cp.random.rand(1024, 1000, dtype="float32")-1)

log_a = []
log_b = []
log_c = []
log_d = []

start = perf_counter()

# 1000 iterations
# 4 islands
for i in range(1000):
	print(i)

	a = advance_island(a)
	b = advance_island(b)
	c = advance_island(c)
	d = advance_island(d)

	log_a.append( cp.min(objective(a)).get() )
	log_b.append( cp.min(objective(b)).get() )
	log_c.append( cp.min(objective(c)).get() )
	log_d.append( cp.min(objective(d)).get() )

	# Migration event
	if i % 25 == 0:
		u = a[:256]
		v = b[:256]
		w = c[:256]
		x = d[:256]

		pool = cp.vstack([u, v, w, x])
		cp.random.shuffle(pool)

		a[:256] = pool[    : 256]
		b[:256] = pool[ 256: 512]
		c[:256] = pool[ 512: 768]
		d[:256] = pool[ 768:1024]

elapsed = perf_counter() - start

print(f"Elapsed: {elapsed:.1f}s")

print("Best fitness", cp.min(objective(a)).get() )
print("Best fitness", cp.min(objective(b)).get() )
print("Best fitness", cp.min(objective(c)).get() )
print("Best fitness", cp.min(objective(d)).get() )

plt.semilogy(log_a)
plt.semilogy(log_b)
plt.semilogy(log_c)
plt.semilogy(log_d)
plt.show()
