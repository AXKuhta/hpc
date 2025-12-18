
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
#include <curand_kernel.h>

extern "C" {
__global__ void my_add(const float* a, const float* b, const float* u, const float* v, float* z) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * 1000 + x;

	// drop random selection flips they're too troublesome in custom kernel
	//curandState_t state;
	//curand_init(42, idx, 0, &state);
	//z[idx] = u[y] < v[y] && (curand_uniform(&state) < 0.99f) ? a[idx] : b[idx];

	z[idx] = u[y] < v[y] ? a[idx] : b[idx];
}}
""", "my_add")


mutation_k = cp.RawKernel(r"""
extern "C" {
typedef unsigned long long uint64_t;

// Fast xorshift128+ PRNG
__device__ __forceinline__ uint64_t xorshift128plus(uint64_t* s0, uint64_t* s1) {
    uint64_t x = *s0;
    uint64_t y = *s1;
    *s0 = y;
    x ^= x << 23;
    *s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return *s1 + y;
}

__device__ __forceinline__ float rand_float(uint64_t* s0, uint64_t* s1) {
    // Convert to float in [0, 1)
    return (xorshift128plus(s0, s1) >> 40) * (1.0f / 16777216.0f);
}


__global__ void my_add(float* z, double decay, unsigned long long seed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * 1000 + x;

	// Initialize state from seed and thread index
	uint64_t s0 = seed + idx * 2;
	uint64_t s1 = seed + idx * 2 + 1;

	// Warm up (optional, improves distribution)
	xorshift128plus(&s0, &s1);


	//curandState_t state;
	//curand_init(42, idx, 0, &state);
	//z[idx] = u[y] < v[y] && (curand_uniform(&state) < 0.99f) ? a[idx] : b[idx];

	float r = rand_float(&s0, &s1);
	if (r < 0.1f) {
		float r2 = rand_float(&s0, &s1);
		z[idx] += (2*r2 - 1)*decay;
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
	a = x[:512]
	b = x[512:]
	u = z[:512]
	v = z[512:]

	# Winner should actually have index 0
	# So compare u > v
	winner = 0 + (u > v)

	# For some small fraction, the unfittest actually survives
	winner[cp.random.rand(512, dtype="float32") > 0.99] ^= 1
	x = cp.where(winner[:, None], b, a)

	"""
	z = cp.zeros((512, 1000), dtype="float32")

	# Try grid (1000, 512) block (1, 1) first
	# after that increase block size in steps
	selection_k((50, 16), (20, 32), (a, b, u, v, z))
	x = z
	"""

	#
	# Cross polination (probabilistic)
	#

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

	#mask = cp.random.rand(1024, 1000, dtype="float32") < 0.1
	#x += (2*cp.random.rand(1024, 1000, dtype="float32")-1) * mask * decay
	mutation_k((50, 32), (20, 32), (x, decay, i))

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
