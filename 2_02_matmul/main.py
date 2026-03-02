from time import perf_counter
import json

import numpy as np
import cupy as cp

np.random.seed(42)

#
# numpy native matmul
#
def test_native(n = 1024):
	a = np.random.rand(n, n)
	b = np.random.rand(n, n)

	start = perf_counter()
	c = a @ b
	elapsed = perf_counter() - start

	print(f"Numpy native: {elapsed*1000:.1f}ms")

	return elapsed

#
# i-j-k matmul
#
def test_ijk(n = 1024):
	a = np.random.rand(n, n)
	b = np.random.rand(n, n)
	c = np.zeros_like(a)

	start = perf_counter()

	for i in range(n):
		for j in range(n):
			c[i, j] = np.dot(a[i, :], b[:, j])

	elapsed = perf_counter() - start

	assert np.allclose(c, a@b), "Veriftication failed"

	print(f"Numpy ijk: {elapsed*1000:.1f}ms")

	return elapsed

#
# k-i-j
#
def test_kij(n = 1024):
	a = np.random.rand(n, n)
	b = np.random.rand(n, n)
	c = np.zeros_like(a)

	start = perf_counter()

	for k in range(n):
		c += np.outer(a[:, k], b[k, :])

	elapsed = perf_counter() - start

	assert np.allclose(c, a@b), "Veriftication failed"

	print(f"Numpy kji: {elapsed*1000:.1f}ms")

	return elapsed

#
# cupy native matmul
#
def cupy_native(n = 1024):
	a = cp.random.rand(n, n, dtype=cp.float32)
	b = cp.random.rand(n, n, dtype=cp.float32)
	c = cp.zeros_like(a)

	cp.cuda.Stream.null.synchronize() # Cupy будет откладывать вычисления, если не заставить его выполнить их сейчас!

	start = perf_counter()
	c = a @ b
	cp.cuda.Stream.null.synchronize()
	elapsed = perf_counter() - start

	print(f"Cupy native: {elapsed*1000:.1f}ms")

	return elapsed


#
# cupy striped
#
def cupy_striped(n = 1024):
	a = cp.random.rand(n, n, dtype=cp.float32)
	b = cp.random.rand(n, n, dtype=cp.float32)
	c = cp.zeros_like(a)

	stripe_matmul_k = cp.RawKernel(fr"#define SZ ({n})" + r"""
	extern "C" {
	__global__ void matmul(float* c, const float* a, const float* b) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int idx = y * SZ + x;

		for (int i = 0; i < SZ; i++)
			c[idx] = c[idx] + a[y * SZ + i] * b[i * SZ + x];
	}}
	""", "matmul")


	# Делаем 256 нитей на блок
	size = np.array([n, n])
	rect = (16, 16)

	grid = size/rect

	assert np.all(np.int64(grid) == grid)

	grid = tuple(np.int64(grid))

	start = perf_counter()
	stripe_matmul_k(grid, rect, (c, a, b))
	cp.cuda.Stream.null.synchronize()
	elapsed = perf_counter() - start

	print(f"Cupy striped: {elapsed*1000:.1f}ms")

	assert cp.allclose(c, a @ b), "Verification failed"

	return elapsed

#
# cupy tiled
#
def cupy_tiled(n = 1024):
	a = cp.random.rand(n, n, dtype=cp.float32)
	b = cp.random.rand(n, n, dtype=cp.float32)
	c = cp.zeros_like(a)

	tile_sm_matmul_k = cp.RawKernel(fr"#define N ({n})" + r"""
	extern "C" {

	// Must match block dimensions
	#define SZ (16)

	__global__ void matmul(float* c, const float* a, const float* b) {

		// Result location
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int idx = y * N + x;

		// Tile offset
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// Tile index
		int bx = blockIdx.x;
		int by = blockIdx.y;

		int aBegin = N * SZ * by;
		int aEnd = aBegin + N - 1;
		int bBegin = SZ * bx;
		int aStep = SZ, bStep = SZ * N;
		float sum = 0.0f;

		for ( int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep ){

			// Loading subsequent tile into Shared Memory
			// Shared memory only valid inside a block
			__shared__ float as [SZ][SZ];
			__shared__ float bs [SZ][SZ];
			as [ty][tx] = a [ia + N * ty + tx];
			bs [ty][tx] = b [ib + N * ty + tx];
			__syncthreads();

			// Reducing into one sum from a bunch of threds
			for ( int k = 0; k < SZ; k++ )
				sum += as [ty][k] * bs [k][tx];

			__syncthreads();
		}

		// Writeback
		c[idx] = sum;
	}}
	""", "matmul")

	# Делаем 256 нитей на блок
	size = np.array([n, n])
	rect = (16, 16)

	grid = size/rect

	assert np.all(np.int64(grid) == grid)

	grid = tuple(np.int64(grid))

	start = perf_counter()
	tile_sm_matmul_k(grid, rect, (c, a, b))
	cp.cuda.Stream.null.synchronize()
	elapsed = perf_counter() - start

	print(f"Cupy tiled: {elapsed*1000:.1f}ms")

	assert cp.allclose(c, a @ b), "Verification failed"

	return elapsed

size = [128, 256, 512, 1024, 2048]

native = [test_native(x) for x in size]
np_ijk = [test_ijk(x) for x in size]
np_kij = [test_kij(x) for x in size]

# Cuda tests run twice to wake gpu
cp_native = [cupy_native(x) for x in size]
cp_native = [cupy_native(x) for x in size]

cp_striped = [cupy_striped(x) for x in size]
cp_striped = [cupy_striped(x) for x in size]

cp_tiled = [cupy_tiled(x) for x in size]
cp_tiled = [cupy_tiled(x) for x in size]

results = dict(
	size=size,
	native=native,
	np_ijk=np_ijk,
	np_kij=np_kij,
	cp_native=cp_native,
	cp_striped=cp_striped,
	cp_tiled=cp_tiled
)

with open("results.json", "w") as f:
	json.dump(results, f)
