from time import perf_counter
import concurrent.futures

import numpy as np
import cupy as cp

np.random.seed(42)

#
# Формулка:
# x.i = A.i x.i-1 + B.i
#
# Задача:
# найти каждый x.i
#

A = np.random.rand(10000000).tolist()
B = np.random.rand(10000000).tolist()

#
# Обычная последовательная версия
#
def test_sequential():
	x = 5
	X = [x]

	start = perf_counter()

	for i in range(10000000 - 1):
		x = A[i]*x + B[i]
		X.append(x)

	elapsed = perf_counter() - start

	print(f"test_sequential elapsed {elapsed*1000:.1f}ms")

	return X

#
# Параллельная версия, четыре исполнителя
#
# Из-за GIL не может разогнаться быстрее последовательной...
#
def test_parallel():
	x = 5

	X = [x]

	A_ = [
		A[       :2500000],
		A[2500000:5000000],
		A[5000000:7500000],
		A[7500000:       ],
	]

	B_ = [
		B[       :2500000],
		B[2500000:5000000],
		B[5000000:7500000],
		B[7500000:       ],
	]

	# здесь считаем суперпозиции
	def do_superpositions(A_, B_):
		a = 1
		b = 0

		for u, v in zip(A_, B_):
			a = u * a
			b = u * b + v

		return a, b

	start = perf_counter()

	with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
		a = executor.submit(do_superpositions, A_[0], B_[0])
		b = executor.submit(do_superpositions, A_[1], B_[1])
		c = executor.submit(do_superpositions, A_[2], B_[2])
		#d = executor.submit(do_superpositions, A_[3], B_[3])

	a = a.result()
	b = b.result()
	c = c.result()
	#d = d.result()

	# здесь будем в четыре потока находить элементы
	def do_affine_transforms(x, A, B):
		X = [x]

		for a, b in zip(A, B):
			x = a*x + b
			X.append(x)

		return X

	init_x_1 = 5
	init_x_2 = a[0] * init_x_1 + a[1]
	init_x_3 = b[0] * init_x_2 + b[1]
	init_x_4 = c[0] * init_x_3 + c[1]

	with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
		a = executor.submit(do_affine_transforms, init_x_1, A_[0], B_[0])
		b = executor.submit(do_affine_transforms, init_x_2, A_[1], B_[1])
		c = executor.submit(do_affine_transforms, init_x_3, A_[2], B_[2])
		d = executor.submit(do_affine_transforms, init_x_4, A_[3], B_[3])

	a = a.result()
	b = b.result()
	c = c.result()
	d = d.result()

	elapsed = perf_counter() - start

	print(f"test_parallel elapsed {elapsed*1000:.1f}ms")

	a.pop(-1)
	b.pop(-1)
	c.pop(-1)
	d.pop(-1)

	return a + b + c + d

#
# Параллельная версия на numpy, 1000 "исполнителей"
#
def test_numpy(k=1000):
	A_ = np.array(np.float32(A)).reshape(k, -1)
	B_ = np.array(np.float32(B)).reshape(k, -1)

	acc_a = np.float32(np.ones(k))
	acc_b = np.float32(np.zeros(k))

	n = len(A)//k

	start = perf_counter()

	# здесь считаем суперпозиции
	# inplace операции на ~10% быстрее
	for i in range(n):
		acc_a *= A_[:, i]
		acc_b *= A_[:, i]
		acc_b += B_[:, i]

	init_x = np.zeros_like(acc_a)
	init_x[0] = 5

	for i in range(k-1):
		init_x[i+1] = init_x[i] * acc_a[i] + acc_b[i]

	result = np.zeros_like(A_)
	result[:, 0] = init_x

	# здесь находим все элементы
	for i in range(1, n):
		result[:, i] = A_[:, i - 1] * result[:, i - 1] + B_[:, i - 1]

	elapsed = perf_counter() - start

	print(f"test_numpy elapsed {elapsed*1000:.1f}ms")

	return result.flatten()

#
# Параллельная версия на операциях cupy, прямой перенос с numpy
#
# Реализовать что-то похожее на обычную параллельную версию сложно,
# так как память между запусками блоков не синхронизируется, а это
# очень нужно, так как на очередном блоке нужны значения с предыдущего
#
def test_cupy(k=5000):
	A_ = cp.array(cp.float32(A)).reshape(k, -1)
	B_ = cp.array(cp.float32(B)).reshape(k, -1)

	acc_a = cp.ones(k)
	acc_b = cp.zeros(k)

	n = len(A)//k

	start = perf_counter()

	# здесь считаем суперпозиции
	# inplace операции на ~10% быстрее
	for i in range(n):
		acc_a *= A_[:, i]
		acc_b *= A_[:, i]
		acc_b += B_[:, i]

	init_x = cp.zeros_like(acc_a)
	init_x[0] = 5

	for i in range(k-1):
		init_x[i+1] = init_x[i] * acc_a[i] + acc_b[i]

	result = cp.zeros_like(A_)
	result[:, 0] = init_x

	# здесь находим все элементы
	for i in range(1, n):
		result[:, i] = A_[:, i - 1] * result[:, i - 1] + B_[:, i - 1]

	cp.cuda.Stream.null.synchronize()
	elapsed = perf_counter() - start

	print(f"test_cupy elapsed {elapsed*1000:.1f}ms")

	return result.flatten()

a = test_sequential()
b = test_parallel()
c = test_numpy()
d = test_cupy()

assert np.allclose(a, b)
assert np.allclose(a, c)
assert np.allclose(a, d)
