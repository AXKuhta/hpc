#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>


// Doesn't work? Try:
// sudo rmmod nvidia_uvm
// sudo modprobe nvidia_uvm


// High resolution time
static double hrtime() {
	struct timeval tv;

	if (gettimeofday(&tv, NULL) == -1) {
		perror("gettimeofday");
		return 1;
	}

	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int M[] = {1, 10, 100, 1000};
static int m_count = sizeof(M)/sizeof(int);
static double runtime[sizeof(M)/sizeof(int)] = {0.0};

//
// 3.7. Thrust and cuRAND Example
// https://docs.nvidia.com/cuda/pdf/CURAND_Library.pdf
//
// Also 2.2:
// https://docs.nvidia.com/cuda/archive/11.8.0/thrust/index.html
//
struct estimate_area : public thrust::unary_function<unsigned int, float> {
	float a, b, u, v;

	estimate_area(
		float a,
		float b,
		float u,
		float v
	) : a(a), b(b), u(u), v(v) {}

	__device__
	float f(float x) {
		return x*x/(x+1) + 1/x;
	}

	__device__
	float operator() (unsigned int thread_id) {
		int hits = 0;
		int n = 100;

		curandState s;

		curand_init(thread_id, 0, 0, &s);

		for (int j = 0; j < n; j++) {
			float pt_x = curand_uniform(&s) * (b - a) + a;
			float pt_y = curand_uniform(&s) * (v - u) + u;

			float y = f(pt_x);

			if (pt_y < y)
				hits++;
		}

		return hits;
	}
};

int main() {
	double a = 1, b = 3;
	double u = 0, v = 3;

	double box_area = (b-a)*(v-u);

	for (int i = 0; i < m_count; i++) {
		int m = M[i];
		int n = m*100;

		double start = hrtime();

		int hits = thrust::transform_reduce(thrust::device,
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(m),
			estimate_area(a, b, u, v),
			0.0f,
			thrust::plus<float>()
		);

		double elapsed = hrtime() - start;

		runtime[i] = elapsed*1000.0;

		printf("%d %d\n", n, hits);
		printf("area %lf\n", (0.0+hits)/n * box_area);
	}

	FILE* results = fopen("results_cuda.json", "w");

	fprintf(results, "{ \"x\": [");

	for (int i = 0; i < m_count; i++) {
		fprintf(results, "%d", M[i]*100);
		fprintf(results, i + 1 == m_count ? "]" : ",");
	}

	fprintf(results, ", \"y\": [");

	for (int i = 0; i < m_count; i++) {
		fprintf(results, "%lf", runtime[i]);
		fprintf(results, i + 1 == m_count ? "]" : ",");
	}

	fprintf(results, "}");

	fclose(results);
}
