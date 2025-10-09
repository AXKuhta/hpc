#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cstdio>
#include <cstdlib>

struct linear_combo_3 {
	float a, b, c;

	linear_combo_3(float a, float b, float c) : a(a), b(b), c(c) {}

	__host__ __device__
	float operator() (thrust::tuple<float, float, float> t) {
		float x, y, z;
		thrust::tie(x, y, z) = t;
		return a*x + b*y + c*z;
	}
};

void usage(const char* filename)
{
	printf("Calculating a saxpy transform for two random vectors of the given length.\n");
	printf("Usage: %s <n>\n", filename);
}

int main(int argc, char* argv[])
{
	const int printable_n = 128;


	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}
	//thrust::cudaSetDevice(2);

	// Generate 3 vectors on host
	thrust::host_vector<float> X(n), Y(n), Z(n);

	thrust::generate(X.begin(), X.end(), []() -> float {return rand()*1.0f/RAND_MAX;});
	thrust::generate(Y.begin(), Y.end(), []() -> float {return rand()*1.0f/RAND_MAX;});
	thrust::generate(Z.begin(), Z.end(), []() -> float {return rand()*1.0f/RAND_MAX;});

	float a, b, c;

	printf("Enter a, b, c:\n");

	int rc = scanf("%f %f %f", &a, &b, &c);

	assert(rc == 3);

	// Print out the input data if n is small.
	if (n <= printable_n)
	{
		printf("Input data:\n");
		for (int i = 0; i < X.size(); i++)
			printf("%f   %f   %f\n", X[i], Y[i], Z[i]);
		printf("\n");
	}

	// Device upload
	thrust::device_vector<float> Xd(X), Yd(Y), Zd(Z);

	// Device output vector.
	thrust::device_vector<float> Rd(n);

	// Host readout vector.
	thrust::host_vector<float> R(n);

	// Use transform to make an saxpy operation
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(Xd.begin(), Yd.begin(), Zd.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(Xd.end(), Yd.end(), Zd.end())),
		Rd.begin(),
		linear_combo_3(a, b, c)
	);

	// Transfer data back to host.
	thrust::copy(Rd.begin(), Rd.end(), R.begin());

	// Print out the output data if n is small.
	if (n <= printable_n)
	{
		printf("Output data:\n");
		for (int i = 0; i < X.size(); i++)
			printf("%f*%f + %f*%f + %f*%f = %f\n", a, X[i], b, Y[i], c, Z[i], R[i]);
		printf("\n");
	}

	return 0;
}
