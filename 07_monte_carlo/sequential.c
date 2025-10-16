#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

// High resolution time
static double hrtime() {
	struct timeval tv;

	if (gettimeofday(&tv, NULL) == -1) {
		perror("gettimeofday");
		return 1;
	}

	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static double f(double x) {
	return x*x/(x+1) + 1/x;
}

static int N[] = {100, 1000, 10000, 100000};
static int n_count = sizeof(N)/sizeof(int);
static double runtime[sizeof(N)/sizeof(int)] = {0.0};

int main() {
	double a = 1, b = 3;
	double u = 0, v = 3;

	double box_area = (b-a)*(v-u);

	for (int i = 0; i < n_count; i++) {
		int n = N[i];
		int n_hits = 0;

		double start = hrtime();

		for (int j = 0; j < n; j++) {
			double pt_x = drand48() * (b - a) + a;
			double pt_y = drand48() * (v - u) + u;

			double y = f(pt_x);

			if (pt_y < y)
				n_hits++;
		}

		double elapsed = hrtime() - start;

		runtime[i] = elapsed*1000.0;

		printf("%d %d\n", n, n_hits);
		printf("area %lf\n", (0.0+n_hits)/n * box_area);
	}

	FILE* results = fopen("results_c.json", "w");

	fprintf(results, "{ \"x\": [");

	for (int i = 0; i < n_count; i++) {
		fprintf(results, "%d", N[i]);
		fprintf(results, i + 1 == n_count ? "]" : ",");
	}

	fprintf(results, ", \"y\": [");

	for (int i = 0; i < n_count; i++) {
		fprintf(results, "%lf", runtime[i]);
		fprintf(results, i + 1 == n_count ? "]" : ",");
	}

	fprintf(results, "}");

	fclose(results);
}
