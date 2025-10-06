---------MATRIX VECTOR MULTIPLICATION----------------

01)ROUND ROBIN WITH ROW AND THREADS

#include <omp.h>
#include <stdio.h>

int main() {
    int nrows = 10;  // More rows than threads
    int cols = 3;
    int matrix[10][3] = { {1, 0, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1},
                         {1, 1, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0},
                         {0, 0, 1}, {1, 0, 1} };
    int vector[3] = { 1, 2, 3 };
    int result[10] = { 0 };

    omp_set_num_threads(4);  // Fewer threads than rows

    #pragma omp parallel
    {
        int myid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        for (int i = myid; i < nrows; i += nthreads) {
            result[i] = 0;
            for (int j = 0; j < cols; j++) {
                result[i] += matrix[i][j] * vector[j];  
            }
            printf("\n Thread %d >>>>> result[%d] = %d", myid, i, result[i]);
        }
    }
    return 0;
}
----------------------------------------------------------------------
02)USING REDUCTION

#include <stdio.h>
#include <omp.h>

int main() {
    int n = 4;
    int A[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int x[4] = { 1, 1, 1, 1 };
    int y[4] = { 0 };

    // Parallelize the outer loop
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int sum = 0;

        // Use reduction inside inner loop
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }

        y[i] = sum;
    }

    printf("Result vector: ");
    for (int i = 0; i < n; i++) 
    printf("%d ", y[i]);
    printf("\n");

    return 0;
}
---------------------------DOT PRODUCT----------------------------

01)Using Parallel

#include <omp.h>
#include <stdio.h>

const int n = 10;  
const int nthreads = 4;
    
int main() {
    int a[n] = { 2, 5, 3, 6, 1, 9, 4, 7, 3, 5 };
    int b[n] = { 4, 2, 0, 6, 9, 3, 1, 4, 2, 8 };
    int c = 0;   // = 168

    int result[nthreads] = { 0 };

    omp_set_num_threads(4); 

#pragma omp parallel
    {
        int myId = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        for (int i = myId; i < n; i += nthreads) {
            result[myId] += a[i] * b[i];
        }
    }

    for (int j = 0; j < nthreads; j++) 
	c += result[j];

    printf("\n Dot product is %d", c);

    return 0;
}
-------------------------------------------------------------
02)Using Reduction


#include <stdio.h>
#include <omp.h>

int main() {
    int n = 5;
    int x[5] = { 1,2,3,4,5 };
    int y[5] = { 10,20,30,40,50 };
    int result = 0;

#pragma omp parallel for reduction(+:result)

    for (int i = 0; i < n; i++)
        result += x[i] * y[i];
    printf("Dot product = %d\n", result);
    return 0;
}

------------------------PI------------------------------
01)Using REDUCTION

#include <stdio.h>
#include <omp.h>

long N = 100000000;  // number of steps
double dx;

int main() {
    long i;
    double pi, sum = 0.0;
    double start_time, run_time;

    dx = 1.0 / (double)N;

    start_time = omp_get_wtime();

    // Parallel for loop with reduction
    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < N; i++) {
        double x = (i + 0.5) * dx;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = sum * dx;
    run_time = omp_get_wtime() - start_time;

    printf("\npi with %ld steps is %.15lf in %lf seconds\n", N, pi, run_time);
    return 0;
}
-------------------------------------------------------------------------

02)PARALLEL

#include <stdio.h>
#include <omp.h>

long N = 100000000;
double dx;

const int nt = 4;

int main()
{
	double pi = 0.0, sum[nt] = { 0.0 };
	double start_time, run_time;

	dx = 1.0 / (double)N;

	start_time = omp_get_wtime();

#pragma omp parallel num_threads(nt)
	{
		int myId = omp_get_thread_num();
		int i;
		double x;

		for (i = myId; i < N; i += nt) {
			x = (i + 0.5) * dx;
			sum[myId] += 4.0 / (1.0 + x * x);
		}
	}

	for (int j = 0; j < nt; j++) 
		pi += sum[j] * dx;

	run_time = omp_get_wtime() - start_time;
	printf("\n pi with %ld steps is %lf in %lf seconds\n ", N, pi, run_time);
}

----------------------------------------------------------------------
03)PARALLEL WITH PADDING 

#include <stdio.h>
#include <omp.h>

long N = 100000000;
double dx;

const int nt = 4;

int main()
{
	double pi = 0.0;
	double start_time, run_time;

	double sum[nt][8] = { {0} };

	dx = 1.0 / (double)N;

	start_time = omp_get_wtime();

	omp_set_num_threads(nt);

	#pragma omp parallel
	{
		int myId = omp_get_thread_num();

		double x;
		int i;

		for (i = myId; i < N; i += nt) {
			x = (i + 0.5) * dx;
			sum[myId][0] += 4.0 / (1.0 + x * x);
		}
	}

	for (int j = 0; j < nt; j++)
		pi += sum[j][0] * dx;

	run_time = omp_get_wtime() - start_time;
	printf("\n pi with %ld steps is %lf in %lf seconds\n ", N, pi, run_time);
}
 -----------------------------------------------------------------
04)PARALLEL WITH CRITIACL

#include <stdio.h>
#include <omp.h>

long N = 100000000;
double dx;

const int nt = 4;

int main()
{
	double pi = 0.0;
	double start_time, run_time;

	dx = 1.0 / (double)N;

	start_time = omp_get_wtime();

	#pragma omp parallel num_threads(nt)
	{
		int myId = omp_get_thread_num();
		int i;
		double x, sum = 0.0;

		for (i = myId; i < N; i += nt) {
			x = (i + 0.5) * dx;
			sum += 4.0 / (1.0 + x * x);
		}

		#pragma omp critical
			pi += sum * dx;
	}


	run_time = omp_get_wtime() - start_time;
	printf("\n pi with %ld steps is %lf in %lf seconds\n ", N, pi, run_time);
}
--------------------------------------------------------------------------
01)#include <stdio.h>
   #include <omp.h>

int main() {
    int n = 10;
    int a[10] = { 1,2,3,4,5,6,7,8,9,10 };
    int sum = 0;

    //Compute the sum of all elements in the array

#pragma omp parallel for reduction(+:sum)

    for (int i = 0; i < n; i++)
        sum += a[i];

    printf("Sum of array = %d\n", sum);
    return 0;
}
-----------------------------------------------------------------
02)     #include <stdio.h>
		#include <omp.h>
		#include <limits.h>

		int main() {
			int n = 8;
			int a[8] = { 3, 7, 2, 9, 12, 5, 8, 4 };
			int maxVal = INT_MIN;

			// Find the maximum value using OpenMP reduction
			#pragma omp parallel for reduction(max:maxVal)
			for (int i = 0; i < n; i++) {
				if (a[i] > maxVal)
					maxVal = a[i];
			}

			printf("Maximum value = %d\n", maxVal);
			return 0;
		}


03)
#include <stdio.h>
#include <omp.h>

int main() {
    int n = 6;
    double a[6] = { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0 };
    double sum = 0.0, sumSq = 0.0, mean, variance;

    // Parallelize the loop with reduction
#pragma omp parallel for reduction(+:sum,sumSq)
    for (int i = 0; i < n; i++) {
        sum += a[i];
        sumSq += a[i] * a[i];
    }

    mean = sum / n;
    variance = (sumSq / n) - (mean * mean);

    printf("Mean = %.2f, Variance = %.2f\n", mean, variance);
    return 0;
}
-----------------------RACE WITH ARRAY--------------------------------

#include <stdio.h>
#include <omp.h>

const int nt = 8;   //Number of threads

int main() {
    long sum = 0;	//Shared variable
    long partial_sum[nt] = { 0 };   //Shared array

    double start_time = omp_get_wtime(); // Start timing

    #pragma omp parallel num_threads(nt)
    {
        int myID = omp_get_thread_num();

        for (int i = 0; i < 100000000; i++) {
            partial_sum[myID] ++;
        }
    }

    for (int j = 0; j < nt; j++)
        sum += partial_sum[j];	//Calculating final sum

    double end_time = omp_get_wtime(); // End timing

    printf("Final value of shared_variable is %ld.\n", sum);
    printf("Runtime: %f seconds\n", end_time - start_time); // Calculate and print runtime

    return 0;
}
-----------------------RACE WITH PADED-------------------------------------
#include <stdio.h>
#include <omp.h>

const int nt = 8;   //Number of threads
const int pad_size = 8;     //Cache line size

int main() {
    long sum = 0;	//Shared variable
    long partial_sum[nt][pad_size] = { {0} };   //Shared array

    double start_time = omp_get_wtime(); // Start timing

    #pragma omp parallel num_threads(nt)
    {
        int myID = omp_get_thread_num();

        for (int i = 0; i < 100000000; i++) {
            partial_sum[myID][0] ++;
        }
    }

    for (int j = 0; j < nt; j++)
        sum += partial_sum[j][0];   //Calculating final sum

    double end_time = omp_get_wtime(); // End timing

    printf("Final value of shared_variable is %ld.\n", sum);
    printf("Runtime: %f seconds\n", end_time - start_time); // Calculate and print runtime

    return 0;
}

#include <stdio.h>
#include <omp.h>

int main() {
    int N = 20;
    int a[20];

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }

    printf("Array elements:\n");
    for (int i = 0; i < N; i++)
        printf("%d ", a[i]);
    printf("\n");

    return 0;
}
#include <stdio.h>
#include <omp.h>

int main() {
    int A[5][5];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            A[i][j] = i + j;
            printf("Thread %d set A[%d][%d] = %d\n", omp_get_thread_num(), i, j, A[i][j]);
        }
    }

    printf("\nMatrix A:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%3d ", A[i][j]);
        printf("\n");
    }

    return 0;
}
#include <stdio.h>
#include <omp.h>

#define N 3

int main() {
    int A[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[N][N] = {0};

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }

    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%3d ", C[i][j]);
        printf("\n");
    }

    return 0;
}
#include <stdio.h>
#include <omp.h>

int main() {
    int N = 8;
    int pascal[9][9] = {0};

    pascal[0][0] = 1;

    for (int i = 1; i <= N; i++) {
        pascal[i][0] = 1;
        pascal[i][i] = 1;

        #pragma omp parallel for
        for (int j = 1; j < i; j++) {
            pascal[i][j] = pascal[i - 1][j - 1] + pascal[i - 1][j];
        }
    }

    printf("Pascal's Triangle (N=8):\n\n");
    for (int i = 0; i <= N; i++) {
        // Print leading spaces for centering
        for (int s = 0; s < N - i; s++)
            printf("   ");  // Adjust spacing for alignment

        // Print numbers
        for (int j = 0; j <= i; j++)
            printf("%6d", pascal[i][j]);
        printf("\n");
    }

    return 0;
}
--------------------------------------------------------------------------

#include <stdio.h>
#include <omp.h>  

int main() {

    omp_set_num_threads(4);

    #pragma omp parallel for schedule(static, 2)
    for (int i = 0; i < 10; i++) {
        printf("Thread %d processing %d\n", omp_get_thread_num(), i);
    }
    return 0;
}
-------------------------------------
#include <omp.h>
#include <stdio.h>

int main() {
    int x = 0;
    printf("\n Initially, x = %d", x);

    omp_set_num_threads(4);

#pragma omp parallel
    {
        int myid = omp_get_thread_num();
        x = x + myid; 
        printf("\n Thread %d >>>>> x + myid = %d", myid, x);
    }

    return 0;
}
----------------bigjob with barrier-------------------------
#include <stdio.h>
#include <omp.h>

void doBigJob(int id) {
    long long dummy, workload = (id + 1) * 100000000LL; 
    for (long long i = 0; i < workload; i++)
        dummy = i * i; 
}

int main() {
    #pragma omp parallel num_threads(4)
    {
        int id = omp_get_thread_num();

        doBigJob(id);

        printf("First job done by %d\n", id);

        #pragma omp barrier
        doBigJob(id);

        printf("Second job done by %d\n", id);
    }

    return 0;
}
-------------no race with critical-----------------------
#include <stdio.h>
#include <omp.h>

int main() {
    long final_sum = 0;     //Shared variable

    double start_time = omp_get_wtime(); // Start timing

    #pragma omp parallel num_threads(8)
    {
        long local_sum = 0;     //private variable

        for (int i = 0; i < 1000000; i++) {
            local_sum++; 
        }
        #pragma omp critical
            final_sum += local_sum;
    }

    double end_time = omp_get_wtime(); // End timing

    printf("Final value of shared_variable is %d.\n", final_sum);
    printf("Runtime: %f seconds\n", end_time - start_time);

    return 0;
}
---------------------------parallel for nowait-------------------------
#include <omp.h>
#include <stdio.h>

#define WORKLOAD_SIZE 8

// Function to compute factorial (simplified for large numbers)
unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main() {
    int workload[WORKLOAD_SIZE] = { 100000, 10000, 20000, 800000, 1000, 400000, 5000, 25000 }; // Example workload sizes
    double start, end;

    // Measure time without 'nowait'
    start = omp_get_wtime();

    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for (int i = 0; i < WORKLOAD_SIZE; i++) {
            printf("Thread %d calculating factorial of %d\n", omp_get_thread_num(), workload[i]);
            unsigned long long result = factorial(workload[i]);
        }
    }

    end = omp_get_wtime();
    printf("Time without nowait: %f seconds\n\n", end - start);

    // Measure time with 'nowait'
    start = omp_get_wtime();
    
    #pragma omp parallel num_threads(8)
    {
        #pragma omp for nowait
        for (int i = 0; i < WORKLOAD_SIZE; i++) {
            printf("Thread %d calculating factorial of %d\n", omp_get_thread_num(), workload[i]);
            unsigned long long result = factorial(workload[i]);
        }
        // No barrier here due to nowait
    }

    end = omp_get_wtime();
    printf("Time with nowait: %f seconds\n", end - start);

    return 0;
}
-------------------race---------------------
#include <stdio.h>
#include <omp.h>

int main() {
    long sum = 0;	//Shared variable

    double start_time = omp_get_wtime(); // Start timing

    #pragma omp parallel num_threads(8)
    {
        for (int i = 0; i < 1000000; i++) {
            sum++; 
        }
    }

    double end_time = omp_get_wtime(); // End timing

    printf("Final value of shared_variable is %ld.\n", sum);
    printf("Runtime: %f seconds\n", end_time - start_time); // Calculate and print runtime

    return 0;
}
