
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
