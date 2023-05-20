#include <stdio.h>
#include <math.h>
#include <time.h>
#include "nn.h"

int main(void)
{
    srand(time(NULL));

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN gradient = nn_alloc(arch, ARRAY_LEN(arch));
    nn_randomize(nn);

    NN_PRINT(nn);

    nn_free(gradient);
    nn_free(nn);

    return 0;
}
