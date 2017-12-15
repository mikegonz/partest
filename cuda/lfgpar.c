#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>

/*
 * The Lagged Fibonacci Generator has a state that is an array of lag2
 * elements. The state begins as initvals. To produce a new random number,
 * the function returns
 *     (state[lag2-lag1] + state[0]) mod modval
 * It then shift all the states left by 1 to add the new
 * values to the state values.
 * NOTE: lag1 should be less than lag2
 *       modval should be a power of 2 minus 1
 *       state should be an array with lag2 elements
 */

int iters = 100000;
int nprocs;
int rank;
int TAG_J = 111;
int TAG_K = 222;
int TAG_RESULT = 555;
uint32_t modval = 4294967296 - 1;

uint32_t lag1 = 9739; //5;
uint32_t lag2 = 23209; //17;
uint32_t * statearray;

uint32_t lfg(uint32_t* lag1, uint32_t* lag2, uint32_t* modval,
             uint32_t* state)
{
    uint32_t result = (state[(*lag2) - (*lag1)] + state[0]) & *modval;
    //#pragma omp parallel for shared(state,lag2)
    for(int i = 0; i < *lag2; i++)
    {
        state[i] = state[i + 1];
    }

    state[*lag2 - 1] = result;
    return result;
}

void * lfg_threaded(void * input){
    int offset = *((int *)input);
    uint32_t * result = (uint32_t *)malloc(sizeof(uint32_t));
    *result = (statearray[lag2 - lag1 + offset] + statearray[offset]) & modval;
    //printf("%u\n", *result);
    //fflush(stdout);
    return (void*)(result);
}

void manager(){
    //char* mode = "w";
    //FILE* filepntr = fopen("lfgpar_out.txt", mode);
    int numWorkers = nprocs - 1;
    srand(0);
    statearray = (uint32_t *)malloc(lag2 * sizeof(uint32_t));
    for(int i = 0; i < lag2; i++) statearray[i] = rand(); //this should suck maybe do something about this
    for(int c = 0; c < iters; c++){
        uint32_t * temparray = (uint32_t *)malloc(lag2 * sizeof(uint32_t));
        MPI_Request * sendreqs = (MPI_Request *)malloc(2 * numWorkers * sizeof(MPI_Request));
        MPI_Request * recvreqs = (MPI_Request *)malloc(numWorkers * sizeof(MPI_Request));
        for(int p = 0; p < numWorkers; p++){
            int first = floor(p * lag1 / numWorkers);
            int numowned = floor((p+1) * lag1 / numWorkers) - first;
            MPI_Isend(statearray + (lag2 - lag1) + first,
                    numowned,
                    MPI_UINT32_T, p, TAG_J, MPI_COMM_WORLD, sendreqs+2*p);
            MPI_Isend(statearray + first, numowned,
                    MPI_UINT32_T, p, TAG_K, MPI_COMM_WORLD, sendreqs+2*p + 1);
            MPI_Irecv(temparray + (lag2 - lag1) + first, numowned,
                    MPI_UINT32_T, p, TAG_RESULT, MPI_COMM_WORLD, recvreqs+p);
        }
#pragma omp parallel for shared(lag2,lag1,temparray,statearray)
        for(int i = 0; i < (lag2 - lag1); i++){
            temparray[i] = statearray[i + lag1];
        }
        /*
        MPI_Status * status = (MPI_Status *)malloc(sizeof(MPI_Status));
        int doneIndex;
        for(int i = 0; i < numWorkers; i++){
            MPI_Waitany(numWorkers, recvreqs, &doneIndex, status);
        }
        */
        MPI_Status * statuses = (MPI_Status *)malloc(numWorkers * sizeof(MPI_Status));
        MPI_Waitall(numWorkers, recvreqs, statuses);
        uint32_t * trasharray = statearray;
        statearray = temparray;
        free(trasharray);
        free(sendreqs);
        free(recvreqs);
        free(statuses);
        //for(int i = lag2 - lag1; i < lag2; i++) fprintf(filepntr, "%u\n", temparray[i]);
    }
    //fclose(filepntr);
}

void worker(){
    for(int c = 0; c < iters; c++){
        int first = floor(rank * lag1 / (nprocs-1));
        int numowned = floor((rank+1) * lag1 / (nprocs-1)) - first;
        uint32_t * js = (uint32_t *)malloc(numowned * sizeof(uint32_t));
        uint32_t * ks = (uint32_t *)malloc(numowned * sizeof(uint32_t));
        uint32_t * results = (uint32_t *)malloc(numowned * sizeof(uint32_t));
        MPI_Request * reqs = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
        MPI_Irecv(js, numowned, MPI_UINT32_T, nprocs-1, TAG_J, MPI_COMM_WORLD, reqs);
        MPI_Irecv(ks, numowned, MPI_UINT32_T, nprocs-1, TAG_K, MPI_COMM_WORLD, reqs + 1);
        int doneIndex;
        MPI_Status * statuses = (MPI_Status *)malloc(2 * sizeof(MPI_Status));
        MPI_Waitany(2, reqs, &doneIndex, statuses);
        if(doneIndex == 0) for(int i = 0; i < numowned; i++) results[i] = js[i];
        else for(int i = 0; i < numowned; i++) results[i] = ks[i];
        MPI_Waitany(2, reqs, &doneIndex, statuses + 1);
        if(doneIndex == 0) for(int i = 0; i < numowned; i++) results[i] = (results[i] + js[i]) & modval;
        else for(int i = 0; i < numowned; i++) results[i] = (results[i] + ks[i]) & modval;
        MPI_Send(results, numowned, MPI_UINT32_T, nprocs-1, TAG_RESULT, MPI_COMM_WORLD);
        free(js);
        free(ks);
        free(results);
    }
}

int main(int argc, char * argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == nprocs-1) manager();
    else worker();
    MPI_Finalize();
    return 0;
}
