#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

int seed = 0;
int j = 9739;
int k = 23209;
uint32_t m = 0xFFFFFFFF;
uint32_t * array;

uint32_t fib(){
  uint32_t result = (array[k - j] + array[0]) & m;
  for(int i = 1; i < k; i++) array[i-1] = array[i];
  array[k-1] = result;
  return result;
}

uint32_t fibpointed(uint32_t * jth, uint32_t * kth){
    return (*jth + *kth) & m;
}

int main(){
    //char* mode = "w";
    //FILE * fileptr = fopen("fibchunk.txt", mode);
  srand(seed);
  array = (uint32_t *)malloc(k*sizeof(uint32_t));
  for(int i = 0; i < k; i++) array[i] = rand() & m;
  for(int c = 0; c < 100000; c++){
    uint32_t * firstj = array + (k - j);
    uint32_t * firstk = array;
    uint32_t * temparray = (uint32_t *)malloc(k*sizeof(uint32_t));
    uint32_t * tempptr = temparray + (k - j);
    while(firstk < array + j){
      *(tempptr++) = fibpointed(firstj++, firstk++);
      //*(tempptr++) = (*(firstj++) + *(firstk++)) & m;
      //fprintf(fileptr, "%u\n", *(tempptr-1));
    }
    for(int i = 0; i < k - j; i++){
        temparray[i] = array[i + j];
    }
    uint32_t * trasharray = array;
    array = temparray;
    free(trasharray);
    //sleep(1);
  }
  //fclose(fileptr);
  return 0;
}
