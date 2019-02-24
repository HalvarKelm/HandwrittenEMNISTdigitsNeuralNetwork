#include <stdint.h>

typedef uint8_t  BYTE;
typedef uint32_t DWORD;
typedef uint64_t HEADER;
typedef int32_t  LONG;
typedef uint16_t WORD;

typedef struct
{
    DWORD magicNumber;
    DWORD items;
} __attribute__((__packed__))
TRAINLABELSHEADER;

typedef struct
{
    DWORD magicNumber;
    DWORD items;
    DWORD rows;
    DWORD columns;
} __attribute__((__packed__))
TRAINIMAGESHEADER;

typedef struct
{
    BYTE value[28][28];
    BYTE label;
}__attribute__((__packed__))
DATASET;

//sigmoid activation function
//calculate the sigmoid for a given double
float sigmoid(float x){
    return (1.0/(1.0+(exp(-1.0 * x))));
}

//calculate the sigmoid prime for a given double
float sigmoid_prime(float x){
    return sigmoid(x) * (1.0 - sigmoid(x));
}

//reads the byte size of a file
int FileSize(FILE* file) {
	fseek(file, 0L, SEEK_END);
	int size = ftell(file);
	fseek(file, 0L, SEEK_SET);
	return size;
}