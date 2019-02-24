#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "NNHelper.h"

#define IMAGESTOREAD 200
#define EPOCHS 1
#define LEARNINGRATE 0.2
#define LENGTH 28
#define INPUTLAYERNODES 28*28
#define HIDDENLAYERNODES 16
#define OUTPUTLAYERNODES 10

//index of highest value from an array
int getIndexOfHighestValue(float *array,int size){
	int index = 0;
	for(int i = 1; i < size; i++){
    	if(array[i] > array[index]){
    	    index = i;
    	}
	}
	return index;
}

//update the weights and biases of the network
void updateWeights(float *input, float *hidden, float *deltaHidden, float *deltaOutput, float *weightsFromInputToHidden, float *weightsFromHiddenToOutput, float *biasesOfHidden, float *biasesOfOutput){
    //update the last layer
	int nodeOfHidden = 0;
	int nodeOfOutput = 0;
	for(int i = 0; i < HIDDENLAYERNODES * OUTPUTLAYERNODES; i++){
        nodeOfHidden = i%HIDDENLAYERNODES;
        if(i%HIDDENLAYERNODES == 0 && i != 0){
			nodeOfOutput++;
		}
		//calculate weight change with learning rate, the calculated error and the hidden node value
		weightsFromHiddenToOutput[i] += LEARNINGRATE * (deltaOutput[nodeOfOutput] * sigmoid(hidden[nodeOfHidden]));
	}
	for(int i = 0; i < OUTPUTLAYERNODES; i++){
		biasesOfOutput[i] += LEARNINGRATE * deltaOutput[i];
	}
	//update the hidden layer
	nodeOfHidden = 0;
	int nodeOfInput = 0;
	for(int i = 0; i < HIDDENLAYERNODES * INPUTLAYERNODES; i++){
        nodeOfInput = i%INPUTLAYERNODES;
        if(i%INPUTLAYERNODES == 0 && i != 0){
	    	nodeOfHidden++;
		}
		//calculate weight change with learning rate, the calculated error and the input node value
		weightsFromInputToHidden[i] += LEARNINGRATE * (deltaHidden[nodeOfHidden] * input[nodeOfInput]);
	}
	for(int i = 0; i < HIDDENLAYERNODES; i++){
    	biasesOfHidden[i] += LEARNINGRATE * deltaHidden[i];
	}
}

//calculate the error for every node in the network
void backPropagateError(float *input, float *hidden, float *output, float *target, float *deltaHidden, float *deltaOutput, float *weightsFromHiddenToOutput){
    
	//error of last layer
    float errorOutput[OUTPUTLAYERNODES];
	for(int i = 0; i < OUTPUTLAYERNODES; i++){
        errorOutput[i] = (target[i] - sigmoid(output[i]));
    }

    for(int i = 0; i < OUTPUTLAYERNODES; i++){ 
		deltaOutput[i] = errorOutput[i] * sigmoid_prime(output[i]);
	}

	//error of hidden layer
	float errorHidden[HIDDENLAYERNODES];
	for(int i = 0; i < HIDDENLAYERNODES; i++){
		errorHidden[i] = 0.0;
		for(int j = 0; j < OUTPUTLAYERNODES; j++){
			errorHidden[i] += weightsFromHiddenToOutput[j*OUTPUTLAYERNODES+i] * deltaOutput[j];
		}
	}
	for(int i = 0; i < HIDDENLAYERNODES; i++){
		deltaHidden[i] = errorHidden[i] * sigmoid_prime(hidden[i]);
    }
}

//feed forward through the network
void feedForward(float *input, float* hidden, float* output, DATASET *trainData,int data,float *weightsFromInputToHidden,float *weightsFromHiddenToOutput,float *biasesOfHidden, float *biasesOfOutput){
    for (int j = 0; j < INPUTLAYERNODES; j++) {
		input[j] = (float)((int)trainData[data].value[j%LENGTH][j/LENGTH]) / 255.0f;
	}
    //setting up NNs fist hidden layer
	for (int j = 0; j < HIDDENLAYERNODES; j++) {
		//calculate NNs hidden layer
		float z = 0;
		for (int k = 0; k < INPUTLAYERNODES; k++) {
			z += input[k] * weightsFromInputToHidden[j*INPUTLAYERNODES+k];
		}
		z += biasesOfHidden[j];
		hidden[j] = z;
	}

    //setting up the output layer
	for (int j = 0; j < OUTPUTLAYERNODES; j++) {
		//calculate NNs output layer
		float z = 0;
		for (int k = 0; k < HIDDENLAYERNODES; k++) {
			z += sigmoid(hidden[k]) * weightsFromHiddenToOutput[j*HIDDENLAYERNODES+k];
		}
		z += biasesOfOutput[j];
		output[j] = z;
	}
}

//training loop
void trainNetwork(DATASET *trainData, float *weightsFromInputToHidden,
		float *weightsFromHiddenToOutput, float *biasesOfHidden,
		float *biasesOfOutput) {
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		float error = 0;
		for (int train = 0; train < IMAGESTOREAD; train++) {

			float input[INPUTLAYERNODES];
			float hidden[HIDDENLAYERNODES];
			float output[OUTPUTLAYERNODES];
			float target[OUTPUTLAYERNODES];
			for (int j = 0; j < OUTPUTLAYERNODES; j++) {
				if (trainData[train].label == j) {
					target[j] = 1;
				} else {
					target[j] = 0;
				}
			}
			feedForward(input, hidden, output, trainData, train,
					weightsFromInputToHidden, weightsFromHiddenToOutput,
					biasesOfHidden, biasesOfOutput);
				float oldError = 0;
			for(int i = 0; i < OUTPUTLAYERNODES; i++){
                error += pow(target[i] - sigmoid(output[i]),2);
				oldError += pow(target[i] - sigmoid(output[i]), 2);
			}
			float deltaHidden[HIDDENLAYERNODES];
			float deltaOutput[OUTPUTLAYERNODES];
			backPropagateError(input, hidden, output, target, deltaHidden,
					deltaOutput, weightsFromHiddenToOutput);
			updateWeights(input, hidden, deltaHidden, deltaOutput,
					weightsFromInputToHidden, weightsFromHiddenToOutput,
					biasesOfHidden, biasesOfOutput); 
		}
		printf("EPOCH %i | error = %f\n", epoch, 1.0/IMAGESTOREAD*error);
	}
}

//prints an image to the terminal
void printImage(DATASET image, int XLENGTH, int YLENGTH){
    for(int i = 0; i < YLENGTH ; i++){
        for(int j = 0; j < XLENGTH; j++){
            printf(" %03i ", (int)image.value[i][j]);
        }
        printf("\n");
    }
    printf("THE RESULT IS %f \n", (float)image.label);
    return;
}

//predict the result of a given image
int predict(float* data,float * weightsFromInputToHidden,float * weightsFromHiddenToOutput,float * biasesOfHidden,float * biasesOfOutput){
    float input[INPUTLAYERNODES];
    for (int j = 0; j < INPUTLAYERNODES; j++) {
		input[j] = data[j];
	}

    //setting up NNs fist hidden layer
	float hidden[HIDDENLAYERNODES];
	for (int j = 0; j < HIDDENLAYERNODES; j++) {
		//calculate NNs hidden layer
		float z = 0;
		for (int k = 0; k < INPUTLAYERNODES; k++) {
			z += input[k] * weightsFromInputToHidden[j*INPUTLAYERNODES+k];
		}
		z += biasesOfHidden[j];
		hidden[j] = z;
	}

    //setting up the output layer
	float output[OUTPUTLAYERNODES];
	for (int j = 0; j < OUTPUTLAYERNODES; j++) {
		//calculate NNs output layer
		float z = 0;
		for (int k = 0; k < HIDDENLAYERNODES; k++) {
			z += sigmoid(hidden[k]) * weightsFromHiddenToOutput[j*HIDDENLAYERNODES+k];
		}
		z += biasesOfOutput[j];
		output[j] = sigmoid(z);
		printf("OUTPUT[%i] = %f | %f = TARGET[%i]\n", j,output[j],data[INPUTLAYERNODES],j);
	}
    return getIndexOfHighestValue(output, OUTPUTLAYERNODES);
}

int main(int argc, char* argv[]) {

	//random time for random values later for weights and biases
	srand(time( NULL));

	//load the input files
	FILE* trainImages = fopen(argv[1], "r");
	FILE* trainLabels = fopen(argv[2], "r");

	//give feedback on the files
	if (trainImages == NULL || trainLabels == NULL) {
		printf("does not exist");
		return 1;
	} else {
		printf("Image file exists with byte length %i\n",FileSize(trainImages));
		printf("Label file exists with byte length %i\n",FileSize(trainLabels));
	}

	//read the header data from the EMNIST train labels
	TRAINLABELSHEADER tlh;
	fread(&tlh, sizeof(TRAINLABELSHEADER) - 1, 1, trainLabels);

	//read the header data from the EMNIST train images
	TRAINIMAGESHEADER tih;
	fread(&tih, sizeof(TRAINIMAGESHEADER), 1, trainImages);

	//load the training images to memory
	DATASET trainData[IMAGESTOREAD];
	BYTE b = 0;
	int pic = 0;
	int pixelPos = 0;
	while (fread(&b, sizeof(BYTE), 1, trainImages) != EOF && pic < IMAGESTOREAD) {

		//label the images
		if (pixelPos == 0 || pixelPos >= LENGTH * LENGTH) {
			BYTE l;
			fread(&l, sizeof(BYTE), 1, trainLabels);
			trainData[pic].label = l;
		}

		//write data from image file to the correct picture it belongs to
		if (pixelPos >= LENGTH * LENGTH) {
			pixelPos = 0;
			pic++;
		}
		trainData[pic].value[pixelPos%LENGTH][pixelPos/LENGTH] = (float)b;
		pixelPos++;
	}

	//FOR TESTING
	//printImage(trainData[0],28,28);

	//setting up the network
	float weightsFromInputToHidden[INPUTLAYERNODES * HIDDENLAYERNODES];
	for (int i = 0; i < INPUTLAYERNODES * HIDDENLAYERNODES; i++) {
		weightsFromInputToHidden[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	float weightsFromHiddenToOutput[HIDDENLAYERNODES * OUTPUTLAYERNODES];
	for (int i = 0; i < HIDDENLAYERNODES * OUTPUTLAYERNODES; i++) {
		weightsFromHiddenToOutput[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	float biasesOfHidden[HIDDENLAYERNODES];
	for (int i = 0; i < HIDDENLAYERNODES; i++) {
		biasesOfHidden[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	float biasesOfOutput[OUTPUTLAYERNODES];
	for (int i = 0; i < OUTPUTLAYERNODES; i++) {
		biasesOfOutput[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	trainNetwork(trainData, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput);
    printf("predicting:\n");
	int right = 0;
    for(int i = 0; i < IMAGESTOREAD; i++){
        float data[INPUTLAYERNODES+1];
		for(int x = 0; x < INPUTLAYERNODES; x++){
        	data[x] = (float)((int)trainData[i].value[x%LENGTH][x/LENGTH]) / 255.0f;
		}
        data[INPUTLAYERNODES] = (float)((int)trainData[i].label);
		//printImage(trainData[i], 28,28);
        int result = predict(data, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput);
		printf("result = %i | %i\n", result, (int)trainData[i].label);
		if(result == (int)trainData[i].label){
			right++;
		}
    }
	printf("Percentage %.2f\n", (float)right/(float)IMAGESTOREAD);
	return 0;
}
