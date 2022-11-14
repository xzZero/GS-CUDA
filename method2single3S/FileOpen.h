#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <cufft.h>

typedef double real_t;

void FileOpen(char *link, float *matrix) { //int row, int column, int nnz
	FILE *fp;
	fp = fopen(link, "r");
	char c;
	char num_string[30];
	float num;
	int i = 0;
	int index = 0;
	//int *matrix = (int*)malloc(600000 * sizeof(int));
	//int row = 0, column = 0;
	//int nnz = 0;
	//*row = 0; *column = 0;
	//*nnz = 0;

	while (1) {
		c = fgetc(fp);
		if (c == EOF) {
			break;
		}
		if ((c != ',') && (c != EOF) && (c != '\n')) {
			num_string[i] = c;
			i++;
		}
		else {

			num_string[i] = '\0';
			num = (float)atof(num_string);
			//printf("%d\t", num);
			//if (num != 0) {
			//	*nnz = *nnz + 1; 
			//}
			matrix[index++] = num;
			i = 0;
			strcpy(num_string, "\0");
			if (c == '\n') {
				//	printf("\n");
				//*row = *row+1;

			}
			//

			if (feof(fp)) {
				break;
			}

		}



	}
	//*column = (index + 1) / (*row);
	//printf("\n number of row: %d", *row);
	//printf("\n number of column: %d", *column);
	//printf("\n number of non zero: %d", *nnz);


	/*for (i = 0; i < 64; i++) {
	printf("\na[%d]: %f", i, matrix[i]);
	}*/



	fclose(fp);
}

void FileOpen(char *link, double *matrix) { //int row, int column, int nnz
	FILE *fp;
	fp = fopen(link, "r");
	char c;
	char num_string[30];
	double num;
	int i = 0;
	int index = 0;
	//int *matrix = (int*)malloc(600000 * sizeof(int));
	//int row = 0, column = 0;
	//int nnz = 0;
	//*row = 0; *column = 0;
	//*nnz = 0;

	while (1) {
		c = fgetc(fp);
		if (c == EOF) {
			break;
		}
		if ((c != ',') && (c != EOF) && (c != '\n')) {
			num_string[i] = c;
			i++;
		}
		else {

			num_string[i] = '\0';
			num = (double)atof(num_string);
			//printf("%d\t", num);
			//if (num != 0) {
			//	*nnz = *nnz + 1; 
			//}
			matrix[index++] = num;
			i = 0;
			strcpy(num_string, "\0");
			if (c == '\n') {
				//	printf("\n");
				//*row = *row+1;

			}
			//

			if (feof(fp)) {
				break;
			}

		}



	}
	//*column = (index + 1) / (*row);
	//printf("\n number of row: %d", *row);
	//printf("\n number of column: %d", *column);
	//printf("\n number of non zero: %d", *nnz);


	/*for (i = 0; i < 64; i++) {
	printf("\na[%d]: %f", i, matrix[i]);
	}*/



	fclose(fp);
}
void FileOpen(char *link, long *matrix_) { //int row, int column, int nnz 
	FILE *fp;
	double *matrix = (double*)malloc(56000 * sizeof(double));
	fp = fopen(link, "r");
	char c;
	char num_string[30];
	double num;
	long i = 0;
	long index = 0;
	//int *matrix = (int*)malloc(600000 * sizeof(int));
	//int row = 0, column = 0;
	//int nnz = 0;
	//*row = 0; *column = 0;
	//*nnz = 0;

	while (1) {
		c = fgetc(fp);
		if (c == EOF) {
			break;
		}
		if ((c != ',') && (c != EOF) && (c != '\n')) {
			num_string[i] = c;
			i++;
		}
		else {

			num_string[i] = '\0';
			num = atof(num_string);
			//printf("%d\t", num);
			//if (num != 0) {
			//	*nnz = *nnz + 1;
			//}
			matrix[index++] = num;
			i = 0;
			strcpy(num_string, "\0");
			if (c == '\n') {
				//	printf("\n");
				//*row = *row + 1;

			}
			//

			if (feof(fp)) {
				break;
			}

		}



	}
	//*column = (index + 1) / (*row);
	//printf("\n number of row: %d", *row);
	//printf("\n number of column: %d", *column);
	//printf("\n number of non zero: %d", *nnz);
	for (i = 0; i < index; i++) {
		matrix_[i] = (long)matrix[i];
	}
	for (long j = 0; j < index; j++) {
		printf("\na[%ld]: %ld", j, matrix_[j]);
	}
	free(matrix);


	fclose(fp);
}

void matrixCol(real_t *omatrix, real_t *imatrix, int row, int col) {
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			omatrix[j*row + i] = imatrix[i*col + j];
		}
	}
	for (int i = 0; i < row*col; i++) {
		printf("\na[%d]: %e", i, omatrix[i]);
	}
}

