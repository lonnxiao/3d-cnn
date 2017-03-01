/******************************************************************************
* filename: 
*******************************************************************************/
#ifndef matrix_H
#define matrix_H
typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);

float matrix_topk_accuracy(matrix truth, matrix guess, int k);

#endif
