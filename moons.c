#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N_COLS 3

// load csv file into a 3d array
float** load_dataset(char* filename, int* rows) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    // count the number of rows
    *rows = 0;
    char c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            (*rows)++;
        }
    }
    rewind(fp);

    // allocate memory for the 2d array
    float** data = (float**)malloc(*rows * sizeof(float*));
    for (int i = 0; i < *rows; i++) {
        data[i] = (float*)malloc(N_COLS * sizeof(float));
    }

    // read the data into the 2d array
    int i = 0;
    while (fscanf(fp, "%f,%f,%f", &data[i][0], &data[i][1], &data[i][2]) != EOF) {
        i++;
    }

    fclose(fp);
    return data;
}

#define NUM_TRAINING_SAMPLES 950
#define NUM_TEST_SAMPLES 50
#define NUM_FEATURES 2
#define NUM_LABELS 1

typedef struct
{
    float w1;
    float w2;
    float b;
} Neuron;

typedef struct
{
    Neuron l1_p1;
    Neuron l1_p2;
    Neuron l2_p;
} Model;

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}
void initialize_neuron(Neuron *n)
{
    n->w1 = rand_float();
    n->w2 = rand_float();
    n->b = rand_float();
}

void init_model(Model *m)
{
    initialize_neuron(&(m->l1_p1));
    initialize_neuron(&(m->l1_p2));
    initialize_neuron(&(m->l2_p));
}

float sigmoidf(float x)
{
    return (1.f / (1.f + expf(-x)));
}
void compute_neuron(float arr[][NUM_FEATURES], Neuron n, float **result)
{
    float w1 = n.w1;
    float w2 = n.w2;
    float b = n.b;
    for (size_t i = 0; i < NUM_TRAINING_SAMPLES; i++)
    {
        float x1 = arr[i][0];
        float x2 = arr[i][1];
        float y = sigmoidf(w1 * x1 + w2 * x2 + b);
        result[i][0] = y;
    }
}

void print_matrix(float *arr, size_t num_rows, size_t num_columns)
{
    for (size_t i = 0; i < num_rows; ++i)
    {
        for (size_t j = 0; j < num_columns; ++j)
        {
            printf("%.2f ", *(arr + i * num_columns + j));
        }
        printf("\n");
    }
}



float** forward_with_return(Model m, float init_data[][2], float y[][1]) {
    float** or_result = (float**)malloc(NUM_TRAINING_SAMPLES * sizeof(float*));
    float** nand_result = (float**)malloc(NUM_TRAINING_SAMPLES * sizeof(float*));
    float** y_hat = (float**)malloc(NUM_TRAINING_SAMPLES * sizeof(float*));

    for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
        or_result[i] = (float*)malloc(sizeof(float));
        nand_result[i] = (float*)malloc(sizeof(float));
        y_hat[i] = (float*)malloc(sizeof(float));
    }

    compute_neuron(init_data, m.l1_p1, or_result);
    compute_neuron(init_data, m.l1_p2, nand_result);

    float merged[NUM_TRAINING_SAMPLES][NUM_FEATURES];
    for (size_t i = 0; i < NUM_TRAINING_SAMPLES; ++i)
    {
        merged[i][0] = or_result[i][0];
        merged[i][1] = nand_result[i][0];
    }
    compute_neuron(merged, m.l2_p, y_hat);
    // free memory
    for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
        free(or_result[i]);
        free(nand_result[i]);
    }
    return y_hat;
}



float loss(Model m, float init_data[][2], float y[][1])
{
    float** y_hat = forward_with_return(m, init_data, y);
    float loss = 0.f;
    for (size_t i = 0; i < NUM_TRAINING_SAMPLES; ++i)
    {
        float d = 0.f;
        d = y_hat[i][0] - y[i][0];
        loss += d * d;
    }

    loss = loss /NUM_TRAINING_SAMPLES;
    // free memory
    for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
        free(y_hat[i]);
    }
    return loss;
}



Model update_gradients(Model m, float cost, float eps, float lr, float init_data[][2], float labels[][1]) {
    Neuron g_l1_p1;
    Neuron g_l1_p2;
    Neuron g_l2_p;

    float dw1_eps;
    float dw1_cost;

    float dw2_eps;
    float dw2_cost;

    float db_eps;
    float db_cost;

    Neuron original_l1_p1 = m.l1_p1;
    Neuron original_l1_p2 = m.l1_p2;
    Neuron original_l2_p = m.l2_p;

    m.l1_p1.w1 += eps;
    dw1_eps = loss(m, init_data, labels);
    dw1_cost = (dw1_eps - cost) / eps;
 
    m.l1_p1 = original_l1_p1; // Reset to original values

    m.l1_p1.w2 += eps;
    dw2_eps = loss(m, init_data, labels);
    dw2_cost = (dw2_eps - cost) / eps;
        // printf("11111**** %f,%f\n",dw2_eps,cost);
    m.l1_p1 = original_l1_p1; // Reset to original values

    m.l1_p1.b += eps;
    db_eps = loss(m, init_data, labels);
    db_cost = (db_eps - cost) / eps;
    m.l1_p1 = original_l1_p1; // Reset to original values

    g_l1_p1.w1 = original_l1_p1.w1 - lr * dw1_cost;
    g_l1_p1.w2 = original_l1_p1.w2 - lr * dw2_cost;
    g_l1_p1.b = original_l1_p1.b - lr * db_cost;

    m.l1_p2.w1 += eps;
    dw1_eps = loss(m, init_data, labels);
    dw1_cost = (dw1_eps - cost) / eps;
    m.l1_p2 = original_l1_p2; // Reset to original values

    m.l1_p2.w2 += eps;
    dw2_eps = loss(m, init_data, labels);
    dw2_cost = (dw2_eps - cost) / eps;
    m.l1_p2 = original_l1_p2; // Reset to original values

    m.l1_p2.b += eps;
    db_eps = loss(m, init_data, labels);
    db_cost = (db_eps - cost) / eps;
    m.l1_p2 = original_l1_p2; // Reset to original values

    g_l1_p2.w1 = original_l1_p2.w1 - lr * dw1_cost;
    g_l1_p2.w2 = original_l1_p2.w2 - lr * dw2_cost;
    g_l1_p2.b = original_l1_p2.b - lr * db_cost;

    m.l2_p.w1 += eps;
    dw1_eps = loss(m, init_data, labels);
    dw1_cost = (dw1_eps - cost) / eps;
    m.l2_p = original_l2_p; // Reset to original values

    m.l2_p.w2 += eps;
    dw2_eps = loss(m, init_data, labels);
    dw2_cost = (dw2_eps - cost) / eps;
    m.l2_p = original_l2_p; // Reset to original values

    m.l2_p.b += eps;
    db_eps = loss(m, init_data, labels);
    db_cost = (db_eps - cost) / eps;
    m.l2_p = original_l2_p; // Reset to original values

    g_l2_p.w1 = original_l2_p.w1 - lr * dw1_cost;
    g_l2_p.w2 = original_l2_p.w2 - lr * dw2_cost;
    g_l2_p.b = original_l2_p.b - lr * db_cost;

    // Assign the updated neuron parameters back into the model
    m.l1_p1 = g_l1_p1;
    m.l1_p2 = g_l1_p2;
    m.l2_p = g_l2_p;

    return m;
}



#define EPS 1e-1
#define LR 3e-1

int main() {
    srand(69);
    int rows, cols;
    float** data = load_dataset("./moons.csv", &rows);

    Model m;
    init_model(&m);
    float train_data[NUM_TRAINING_SAMPLES][NUM_FEATURES];
    float labels[NUM_TRAINING_SAMPLES][NUM_LABELS];
    float test_data[NUM_TEST_SAMPLES][NUM_FEATURES];
    float test_labels[NUM_TEST_SAMPLES][NUM_LABELS];
    
    for (size_t i = 0; i < NUM_TRAINING_SAMPLES; ++i)
    {
        train_data[i][0] = data[i][0];
        train_data[i][1] = data[i][1];
        labels[i][0] = data[i][2];
    }
    for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i)
    {
        test_data[i][0] = data[i+NUM_TRAINING_SAMPLES][0];
        test_data[i][1] = data[i+NUM_TRAINING_SAMPLES][1];
        test_labels[i][0] = data[i+NUM_TRAINING_SAMPLES][2];
    }

    for (size_t i = 0; i < 100*1000; i++) // Increase iterations
    {
        float ct = loss(m, train_data, labels);
        m = update_gradients(m, ct, EPS, LR, train_data, labels);
        printf("====loss====%f\n", ct);
    }
    // print accuracy of the test data
    printf("================= training done =================\n");
    float accuracy = 0;
    for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i)
    {
        float **y_hat = forward_with_return(m, test_data,test_labels);
        //loop through the y_hat and test_labels and compare
        for (size_t j = 0; j < NUM_TEST_SAMPLES; ++j)
        {
            float value = y_hat[j][0];
            if (value > 0.5)
            {
                value = 1;
            }
            else
            {
                value = 0;
            }

            if (value == test_labels[j][0])
            {
                accuracy++;
            }
        }
    }
    accuracy = accuracy / NUM_TEST_SAMPLES;
    printf("Accuracy: %f\n", accuracy);
    return 0;
}
