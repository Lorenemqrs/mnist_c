#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Bmp2Matrix.h"

// Structure pour stocker les poids d'une couche
typedef struct {
    char layer_name[50];
    int weight_shape[2];
    double *weights;
    double *biases;
} LayerWeights;

// Fonction pour lire les poids (et les biais) d'une couche à partir de deux fichiers
LayerWeights read_weights_from_files(const char *weights_filename, const char *biases_filename) {
    LayerWeights layer_weights;

    // Lire les informations des poids
    FILE *weights_file = fopen(weights_filename, "r");
    if (weights_file == NULL) {
        perror("Erreur lors de l'ouverture du fichier de poids");
        exit(EXIT_FAILURE);
    }

    fscanf(weights_file, "Layer Name: %s\n", layer_weights.layer_name);
    printf("%s\n", layer_weights.layer_name);
    fscanf(weights_file, "Weight Shape: (%d, %d)\n", &layer_weights.weight_shape[0], &layer_weights.weight_shape[1]);
    printf("%d\n", layer_weights.weight_shape[0]);

    layer_weights.weights = (double *)malloc(layer_weights.weight_shape[0] * layer_weights.weight_shape[1] * sizeof(double));

    for (int i = 0; i < layer_weights.weight_shape[0] * layer_weights.weight_shape[1]; i++) {
        fscanf(weights_file, "%lf ", &layer_weights.weights[i]);
    }

    fclose(weights_file);

    // Lire les informations des biais
    FILE *biases_file = fopen(biases_filename, "r");
    if (biases_file == NULL) {
        perror("Erreur lors de l'ouverture du fichier de biais");
        exit(EXIT_FAILURE);
    }

    fscanf(biases_file, "Layer Name: %s\n", layer_weights.layer_name);
    fscanf(biases_file, "Weight Shape: %d\n", &layer_weights.weight_shape[1]);

    layer_weights.biases = (double *)malloc(layer_weights.weight_shape[1] * sizeof(double));

    for (int i = 0; i < layer_weights.weight_shape[1]; i++) {
        fscanf(biases_file, "%lf,", &layer_weights.biases[i]);
    }

    fclose(biases_file);

    return layer_weights;
}

// Fonction d'activation ReLU
float relu(float x) {
    if (x < 0.0) {
        return 0.0;
    } else {
        return x;
    }
}

// Fonction d'activation Softmax
double *softmax(double *input, int size) {
    double *output = (double *)malloc(size * sizeof(double));
    double sum_exp = 0.0;

    // Calculer l'exponentielle pour chaque élément
    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i]);
        sum_exp += output[i];
    }

    // Normaliser en divisant par la somme des exponentielles
    for (int i = 0; i < size; ++i) {
        output[i] /= sum_exp;
    }

    return output;
}

// Fonction de propagation avant
double *forward(LayerWeights layer, double *input, int use_relu) {
    double *output = (double *)malloc(layer.weight_shape[1] * sizeof(double));
    int scale_factor = 1e6;

    for (int i = 0; i < layer.weight_shape[1]; i++) {
        output[i] = 0.0;
        for (int j = 0; j < layer.weight_shape[0]; j++) {
            output[i] += input[j] * (layer.weights[j * layer.weight_shape[1] + i]/ scale_factor);
        }
        output[i] += (layer.biases[i]/scale_factor);

        if (isnan(output[i]) || isinf(output[i])) {
            printf("NaN or infinite value detected in output[%d]\n", i);
        }
        if (use_relu==1){
            output[i] = relu(output[i]);
        }
    }

    return output;
}

int prediction(double *output){
    int max = 0;
    for(int i=0; i<10; i++){
        if(output[i]>output[max]){
            max = i;
        }
    }
    return max;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <photo_path>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *photo_path = argv[1];

    BMP bitmap;
    FILE *pFichier = NULL;

    pFichier = fopen(photo_path, "rb");
    if (pFichier == NULL) {
        perror("Erreur dans la lecture du fichier");
        return EXIT_FAILURE;
    }
    LireBitmap(pFichier, &bitmap);
    fclose(pFichier);               //Fermeture du fichier contenant l'image

    ConvertRGB2Gray(&bitmap);
    double *input = (double *)malloc(28*28*sizeof(double));
    for(int i=0; i<28; i++){
        for(int j=0; j<28*3; j=j+3){
            input[i*28+j] = bitmap.mPixelsGray[i][j];
        }
    }

    printf("Couche 1\n");
    // Lire les poids de la première couche
    LayerWeights layer1_weights = read_weights_from_files("weights/dense_weights.txt", "weights/dense_biais.txt");
    printf("Couche 2\n");

    // Lire les poids de la deuxième couche, etc.
    LayerWeights layer2_weights = read_weights_from_files("weights/dense_1_weights.txt", "weights/dense_1_biais.txt");
    printf("Foward dense 1\n");
    double *output1 = forward(layer1_weights, input, 1);
    printf("Foward dense 2\n");
    double *output2 = forward(layer2_weights, output1,0);
    double *softmax_output = softmax(output2, layer2_weights.weight_shape[1]);
    printf("Output softmax: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
           softmax_output[0], softmax_output[1], softmax_output[2], softmax_output[3], softmax_output[4],
           softmax_output[5], softmax_output[6], softmax_output[7], softmax_output[8], softmax_output[9]);
    printf("Prediction: %d\n", prediction(softmax_output));
    // Libérer la mémoire après utilisation
    free(layer1_weights.weights);
    free(layer1_weights.biases);
    free(layer2_weights.weights);
    free(layer2_weights.biases);
    DesallouerBMP(&bitmap);

    return 0;
}
