#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "Bmp2Matrix.h"

void saveImageDataToTxt(const char *inputImagePath, const char *outputTxtPath) {
    BMP bitmap;
    FILE *pFichier = NULL;
    FILE *outputFile = NULL;

    pFichier = fopen(inputImagePath, "rb");
    if (pFichier == NULL) {
        printf("Erreur dans la lecture du fichier\n");
        return;
    }

    LireBitmap(pFichier, &bitmap);
    fclose(pFichier);

    ConvertRGB2Gray(&bitmap);

    outputFile = fopen(outputTxtPath, "w");
    printf("outputTxtPath: %s\n", outputTxtPath);
    if (outputFile == NULL) {
        //creer un fichier texte pour stocker les valeurs des pixels
        
        printf("Erreur dans l'écriture du fichier de sortie\n");
        DesallouerBMP(&bitmap);
        return;
    }

    // Enregistrement des valeurs des pixels dans le fichier texte
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            fprintf(outputFile, "%d ", bitmap.mPixelsGray[i][j]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);
    DesallouerBMP(&bitmap);
}

void processDirectory(const char *dirPath,const char *outputDirectoryPath) {
    DIR *directory;
    struct dirent *entry;

    directory = opendir(dirPath);

    if (directory == NULL) {
        perror("Erreur lors de l'ouverture du répertoire");
        return;
    }

    while ((entry = readdir(directory)) != NULL) {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".bmp")) {
            char inputImagePath[256];
            char outputTxtPath[256];

            snprintf(inputImagePath, sizeof(inputImagePath), "%s/%s", dirPath, entry->d_name);
            snprintf(outputTxtPath, sizeof(outputTxtPath), "%s/%s.txt", outputDirectoryPath, entry->d_name);

            saveImageDataToTxt(inputImagePath, outputTxtPath);
        }
    }

    closedir(directory);
}

int main() {
    for (int i = 0; i < 10; i++) {
        char inputDirectoryPath[256];
        char outputDirectoryPath[256];
        snprintf(inputDirectoryPath, sizeof(inputDirectoryPath), "dataset/%d", i);
        snprintf(outputDirectoryPath, sizeof(outputDirectoryPath), "dataset/txt");
        processDirectory(inputDirectoryPath, outputDirectoryPath);
    }


    return 0;
}
