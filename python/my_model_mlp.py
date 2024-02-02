import keras as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import os


class Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        inputs = Input(shape=self.input_shape)
        # transforme input en gray scale
        inputs = K.layers.Lambda(lambda x: K.backend.expand_dims(x, axis=-1))(inputs)
        x = Flatten()(inputs)
        x = Dense(28*28*1.5, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = K.models.Model(inputs, outputs)
        return model
    
    def train(self, model, train_path, epochs, batch_size):
        model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        # Load data from text files
        train_data, train_labels = self.load_data(train_path)
        


        # Convert labels to categorical
        train_labels_categorical = K.utils.to_categorical(train_labels, num_classes=self.num_classes)
        print("Train Data Shape:", train_data.shape)
        print("Train Labels Shape:", train_labels_categorical.shape)

        model.fit(
            train_data,
            train_labels_categorical,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )
        return model

    def load_data(self, train_path):
        data = []
        labels = []
        
        for filename in os.listdir(train_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(train_path, filename)
                
                # Load data from text file
                loaded_data = np.loadtxt(file_path)
                
                # Assuming the label is extracted from the file name
                label = int(filename.split("_")[0])
                
                data.append(loaded_data)
                labels.append(label)

        return np.array(data), np.array(labels)
    
    def evaluate(self, model, test_data):

        datagen = ImageDataGenerator()
        test_generator = datagen.flow_from_directory(
            test_data,
            target_size=(28, 28),
            batch_size=3,
            class_mode='categorical',
            shuffle=False,
        )
        
        # Utilisez le générateur pour évaluer le modèle
        test_loss, test_acc = model.evaluate(test_generator)
        print('Test accuracy:', test_acc)
        return test_acc


    def predict(self, model, data):
        predictions = model.predict(data)
        return predictions
    
    def save(self, model, path):
        os.makedirs(path, exist_ok=True)

        # Save only the weights of the specified layers
        for layer in model.layers:
            layer_name = layer.name
            file_path = os.path.join(path, f"{layer_name}_weights.txt")
            biais_path = os.path.join(path, f"{layer_name}_biais.txt")

            # Check if the layer has weights before saving
            if layer.get_weights():
                layer_weights_float = layer.get_weights()[0]
                biais_float = layer.get_weights()[1]

                # Scale the weights for better integer representation
                scale_factor = 10**6  # Adjust as needed
                layer_weights_int = (layer_weights_float * scale_factor).astype(int)
                biais_int = (biais_float * scale_factor).astype(int)

                # Save the information to the file
                with open(file_path, 'w') as file:
                    file.write(f"Layer Name: {layer_name}\n")
                    file.write(f"Weight Shape: {layer_weights_int.shape}\n")
                    np.savetxt(file, layer_weights_int, delimiter=' ', fmt='%d')  # Save as integers

                with open(biais_path, 'w') as file:
                    file.write(f"Layer Name: {layer_name}\n")
                    file.write(f"Weight Shape: {biais_int.shape}\n")
                    np.savetxt(file, biais_int, delimiter=' ', fmt='%d')  # Save as integers

        model.save(os.path.join(path, 'mlp.h5'))
        print('Model saved to', path)
        return
    
    def load(self, path):
        model = K.models.load_model(path)
        print('Model loaded from', path)
        return model
    

