import keras as K
import my_model_mlp as my_model_mlp

train_dataset = 'C/dataset/txt'

model_mnist = my_model_mlp.Model(input_shape=(28, 28, 1), num_classes=10)

model_mnist_instance = model_mnist.build()

acc_test_tab = []
model_mnist.train(model_mnist_instance, train_dataset, epochs=100, batch_size=10)

model_mnist.save(model_mnist_instance, 'weights')