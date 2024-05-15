from train_functions import *
import math
import tensorflow as tf
import os

# Check if the directory exists, if not, create it
directory = '/content/models'
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Initialize lists to store metrics
train_losses = []
test_losses = []

EPOCHS = 10
old_test_loss = math.inf  # extremely high number here.

for epoch in range(EPOCHS):
    train_loss.reset_state()
    test_loss.reset_state()

    enc_hidden = encoder.initialize_hidden_state()
    steps_per_epoch = len(train_dataset)
    bar = tf.keras.utils.Progbar(target=steps_per_epoch)

    for step, (inputs, target) in enumerate(train_dataset):
        batch_loss = train_step(inputs, target, enc_hidden)
        bar.update(step + 1)

    for step, (inputs, target) in enumerate(test_dataset):
        test_step(inputs, target, enc_hidden)

    if old_test_loss > test_loss.result():
        old_test_loss = test_loss.result()
        encoder.save(filepath='/content/models/encoder.keras')  # Save encoder model
        decoder.save(filepath='/content/models/decoder.keras')  # Save decoder model
        print('Model is saved')

    train_losses.append(train_loss.result())
    test_losses.append(test_loss.result())
    
    print('#' * 50)
    print(f'Epoch #{epoch + 1}')
    print(f'Training Loss {train_loss.result()}')
    print(f'Testing Loss {test_loss.result()}')
    print('#' * 50)