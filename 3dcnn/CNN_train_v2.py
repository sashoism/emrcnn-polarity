import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Reshape
from disp_cell3D import plot4d
import pickle

# Define the input shape
input_shape = (32, 32, 32, 1) # Assuming single-channel input images

# Create the model
model = Sequential([
    # First convolutional block, input shape is (32, 32, 32, 1), output shape is (16, 16, 16, 32)
    Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
    MaxPooling3D((2, 2, 2)),
    
    # Second convolutional block, output shape is (8, 8, 8, 64)
    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D((2, 2, 2)),
    
    # Third convolutional block, output shape is (4, 4, 4, 128)
    Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D((2, 2, 2)),
    
    # Flatten the output of the convolutional blocks
    Flatten(),
    
    # Fully connected layers
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    
    # Reshape the output to match the desired output shape
    Dense(32*32*32, activation='sigmoid'),
    Reshape((32, 32, 32, 1)) # Output shape is (32, 32, 32, 1) for binary masks
])

def DiceLoss(y_true, y_pred):
    smooth = 1e-6
    y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
    denominator = tf.reduce_sum(y_pred ** 2) + tf.reduce_sum(y_true ** 2) + smooth
    result = 1 - tf.divide(nominator, denominator)
    return result

# Compile the model
model.compile(optimizer='adam', loss=DiceLoss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

# Print the model summary
model.summary()

dp = 'CNN_data'
train_images = np.load(dp + '/X_train.npy')
train_masks = np.load(dp + '/y_train.npy')
val_images = np.load(dp + '/X_test.npy')
val_masks = np.load(dp + '/y_test.npy')

# Train the model
history = model.fit(train_images, train_masks, epochs=50, batch_size=32, validation_data=(val_images, val_masks))
# Save the history object
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Evaluate the model on the validation set
val_loss, val_acc, val_iou = model.evaluate(val_images, val_masks, verbose=2)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

tf.keras.models.save_model(model, 'CNN_model',overwrite=True) # Save the model

input_image = train_images
output_mask = model.predict(input_image)

plot4d(input_image, output_mask)

