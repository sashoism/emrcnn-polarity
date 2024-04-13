import numpy as np
import tensorflow as tf
from disp_cell3D import plot4d
# Load the model
# Custom loss function that applies class weightsd
def DiceLoss(y_true, y_pred):
    smooth = 1e-6
    y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
    denominator = tf.reduce_sum(y_pred ** 2) + tf.reduce_sum(y_true ** 2) + smooth
    result = 1 - tf.divide(nominator, denominator)
    return result

with tf.keras.utils.custom_object_scope({'DiceLoss': DiceLoss}):
    CNN_model = tf.keras.models.load_model('CNN_model_100_epochs')

print(CNN_model.summary())
X_test = np.load('CNN_data/X_test.npy')
y_test = np.load('CNN_data/y_test.npy')
print(X_test.shape, y_test.shape)
input_image = X_test
output_mask = CNN_model.predict(input_image)

print(output_mask.max())
#print index of the maximum value in the output_mask
print(np.unravel_index(np.argmax(output_mask), output_mask.shape))
# print average value of the output_mask
print(np.mean(output_mask))
# make all values in the output_mask greater than 0.5 equal to 1
output_mask = output_mask > 0.5

plot4d(input_image[4:], output_mask[4:])

# Print test loss and accuracy
loss, accuracy, mean_iou = CNN_model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy},' + f' Test mean IoU: {mean_iou}')

np.save('CNN_data/y_pred.npy', output_mask)