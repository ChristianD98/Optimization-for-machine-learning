import tensorflow as tf

print("GPUs disponibles :")
print(tf.config.list_physical_devices('GPU'))