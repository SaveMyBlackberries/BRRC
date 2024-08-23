import tensorflow as tf

log_file = "E:\rvc2-7537\logs\DanjunV2-48k\events.out.tfevents.1721955882.4090-AISound.8660.0"
dataset = tf.data.TFRecordDataset(log_file)

for record in dataset:
    print(record)
