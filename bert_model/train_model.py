import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.model_selection import train_test_split
from bert_model.config.core import config , PACKAGE_ROOT
from official.nlp import optimization     # to create AdamW optimizer
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
import pandas as pd
import os

dataset_dir = config['dataset_dir']
data_path = os.path.join(PACKAGE_ROOT,str(dataset_dir),"preprocessed_Reviews.csv")

data_sample = pd.read_csv(data_path)

# split the data into train , val and test set

X = data_sample['preprocessed_text']
y = data_sample['sentiment_score']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=40000, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=20000, stratify=y_temp, random_state=42)

# Training set: X_train, y_train (20K samples)
# Validation set: X_val, y_val (20K samples)
# Test set: X_test, y_test (20K samples)

# Convert the x_train and y_train into a tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# prefetch the batch size
batch_size = int(config['batch_size'])
print(f"Type of batch_size---{type(batch_size)}")
train_ds = train_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

map_name_to_handle = config['map_name_to_handle']
map_model_to_preprocess = config['map_model_to_preprocess']


bert_model_name = 'small_bert/bert_en_uncased_L-2_H-128_A-2'
tfhub_handle_encoder = str(map_name_to_handle[bert_model_name])
tfhub_handle_preprocess = str(map_model_to_preprocess[bert_model_name])

# Load BERT model
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

# Define a function to create custom classifier model using BERT

def build_classifier_model():
    # YOUR CODE HERE
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')          # input layer
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing') # text processing (text to numbers)
    encoder_inputs = preprocessing_layer(text_input)                                    # one array to three array(mask, ids, sentence ids)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder') # define encoding layer
    outputs = encoder(encoder_inputs)                                                   # encoding
    net = outputs['pooled_output']                                                      # input array for the classification model
    net = tf.keras.layers.Dropout(0.1)(net)                                        # drouput layer
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)              # classification
    return tf.keras.Model(text_input, net)



# Custom classifier model
classifier_model = build_classifier_model()


# Visualize model's structure
# print(tf.keras.utils.plot_model(classifier_model))
print(classifier_model.summary())

# Define loss function and metric
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# Hyperparameters ad Optimizer
epochs = int(config['epochs'])
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')



# Compile model
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# Train the model
print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(train_ds,
                               validation_data=val_ds,
                               epochs=epochs,
                               )

# Predection on test dataset
loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')


# Save model for later use
saved_model_path = 'trained_models'
classifier_model.save(saved_model_path, include_optimizer=False)
