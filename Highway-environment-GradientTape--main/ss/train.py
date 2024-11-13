import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 超參數設置
learning_rate = 0.001
epochs = 50
batch_size = 32

def load_training_data():
    train_dataset = np.load(os.path.join(root_path, 'dataset', 'train.npz'))
    train_data = train_dataset['data']
    train_label = to_categorical(train_dataset['label'])
    return train_data, train_label

def load_validation_data():
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))
    valid_data = valid_dataset['data']
    valid_label = to_categorical(valid_dataset['label'])
    return valid_data, valid_label

# 使用 MirroredStrategy 進行多 GPU 訓練
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    @tf.function
    def train_step(inputs):
        x_batch, y_batch = inputs
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions) / strategy.num_replicas_in_sync
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train_model():
        train_data, train_label = load_training_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in train_dist_dataset:
                per_replica_losses = strategy.run(train_step, args=(batch,))
                total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                num_batches += 1

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches}")

        model.save('YOURMODEL.h5')

    def evaluate_model():
        valid_data, valid_label = load_validation_data()
        predictions = model.predict(valid_data, batch_size=batch_size)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(valid_label, axis=1)
        accuracy = np.mean(true_labels == predicted_labels)
        print(f'Predicted labels: {predicted_labels}')
        print(f'True labels: {true_labels}')
        print(f'Accuracy: {accuracy:.2f}')

    if __name__ == "__main__":
        train_model()
        evaluate_model()
