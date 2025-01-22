import tensorflow as tf
import tensorflow_datasets as tfds

# Load dataset
dataset, info = tfds.load("rock_paper_scissors", as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]

# Prepare data
train_dataset = (
    train_dataset
    .map(lambda x, y: (tf.image.resize(x, (150, 150)) / 255.0, y)) 
    .shuffle(1000)
    .batch(32)
)
test_dataset = (
    test_dataset
    .map(lambda x, y: (tf.image.resize(x, (150, 150)) / 255.0, y))
    .batch(32)
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Summary
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train model
history = model.fit(train_dataset, epochs=25, validation_data=test_dataset, verbose=1)

# Evaluate model
model.evaluate(test_dataset)

# Save model
model.save("rps_model.h5")
