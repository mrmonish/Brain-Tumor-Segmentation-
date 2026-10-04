import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("--- Step 1: Generating Dataset ---")
def generate_dummy_data(num_samples=200, img_size=128):
    X = np.random.rand(num_samples, img_size, img_size, 1).astype(np.float32)
    Y = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    for i in range(num_samples):
        cx, cy = np.random.randint(40, 88, 2)
        r = np.random.randint(12, 25)
        y, x = np.ogrid[:img_size, :img_size]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        Y[i, mask, 0] = 1.0
    return X, Y

X_train, Y_train = generate_dummy_data(num_samples=300, img_size=128)
X_val, Y_val = generate_dummy_data(num_samples=60, img_size=128)
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

print("\n--- Step 2: Building U-Net Architecture ---")
def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)
    
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = build_unet(input_shape=(128, 128, 1))

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coefficient])
model.summary()

print("\n--- Step 3: Training the Model ---")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=5,
    shuffle=True
)

print("\n--- Step 4: Visualizing Results ---")
preds = model.predict(X_val[:3])

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(X_val[i].squeeze(), cmap='gray')
    plt.title(f"Sample {i+1}")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(Y_val[i].squeeze(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(preds[i].squeeze() > 0.5, cmap='gray')
    plt.title("AI Prediction")
    plt.axis('off')

plt.tight_layout()
plt.show()