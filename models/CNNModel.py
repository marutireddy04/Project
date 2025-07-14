def main():
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    import os  # Make sure to import os for path manipulation

    basepath = "C:\\Users\\shital\\Desktop\\python\\dataset"

    # Initialize the CNN
    classifier = Sequential()

    # Add convolutional, pooling, and dense layers
    classifier.add(Convolution2D(32, (1, 1), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, (1, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(64, (1, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(3, activation='softmax'))

    # Compile the model
    classifier.compile(optimizer=SGD(learning_rate=0.1),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # Prepare data generators
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        os.path.join(basepath, 'C:\\Users\\shital\\Desktop\\python\\dataset\\training_set'),  # Correct path concatenation
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        os.path.join(basepath, 'C:\\Users\\shital\\Desktop\\python\\dataset\\test_set'),  # Correct path concatenation
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    steps_per_epoch = int(np.ceil(training_set.samples / 32))
    val_steps = int(np.ceil(test_set.samples / 32))

    print("Starting model training...")
    model = classifier.fit(training_set, steps_per_epoch=steps_per_epoch, epochs=10, validation_data=test_set, validation_steps=val_steps)

    # Save the model
    classifier.save(os.path.join(basepath, 'lung_model.h5'))

    # Evaluate the model
    test_scores = classifier.evaluate(test_set, verbose=1)
    train_scores = classifier.evaluate(training_set, verbose=1)

    print(f"Testing Accuracy: {test_scores[1]*100:.2f}%")
    print(f"Training Accuracy: {train_scores[1]*100:.2f}%")

    # Plot results
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(basepath, "accuracy.png"), bbox_inches='tight')
    plt.show()

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(basepath, "loss.png"), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
