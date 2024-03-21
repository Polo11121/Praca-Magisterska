import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.file_utils import find_files, features_extractor


def new_model(input_shape, num_labels):
    return Sequential([
            Dense(256, input_shape=(input_shape,)),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(512),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_labels, activation='softmax')
        ])

def old_model(input_shape, num_labels):
    return Sequential([
            Dense(125, input_shape=(input_shape,)),
            Activation('relu'),
            Dropout(0.5),
            Dense(250),
            Activation('relu'),
            Dropout(0.5),
            Dense(125),
            Activation('relu'),
            Dropout(0.5),
            Dense(num_labels),
            Activation('softmax')
        ])

def train_and_save_model(model_file):
    if not os.path.exists(model_file):
        dataset = []
        for filename in find_files("dataSources/UrbanSound8K/audio", "*.wav"):
            label = filename.split(".wav")[0][-5]
            if label == '-':
                label = filename.split(".wav")[0][-6]
            dataset.append({"file_name": filename, "label": label})

        dataset = pd.DataFrame(dataset)
        dataset['data'] = dataset['file_name'].apply(features_extractor)

        X = np.array(dataset['data'].tolist())
        y = np.array(dataset['label'].tolist())

        labelencoder = LabelEncoder()
        y = to_categorical(labelencoder.fit_transform(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        num_labels = y.shape[1]

        model = old_model(X_train.shape[1],num_labels)

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        history = model.fit(X_train, y_train, batch_size=32, epochs=300, validation_data=(X_test, y_test), verbose=1,
                            callbacks=[early_stopping])

        best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
        print(f"Najlepsza epoka: {best_epoch}")

        best_val_loss = history.history['val_loss'][best_epoch]
        print(f"Strata na zbiorze walidacyjnym w najlepszej epoce: {best_val_loss}")

        best_val_accuracy = history.history['val_accuracy'][best_epoch]
        print(f"Dokładność na zbiorze walidacyjnym w najlepszej epoce: {best_val_accuracy}")

        model.save(f'models/{best_val_accuracy}_old_model_mfcc.h5')

    else:
        model = load_model(model_file)

    return model
