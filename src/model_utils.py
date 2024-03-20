import os
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.file_utils import find_files, features_extractor


def train_and_save_model(model_file='models/model_mfcc.h5'):
    if not os.path.exists(model_file):
        dataset = []
        for filename in find_files("dataSources/UrbanSound8K/audio", "*.wav"):
            label = filename.split(".wav")[0][-5]
            if label == '-':
                label = filename.split(".wav")[0][-6]
            dataset.append({"file_name": filename, "label": label})

        dataset = pd.DataFrame(dataset)
        dataset['dataSources'] = dataset['file_name'].apply(features_extractor)

        extracted_features_df = pd.DataFrame(dataset, columns=['class', 'feature'])
        X = np.array(extracted_features_df['feature'].tolist())
        y = np.array(extracted_features_df['class'].tolist())

        labelencoder = LabelEncoder()
        y = to_categorical(labelencoder.fit_transform(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        num_labels = 10

        model = Sequential([
            Dense(125, input_shape=(65,)),
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

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        model.fit(X_train, y_train, batch_size=32, epochs=300, validation_data=(X_test, y_test), verbose=1)
        model.save(model_file)
    else:
        model = load_model(model_file)

    return model
