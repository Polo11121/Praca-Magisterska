from src.model_utils import train_and_save_model
from src.mfcc import mfcc
import numpy as np
import time


start_time = time.time()

model = train_and_save_model()

filename = ""

mfccs_features = mfcc(filename)

mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

result_array = model.predict(mfccs_scaled_features)
#  ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling",
#                   "gun_shot", "jackhammer", "siren", "street_music"]
result_classes = ["klimatyzacja", "klakson", "bawiące się dzieci", "szczekanie psa", "wiercenie",
               "praca silnika", "strzał z broni", "młot pneumatyczny", "syreny", "muzyka uliczna"]
result = np.argmax(result_array[0])

print(result_classes[result])

elapsed_time = time.time() - start_time

print(f"Czas wykonania programu: {elapsed_time} sekund")