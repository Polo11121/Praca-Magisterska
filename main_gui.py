import time
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pygame
from src.mfcc import mfcc
from src.model_utils import train_and_save_model


def init_pygame():
    pygame.mixer.init()


def play_sound():
    if selected_filename:
        try:
            pygame.mixer.music.load(selected_filename)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można odtworzyć pliku: {e}")
    else:
        messagebox.showinfo("Informacja", "Najpierw wybierz plik audio.")


def choose_file():
    global selected_filename
    filename = filedialog.askopenfilename(title="Wybierz plik audio",
                                          filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
    if filename:
        selected_filename = filename
        prediction_text.set("Wybrano plik audio.")
        elapsed_time_label.config(text="")
        choose_file_button.config(text="Wybierz inny plik audio")
        play_button.pack(pady=(5, 5))
        analyze_button.pack(pady=(5, 0))


def predict_sound():
    global selected_filename
    if not selected_filename:
        messagebox.showinfo("Informacja", "Proszę najpierw wybrać plik audio.")
        return

    prediction_text.set("Analizuję dźwięk...")
    root.update()

    start_time = time.time()

    model = train_and_save_model()

    mfccs_features = mfcc(selected_filename)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    result_array = model.predict(mfccs_scaled_features)
    result_classes = ["klimatyzacja", "klakson", "bawiące się dzieci", "szczekanie psa", "wiercenie",
                      "praca silnika", "strzał z broni", "młot pneumatyczny", "syreny", "muzyka uliczna"]
    result = np.argmax(result_array[0])
    prediction_text.set(f"Przewidziana klasa dźwięku: {result_classes[result]}")

    elapsed_time = time.time() - start_time
    elapsed_time_label.config(text=f"Czas wykonania analizy: {elapsed_time:.2f} sekund")


root = tk.Tk()
root.title("Rozpoznawanie dźwięków")
root.geometry("400x250")

selected_filename = ""
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

prediction_text = tk.StringVar()
prediction_label = tk.Label(frame, textvariable=prediction_text, wraplength=380, justify="center")
prediction_label.pack(pady=(0, 10))

elapsed_time_label = tk.Label(frame, text="")
elapsed_time_label.pack(pady=(0, 10))

choose_file_button = tk.Button(frame, text="Wybierz plik audio", command=choose_file, width=20, height=2)
choose_file_button.pack()

play_button = tk.Button(frame, text="Odtwórz wybrany dźwięk", command=play_sound, width=20, height=2)

analyze_button = tk.Button(frame, text="Analizuj dźwięk", command=predict_sound, width=20, height=2)

init_pygame()

root.mainloop()
