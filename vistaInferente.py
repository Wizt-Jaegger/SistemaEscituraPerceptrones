import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
import os
import random

# Cargar el modelo
modelo = tf.keras.models.load_model("modelo_perceptronFigG.h5")

nombreClase = [f"letra-{chr(65+i)}" for i in range(26)]  # ["letra-A", ..., "letra-Z"]

# Diccionario de palabras (una por letra)
diccionario_palabras = {
    'A': ["Apple"], 'B': ["Ball"], 'C': ["Cat"], 'D': ["Dog"], 'E': ["Elephant"],
    'F': ["Fish"], 'G': ["Giraffe"], 'H': ["Hat"], 'I': ["Igloo"], 'J': ["Juice"],
    'K': ["Kite"], 'L': ["Lion"], 'M': ["Monkey"], 'N': ["Nest"], 'O': ["Orange"],
    'P': ["Pencil"], 'Q': ["Queen"], 'R': ["Rabbit"], 'S': ["Sun"], 'T': ["Tiger"],
    'U': ["Umbrella"], 'V': ["Violin"], 'W': ["Whale"], 'X': ["Xylophone"],
    'Y': ["Yogurt"], 'Z': ["Zebra"]
}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Learn to Write in English")
        self.canvas_size = 200

        self.word = ""
        self.current_letter_index = 0

        self.label_instruction = tk.Label(root, text="Click 'Start' to begin!", font=("Arial", 14))
        self.label_instruction.pack(pady=10)

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.imagen_pil = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.dibujo_pil = ImageDraw.Draw(self.imagen_pil)

        self.canvas.bind("<B1-Motion>", self.dibujar)

        self.button_start = tk.Button(root, text="Start", command=self.iniciar_palabra)
        self.button_start.pack(pady=5)

        self.button_limpiar = tk.Button(root, text="Clear", command=self.limpiar_canvas)
        self.button_limpiar.pack(side=tk.LEFT, padx=20)

        self.button_predecir = tk.Button(root, text="Check Letter", command=self.predecir)
        self.button_predecir.pack(side=tk.RIGHT, padx=20)

    def iniciar_palabra(self):
        letra = random.choice(list(diccionario_palabras.keys()))
        self.word = random.choice(diccionario_palabras[letra]).upper()
        self.current_letter_index = 0
        self.label_instruction.config(text=f"Let's write: {self.word}\nDraw letter: {self.word[0]}")
        self.limpiar_canvas()

    def dibujar(self, event):
        x, y = event.x, event.y
        r = 7
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.dibujo_pil.ellipse((x - r, y - r, x + r, y + r), fill=0)

    def limpiar_canvas(self):
        self.canvas.delete("all")
        self.imagen_pil = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.dibujo_pil = ImageDraw.Draw(self.imagen_pil)

    def predecir(self):
        if not self.word:
            messagebox.showinfo("Info", "Click 'Start' to begin!")
            return

        letra_esperada = self.word[self.current_letter_index]

        img_resized = self.imagen_pil.resize((64, 64))
        img_array = np.array(img_resized)
        img_array = cv2.Canny(img_array, 100, 200).astype('float32') / 255.0
        x_test = tf.constant(img_array.reshape((1, 64, 64, 1)), dtype=tf.float32)

        prediccion = modelo.predict(x_test)
        letra_predicha = nombreClase[np.argmax(prediccion)].split("-")[1]

        if letra_predicha == letra_esperada:
            self.current_letter_index += 1
            if self.current_letter_index < len(self.word):
                siguiente = self.word[self.current_letter_index]
                self.label_instruction.config(text=f"Correct! Now draw: {siguiente}")
                self.limpiar_canvas()
            else:
                messagebox.showinfo("Great job!", f"You wrote '{self.word}' correctly!")
                self.word = ""
                self.label_instruction.config(text="Click 'Start' to try another word!")
                self.limpiar_canvas()
        else:
            messagebox.showwarning("Try again", f"That's not correct. Try drawing '{letra_esperada}' again.")
            self.limpiar_canvas()

# Ejecutar aplicación
root = tk.Tk()
app = App(root)
root.mainloop()
