import flet as ft
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle

# Load necessary data and model
with open("intents.json") as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model('chat_model_new.keras')

# Load tokenizer and label encoder objects
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

# Flet App
def main(page: ft.Page):

    # Set the title and theme
    page.title = "Chatbot"
    page.window.width = 570
    page.window.height = 700
    page.padding = 10
    page.theme_mode = ft.ThemeMode.LIGHT

    # Chat history container
    chat_history = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)

    # Function to handle user input
    def send_message(e):
        user_message = message_box.value.strip()
        if user_message:
            # Clear input field
            message_box.value = ""
            message_box.update()

            # Add user message to chat (right side)
            user_text = ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Text(user_message, size=16, color=ft.colors.WHITE),
                        padding=10,
                        margin=5,
                        bgcolor=ft.colors.BLUE,
                        border_radius=ft.border_radius.all(15),
                    )
                ],
                alignment=ft.MainAxisAlignment.END  # Align to right
            )
            chat_history.controls.append(user_text)

            # Predict chatbot response
            result = model.predict(keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([user_message]), truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])

            # Get the response
            response = ""
            for i in data['intents']:
                if i['tag'] == tag:
                    response = np.random.choice(i['responses'])

            # Add bot response to chat (left side)
            bot_text = ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Text(response, size=16, color=ft.colors.WHITE),
                        padding=10,
                        margin=5,
                        bgcolor=ft.colors.GREEN,
                        border_radius=ft.border_radius.all(15),
                        width= 300,
                        height = None
                    )
                ],
                alignment=ft.MainAxisAlignment.START  # Align to left
            )
            chat_history.controls.append(bot_text)
            chat_history.update()

    # Input field for user messages
    message_box = ft.TextField(hint_text="Type a message...", expand=True)

    # Send button
    send_button = ft.ElevatedButton(text="SEND", on_click=send_message)

    # Layout for input field and button
    input_row = ft.Row(
        controls=[message_box, send_button],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )

    # Page layout
    page.add(
        ft.Column(
            controls=[
                ft.Text("LAW bot", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                chat_history,
                input_row,
            ], expand=True
        )
    )

# Run the Flet app
ft.app(target=main)
