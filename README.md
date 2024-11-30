# Indian Law Bot
Indian Law Bot is a chatbot designed to assist users with basic information on Indian laws and legal procedures. Built using machine learning and NLP techniques, this bot can interpret user queries and provide relevant responses from its knowledge base of Indian legal concepts.

# Features
- **Interactive Chatbot:** Uses a graphical interface with Flet for user-friendly interaction.
- **Comprehensive Legal Responses:** Covers various aspects of Indian laws, including criminal, civil, constitutional, family, and corporate law.
- **Machine Learning Model:** Trained using TensorFlow to classify user queries into appropriate legal categories and provide accurate responses.

# Outcome of the Project:

https://github.com/user-attachments/assets/ebf961a3-5834-4ec2-8592-cf02f5dbc72a

# Technologies Used
- **Python:** Core programming language.
- **TensorFlow/Keras:** Machine learning framework for training the chatbot model.
- **Flet:** For the user interface.
- **Scikit-learn:** Used for label encoding.
- **JSON:** To structure and store intents.
- **Pickle:** For serializing and deserializing the model and tokenizer objects.
## Installation

**Prerequisites**
- Python 3.8+
- pip (Python package manager)

**Steps to Run**

**Step - 1. Clone the Repository:**
```bash
git clone https://github.com/Sriranga1105/Indian-law-bot.git 
cd Indian-Law-Bot
```

**Step - 2: Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Step - 3: Train the Model (Optional): If you want to retrain the model:**
```bash
python train.py
```

**Step - 4: Run the Chatbot:**
```bash
python chat.py
```

## File Structure
Indian-Law-Bot/

├── chat.py                # Main application file for Flet UI

├── train.py               # Script for training the model

├── intents.json           # JSON file containing intents and responses

├── tokenizer.pickle       # Serialized tokenizer object

├── label_encoder.pickle   # Serialized label encoder object

├── chat_model_new.keras   # Trained model file

├── requirements.txt       # Project dependencies

└── README.md              # Project documentation

## How It Works
- **User Interface:** The chatbot runs in a simple Flet-based UI, where users can input questions about Indian laws.
- **Intent Classification:** User messages are classified into predefined intents using the trained neural network.
- **Response Generation:** Based on the classified intent, the bot fetches an appropriate response from intents.json.

## Dataflow Diagram
![image](https://github.com/user-attachments/assets/abaad5dd-9aa3-4bd0-9d20-39a369777ed6)

![image](https://github.com/user-attachments/assets/35bbc5d4-27d5-4a39-81b4-e27f3d89bfeb)

## Customization
- **Adding New Intents:** Modify intents.json to add new patterns and responses.
- **Retraining:** Run train.py after modifying intents.json to retrain the model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License

Copyright (c) 2024 Sriranganathan M

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

