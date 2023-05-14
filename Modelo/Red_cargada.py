import os
import torch
import librosa
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

# Obtener la ruta absoluta de la carpeta actual
ruta_absoluta = os.path.dirname(os.path.abspath(__file__))

# Crea una nueva instancia del modelo
modelo_cargado = nn.Sequential(nn.Linear(10338, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 6),
                            nn.LogSoftmax(dim=1))

# Restaura el estado del modelo y otros elementos desde el checkpoint
checkpoint = torch.load(ruta_absoluta + "\\modelo.pth")
modelo_cargado.load_state_dict(checkpoint["modelo"])

criterion = nn.NLLLoss()
criterion.load_state_dict(checkpoint["criterion"])
optimizer = optim.Adam(modelo_cargado.parameters(), lr=0.003)
optimizer.load_state_dict(checkpoint["optimizador"])
epochs = checkpoint["epoch"]


def evaluar_cancion(ruta):
    audio, sr = librosa.load(ruta)
    caracteristicas = librosa.feature.melspectrogram(y=audio, sr=sr)
    tensor_prueba = torch.from_numpy(caracteristicas).float()

    logits = modelo_cargado.forward(tensor_prueba)


    # logits is the output of the neural network
    # logits = model(tensor_prueba)

    # Apply softmax to the logits to get probabilities
    ps = F.softmax(logits, dim=1)
    promedio = torch.mean(ps, dim=0)
    print(promedio)
    # Get the probabilities for each class
    probs = promedio.detach().numpy()

    # Create a bar chart of the probabilities
    fig, ax = plt.subplots()
    ax.bar(range(6), probs)
    ax.set_xticks(range(6))
    ax.set_xticklabels(['Desde Lejos', 'La Playa', 'Lo noto', 'Me voy a olvidar.mp3','Por las noches', 'Te conozco'])
    ax.set_ylabel('Probability')
    ax.set_title('Classification Probabilities')
    plt.show()

evaluar_cancion(ruta_absoluta + "\\Testing\\Me voy a olvidar_robinson.mp3" )    
evaluar_cancion(ruta_absoluta + "\\Testing\\Por las noches_prueba.mp3" )    