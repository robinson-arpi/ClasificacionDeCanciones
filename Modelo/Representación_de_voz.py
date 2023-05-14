import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo de audio y extraer la componente armónica
audio_file = "ClasificacionDeCanciones\Modelo\Testing\Me voy a olvidar_robinson.mp3"
audio, sr = librosa.load(audio_file)
y_harmonic, y_percussive = librosa.effects.hpss(audio)

# Seleccionar solo los primeros 5 segundos
audio_length = audio.shape[0] / sr
start_time = 18.3
end_time = 18.6
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)
y_harmonic_5s = y_harmonic[start_sample:end_sample]

# Graficar la señal de la componente armónica de los primeros 5 segundos
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y_harmonic_5s, sr=sr, alpha=0.5)
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Amplitud")
titulo = "Gráfico de la señal de la componente armónica (voz) en el intervalo [" + str(start_time) + ", " + str(end_time) + "] segundos"
plt.title(titulo)
plt.ylim((-1, 1))
plt.show()





