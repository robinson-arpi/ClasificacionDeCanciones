import librosa
import matplotlib.pyplot as plt

# Leer la grabación de audio en formato .wav
ruta = 'ClasificacionDeCanciones\Modelo\Canciones\Me voy a olvidar.mp3'
audio, sr = librosa.load(ruta)

# Extraer el pitch contour
f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Visualizar el pitch contour en un gráfico de entonación en líneas
plt.figure(figsize=(12, 4))
plt.plot(f0)
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Gráfico de entonación en líneas')
plt.show()


