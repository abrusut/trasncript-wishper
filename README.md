# Transcriptor de Audios y Videos con Whisper

Script para transcribir archivos de audio/video a texto usando OpenAI Whisper.

## Requisitos

- Python con dependencias compatibles con OpenAI Whisper.
- FFmpeg y ffprobe (en Ubuntu, ambos se instalan con el paquete `ffmpeg`).
- OpenAI Whisper instalado en un entorno virtual del proyecto.
- Conexión a Internet para instalar dependencias y descargar el modelo en su primer uso.
- Espacio disponible para las dependencias, modelos y archivos de salida.

## Instalación en Ubuntu

Ejecutar desde la carpeta del proyecto. Adaptar la ruta si está en otra ubicación:

```bash
cd ~/Documentos/Andres/transcribe

sudo apt update
sudo apt install -y python3-venv ffmpeg

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade openai-whisper
```

El entorno `.venv` mantiene las dependencias del proyecto separadas del Python del sistema. No usar `sudo pip` ni `--break-system-packages`.

### Verificar la instalación

```bash
python --version
python -m pip --version
python -c "import whisper; print('Whisper instalado correctamente')"
ffmpeg -version
ffprobe -version
python transcribir_tickets.py --help
```

La ruta mostrada por `python -m pip --version` debe apuntar a `.venv`.

### Ejecutar en una terminal nueva

Activar el entorno cada vez que se abre una terminal:

```bash
cd ~/Documentos/Andres/transcribe
source .venv/bin/activate
```

Todos los ejemplos siguientes suponen que el entorno está activo. Para salir:

```bash
deactivate
```

También es posible ejecutar sin activar el entorno, usando su intérprete directamente:

```bash
cd ~/Documentos/Andres/transcribe
./.venv/bin/python transcribir_tickets.py --all ./audios --out ./tickets_out --model small --language es
```

### Primera transcripción

Crear la carpeta de entrada y colocar allí los audios o videos:

```bash
mkdir -p audios
```

Luego ejecutar:

```bash
python transcribir_tickets.py --all ./audios --out ./tickets_out --model small --language es
```

El modelo se descarga automáticamente la primera vez que se utiliza. El script usa `turbo` por defecto; `small` permite empezar con un modelo más liviano. El tiempo de procesamiento depende del equipo, el modelo y la duración del archivo.

### Actualizar Whisper

```bash
cd ~/Documentos/Andres/transcribe
source .venv/bin/activate
python -m pip install --upgrade openai-whisper
```

## Solución de problemas de instalación

### `externally-managed-environment`

Este mensaje indica que `pip` está intentando instalar en el Python administrado por Ubuntu. Crear y activar `.venv` con los pasos anteriores y volver a ejecutar:

```bash
python -m pip install --upgrade openai-whisper
```

### No se puede crear el entorno virtual

Si aparece un error por falta de `venv` o `ensurepip`, instalar el soporte de entornos virtuales:

```bash
sudo apt install -y python3-venv
```

Después volver a ejecutar `python3 -m venv .venv` desde el proyecto.

### `ModuleNotFoundError: No module named 'whisper'`

Verificar que se ejecuta con el entorno correcto:

```bash
./.venv/bin/python -m pip show openai-whisper
./.venv/bin/python transcribir_tickets.py --help
```

Si el paquete no está instalado:

```bash
./.venv/bin/python -m pip install --upgrade openai-whisper
```

### Errores instalando `torch`, `numba` o `tiktoken`

Son problemas distintos de la protección del Python del sistema. La compatibilidad depende de la versión de Python y de las versiones disponibles de cada dependencia. En particular, no asumir que una instalación sobre Python 3.14 es compatible solo porque `.venv` se creó correctamente.

Registrar la versión y el error completo para diagnosticarlo:

```bash
python --version
python -m pip --version
```

Si una dependencia no admite la versión instalada, usar una versión de Python compatible en un entorno virtual separado, sin reemplazar el Python del sistema.

### No se encuentra `ffmpeg` o `ffprobe`

```bash
sudo apt install -y ffmpeg
ffmpeg -version
ffprobe -version
```

## Uso

### Transcribir todos los archivos soportados de una carpeta

```bash
python transcribir_tickets.py --all ./audios --out ./tickets_out --model turbo --language es
```

### Transcribir un archivo específico

```bash
python transcribir_tickets.py --file ./audios/mi_audio.ogg --out ./tickets_out --model turbo --language es
```

### Transcribir un video específico

```bash
python transcribir_tickets.py --file ./videos/mi_video.mp4 --out ./tickets_out --model turbo --language es
```

## Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--all <carpeta>` | Carpeta con audios y videos a transcribir (sin recorrer subcarpetas) | - |
| `--file <archivo>` | Ruta a un audio o video específico | - |
| `--out <carpeta>` | Carpeta de salida | `out` |
| `--model <modelo>` | Modelo Whisper (tiny, base, small, medium, large, turbo) | `turbo` |
| `--language <idioma>` | Idioma del audio (es, en, auto, etc.) | `es` |
| `--date <YYYY-MM-DD>` | Fecha para la carpeta de salida | Fecha actual |
| `--start <número>` | Número inicial para INC (ej: 5) | Auto |
| `--no-title-from-text` | No usar el texto transcrito para el nombre de carpeta | - |
| `--keep-extracted-audio` | Guarda audio extraído (solo para video) junto al original | `false` |
| `--quiet` | Menos logs en consola | - |

> Nota: `--all` y `--file` son mutuamente excluyentes (usar uno u otro).

## Formatos soportados

Audio:
- `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`, `.webm`

Video:
- `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`, `.webm`

### Nota sobre `.webm`

- `.webm` puede ser audio-only o video.
- Si `ffprobe` está disponible, el script detecta si hay stream de video.
- Si `ffprobe` no está, intenta extraer audio con `ffmpeg`; si falla, intenta transcribir directo como audio.

## Estructura de salida

```
tickets_out/
└── 2026-02-06/
    ├── INC-001__primera-linea-del-texto/
    │   ├── audio_original.ogg
    │   └── audio_original.txt
    ├── INC-002__otro-audio-transcrito/
    │   ├── grabacion.mp3
    │   └── grabacion.txt
    └── ...
```

Cada carpeta contiene:
- Copia del archivo original (audio o video)
- Archivo `.txt` con la transcripción (mismo stem que el archivo original)
- Opcionalmente, audio extraído (`--keep-extracted-audio`) cuando el input es video

## Ejemplos

Transcribir audios con detección automática de idioma:

```bash
python transcribir_tickets.py --all ./audios --out ./tickets_out --language auto
```

Transcribir con modelo más ligero (más rápido, menos preciso):

```bash
python transcribir_tickets.py --all ./audios --out ./tickets_out --model small
```

Continuar numeración desde INC-010:

```bash
python transcribir_tickets.py --all ./audios --out ./tickets_out --start 10
```

Transcribir videos de una carpeta:

```bash
python transcribir_tickets.py --all ./videos --out ./tickets_out --model turbo --language es
```

Mantener audio extraído para depuración:

```bash
python transcribir_tickets.py --file ./videos/mi_video.mkv --out ./tickets_out --keep-extracted-audio
```

## Tests manuales rápidos

```bash
# 1) Archivo único de video
python transcribir_tickets.py --file ./videos/video.mp4 --out ./tickets_out

# 2) Lote de carpeta (audio + video)
python transcribir_tickets.py --all ./videos --out ./tickets_out
```
