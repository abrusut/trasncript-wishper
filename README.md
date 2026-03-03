# Transcriptor de Audios y Videos con Whisper

Script para transcribir archivos de audio/video a texto usando OpenAI Whisper.

## Requisitos

- Python 3.8+
- FFmpeg
- OpenAI Whisper

### Instalar FFmpeg

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

### Instalar Whisper

```bash
pip install -U openai-whisper
```

## Uso

### Transcribir todos los archivos soportados de una carpeta

```bash
python3 transcribir_tickets.py --all ./audios --out ./tickets_out --model turbo --language es
```

### Transcribir un archivo específico

```bash
python3 transcribir_tickets.py --file ./audios/mi_audio.ogg --out ./tickets_out --model turbo --language es
```

### Transcribir un video específico

```bash
python3 transcribir_tickets.py --file ./videos/mi_video.mp4 --out ./tickets_out --model turbo --language es
```

## Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--all <carpeta>` | Carpeta con audios a transcribir | - |
| `--file <archivo>` | Ruta a un audio específico | - |
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
python3 transcribir_tickets.py --all ./audios --out ./tickets_out --language auto
```

Transcribir con modelo más ligero (más rápido, menos preciso):

```bash
python3 transcribir_tickets.py --all ./audios --out ./tickets_out --model small
```

Continuar numeración desde INC-010:

```bash
python3 transcribir_tickets.py --all ./audios --out ./tickets_out --start 10
```

Transcribir videos de una carpeta:

```bash
python3 transcribir_tickets.py --all ./videos --out ./tickets_out --model turbo --language es
```

Mantener audio extraído para depuración:

```bash
python3 transcribir_tickets.py --file ./videos/mi_video.mkv --out ./tickets_out --keep-extracted-audio
```

## Tests manuales rápidos

```bash
# 1) Archivo único de video
python3 transcribir_tickets.py --file ./videos/video.mp4 --out ./tickets_out

# 2) Lote de carpeta (audio + video)
python3 transcribir_tickets.py --all ./videos --out ./tickets_out
```
