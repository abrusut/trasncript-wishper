# Transcriptor de Audios con Whisper

Script para transcribir archivos de audio a texto usando OpenAI Whisper.

## Requisitos

- Python 3.8+
- OpenAI Whisper

```bash
pip install -U openai-whisper
```

## Uso

### Transcribir todos los audios de una carpeta

```bash
python3 transcribir_tickets.py --all ./audios --out ./tickets_out --model turbo --language es
```

### Transcribir un archivo específico

```bash
python3 transcribir_tickets.py --file ./audios/mi_audio.ogg --out ./tickets_out --model turbo --language es
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
| `--quiet` | Menos logs en consola | - |

> Nota: `--all` y `--file` son mutuamente excluyentes (usar uno u otro).

## Formatos de audio soportados

- `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`, `.webm`

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
- Copia del archivo de audio original
- Archivo `.txt` con la transcripción (mismo nombre que el audio)

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
