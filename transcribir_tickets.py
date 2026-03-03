#!/usr/bin/env python3
"""
Transcriptor de audios con Whisper.
- Soporta --file <audio> o --all <carpeta>
- Produce estructura: out/YYYY-MM-DD/INC-###__slug/
  - <nombre_audio>.txt (transcripción)
  - audio.<ext> (copia)
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import whisper  # pip install -U openai-whisper


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}


def slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    # reemplazos básicos
    text = re.sub(r"[áàäâ]", "a", text)
    text = re.sub(r"[éèëê]", "e", text)
    text = re.sub(r"[íìïî]", "i", text)
    text = re.sub(r"[óòöô]", "o", text)
    text = re.sub(r"[úùüû]", "u", text)
    text = re.sub(r"ñ", "n", text)

    # limpia símbolos
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")

    if not text:
        return "sin-titulo"
    return text[:max_len].rstrip("-")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_media_files(folder: Path) -> List[Path]:
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in (AUDIO_EXTS | VIDEO_EXTS):
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())


def summarize_stderr(stderr_text: str, max_lines: int = 8) -> str:
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    if not lines:
        return "(sin detalle de stderr)"
    return " | ".join(lines[-max_lines:])


def ffprobe_has_video_stream(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "No se encontró ffprobe en PATH. Instalá FFmpeg (ej: sudo apt install ffmpeg)."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = summarize_stderr(e.stderr or "")
        raise RuntimeError(f"ffprobe falló para {path.name}: {stderr}") from e

    return bool((proc.stdout or "").strip())


def extract_audio(video_path: Path, tmp_dir: Path) -> Path:
    """
    Extrae audio a MP3 o WAV temporal (mono, 16kHz). Devuelve ruta del temporal.
    """
    ensure_dir(tmp_dir)

    base_name = f"{video_path.stem}__{uuid.uuid4().hex}"
    candidates = [
        (
            ".mp3",
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
            ],
        ),
        (
            ".wav",
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
            ],
        ),
    ]
    errors: List[str] = []

    for suffix, base_cmd in candidates:
        out_audio = tmp_dir / f"{base_name}{suffix}"
        cmd = [*base_cmd, str(out_audio)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return out_audio
        except FileNotFoundError as e:
            raise RuntimeError(
                "No se encontró ffmpeg en PATH. Instalalo con: sudo apt install ffmpeg"
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = summarize_stderr(e.stderr or "")
            errors.append(f"{suffix}: {stderr}")

    detail = "; ".join(errors) if errors else "sin detalle"
    raise RuntimeError(f"No se pudo extraer audio de '{video_path.name}'. Detalle: {detail}")


def detect_webm_is_video(path: Path) -> Optional[bool]:
    """
    Devuelve:
      - True: webm con stream de video
      - False: webm sin stream de video
      - None: no se pudo determinar (ej: no ffprobe)
    """
    try:
        return ffprobe_has_video_stream(path)
    except RuntimeError as e:
        msg = str(e)
        if "No se encontró ffprobe" in msg:
            return None
        raise


def next_inc_number(out_day_folder: Path) -> int:
    """
    Busca carpetas INC-###__* y devuelve el siguiente número disponible.
    """
    if not out_day_folder.exists():
        return 1

    max_n = 0
    for p in out_day_folder.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"INC-(\d{3})__", p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


@dataclass
class TranscriptionResult:
    audio_path: Path
    text: str
    language: str
    prob: float


def transcribe_one(
    model,
    audio_path: Path,
    force_language: Optional[str] = "es",
    verbose: bool = True,
) -> TranscriptionResult:
    """
    Usa whisper transcribe (más simple y robusto que decode manual) y retorna texto + idioma.
    """
    # fp16 puede fallar en CPU en algunas máquinas
    use_fp16 = False

    # En whisper, language=None => autodetect
    language_arg = None if force_language in (None, "", "auto") else force_language

    # Task: "transcribe" (no translate)
    result = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=language_arg,
        fp16=use_fp16,
        verbose=False,
    )

    # idioma detectado (si language_arg es None, detecta; si está forzado, devuelve eso)
    detected_lang = result.get("language") or (language_arg or "unknown")

    # probabilidad: si autodetect, podemos intentar detect_language con mel (opcional).
    # Para no recalcular, dejamos prob=1.0 cuando está forzado.
    prob = 1.0 if language_arg else 0.0

    text = (result.get("text") or "").strip()

    if verbose:
        print(f"[OK] {audio_path.name} -> lang={detected_lang} chars={len(text)}")

    return TranscriptionResult(audio_path=audio_path, text=text, language=detected_lang, prob=prob)


def write_outputs(
    out_root: Path,
    source_path: Path,
    transcription: str,
    inc_number: int,
    date_str: str,
    title_from_text: bool = True,
    extracted_audio_path: Optional[Path] = None,
) -> Path:
    """
    Crea carpeta OUT/DATE/INC-###__slug/ y guarda transcripción + copia del audio.
    El archivo .txt tiene el mismo nombre base del archivo original.
    Devuelve la carpeta creada.
    """
    out_day = out_root / date_str
    ensure_dir(out_day)

    # título preliminar (1era línea / primeras palabras)
    raw_title = ""
    if title_from_text and transcription:
        raw_title = transcription.splitlines()[0].strip()
        # si la 1ra línea es muy larga, recortamos para slug/título
        raw_title = raw_title[:120]
    else:
        raw_title = source_path.stem

    slug = slugify(raw_title)
    folder_name = f"INC-{inc_number:03d}__{slug}"
    out_folder = out_day / folder_name
    ensure_dir(out_folder)

    # copia archivo original (audio o video)
    dest_source = out_folder / source_path.name
    shutil.copy2(source_path, dest_source)

    if extracted_audio_path is not None and extracted_audio_path.exists():
        extracted_name = f"{source_path.stem}__extracted{extracted_audio_path.suffix.lower()}"
        dest_extracted = out_folder / extracted_name
        shutil.copy2(extracted_audio_path, dest_extracted)

    # transcript con el nombre base del archivo original
    transcript_name = source_path.stem + ".txt"
    transcript_path = out_folder / transcript_name
    transcript_path.write_text(transcription + "\n", encoding="utf-8")

    return out_folder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe audios y videos con Whisper",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Ruta a un archivo de audio/video")
    g.add_argument("--all", type=str, help="Carpeta con archivos de audio/video a transcribir")

    p.add_argument("--out", type=str, default="out", help="Carpeta de salida (default: out)")
    p.add_argument("--model", type=str, default="turbo", help="Modelo whisper (default: turbo)")
    p.add_argument(
        "--language",
        type=str,
        default="es",
        help="Idioma (es/auto). default: es (forzado). Usar 'auto' para autodetect.",
    )
    p.add_argument("--date", type=str, default=None, help="Fecha YYYY-MM-DD (default: hoy)")
    p.add_argument("--start", type=int, default=None, help="Forzar INC inicial (ej: 5)")
    p.add_argument("--no-title-from-text", action="store_true", help="No usar texto para título/slug")
    p.add_argument(
        "--keep-extracted-audio",
        action="store_true",
        help="Guardar el audio extraído junto al video en la carpeta final",
    )
    p.add_argument("--quiet", action="store_true", help="Menos logs")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    date_str = args.date or dt.date.today().isoformat()

    # Carga modelo
    if not args.quiet:
        print(f"[INFO] Loading Whisper model: {args.model}")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"[ERROR] No pude cargar el modelo '{args.model}'. Detalle: {e}", file=sys.stderr)
        return 2

    force_language = None if args.language == "auto" else args.language

    # Selección de archivos
    media_files: List[Path] = []
    if args.file:
        input_path = Path(args.file).resolve()
        if not input_path.exists():
            print(f"[ERROR] Archivo no encontrado: {input_path}", file=sys.stderr)
            return 2
        ext = input_path.suffix.lower()
        if ext not in AUDIO_EXTS and ext not in VIDEO_EXTS:
            print(
                f"[ERROR] Extensión no soportada: {ext}. "
                f"Audio: {sorted(AUDIO_EXTS)} | Video: {sorted(VIDEO_EXTS)}",
                file=sys.stderr,
            )
            return 2
        media_files = [input_path]
    else:
        folder = Path(args.all).resolve()
        if not folder.exists() or not folder.is_dir():
            print(f"[ERROR] Carpeta inválida: {folder}", file=sys.stderr)
            return 2
        media_files = list_media_files(folder)
        if not media_files:
            print(
                f"[WARN] No encontré archivos soportados en {folder} "
                f"(audio: {sorted(AUDIO_EXTS)} | video: {sorted(VIDEO_EXTS)})"
            )
            return 0

    out_day = out_root / date_str
    inc = args.start if args.start is not None else next_inc_number(out_day)
    tmp_dir = out_root / "_tmp_audio"
    ensure_dir(tmp_dir)

    if not args.quiet:
        print(f"[INFO] Output root: {out_root}")
        print(f"[INFO] Date folder: {out_day}")
        print(f"[INFO] Starting INC: {inc:03d}")
        print(f"[INFO] Archivos: {len(media_files)}")

    errors = 0

    for source_path in media_files:
        extracted_audio: Optional[Path] = None
        transcribe_path = source_path
        ext = source_path.suffix.lower()

        try:
            if ext in VIDEO_EXTS and ext != ".webm":
                if not args.quiet:
                    print(f"[INFO] Video detectado: {source_path.name}")
                extracted_audio = extract_audio(source_path, tmp_dir)
                transcribe_path = extracted_audio
                if not args.quiet:
                    print(f"[INFO] Audio extraído temporal: {transcribe_path}")
            elif ext == ".webm":
                webm_is_video = detect_webm_is_video(source_path)
                if webm_is_video is True:
                    if not args.quiet:
                        print(f"[INFO] .webm con stream de video: {source_path.name}")
                    extracted_audio = extract_audio(source_path, tmp_dir)
                    transcribe_path = extracted_audio
                    if not args.quiet:
                        print(f"[INFO] Audio extraído temporal: {transcribe_path}")
                elif webm_is_video is False:
                    if not args.quiet:
                        print(f"[INFO] .webm detectado como audio: {source_path.name}")
                    transcribe_path = source_path
                else:
                    if not args.quiet:
                        print(
                            "[WARN] ffprobe no disponible para .webm, intento extraer audio "
                            f"de {source_path.name}"
                        )
                    try:
                        extracted_audio = extract_audio(source_path, tmp_dir)
                        transcribe_path = extracted_audio
                        if not args.quiet:
                            print(f"[INFO] Audio extraído temporal: {transcribe_path}")
                    except RuntimeError as e:
                        if not args.quiet:
                            print(
                                f"[WARN] No se pudo extraer audio de {source_path.name} ({e}). "
                                "Se intentará transcribir directo."
                            )
                        extracted_audio = None
                        transcribe_path = source_path
            else:
                transcribe_path = source_path

            res = transcribe_one(
                model=model,
                audio_path=transcribe_path,
                force_language=force_language,
                verbose=not args.quiet,
            )

            out_folder = write_outputs(
                out_root=out_root,
                source_path=source_path,
                transcription=res.text,
                inc_number=inc,
                date_str=date_str,
                title_from_text=not args.no_title_from_text,
                extracted_audio_path=extracted_audio if args.keep_extracted_audio else None,
            )

            if not args.quiet:
                print(f"[DONE] {source_path.name} -> {out_folder}")

            inc += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] Falló '{source_path.name}': {e}", file=sys.stderr)
            if args.file:
                return 1
        finally:
            if extracted_audio is not None and extracted_audio.exists():
                try:
                    extracted_audio.unlink()
                except Exception as cleanup_err:
                    if not args.quiet:
                        print(
                            f"[WARN] No se pudo borrar temporal '{extracted_audio}': {cleanup_err}",
                            file=sys.stderr,
                        )

    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
