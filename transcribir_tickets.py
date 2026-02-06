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
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List

import whisper  # pip install -U openai-whisper


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".webm"}


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


def list_audio_files(folder: Path) -> List[Path]:
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())


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
    audio_path: Path,
    transcription: str,
    inc_number: int,
    date_str: str,
    title_from_text: bool = True,
) -> Path:
    """
    Crea carpeta OUT/DATE/INC-###__slug/ y guarda transcripción + copia del audio.
    El archivo .txt tiene el mismo nombre que el audio original.
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
        raw_title = audio_path.stem

    slug = slugify(raw_title)
    folder_name = f"INC-{inc_number:03d}__{slug}"
    out_folder = out_day / folder_name
    ensure_dir(out_folder)

    # copia audio
    dest_audio = out_folder / audio_path.name
    shutil.copy2(audio_path, dest_audio)

    # transcript con el nombre del audio original
    transcript_name = audio_path.stem + ".txt"
    transcript_path = out_folder / transcript_name
    transcript_path.write_text(transcription + "\n", encoding="utf-8")

    return out_folder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe audios con Whisper",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Ruta a un audio específico (ej: 1.ogg)")
    g.add_argument("--all", type=str, help="Carpeta con audios a transcribir")

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

    # Selección de audios
    audios: List[Path] = []
    if args.file:
        ap = Path(args.file).resolve()
        if not ap.exists():
            print(f"[ERROR] Archivo no encontrado: {ap}", file=sys.stderr)
            return 2
        if ap.suffix.lower() not in AUDIO_EXTS:
            print(f"[ERROR] Extensión no soportada: {ap.suffix}", file=sys.stderr)
            return 2
        audios = [ap]
    else:
        folder = Path(args.all).resolve()
        if not folder.exists() or not folder.is_dir():
            print(f"[ERROR] Carpeta inválida: {folder}", file=sys.stderr)
            return 2
        audios = list_audio_files(folder)
        if not audios:
            print(f"[WARN] No encontré audios en {folder} (ext: {sorted(AUDIO_EXTS)})")
            return 0

    out_day = out_root / date_str
    inc = args.start if args.start is not None else next_inc_number(out_day)

    if not args.quiet:
        print(f"[INFO] Output root: {out_root}")
        print(f"[INFO] Date folder: {out_day}")
        print(f"[INFO] Starting INC: {inc:03d}")
        print(f"[INFO] Audios: {len(audios)}")

    for audio_path in audios:
        res = transcribe_one(
            model=model,
            audio_path=audio_path,
            force_language=force_language,
            verbose=not args.quiet,
        )

        out_folder = write_outputs(
            out_root=out_root,
            audio_path=audio_path,
            transcription=res.text,
            inc_number=inc,
            date_str=date_str,
            title_from_text=not args.no_title_from_text,
        )

        if not args.quiet:
            print(f"[DONE] {audio_path.name} -> {out_folder}")

        inc += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
