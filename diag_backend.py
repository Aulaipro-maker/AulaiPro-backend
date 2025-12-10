# diag_backend.py
# ==========================================
# Utilitários de diagnóstico para o backend
# ==========================================

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, Optional

# Ligue/desligue o modo detalhado de debug
DEBUG_DIAGNOSTICO: bool = True

# Arquivos que queremos destacar nos tracebacks
ARQUIVOS_INTERESSE = {
    "main.py",
    "utils_norm.py",
    "schema_validator.py",
    "field_map.json",
}

def formatar_erro_detalhado(
    exc: Exception,
    contexto: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Monta um dicionário com:
      - tipo e mensagem da exceção
      - contexto fornecido (endpoint, etapa, disciplina, etc.)
      - frames relevantes do traceback (arquivo, linha, função, código)
    """
    tb = traceback.extract_tb(exc.__traceback__)
    frames_interessantes = []

    for fr in tb:
        base = os.path.basename(fr.filename)
        if base in ARQUIVOS_INTERESSE:
            frames_interessantes.append(
                {
                    "file": base,
                    "line": fr.lineno,
                    "func": fr.name,
                    "code": fr.line,
                }
            )

    # Se não achou nenhum frame interessante, pega o último mesmo assim
    if not frames_interessantes and tb:
        fr = tb[-1]
        frames_interessantes.append(
            {
                "file": os.path.basename(fr.filename),
                "line": fr.lineno,
                "func": fr.name,
                "code": fr.line,
            }
        )

    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "context": contexto or {},
        "frames": frames_interessantes,
    }


def log_erro_backend(
    tag: str,
    exc: Exception,
    contexto: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Usa formatar_erro_detalhado, imprime no console e retorna o dicionário.
    """
    debug_info = formatar_erro_detalhado(exc, contexto=contexto)
    print(f"[ERRO {tag}] Detalhes do erro:")
    print(json.dumps(debug_info, ensure_ascii=False, indent=2))
    return debug_info
