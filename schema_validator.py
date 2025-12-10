# schema_validator.py
# ============================================
# Validador de Esquema de Dados da LessonAI
# - Fase 2.1.A: Extração de chaves
# - Fase 2.2: Validação contra Esquema Mestre
# ============================================

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

# -------------------------------------------------
# 1) Definição (provisória) do Esquema Mestre
# -------------------------------------------------
# Depois de rodar a Fase 2.1.A e analisar as chaves,
# você preenche essas estruturas com os nomes padronizados.

# Exemplo ilustrativo – substitua pelas suas chaves reais:
MASTER_KEYS = {
    "Ciclo",
    "Etapa",
    "Ano/Série",
    "Disciplina",
    "Tema",
    "Objeto de Conhecimento",
    "Título",
    "Conteúdo",
    "Habilidade",
    "Código da Habilidade",
    "Aula",
    "Bimestre",
    "Componente Curricular",
    # ... adicionar aqui todas as chaves padronizadas ...
}

# Quais chaves um "objeto de aula" DEVE ter obrigatoriamente?
REQUIRED_KEYS = {
    "Ciclo",
    "Disciplina",
    "Aula",
    "Habilidade",
    "Código da Habilidade",
    "Título",
}


# -------------------------------------------------
# 2) Funções de leitura e extração de chaves
# -------------------------------------------------
def load_json(path: Path) -> Any:
    """Carrega JSON de um arquivo, com erro amigável."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar {path}: {e}") from e


def walk_and_collect_keys(obj: Any, counter: Counter) -> None:
    """Varre o objeto JSON e acumula chaves de todos os dicionários."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            counter[k] += 1
            walk_and_collect_keys(v, counter)
    elif isinstance(obj, list):
        for item in obj:
            walk_and_collect_keys(item, counter)
    # Tipos primitivos são ignorados


def scan_json_files(json_paths: List[Path]) -> Dict[str, Any]:
    """
    Extração de chaves (Fase 2.1.A).

    Retorna estrutura:
    {
      "files": {
        "Matemática.json": {
          "total_keys": 123,
          "unique_keys": [...],
          "key_counts": [["Título", 50], ["TÍTULO ", 3], ...]
        },
        ...
      },
      "global": {
        "unique_keys": [...],
        "key_counts": [["Título", 200], ["TÍTULO ", 5], ...]
      }
    }
    """
    from collections import Counter

    global_counter: Counter = Counter()
    per_file_counters: Dict[str, Counter] = {}

    for path in json_paths:
        data = load_json(path)
        c = Counter()
        walk_and_collect_keys(data, c)
        per_file_counters[path.name] = c
        global_counter.update(c)

    files_info: Dict[str, Any] = {}
    for fname, counter in per_file_counters.items():
        files_info[fname] = {
            "total_keys": sum(counter.values()),
            "unique_keys": sorted(counter.keys()),
            "key_counts": sorted(counter.items(), key=lambda x: (-x[1], x[0])),
        }

    result = {
        "files": files_info,
        "global": {
            "unique_keys": sorted(global_counter.keys()),
            "key_counts": sorted(global_counter.items(), key=lambda x: (-x[1], x[0])),
        },
    }
    return result


def save_schema_scan(result: Dict[str, Any], out_path: Path) -> None:
    """Salva o resultado da extração de chaves em JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# -------------------------------------------------
# 3) Validação contra Esquema Mestre
# -------------------------------------------------
class SchemaError:
    """Representa um erro de schema encontrado em um objeto."""

    def __init__(
        self,
        filename: str,
        index: int,
        msg: str,
        key: str | None = None,
    ) -> None:
        self.filename = filename
        self.index = index  # índice do objeto no array (se aplicável)
        self.msg = msg
        self.key = key

    def __str__(self) -> str:
        base = f"ERRO DE SCHEMA: {self.filename} [objeto {self.index}] {self.msg}"
        if self.key is not None:
            base += f": {self.key!r}"
        return base


def validate_object_keys(
    obj: Dict[str, Any],
    filename: str,
    index: int,
    master_keys: set[str],
    required_keys: set[str],
) -> List[SchemaError]:
    """Valida UM objeto (uma aula) contra o Esquema Mestre."""
    errors: List[SchemaError] = []

    # 1) Chaves inválidas (não pertencem ao Esquema Mestre)
    for key in obj.keys():
        if key not in master_keys:
            errors.append(
                SchemaError(
                    filename=filename,
                    index=index,
                    msg="contém chave inválida",
                    key=key,
                )
            )

    # 2) Chaves obrigatórias ausentes
    for req in required_keys:
        if req not in obj:
            errors.append(
                SchemaError(
                    filename=filename,
                    index=index,
                    msg="falta chave obrigatória",
                    key=req,
                )
            )

    return errors


def validate_file(
    path: Path,
    master_keys: set[str] | None = None,
    required_keys: set[str] | None = None,
) -> List[SchemaError]:
    """
    Valida um arquivo JSON de dados de aulas.
    Assume que o arquivo é:
        - uma lista de objetos, ou
        - um dict com alguma chave contendo a lista (você pode adaptar aqui).
    """
    if master_keys is None:
        master_keys = MASTER_KEYS
    if required_keys is None:
        required_keys = REQUIRED_KEYS

    data = load_json(path)
    errors: List[SchemaError] = []

    # Caso mais comum: arquivo é uma LISTA de aulas
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                errors.extend(
                    validate_object_keys(
                        item,
                        filename=path.name,
                        index=idx,
                        master_keys=master_keys,
                        required_keys=required_keys,
                    )
                )
            else:
                errors.append(
                    SchemaError(
                        filename=path.name,
                        index=idx,
                        msg="item não é um objeto JSON (dict)",
                        key=None,
                    )
                )
    # Caso alternativo: dicionário com lista em alguma chave
    elif isinstance(data, dict):
        # Aqui você pode especializar para o seu formato
        # Exemplo: data["aulas"] é a lista
        maybe_list = data.get("aulas") or data.get("Aulas") or None
        if isinstance(maybe_list, list):
            for idx, item in enumerate(maybe_list):
                if isinstance(item, dict):
                    errors.extend(
                        validate_object_keys(
                            item,
                            filename=path.name,
                            index=idx,
                            master_keys=master_keys,
                            required_keys=required_keys,
                        )
                    )
                else:
                    errors.append(
                        SchemaError(
                            filename=path.name,
                            index=idx,
                            msg="item não é um objeto JSON (dict)",
                            key=None,
                        )
                    )
        else:
            errors.append(
                SchemaError(
                    filename=path.name,
                    index=-1,
                    msg="formato inesperado: não encontrei lista de aulas",
                    key=None,
                )
            )
    else:
        errors.append(
            SchemaError(
                filename=path.name,
                index=-1,
                msg="formato inesperado: JSON raiz não é list nem dict",
                key=None,
            )
        )

    return errors


def validate_files_or_die(
    json_paths: List[Path],
    master_keys: set[str] | None = None,
    required_keys: set[str] | None = None,
) -> None:
    """
    Valida vários arquivos.
    Se houver qualquer erro, imprime no console e levanta RuntimeError.
    Ideal para ser chamado no startup da API.
    """
    all_errors: List[SchemaError] = []
    for path in json_paths:
        errs = validate_file(path, master_keys=master_keys, required_keys=required_keys)
        all_errors.extend(errs)

    if all_errors:
        print("\n========== ERROS DE SCHEMA DETECTADOS ==========")
        for e in all_errors:
            print(str(e))
        print("================================================\n")
        raise RuntimeError(
            f"Foram encontrados {len(all_errors)} erros de schema nos arquivos de dados."
        )
    else:
        print("[SchemaValidator] Todos os JSONs passaram na validação de schema.")


# -------------------------------------------------
# 4) Pequeno CLI para rodar na linha de comando
# -------------------------------------------------
if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        print(
            "Uso:\n"
            "  python -m schema_validator scan <caminho_json_ou_pasta>...\n"
            "  python -m schema_validator validate <caminho_json_ou_pasta>...\n"
        )
        raise SystemExit(1)

    mode = args[0]
    paths = [Path(p) for p in args[1:]]
    json_files: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".json":
            json_files.append(p)
        elif p.is_dir():
            json_files.extend(sorted(p.rglob("*.json")))

    if not json_files:
        print("[ERRO] Nenhum arquivo .json encontrado nos caminhos informados.")
        raise SystemExit(1)

    if mode == "scan":
        result = scan_json_files(json_files)
        out_path = Path("schema_raw_keys.json")
        save_schema_scan(result, out_path)
        print(f"[SchemaValidator] Scan concluído. Resultado salvo em {out_path.resolve()}")
        print(
            f"Chaves únicas globais: {len(result['global']['unique_keys'])} "
            f"(veja schema_raw_keys.json)"
        )
    elif mode == "validate":
        validate_files_or_die(json_files)
    else:
        print(f"[ERRO] Modo desconhecido: {mode!r}. Use 'scan' ou 'validate'.")
        raise SystemExit(1)
