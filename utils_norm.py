# utils_norm.py
import unicodedata, re
from typing import Any, Dict, List, Optional, Union, Iterable, Set  # + Union, Iterable



__all__ = [
    "normalize_disciplina_alias", "split_multi", "finalize_list", "keys_from_field_map",
    # Normalização
    "strip_accents", "norm_key_ci", "norm_set", "norm_text",
    "canon_bncc", "normalize_etapa", "unique_preserve_order", "ensure_list",
    # Placeholders / parsing
    "PLACEHOLDERS", "is_real", "split_multi_qs", "split_multiline",
    # Aliases / field map
    "ALIASES", "get_field_ci", "aliases_for", "match_field",
     "is_oc_key", "OC_CANON_KEY",
    # Extratores
    "row_temas", "row_objetos", "row_titles", "row_conteudos", "row_habilidades", "row_aulas",
    # Filtro unificado
    "_check", "match_row", "filter_and_collect", "filter_and_collect_habilidades",
    # BNCC helpers
    "is_hab_code", "only_hab_codes", "split_code_and_text", "row_hab_pairs",
    # Retrocompat
    "_norm_key", "_norm_key_ci", "norm_key",
    "_norm_discipline_key_for_map", "_keys_from_field_map",
    # Compat antiga do export
    "_row_temas_by_context",
]


# ——————————————————————————————————————————————————————————
# Mapa canônico global (acento/caixa/ plural normalizados)
def _cf(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode()
    return re.sub(r"\s+", " ", s).strip().casefold()

DISC_CANON = {
    "arte": "Arte",
    "artes": "Arte",
    "biologia": "Biologia",
    "ciencias": "Ciências",
    "ciência": "Ciências",
    "ciências": "Ciências",
    # adicione: fisica/física, quimica/química, etc.
}

# Se precisar de exceções por etapa, use aqui (senão, deixe vazio)
DISC_ALIASES_BY_ETAPA: Dict[str, Dict[str,str]] = {
    # "medio": {"artes": "Arte"},
}

def normalize_disciplina_alias(etapa: str, disciplina: str) -> str:
    if not disciplina:
        return disciplina
    e = normalize_etapa(etapa) or (etapa or "").strip()
    key = _cf(disciplina)
    espec = (DISC_ALIASES_BY_ETAPA.get(e) or {}).get(key)
    return espec or DISC_CANON.get(key) or disciplina.strip()

# ——————————————————————————————————————————————————————————
# Splits e normalização textual
_SPLIT_RE = re.compile(r"\s*(?:\|\||;)\s*")

def split_multi(text, seps=("||", ";", ",")):
    """
    Divide uma string por múltiplos separadores em uma lista de pedaços limpos.
    Retorna SEMPRE uma lista.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    parts = [text]
    for sep in seps:
        tmp = []
        for chunk in parts:
            tmp.extend(chunk.split(sep))
        parts = tmp

    return [p.strip() for p in parts if p and p.strip()]


def _sort_ci(seq: Iterable[str]) -> List[str]:
    return sorted(
        (str(s) for s in (seq or [])),
        key=lambda s: unicodedata.normalize("NFKD", s).casefold()
    )

def finalize_list(seq: Iterable[str]) -> List[str]:
    """Dedup case-insensitive + sort Unicode-aware."""
    seen = set()
    out = []
    for s in (seq or []):
        k = unicodedata.normalize("NFKD", str(s)).casefold()
        if k in seen: 
            continue
        seen.add(k)
        out.append(str(s).strip())
    return _sort_ci(out)

# ——————————————————————————————————————————————————————————
# Field map helpers (desacoplados do main)
def _norm_discipline_key_for_map(s: str) -> str:
    s = (s or "").replace("_", " ")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip().casefold()



def keys_from_field_map(
    field_map: Dict[str, Any],
    role: str,
    etapa: str,
    disciplina: str,
) -> List[str]:
    """
    Localiza, no field_map, a lista de aliases de coluna para um dado 'role'
    (ex.: 'tema', 'objeto', 'titulo', 'conteudo', 'habilidade', etc.)
    considerando etapa e disciplina.

    Mantém a estrutura original:
      1) tenta na sub-chave "fields" da disciplina (estrutura nested, ex.: Química)
      2) tenta na chave principal da disciplina (estrutura flat, ex.: Fundamental I)
      3) fallback em 'global' da etapa

    Garante que 'disc_map' e 'fm_e' sejam SEMPRE dicionários válidos,
    evitando NameError.
    """

    if not isinstance(field_map, dict):
        return []

    role = (role or "").strip()
    if not role:
        return []

    # -----------------------------
    # Normaliza etapa e disciplina
    # -----------------------------
    etapa_norm = (etapa or "").strip().lower()
    disc_norm = norm_key_ci(disciplina) if disciplina else ""

    fm_e: Dict[str, Any] = {}
    disc_map: Dict[str, Any] = {}

    # -----------------------------
    # 0) Localiza o bloco da ETAPA
    #    (ex.: "fundamental_i", "medio" etc.)
    # -----------------------------
    for k, v in field_map.items():
        if not isinstance(v, dict):
            continue
        if norm_key_ci(k) == etapa_norm:
            fm_e = v
            break

    if not isinstance(fm_e, dict):
        fm_e = {}

    # -----------------------------
    # 1) Localiza o bloco da DISCIPLINA
    #    dentro da etapa (fm_e)
    # -----------------------------
    if disc_norm and fm_e:
        for dk, dv in fm_e.items():
            if not isinstance(dv, dict):
                continue
            if norm_key_ci(dk) == disc_norm:
                disc_map = dv
                break

    # Se não achou disciplina específica, tenta alguns fallbacks
    if not isinstance(disc_map, dict) or not disc_map:
        # alguns nomes comuns para "global" ou "default"
        for fk in ("global", "_global", "default", "_default"):
            maybe = fm_e.get(fk)
            if isinstance(maybe, dict):
                disc_map = maybe
                break

    if not isinstance(disc_map, dict):
        disc_map = {}

    # -----------------------------
    # 2) Agora vem a SUA lógica original
    #    (já usando disc_map e fm_e definidos)
    # -----------------------------

    # 1. Busca na sub-chave 'fields' (Estrutura Nested de Química)
    disc_fields_map = (disc_map.get("fields") or {})
    keys: List[str] = list(disc_fields_map.get(role) or [])

    # 2. Busca na chave principal (Estrutura Flat de Fundamental I)
    keys.extend(disc_map.get(role) or [])

    # 3. Fallback para o global (dentro da etapa)
    global_map = fm_e.get("global") or fm_e.get("_global") or {}
    if isinstance(global_map, dict):
        keys.extend(global_map.get(role) or [])

    # -----------------------------
    # 3) Deduplicar e limpar
    # -----------------------------
    if not keys:
        return []

    # "achata" listas aninhadas e normaliza para string
    flattened: List[str] = []
    for k in keys:
        if isinstance(k, (list, tuple)):
            for sub in k:
                flattened.append(str(sub))
        else:
            flattened.append(str(k))

    # tira vazios / espaços
    flattened = [s.strip() for s in flattened if s and str(s).strip()]

    # remove duplicados preservando ordem
    return unique_preserve_order(flattened)


# -----------------------------------------------------
# NOVAS FUNÇÕES UTILITÁRIAS (COMPATÍVEIS COM FIELD_MAP)
# -----------------------------------------------------
def _norm_ci(s: str) -> str:
    s = (s or "").replace("_", " ")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip().casefold()

def same_ci(a: str, b: str) -> bool:
    return _norm_ci(a) == _norm_ci(b)
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s or ""))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_key_ci(s: str) -> str:
    s = strip_accents(str(s or ""))
    s = re.sub(r"[^\w\s\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().casefold()

# Compatibilidade retroativa
_norm_key = norm_key_ci
_norm_key_ci = norm_key_ci
norm_key = norm_key_ci

# ====================================================
# Normalização de texto genérica (compatível legado)
# ====================================================
def norm_text(value) -> str:
    """
    Normaliza um texto de forma genérica:
    - Converte para string
    - Remove espaços extras nas bordas
    - Remove acentos
    - Coloca tudo em minúsculas
    - Compacta espaços internos em um único espaço

    Essa função é usada pelo main.py e por outros
    pontos que esperam um normalizador "amigável"
    para labels/valores.
    """
    if value is None:
        return ""

    s = str(value)
    # remove espaços nas bordas
    s = s.strip()
    if not s:
        return ""

    # remove acentos
    s = strip_accents(s)

    # tudo minúsculo
    s = s.lower()

    # compacta múltiplos espaços em um só
    s = re.sub(r"\s+", " ", s)

    return s


def norm_set(qs: Optional[str]) -> set[str]:
    if not qs: return set()
    return {norm_key_ci(p) for p in str(qs).split("||") if str(p).strip()}

# utils_norm.py

def split_multiline(text, separators=("\n", "\r\n")):
    """
    Divide uma string por múltiplos separadores de linha.
    Retorna SEMPRE uma lista, mesmo que a entrada seja nula ou vazia.
    """
    # 🔒 Blindagem contra None, números, strings vazias etc.
    if not isinstance(text, str) or not text.strip():
        return []

    parts = [text]
    for sep in separators:
        tmp = []
        for chunk in parts:
            tmp.extend(chunk.split(sep))
        parts = tmp

    # limpa espaços e ignora linhas vazias
    return [p.strip() for p in parts if p and p.strip()]

# [PATCH-HABS: HELPER] — extrai habilidades de forma robusta

# utils_norm.py


# ... aqui vêm as outras funções já existentes ...

def _extract_habs(
    rows: list[dict],
    field_map: dict,
    etapa: str,
    disciplina: str,
    only_codes: bool = False,
) -> list[str]:
    """
    Extrai habilidades a partir das linhas cruas.

    Estratégia:
    1) Tenta o caminho canônico: _row_habilidades(row, field_map, etapa, disciplina).
    2) Se vier vazio, faz fallback:
       - Procura colunas cujo nome contenha 'habilidade'.
       - Quebra texto multilinha se precisar.
    3) Aplica only_codes (se True, tenta extrair só o código BNCC).
    """
    out: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        habs: list[str] = []

        # 1) Caminho canônico via FIELD_MAP recebido por parâmetro
        try:
            habs = row_habilidades(row, field_map, etapa, disciplina) or []
        except Exception:
            habs = []

        # 2) Fallback: qualquer coluna com "habilidade" no nome
        if not habs:
            for k, v in row.items():
                if not isinstance(k, str):
                    continue
                if "habilidade" not in k.lower():
                    continue
                if not isinstance(v, str) or not v.strip():
                    continue

                partes = split_multiline(v) or [v]
                for p in partes:
                    if p and isinstance(p, str) and p.strip():
                        habs.append(p.strip())

        # 3) Normaliza e extrai código, se necessário
        for h in habs:
            if not h:
                continue
            txt = str(h).strip()
            if not txt:
                continue

            if only_codes:
                m = re.search(r'(EF|EM)\d{2}[A-Z]*\d*', txt)
                code = m.group(0) if m else txt
                out.append(code)
            else:
                out.append(txt)

    return finalize_list(out)



def canon_bncc(s: str) -> str:
    s2 = strip_accents(s).upper()
    return re.sub(r"[^A-Z0-9]", "", s2)

# --- Placeholders & filtros de entrada (para QueryStrings) ---
PLACEHOLDERS = {
    "— Selecione Objeto —",
    "— Selecione Tema —",
    "— Selecione Conteúdo —",
    "— Selecione Título —",
    "— Selecione —",
    "", None,
}

# cobre variações como “- Selecione -”, “– Selecione –”, com/sem espaços
PLACEHOLDER_PAT = re.compile(r"^\s*[—\-–]*\s*Selecione\b.*$", re.I)

def is_real(v: Optional[str]) -> bool:
    s = str(v or "").strip()
    if not s:
        return False
    if s in PLACEHOLDERS:
        return False
    if PLACEHOLDER_PAT.match(s):
        return False
    return True

def split_multi_qs(v: Union[str, Iterable[str], None], sep: str = "||") -> List[str]:
    """
    Aceita string 'A||B' (querystring) ou lista iterável.
    Retorna lista já stripada e sem placeholders.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        vals = [str(x).strip() for x in v]
    else:
        vals = [s.strip() for s in str(v).split(sep)]
    return [x for x in vals if is_real(x)]

# ----------------- ETAPA / NORMALIZAÇÃO -----------------

ETAPA_MAP = {
    "fundamental i": "fundamental_I",
    "fundamental 1": "fundamental_I",
    "fundamental ii": "fundamental_II",
    "fundamental 2": "fundamental_II",
    "medio": "medio",
    "ensino medio": "medio",
    "ensino médio": "medio",
}

def normalize_etapa(s: str | None) -> str | None:
    if not s: 
        return None
    k = norm_key_ci(s)
    return ETAPA_MAP.get(k, s)

def unique_preserve_order(seq):
    seen = set(); out = []
    for x in seq or []:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def ensure_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

# ----------------- DETECÇÃO DE CÓDIGO BNCC -----------------

BNCC_RE = re.compile(r'\b[A-Z]{2,}\d{2,}[A-Z]?\d*\b')  # ex.: EF08MA07, EM13MAT302

def is_hab_code(token: str) -> bool:
    if not token: return False
    return bool(BNCC_RE.search(strip_accents(str(token)).upper()))

def only_hab_codes(items: list[str]) -> list[str]:
    return [canon_bncc(x) for x in items if is_hab_code(x)]

def split_code_and_text(raw: str) -> tuple[str | None, str | None]:
    """
    Retorna (codigo, texto) a partir de uma linha potencialmente no formato:
      'EF08MA07: texto ...'  ou  '(EF08MA07) texto ...'  ou  'EF08MA07 - texto ...'
    Faz o match diretamente na string original (case-insensitive), evitando desalinhamento.
    """
    if not raw:
        return (None, None)
    s = str(raw).strip()
    # match direto na string original; códigos BNCC são ASCII → re.IGNORECASE é suficiente
    m = re.match(r'^\s*\(?([A-Za-z]{2,}\d{2,}[A-Za-z]?\d*)\)?\s*[:\-]?\s*(.*)$', s, flags=re.IGNORECASE)
    if not m:
        return (None, s if s else None)

    codigo_raw = m.group(1) or ""
    texto_raw  = (m.group(2) or "").strip()
    codigo = canon_bncc(codigo_raw) if codigo_raw else None
    texto  = texto_raw if texto_raw else None
    return (codigo, texto)

def row_hab_pairs(row: dict, field_map: dict, etapa=None, disciplina=None) -> list[tuple[str | None, str | None]]:
    """
    Retorna lista de pares (codigo_bncc|None, texto|None) a partir do campo de habilidade.
    """
    vals = row_habilidades(row, field_map, etapa, disciplina)
    pairs = []
    for v in vals:
        c, t = split_code_and_text(v)
        pairs.append((c, t))
    return pairs

# -----------------------------------------------------
# ALIASES E FIELD_MAP
# -----------------------------------------------------

# utils_norm.py — ajuste ALIASES

ALIASES = {
    "tema": [
        # existentes…
        "tema", "temas", "Tema", "TEMA",
        "Unidade Temática", "Unidade temática", "UNIDADE TEMÁTICA","unidade","UNIDADE","Unidade",

        # ✅ novas variações (plural e sem acento)
        "Unidades Temáticas", "Unidades temáticas", "UNIDADES TEMÁTICAS",
        "Unidades Tematicas", "UNIDADES TEMATICAS",  # sem acento

        # demais eixos usados em LP/Linguagens
        "Campo de atuação", "Campo de Atuação", "CAMPOS DE ATUAÇÃO",
        "Campo de atuacao", "CAMPOS DE ATUACAO",  # sem acento
        "Práticas de Linguagem", "PRÁTICAS DE LINGUAGEM",
        "Praticas de Linguagem", "PRATICAS DE LINGUAGEM",  # sem acento
        "Linguagem", "LINGUAGEM", "Linguagens", "LINGUAGENS"
    ],
    "objeto": [
        # já cobre bem, mas adiciona mais duas grafias frequentes
        "objeto", "objetos", "OBJETO", "OBJETOS",
        "Objeto do Conhecimento", "Objetos do Conhecimento",
        "OBJETO DO CONHECIMENTO", "OBJETOS DO CONHECIMENTO",
        "Objeto de Conhecimento", "Objetos de Conhecimento",
        "OBJETOS DE CONHECIMENTO",
        "Objeto do Conhecimento (BNCC)", "Objetos do Conhecimento (BNCC)",
        "Objeto de Conhecimento (BNCC)", "Objetos de Conhecimento (BNCC)"      # ✅
    ],
    "titulo": [
        "titulo", "título", "Título", "TÍTULO",
        "Titulo da aula", "Título da aula", "TÍTULO DA AULA",
        "TÍTULO PLANO DE AULA", "TÍTULO PLANO DE AULA SEMANAL",
        "Nome da Aula", "Nome da aula"
    ],
    "conteudo": [
        "conteudo", "conteúdo", "Conteúdo", "CONTEÚDO",
        "Conteudos", "CONTEUDOS",
        "Conteúdos", "CONTEÚDOS",
        "Conteúdo(s)", "Conteúdos (BNCC)", "Conteudo (BNCC)",
        "Tópico", "Tópicos", "Assunto", "Assuntos"
    ],
    "habilidade": [
        "habilidade", "habilidades", "Habilidade", "Habilidades",
        "BNCC", "RA", "Ra", "Habilidades BNCC - Matemática",
        "HABILIDADE", "HABILIDADES",
        "Código BNCC", "Cód. BNCC", "Códigos", "Códigos BNCC"  # ✅ comuns
    ],
    "aula": [
        "aula", "Aula", "AULA", "Nº Aula", "Número da aula", "Numero da aula",
        "num_aula", "n_aula", "lesson", "Lesson"
    ],
    "etapa": [
        "etapa", "Etapa", "ETAPA", "Ciclo", "CICLO"
    ],
    "objetivos": [
        "objetivo", "objetivos", "Objetivos", "OBJETIVO", "OBJETIVOS",
        "Objetivo de Aprendizagem", "OBJETIVO DE APRENDIZAGEM",
        "Objetivos de Aprendizagem", "OBJETIVOS DE APRENDIZAGEM",
        "Objetivos da Aula", "OBJETIVOS DA AULA"
    ],
}

# -----------------------------------------------------
# MAPA: cabeçalho normalizado -> papel lógico (tema, objeto, ...)
# -----------------------------------------------------

def _build_alias_roles(aliases: dict) -> dict[str, str]:
    """
    Constrói um mapa:
        norm_key_ci(cabecalho) -> 'tema' | 'objeto' | 'titulo' | ...
    usando as listas de ALIASES já existentes.
    """
    out: dict[str, str] = {}
    for role, alias_list in aliases.items():
        for raw in alias_list:
            k = norm_key_ci(raw)
            if not k:
                continue
            # não sobrescreve se já existir – 1º ganha
            out.setdefault(k, role)
    return out


ALIAS_ROLES: dict[str, str] = _build_alias_roles(ALIASES)


def resolve_role_from_header(header: str | None) -> str | None:
    """
    Dado um nome de coluna/campo (ex.: 'UNIDADE TEMÁTICA', 'Campo de Atuação'),
    retorna o papel lógico:
        - 'tema', 'objeto', 'titulo', 'conteudo',
        - 'habilidade', 'aula', 'etapa', 'objetivos'
      ou None se não reconhecer.
    """
    if not header:
        return None
    return ALIAS_ROLES.get(norm_key_ci(header))



def get_field_ci(row: Dict[str, Any], keys: List[str]) -> Any:
    """
    Busca por nome exato, case-insensitive e por cabeçalho 'normalizado'
    (sem acento, sem parênteses/colchetes e sem pontuação), aceitando match parcial.
    Prioriza match exato > casefold > normalizado == > contém.
    """
    if not row or not keys:
        return None

    # 0) tentativa exata
    for k in keys:
        if k in row:
            return row[k]

    # 1) casefold direto
    folded = {str(rk).casefold(): rk for rk in row.keys()}
    for k in keys:
        rk = folded.get(str(k).casefold())
        if rk is not None:
            return row[rk]

    # 2) normalização de cabeçalho (remove acento/pontuação/parênteses)
    def _norm_hdr(s: str) -> str:
        s2 = strip_accents(s)
        # remove qualquer coisa entre () ou [] e pontuação comum
        s2 = re.sub(r"[\(\)\[\]]", " ", s2)
        s2 = re.sub(r"[^\w\s]", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip().casefold()
        return s2

    row_norm_map = {_norm_hdr(rk): rk for rk in row.keys()}
    key_norms    = [_norm_hdr(k) for k in keys]

    # 2a) igualdade pelo cabeçalho normalizado
    for kn in key_norms:
        rk = row_norm_map.get(kn)
        if rk is not None:
            return row[rk]

    # 3) match parcial (contém) no cabeçalho normalizado
    for kn in key_norms:
        for rnh, rk in row_norm_map.items():
            if not kn or not rnh:
                continue
            if kn in rnh or rnh in kn:
                return row[rk]

    return None


def aliases_for(field_map: Dict, etapa: Optional[str], disciplina: Optional[str], key: str) -> List[str]:
    """
    Retorna lista de aliases para o campo lógico `key` (ex.: "tema", "objeto"...),
    combinando:
      - field_map[etapa]["disciplinas"][disciplina]["fields"][key]  (se existir; match case-insensitive)
      - fallback em field_map[etapa]["global"][key]
      - fallback final em ALIASES[key] (fixo do utils)
    Faz dedupe (case/acento-insensitive) e normaliza espaços.
    """
    def _norm_dedup(seq: List[str]) -> List[str]:
        seen = set(); out = []
        for s in seq:
            t = (s or "").strip()
            k = norm_key_ci(t)
            if not t or k in seen:
                continue
            seen.add(k); out.append(t)
        return out

    collected: List[str] = []
    try:
        if etapa:
            emap = field_map.get(etapa) or {}
            # 1) disciplina específica
            if disciplina:
                d_all = emap.get("disciplinas") or {}
                dmap = d_all.get(disciplina)
                if not isinstance(dmap, dict):
                    disc_cf = norm_key_ci(disciplina)
                    for k, v in d_all.items():
                        if isinstance(v, dict) and norm_key_ci(k) == disc_cf:
                            dmap = v; break
                if isinstance(dmap, dict):
                    fields = dmap.get("fields") or {}
                    vals = fields.get(key) or fields.get(key.capitalize())
                    if isinstance(vals, list):
                        collected.extend(vals)
                    elif isinstance(vals, str) and vals.strip():
                        collected.append(vals)

            # 2) fallback: global
            g = emap.get("global") or {}
            gvals = g.get(key) or g.get(key.capitalize())
            if isinstance(gvals, list):
                collected.extend(gvals)
            elif isinstance(gvals, str) and gvals.strip():
                collected.append(gvals)
    except Exception:
        pass

    # 3) fallback final: aliases fixos
    collected.extend(ALIASES.get(key, []))
    return _norm_dedup(collected)



def match_field(row: dict, field_alias_key: str, field_map: Dict, etapa: Optional[str], disciplina: Optional[str]) -> Any:
    keys = aliases_for(field_map, etapa, disciplina, field_alias_key)
    return get_field_ci(row, keys) or ""

# -----------------------------------------------------
# EXTRATORES DE CAMPOS
# -----------------------------------------------------

def _clean_label(s: str) -> str:
    """
    Limpa ruídos comuns de labels vindos das planilhas:
    - remove bullets e hífens iniciais ("- Geometria", "• Números")
    - remove pontuação final simples (.,;:)
    - normaliza espaços em branco
    """
    s = str(s or "")
    s = s.strip()
    if not s:
        return ""

    # remove bullets / hífens no início
    s = re.sub(r'^[\-\u2022•]+\s*', '', s)

    # remove pontuação simples no final (.,;:)
    s = re.sub(r'\s*[;:.,]+$', '', s)

    # normaliza espaços internos
    s = re.sub(r'\s+', ' ', s)

    return s.strip()


def _uniq_clean_list(vals: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in vals or []:
        s = _clean_label(x)
        if not s:
            continue
        k = norm_key_ci(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def get_fields_ci_all(row: Dict[str, Any], keys: List[str]) -> List[Any]:
    """
    Versão "multi" do get_field_ci: tenta localizar TODOS os campos possíveis
    (pelos aliases) e retorna uma lista de valores encontrados (sem duplicar colunas).
    Mantém prioridade: exato > casefold > normalizado > contém, POR CHAVE.
    """
    if not row or not keys:
        return []

    # Normalizador de cabeçalho compatível com get_field_ci
    def _norm_hdr(s: str) -> str:
        s2 = strip_accents(str(s or ""))
        s2 = re.sub(r"[\(\)\[\]]", " ", s2)
        s2 = re.sub(r"[^\w\s]", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip().casefold()
        return s2

    row_keys = list(row.keys())
    row_casefold = {str(rk).casefold(): rk for rk in row_keys}
    row_norm_map = {_norm_hdr(rk): rk for rk in row_keys}

    found_cols: List[str] = []
    out_vals: List[Any] = []

    for want in keys:
        want_s = str(want or "").strip()
        if not want_s:
            continue

        # 0) exato
        rk = want_s if want_s in row else None

        # 1) casefold
        if rk is None:
            rk = row_casefold.get(want_s.casefold())

        # 2) normalizado == 
        if rk is None:
            rk = row_norm_map.get(_norm_hdr(want_s))

        # 3) contains parcial (como no get_field_ci)
        if rk is None:
            wn = _norm_hdr(want_s)
            if wn:
                for rnh, real_k in row_norm_map.items():
                    if wn in rnh or rnh in wn:
                        rk = real_k
                        break

        if rk and rk not in found_cols:
            found_cols.append(rk)
            out_vals.append(row.get(rk))

    return out_vals



def row_temas(
    row: dict,
    field_map: dict,
    etapa: Optional[str],
    disciplina: Optional[str]
) -> List[str]:
    """
    Extrai os 'temas' de uma linha respeitando a semântica da disciplina.

    - Língua Portuguesa:
        agrega TODOS os eixos possíveis (Unidade Temática, Práticas de Linguagem,
        Campo de Atuação etc.), pois o conceito de "tema" é multidimensional.
    - Demais disciplinas:
        mantém comportamento tradicional (primeiro campo que casar).
    """

    keys = aliases_for(field_map, etapa, disciplina, "tema")

    # Normaliza disciplina para decisão semântica
    disc_norm = norm_key_ci(disciplina or "")
    is_lp = disc_norm in {
        "lingua portuguesa",
        "língua portuguesa",
        "portugues",
        "português",
    }

    all_items: List[str] = []

    if is_lp:
        # 🔹 LP → agrega múltiplos campos (correção do bug)
        vals = get_fields_ci_all(row, keys)
        for v in vals:
            all_items.extend(split_multiline(v))
    else:
        # 🔹 Outras disciplinas → comportamento clássico
        val = get_field_ci(row, keys)
        all_items.extend(split_multiline(val))

    return _uniq_clean_list(all_items)




def row_objetos(row: dict, field_map: dict, etapa: Optional[str], disciplina: Optional[str]) -> List[str]:
    keys = aliases_for(field_map, etapa, disciplina, "objeto")
    val  = get_field_ci(row, keys)
    return _uniq_clean_list(split_multiline(val))


def row_titles(row: dict, field_map: dict, etapa: Optional[str], disciplina: Optional[str]) -> List[str]:
    keys = aliases_for(field_map, etapa, disciplina, "titulo")
    val  = get_field_ci(row, keys)
    # alguns conjuntos usam TÍTULO como “Tema”; mantemos múltiplos via split_multiline
    return _uniq_clean_list(split_multiline(val))


def row_conteudos(row: dict, field_map: dict, etapa: Optional[str], disciplina: Optional[str]) -> List[str]:
    keys = aliases_for(field_map, etapa, disciplina, "conteudo")
    val  = get_field_ci(row, keys)
    return _uniq_clean_list(split_multiline(val))

def row_aulas(row: dict, field_map: dict, etapa: Optional[str], disciplina: Optional[str]) -> List[str]:
    """
    Extrai o(s) identificador(es) de aula de uma linha.

    - Prioriza os campos definidos no field_map (ex.: "aula", "aula_unidade", "aula_sala").
    - Faz fallback fuzzy procurando qualquer coluna cujo nome contenha "aula".
    - Sempre devolve uma lista de strings (e.g. ["1"], ["Aula 3"], etc.).
    """

    # === ETAPA 1: Busca canônica via field_map ===
    canonical_keys_to_check = [
        "aula",          # mais genérico
        "aula_unidade",  # ex.: "Aula Unidade"
        "aula_sala",     # ex.: "Aula Sala"
    ]

    all_keys: List[str] = []
    for key_role in canonical_keys_to_check:
        keys_for_role = keys_from_field_map(field_map, key_role, etapa, disciplina)
        if keys_for_role:
            all_keys.extend(keys_for_role)

    all_keys = unique_preserve_order(all_keys)

    val = get_field_ci(row, all_keys)

    def _to_items(v: Any) -> List[str]:
        """Normaliza qualquer valor de aula para lista de strings."""
        if v is None:
            return []
        # Se vier número (int/float), transformamos em string simples
        if isinstance(v, (int, float)):
            return [str(int(v))]
        # Se for string, quebramos por linhas/separadores
        if isinstance(v, str):
            return finalize_list(
                unique_preserve_order(
                    split_multiline(v)
                )
            )
        # Qualquer outra coisa, convertemos para string única
        return [str(v)]

    items = _to_items(val)

    # Se a busca canônica funcionou (achou algo), usamos o resultado
    if items:
        return items

    # === ETAPA 2: Fallback fuzzy – procura qualquer coluna com "aula" no nome ===
    target_norm = "aula"
    fuzzy_values: List[str] = []

    for k, v in row.items():
        k_norm = norm_key_ci(k)  # minúsculo, sem acento, sem lixo
        if target_norm in k_norm:
            fuzzy_values.extend(_to_items(v))

    if not fuzzy_values:
        return []

    return finalize_list(
        unique_preserve_order(fuzzy_values)
    )
# ------------------------------------------------------------------
# DEBUG / MAPEAMENTO DE ERROS PARA HABILIDADES
# ------------------------------------------------------------------

HAB_DEBUG = True  # se quiser silenciar logs, mude para False

HAB_ERROR_CODES: Dict[str, str] = {
    "H001_NO_COL": (
        "Nenhuma coluna de habilidade encontrada via field_map nem por fallback "
        "(procurando 'habilidade' no nome das colunas)."
    ),
    "H002_EMPTY_VALS": (
        "Coluna(s) de habilidade identificada(s), mas sem conteúdo não vazio para esta linha."
    ),
    "H003_EXCEPTION": (
        "Exceção inesperada ao processar a linha em row_habilidades."
    ),
}


def _hab_log(*args: Any) -> None:
    """Log simples para debug de habilidades."""
    if HAB_DEBUG:
        print("[HABS]", *args)


def _hab_warn(code: str, msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
    """Log de warning com código de erro mapeado."""
    if not HAB_DEBUG:
        return
    base = HAB_ERROR_CODES.get(code, "Código de erro desconhecido.")
    ctx_str = f" | CTX={ctx}" if ctx else ""
    print(f"[HABS][WARN][{code}] {msg} :: {base}{ctx_str}")


def _split_hab_blocks(text: str) -> List[str]:
    """
    Divide um bloco com POSSÍVEIS múltiplas habilidades em segmentos individuais.

    Exemplo:
      "(EF03MA02) ... (EF03MA27) ..."
    → ["(EF03MA02) ...", "(EF03MA27) ..."]

    Regra:
      - Procura padrões de início de código BNCC tipo "(EF" ou "(EM".
      - Se não encontrar nada, devolve o texto inteiro como 1 item.
    """
    s = str(text or "").strip()
    if not s:
        return []

    # Posições de início de códigos BNCC no estilo "(EF..." ou "(EM..."
    matches = list(re.finditer(r"\((?:EF|EM)[0-9A-Z]", s, flags=re.IGNORECASE))
    if not matches:
        return [s]

    indices = [m.start() for m in matches]
    indices.append(len(s))  # sentinela final

    out: List[str] = []
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]
        chunk = s[start:end].strip(" ;\n\t")
        if chunk:
            out.append(chunk)

    if not out:
        out = [s]

    return out



def row_habilidades(
    row: dict,
    field_map: dict,
    etapa: Optional[str],
    disciplina: Optional[str],
) -> List[str]:
    """
    Extrai habilidades de uma linha, com as seguintes regras:

    - Usa field_map/keys_from_field_map para localizar colunas de habilidade
      (habilidade, habilidade_bncc, habilidade_bncc_computacao, habilidade_diretrizes_tec).
    - Se não achar via field_map, faz fallback procurando "habilidade" no nome das colunas.
    - Junta quebras de linha (\n, \r), espreme múltiplos espaços.
    - Se uma célula contiver múltiplas habilidades (ex.: EF03MA02 e EF03MA27),
      tenta separar em segmentos distintos (_split_hab_blocks).
    - Remove duplicatas preservando a ordem.
    - Loga problemas com códigos H001/H002/H003 para facilitar debug.
    """

    ctx_base = {
        "etapa": etapa,
        "disciplina": disciplina,
    }

    try:
        # -----------------------------
        # 1) Localiza colunas candidatas
        # -----------------------------
        canonical_roles = [
            "habilidade",
            "habilidade_bncc",
            "habilidade_bncc_computacao",
            "habilidade_diretrizes_tec",
        ]

        all_keys: List[str] = []
        for role in canonical_roles:
            cols = keys_from_field_map(field_map, role, etapa or "", disciplina or "")
            if cols:
                all_keys.extend(cols)

        all_keys = unique_preserve_order(all_keys)

        # Fallback: nenhuma coluna encontrada via field_map → procura por "habilidade" no cabeçalho
        if not all_keys:
            target_norm = "habilidade"
            for k in row.keys():
                if target_norm in norm_key_ci(k):
                    all_keys.append(k)

        if not all_keys:
            _hab_warn(
                "H001_NO_COL",
                "Nenhuma coluna de habilidade encontrada para esta linha.",
                {**ctx_base, "row_keys_sample": list(row.keys())[:15]},
            )
            return []

        # -----------------------------
        # 2) Coletar valores dessas colunas
        # -----------------------------
        raw_blocks: List[str] = []

        for col in all_keys:
            val = row.get(col)
            if val is None:
                continue

            # Se for lista, pega cada item; se não, vira string
            if isinstance(val, list):
                candidates = val
            else:
                candidates = [val]

            for v in candidates:
                s = str(v or "").strip()
                if not s:
                    continue

                # junta quebras de linha / \r\n
                s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                s = " ".join(s.split()).strip()
                if not s:
                    continue

                raw_blocks.append(s)

        if not raw_blocks:
            _hab_warn(
                "H002_EMPTY_VALS",
                "Colunas de habilidade encontradas, mas sem conteúdo útil nesta linha.",
                {**ctx_base, "hab_cols": all_keys},
            )
            return []

        # -----------------------------
        # 3) Quebra de blocos em habilidades individuais
        # -----------------------------
        all_habs: List[str] = []
        for block in raw_blocks:
            # Se o texto tiver múltiplas habilidades encadeadas, separa
            segments = _split_hab_blocks(block)
            if not segments:
                continue
            all_habs.extend(segments)

        # -----------------------------
        # 4) Limpeza final (dedupe + trim)
        # -----------------------------
        seen = set()
        final_list: List[str] = []

        for h in all_habs:
            h_clean = str(h or "").strip()
            if not h_clean:
                continue
            k = norm_key_ci(h_clean)
            if k in seen:
                continue
            seen.add(k)
            final_list.append(h_clean)

        _hab_log(
            "row_habilidades → extraídas",
            len(final_list),
            "habilidades.",
            {**ctx_base, "cols": all_keys},
        )

        return final_list

    except Exception as exc:
        _hab_warn(
            "H003_EXCEPTION",
            f"Exceção em row_habilidades: {exc!r}",
            {
                **ctx_base,
                "row_keys_sample": list(row.keys())[:15],
            },
        )
        return []



# utils_norm.py (FUNÇÃO CORRIGIDA)


# ------------------------------------------------------------------
# Filtro específico para HABILIDADES (com debug robusto)
# ------------------------------------------------------------------

def row_aulas(row: dict, field_map: dict, etapa: Optional[str], disciplina: Optional[str]) -> List[str]:
    # aula pode vir em "AULA", "Aulas", etc — o field_map do Médio traz em "aula"
    keys = aliases_for(field_map, etapa, disciplina, "aula")
    val  = get_field_ci(row, keys)
    return _uniq_clean_list(split_multiline(val))

# --- Compat: função esperada por versões antigas do main.py (/export/pdf) ---
def _row_temas_by_context(row: dict, field_map: Dict, etapa=None, disciplina=None) -> List[str]:
    """
    Placeholder de compatibilidade. Algumas versões antigas do backend importam
    esta função. Retorna os temas brutos da linha, respeitando field_map/etapa/disciplina.
    """
    return row_temas(row, field_map, etapa, disciplina)


# -----------------------------------------------------
# FILTRO UNIFICADO
# -----------------------------------------------------

def _check(vals: List[str], sel_set: set[str], contains: bool) -> bool:
    """
    Verifica se há correspondência entre os valores da linha (vals)
    e o conjunto selecionado (sel_set), com normalização bidirecional e suporte a contains parcial.
    """
    if not sel_set:
        return True

    # Normaliza ambos
    row_norm = {norm_key_ci(v) for v in vals if str(v).strip()}
    sel_norm = {norm_key_ci(s) for s in sel_set if str(s).strip()}

    if not row_norm:
        return False

    # match exato
    if row_norm & sel_norm:
        return True

    # match parcial
    if contains:
        for s in sel_norm:
            for r in row_norm:
                if s in r or r in s:
                    return True
        return False

    return False



def _to_str_set(values: Any) -> Set[str]:
    """
    Converte qualquer coisa (None, lista, set, tupla, string) em um set[str] limpo.
    """
    if not values:
        return set()

    # Já é um set de strings
    if isinstance(values, set):
        return {str(v).strip() for v in values if str(v).strip()}

    # String única -> vira um set com 1 item (desde que não seja vazia)
    if isinstance(values, str):
        v = values.strip()
        return {v} if v else set()

    # Qualquer iterável (lista, tupla, etc.)
    try:
        return {str(v).strip() for v in values if str(v).strip()}
    except TypeError:
        # Não é iterável "normal", cai aqui
        v = str(values).strip()
        return {v} if v else set()


def match_row(
    row: dict,
    field_map: Dict,
    # parâmetros “clássicos” (backcompat, usados por quem já passava sets diretamente)
    temas_sel: Any = None,
    objs_sel: Any = None,
    tits_sel: Any = None,
    cont_sel: Any = None,
    aulas_sel: Any = None,
    contains: bool = False,
    etapa=None,
    disciplina=None,
    # aliases que podem estar sendo usados no main.py como kwargs
    tema: Any = None,
    objeto: Any = None,
    titulo: Any = None,
    conteudo: Any = None,
    aula: Any = None,
    **kwargs,
) -> bool:
    """
    Decide se uma linha `row` casa com o filtro (tema/objeto/título/conteúdo/aula).

    Aceita tanto:
      - temas_sel / objs_sel / tits_sel / cont_sel / aulas_sel (sets ou listas)
      - quanto tema / objeto / titulo / conteudo / aula (listas ou sets)

    A prioridade é:
      - se *_sel vier preenchido, usa ele;
      - senão, usa o alias simples (tema, objeto, etc.).
    """

    # 1) Normaliza todos em set[str]
    temas  = _to_str_set(temas_sel  if temas_sel  is not None else tema)
    objs   = _to_str_set(objs_sel   if objs_sel   is not None else objeto)
    tits   = _to_str_set(tits_sel   if tits_sel   is not None else titulo)
    conts  = _to_str_set(cont_sel   if cont_sel   is not None else conteudo)
    aulas  = _to_str_set(aulas_sel  if aulas_sel  is not None else aula)

    # 2) Aplica a lógica de checagem exatamente como você já tinha
    if not _check(row_temas(row, field_map, etapa, disciplina),     temas, contains):
        return False
    if not _check(row_objetos(row, field_map, etapa, disciplina),   objs,  contains):
        return False
    if not _check(row_titles(row, field_map, etapa, disciplina),    tits,  contains):
        return False
    if not _check(row_conteudos(row, field_map, etapa, disciplina), conts, contains):
        return False
    if not _check(row_aulas(row, field_map, etapa, disciplina),     aulas, contains):
        return False

    return True



def filter_and_collect(
    data: List[dict],
    field_map: Dict,
    target_alias_key: str,
    tema: Optional[str] = None,
    objeto: Optional[str] = None,
    titulo: Optional[str] = None,
    conteudo: Optional[str] = None,
    aula: Optional[str] = None,
    contains: bool = False,
    etapa: Optional[str] = None,
    disciplina: Optional[str] = None,
) -> List[str]:

    temas_sel  = norm_set(tema)
    objs_sel   = norm_set(objeto)
    tits_sel   = norm_set(titulo)
    cont_sel   = norm_set(conteudo)
    aulas_sel  = norm_set(aula)

    seen_norm: set[str] = set()
    result: List[str] = []

    for r in data:
        # 🔎 NÃO filtramos mais por 'Ciclo/Etapa' aqui.
        # O load_rows_for(etapa, disciplina) já garante que 'data'
        # só contenha linhas daquela etapa/disciplina.
        if not match_row(
            r,
            field_map,
            temas_sel,
            objs_sel,
            tits_sel,
            cont_sel,
            aulas_sel,
            contains,
            etapa,
            disciplina,
        ):
            continue

        raw = match_field(r, target_alias_key, field_map, etapa, disciplina)
        raw_values: List[str] = []
        if isinstance(raw, list):
            raw_values.extend([str(x).strip() for x in raw])
        elif isinstance(raw, str):
            raw_values.extend(split_multiline(raw))

        for raw_v in raw_values:
            label = _clean_label(raw_v)
            if not label:
                continue
            k = norm_key_ci(label)
            if k and k not in seen_norm:
                seen_norm.add(k)
                result.append(label)


    result.sort(key=lambda s: strip_accents(s).casefold())
    return result


# -----------------------------------------------------
# BACKWARD COMPATIBILITY — aliases para funções antigas
# -----------------------------------------------------

# Muitos trechos do backend ainda usam nomes antigos.
# Para garantir compatibilidade total:
try:
    _norm_key = norm_key_ci
    _norm_key_ci = norm_key_ci
    norm_key = norm_key_ci
except Exception:
    pass


def _keys_from_field_map(role: str, etapa: str, disciplina: str) -> list[str]:
    """
    Essa função era usada pelo /api/dados/temas, objetos, titulos.
    Vamos manter compatível retornando somente a chave normalizada.
    """
    try:
        return [norm_key_ci(disciplina or "")]
    except Exception:
        return []

def filter_and_collect_habilidades(
    rows,
    ctx,
    etapa: str,
    disciplina: str,
    field_map: Dict[str, Any],
    contains: bool = False,
    only_codes: bool = False,
) -> List[str]:
    """
    Filtra linhas por contexto (tema/objeto/título/conteúdo/aula) e devolve
    uma lista de habilidades **sem duplicatas**, usando row_habilidades.

    Parâmetros:
      - rows: lista de linhas cruas (dict) da disciplina/etapa
      - ctx: dict com possíveis chaves "tema", "objeto", "titulo", "conteudo", "aula"
             onde cada valor pode ser string, lista ou set (multi-select → "A||B")
      - etapa / disciplina: usados por row_habilidades e match_row
      - field_map: mapeamento de campos (field_map.json carregado)
      - contains: se True, matching parcial (substring) nos campos de contexto
      - only_codes: se True, tentar retornar só o código BNCC (ex.: EF09MA08)
    """
    if not rows:
        return []

    ctx = ctx or {}

    # 1) Normaliza o contexto em listas (tema, objeto, titulo, conteudo, aula)
    tema_vals     = ensure_list(ctx.get("tema") or ctx.get("temas"))
    objeto_vals   = ensure_list(ctx.get("objeto"))
    titulo_vals   = ensure_list(ctx.get("titulo"))
    conteudo_vals = ensure_list(ctx.get("conteudo"))
    aula_vals     = ensure_list(ctx.get("aula"))

    # 2) Lista bruta de habilidades (antes da deduplicação)
    habs_raw: List[str] = []

    for idx, row in enumerate(rows):
        # Opcional: debug das primeiras linhas
        if idx < 3:
            print(f"[DEBUG HABS] linha[{idx}] chaves=", list(row.keys()))

        # 2a) Se há algum contexto, filtra a linha com match_row
        if any([tema_vals, objeto_vals, titulo_vals, conteudo_vals, aula_vals]):
            try:
                if not match_row(
                    row,
                    field_map,             # ← aqui é SEMPRE field_map como 2º argumento
                    tema=tema_vals,
                    objeto=objeto_vals,
                    titulo=titulo_vals,
                    conteudo=conteudo_vals,
                    aula=aula_vals,
                    contains=contains,
                    etapa=etapa,
                    disciplina=disciplina,
                ):
                    continue
            except Exception as ex:
                print(f"[DEBUG HABS] erro em match_row: {ex!r}")
                continue

        # 2b) Extrai habilidades da linha
        try:
            habs_row = row_habilidades(row, field_map, etapa, disciplina)
        except Exception as ex:
            print(f"[DEBUG HABS] erro em row_habilidades: {ex!r}")
            print(f"[DEBUG HABS] linha com erro chaves=", list(row.keys()))
            continue

        if not habs_row:
            continue

        for h in habs_row:
            if not h:
                continue
            txt = str(h).strip()
            if not txt or txt in PLACEHOLDERS:
                continue
            habs_raw.append(txt)

    if not habs_raw:
        return []

    # 3) Se quiser apenas códigos BNCC, mapeia cada habilidade para o código
    if only_codes:
        codes: List[str] = []
        for h in habs_raw:
            codigo, _texto = split_code_and_text(h)
            codes.append(codigo or h)
        # finalize_list já deduplica (case-insensitive) e ordena
        return finalize_list(codes)

    # 4) Caso normal: retorna habilidades completas, deduplicadas
    return finalize_list(habs_raw)


