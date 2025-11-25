# utils/recomendador.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path, engine='python')
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception:
        return None

_CAMPOS_TEXTO = ["titulo", "tipo", "genero", "tema", "estilo", "contexto", "tags", "descricao"]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    for c in _CAMPOS_TEXTO:
        if c not in base.columns:
            base[c] = ""
    if "id" not in base.columns:
        base.insert(0, "id", range(1, len(base) + 1))
    for c in _CAMPOS_TEXTO + ["id"]:
        base[c] = base[c].astype(str)
    return base

def _build_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["titulo"].fillna("") + " " +
        df["tipo"].fillna("") + " " +
        df["genero"].fillna("") + " " +
        df["tema"].fillna("") + " " +
        df["estilo"].fillna("") + " " +
        df["contexto"].fillna("") + " " +
        df["tags"].fillna("") + " " +
        df["descricao"].fillna("")
    ).str.strip()

def _vetorizar(df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    base = _ensure_cols(df)
    base["caracteristicas"] = _build_text(base)
    if base["caracteristicas"].str.strip().eq("").all():
        raise ValueError("Nenhum texto de características disponível para vetorizar.")
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(base["caracteristicas"])
    return base, vec, X

def recomendar_obras(
    preferencias: Dict[str, str],
    caminho_csv: str,
    quantidade: int = 10,
    filtrar_vistos: bool = False,
    filtrar_curtiu: bool = False,
    caminho_historico: str = "data/historico.csv",
    caminho_feedback: str = "data/feedback.csv",
) -> pd.DataFrame:
    obras = _safe_read_csv(caminho_csv)
    if obras is None or len(obras) == 0:
        raise FileNotFoundError(f"CSV vazio/não encontrado: {caminho_csv}")

    tipo = (preferencias.get("tipo") or "").strip().lower()
    df = obras
    if tipo:
        df = obras[obras.get("tipo", "").astype(str).str.lower() == tipo]
        if df.empty:
            df = obras.copy()
            df["__aviso_tipo"] = f"Tipo '{tipo}' não encontrado; usando catálogo completo."

    df, vec, X = _vetorizar(df)

    consulta = " ".join([
        preferencias.get("genero", ""),
        preferencias.get("tema", ""),
        preferencias.get("estilo", ""),
        preferencias.get("contexto", ""),
        preferencias.get("tags", ""),
    ]).strip()
    if not consulta:
        consulta = tipo or " ".join(df["titulo"].head(3).tolist())

    q = vec.transform([consulta])
    score = cosine_similarity(q, X).ravel()
    df = df.assign(_score=score)

    if filtrar_curtiu:
        fb = _safe_read_csv(caminho_feedback)
        if fb is not None and {"id", "feedback"}.issubset(fb.columns):
            fb["id"] = fb["id"].astype(str)
            ids_curtiu = fb.loc[
                fb["feedback"].astype(str).str.lower().str.strip().isin(["curtiu", "like", "liked", "positivo"]),
                "id"
            ].unique()
            df = df[~df["id"].astype(str).isin(ids_curtiu)]

    if filtrar_vistos:
        hist = _safe_read_csv(caminho_historico)
        if hist is not None and "id" in hist.columns:
            ids_vistos = hist["id"].astype(str).dropna().unique()
            df = df[~df["id"].astype(str).isin(ids_vistos)]

    tokens_q = [t for t in consulta.lower().split() if t]
    def _mk_exp(row):
        tokens_item = [t for t in str(row.get("caracteristicas", "")).lower().split() if t]
        inter, seen = [], set()
        for t in tokens_q:
            if t in tokens_item and t not in seen:
                inter.append(t); seen.add(t)
            if len(inter) >= 3:
                break
        return "Palavras em comum: " + ", ".join(inter) if inter else "Similaridade geral de conteúdo"

    df["explicacao"] = df.apply(_mk_exp, axis=1)

    out = df.sort_values("_score", ascending=False).drop(columns=["_score", "caracteristicas"], errors="ignore").head(quantidade)
    if "__aviso_tipo" in df.columns:
        out["aviso"] = df["__aviso_tipo"].iloc[0]
    return out

def recomendar_personalizado(
    caminho_csv: str,
    caminho_feedback: str,
    quantidade: int = 10,
    caminho_historico: str = "data/historico.csv",
    filtrar_vistos: bool = False,
):
    obras = _safe_read_csv(caminho_csv)
    if obras is None or len(obras) == 0:
        return pd.DataFrame(), "Sem obras"

    fb = _safe_read_csv(caminho_feedback)
    if fb is None or not {"id", "feedback"}.issubset(fb.columns):
        return pd.DataFrame(), "Sem feedback"

    obras = _ensure_cols(obras)
    fb["id"] = fb["id"].astype(str)

    curtiu = fb[fb["feedback"].astype(str).str.lower().str.strip().isin(["curtiu","like","liked","positivo"])]
    if curtiu.empty:
        return pd.DataFrame(), "Sem curtidas"

    obras_like = obras[obras["id"].astype(str).isin(curtiu["id"].astype(str))].copy()
    if obras_like.empty:
        return pd.DataFrame(), "IDs não bateram"

    # 1 vetorizar: TREINA no catálogo todo, TRANSFORMA as curtidas com o MESMO vetor
    obras_all = obras.copy()
    obras_all["caracteristicas"] = _build_text(obras_all)
    if obras_all["caracteristicas"].str.strip().eq("").all():
        return pd.DataFrame(), "Sem texto para vetorizar"

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1,2), min_df=1)
    X_all = vec.fit_transform(obras_all["caracteristicas"])

    obras_like = obras_like.copy()
    obras_like["caracteristicas"] = _build_text(obras_like)
    X_like = vec.transform(obras_like["caracteristicas"])

    # similaridade (agora tem MESMO nº de colunas)
    sim = cosine_similarity(X_like, X_all)      # [n_like, n_all]
    media = sim.mean(axis=0)                    # score médio
    best_like_idx = sim.argmax(axis=0)          # índice da curtida mais parecida p/ cada candidato
    best_like_titles = obras_like.iloc[best_like_idx]["titulo"].values

    ranked = obras_all.assign(
        _score=media,
        explicacao=[f"Parecido com: {t}" for t in best_like_titles]
    ).sort_values("_score", ascending=False)

    # remove já curtidas
    ranked = ranked[~ranked["id"].astype(str).isin(curtiu["id"].astype(str))]

    # filtra vistos (opcional)
    if filtrar_vistos:
        hist = _safe_read_csv(caminho_historico)
        if hist is not None and "id" in hist.columns:
            ids_vistos = hist["id"].astype(str).dropna().unique()
            ranked_sem_vistos = ranked[~ranked["id"].astype(str).isin(ids_vistos)]
        else:
            ranked_sem_vistos = ranked
        if ranked_sem_vistos.empty:
            ranked_sem_vistos = ranked
    else:
        ranked_sem_vistos = ranked

    out = ranked_sem_vistos.drop(columns=["caracteristicas","_score"], errors="ignore").head(quantidade)
    return out, None

