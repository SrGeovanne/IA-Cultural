# app.py
import os
import pandas as pd
import streamlit as st
from utils.recomendador import (
    recomendar_obras,
    recomendar_personalizado,
)

DATA_DIR = "data"
CSV_OBRAS = os.path.join(DATA_DIR, "obras.csv")
CSV_HISTORICO = os.path.join(DATA_DIR, "historico.csv")
CSV_FEEDBACK = os.path.join(DATA_DIR, "feedback.csv")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="🎭 IA Cultural - Sistema de Recomendação", layout="wide")
st.title("🎭 IA Cultural - Sistema de Recomendação de Cultura")

# ---------- Estado ----------
if "last_recs" not in st.session_state:
    st.session_state.last_recs = None   # DataFrame das últimas recomendações mostradas
if "mode" not in st.session_state:
    st.session_state.mode = None        # "conteudo" | "personalizado"

@st.cache_data
def _read_csv_safe(path: str):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path, engine="python")
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception:
        return None

HIST_COLS = ["id","titulo","tipo","genero","tema","estilo","contexto","tags","descricao","explicacao"]

def salvar_historico_rows(rows):
    if not rows:
        return
    df_hist = pd.DataFrame(rows)
    for c in HIST_COLS:
        if c not in df_hist.columns:
            df_hist[c] = ""
    df_hist = df_hist[HIST_COLS]

    try:
        if os.path.exists(CSV_HISTORICO):
            old = pd.read_csv(CSV_HISTORICO, engine="python")
            for c in HIST_COLS:
                if c not in old.columns:
                    old[c] = ""
            old = old[HIST_COLS]
            merged = pd.concat([old, df_hist], ignore_index=True)
        else:
            merged = df_hist
        merged = merged.drop_duplicates(subset=["id", "titulo"], keep="first")
        merged.to_csv(CSV_HISTORICO, index=False)
    except Exception as e:
        st.error(f"Erro ao salvar histórico: {e}")

def salvar_feedback(id_, titulo_, valor_):
    try:
        header_needed = not os.path.exists(CSV_FEEDBACK) or os.path.getsize(CSV_FEEDBACK) == 0
        pd.DataFrame([[str(id_), str(titulo_), str(valor_)]],
                     columns=["id","titulo","feedback"]).to_csv(
            CSV_FEEDBACK, mode="a", header=header_needed, index=False, encoding="utf-8"
        )
        st.toast(f"Feedback salvo: {valor_} → {titulo_}")
    except Exception as e:
        st.error(f"Erro ao salvar feedback: {e}")

# ===================== UI (Tabs) =====================
aba1, aba2, aba3 = st.tabs(["Recomendar", "Histórico", "Feedback"])

with aba1:
    st.header("🔍 Obter Recomendações")
    obra_ok = _read_csv_safe(CSV_OBRAS) is not None
    st.caption(f"📁 Catálogo: {'✅' if obra_ok else '❌'} {os.path.abspath(CSV_OBRAS)}")

    # Essenciais
    tipo = st.selectbox("Escolha o tipo de obra:", ["filme", "livro", "jogo", "série"])
    genero = st.text_input("Gênero (ex: Ficção Científica, RPG):")

    # Opcionais dentro do expander
    with st.expander("⚙️ Opções avançadas (tema, estilo, contexto, tags)"):
        tema = st.text_input("Tema:", "")
        estilo = st.text_input("Estilo:", "")
        contexto = st.text_input("Contexto:", "")
        tags = st.text_input("Tags (separadas por ;):", "")

    # Checkboxes de filtros (fora do expander)
    col_left, col_right = st.columns([1,1])
    with col_left:
        filtrar_vistos = st.checkbox("Ocultar itens já vistos", value=False, help="Usa data/historico.csv")
        filtrar_curtiu = st.checkbox("Ocultar itens que já curti", value=False, help="Usa data/feedback.csv")

    # Botões (fora do expander)
    col_busca, col_perso = st.columns([1, 1])

    # -------- Gerar (conteúdo) --------
    with col_busca:
        if st.button("Gerar Recomendações (Conteúdo)"):
            if not obra_ok:
                st.error("⚠️ `data/obras.csv` não encontrado, vazio ou inválido.")
            elif genero.strip() == "":
                st.error("⚠️ Preencha pelo menos o campo de gênero.")
            else:
                prefs = {"tipo": tipo, "genero": genero, "tema": tema, "estilo": estilo, "contexto": contexto, "tags": tags}
                try:
                    recs = recomendar_obras(
                        prefs, CSV_OBRAS, quantidade=10,
                        filtrar_vistos=filtrar_vistos,
                        filtrar_curtiu=filtrar_curtiu,
                        caminho_historico=CSV_HISTORICO,
                        caminho_feedback=CSV_FEEDBACK,
                    )
                    st.session_state.last_recs = recs if recs is not None and not recs.empty else None
                    st.session_state.mode = "conteudo"
                    if st.session_state.last_recs is None:
                        st.info("Nenhuma recomendação para esses critérios.")
                except Exception as e:
                    st.error(f"Erro ao recomendar: {e}")

    # -------- Personalizado (curtidas) --------
    with col_perso:
        if st.button("Recomendar baseado no que já curti"):
            try:
                recs, motivo = recomendar_personalizado(
                    CSV_OBRAS, CSV_FEEDBACK, quantidade=10,
                    caminho_historico=CSV_HISTORICO, filtrar_vistos=filtrar_vistos
                )
                if recs is None or recs.empty:
                    st.session_state.last_recs = None
                    st.session_state.mode = "personalizado"
                    if motivo in ("Sem feedback", "Sem curtidas"):
                        st.warning("⚠️ Você ainda não curtiu nada. Dê 👍 em algumas obras primeiro.")
                    elif motivo == "IDs não bateram":
                        st.warning("⚠️ Não consegui relacionar suas curtidas às obras (IDs).")
                    elif motivo == "Sem obras":
                        st.error("⚠️ Adicione `data/obras.csv`.")
                    else:
                        st.info("ℹ️ Sem novas sugestões agora. Curta mais obras.")
                else:
                    st.session_state.last_recs = recs
                    st.session_state.mode = "personalizado"
            except Exception as e:
                st.error(f"Erro no personalizado: {e}")

    st.divider()

    # -------- Render da lista persistida + botões de feedback --------
    if st.session_state.last_recs is not None and not st.session_state.last_recs.empty:
        st.subheader("Resultados")
        df_show = st.session_state.last_recs.copy()

        if st.toggle("Salvar esta lista inteira no histórico (vistos)", value=False):
            rows = [{c: row.get(c, "") for c in HIST_COLS} for _, row in df_show.iterrows()]
            salvar_historico_rows(rows)
            st.success("Lista salva no histórico.")

        to_remove = []
        for i, row in df_show.iterrows():
            st.markdown(f"### {row.get('titulo','')}")
            st.write(f"**Tipo:** {row.get('tipo','')} • **Gênero:** {row.get('genero','')} • **Tema:** {row.get('tema','')}")
            st.write(f"**Estilo:** {row.get('estilo','')} • **Contexto:** {row.get('contexto','')}")
            if row.get("tags"):
                st.write(f"**Tags:** {row.get('tags','')}")
            if row.get("descricao"):
                st.info(row.get("descricao",""))
            if row.get("explicacao"):
                st.caption(f"🧠 {row.get('explicacao','')}")
            if row.get("aviso"):
                st.caption(f"ℹ️ {row.get('aviso','')}")

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button(f"👍 Curtir {row.get('id','')}", key=f"like_{row.get('id','')}"):
                    salvar_feedback(row.get("id",""), row.get("titulo",""), "curtiu")
                    to_remove.append(i)
            with c2:
                if st.button(f"👎 Não Curtir {row.get('id','')}", key=f"dislike_{row.get('id','')}"):
                    salvar_feedback(row.get("id",""), row.get("titulo",""), "nao_curtiu")
                    to_remove.append(i)
            with c3:
                if st.button(f"📝 Marcar como visto {row.get('id','')}", key=f"visto_{row.get('id','')}"):
                    salvar_historico_rows([{c: row.get(c,"") for c in HIST_COLS}])
                    to_remove.append(i)

            st.divider()

        if to_remove:
            st.session_state.last_recs = st.session_state.last_recs.drop(index=to_remove)
            if st.session_state.last_recs.empty:
                st.info("Acabaram os itens desta lista. Gere novamente quando quiser.")
