from datasets import load_dataset
import pandas as pd
import os

# Carregar dataset do MovieLens (Hugging Face)
dataset = load_dataset("movielens", "ml-100k")

# Converter para DataFrame
movies = pd.DataFrame(dataset['item'])




# Preparar dados para IA Cultural
df_obras = movies[['movie_id', 'title', 'genres']].copy()
df_obras.rename(columns={
    'movie_id': 'id',
    'title': 'titulo',
    'genres': 'genero'
}, inplace=True)

# Colunas adicionais (simuladas por enquanto)
df_obras['tipo'] = 'filme'
df_obras['tema'] = 'Diversos'
df_obras['estilo'] = 'Narrativa Popular'
df_obras['contexto'] = 'Cinema Global'
df_obras['tags'] = df_obras['genero'].apply(lambda x: x.replace('|', ' #'))
df_obras['descricao'] = df_obras['titulo'] + ' - Filme do gênero ' + df_obras['genero']

# Organizar colunas finais
df_obras = df_obras[['id', 'titulo', 'tipo', 'genero', 'tema', 'estilo', 'contexto', 'tags', 'descricao']]

# Salvar como CSV
os.makedirs('data', exist_ok=True)
df_obras.to_csv('data/obras.csv', index=False)

print("✅ Base de obras culturais gerada com sucesso em data/obras.csv")
