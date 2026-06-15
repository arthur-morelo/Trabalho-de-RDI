import pandas as pd
import re
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Setup inicial e configuração
pio.renderers.default = "notebook_connected"

# Dados Simulados com Desafios_Relatados para Análise de Sentimentos
dados = {
    'UF': ['SP', 'PR', 'RJ', 'DF', 'MG'],
    'Volume_Vagas': [64, 6, 4, 2, 2],
    'Media_Salarial': [9448.41, 8406.50, 7171.59, 10656.71, 7382.81],
    'Descricao_Vaga': [
        "Vaga para Cientista de Dados focado em Python, SQL e machine learning na nuvem.",
        "Desenvolvedor Front-end com alta experiência em React, TypeScript e JavaScript.",
        "Engenheiro de Dados Sênior. Requisitos: PySpark, SQL, Python e arquitetura AWS.",
        "Procuramos Cientista de Dados especialista. Domínio em Python, estatística e deploy de modelos.",
        "Estágio em análise de dados. Essencial saber Python, Pandas, SQL e visualização de dados."
    ],
    'Desafios_Relatados': [
        "Excelente pacote de benefícios e progressão",
        "Ambiente de trabalho tóxico e salário baixo",
        "Ferramentas modernas, mas muita pressão",
        "Equipe muito colaborativa e horários flexíveis",
        "Gestão desorganizada e falta de plano de carreira"
    ]
}
df = pd.DataFrame(dados).sort_values(by='Volume_Vagas', ascending=False)

# Pré-processamento (PLN) para RI
def limpar_texto(texto):
    texto = str(texto).lower()
    return re.sub(r'[^\w\s]', '', texto)

df['Texto_Limpo'] = df['Descricao_Vaga'].apply(limpar_texto)

query = "cientista de dados python"
query_limpa = limpar_texto(query)
query_tokens = query_limpa.split()

# Modelo 1: Booleano
def busca_booleana(texto, tokens_query):
    texto_tokens = set(texto.split())
    return 1 if all(token in texto_tokens for token in tokens_query) else 0

df['Score_Booleano'] = df['Texto_Limpo'].apply(lambda x: busca_booleana(x, query_tokens))

# Modelo 2: BM25
corpus_tokenizado = [texto.split() for texto in df['Texto_Limpo']]
bm25 = BM25Okapi(corpus_tokenizado)
df['Score_BM25'] = bm25.get_scores(query_tokens)

# Modelo 3: Embeddings
modelo_st = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
query_embedding = modelo_st.encode([query_limpa])
corpus_embeddings = modelo_st.encode(df['Texto_Limpo'].tolist())
df['Score_Embeddings'] = cosine_similarity(query_embedding, corpus_embeddings).flatten()

# Análise de Sentimentos Zero-Shot
classificador = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
labels_sentimento = ['Positivo', 'Negativo', 'Neutro']

def classificar_sentimento(texto):
    resultado = classificador(texto, labels_sentimento)
    return resultado['labels'][0]

df['Sentimento_Dominante'] = df['Desafios_Relatados'].apply(classificar_sentimento)

# Preparação de DataFrames auxiliares
df_tabela = df.sort_values(by='Score_Embeddings', ascending=False)
contagem_sentimentos = df['Sentimento_Dominante'].value_counts().reset_index()
contagem_sentimentos.columns = ['Sentimento', 'Contagem']

# Layout do Dashboard de Alta Resolução (Grid Assimétrico 3x2 Invertido)
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}], 
           [{"type": "domain", "colspan": 2}, None], 
           [{"type": "table", "colspan": 2}, None]],
    subplot_titles=(
        "Volume de Vagas por UF", 
        "Média Salarial por UF (R$)", 
        "Sentimento dos Profissionais",
        "Ranking de Vagas: Modelos de RI"
    ),
    vertical_spacing=0.04,
    row_heights=[0.22, 0.48, 0.30]
)

# Linha 1, Col 1: Vagas por UF (Estética Neon)
fig.add_trace(
    go.Bar(
        x=df['UF'],
        y=df['Volume_Vagas'],
        name="Vagas",
        marker_color='#00E5FF',
        opacity=0.9,
        hovertemplate="<b>UF:</b> %{x}<br><b>Vagas:</b> %{y}<extra></extra>"
    ),
    row=1, col=1
)

# Linha 1, Col 2: Salário por UF (Estética Neon)
fig.add_trace(
    go.Scatter(
        x=df['UF'],
        y=df['Media_Salarial'],
        name="Média Salarial",
        mode='lines+markers',
        line=dict(color='#FF3D00', width=4),
        marker=dict(color='#FF3D00', size=12),
        hovertemplate="<b>UF:</b> %{x}<br><b>Média:</b> R$ %{y:,.2f}<extra></extra>"
    ),
    row=1, col=2
)

# Linha 2, Col 1: Gráfico de Rosquinha de Sentimento (Estética Neon)
cores_sentimento = {'Positivo': '#00E5FF', 'Negativo': '#FF3D00', 'Neutro': '#9CA3AF'}
fig.add_trace(
    go.Pie(
        labels=contagem_sentimentos['Sentimento'],
        values=contagem_sentimentos['Contagem'],
        name="Sentimento",
        marker=dict(colors=[cores_sentimento.get(s, '#3498db') for s in contagem_sentimentos['Sentimento']]),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=16),
        hovertemplate="<b>%{label}</b><br>Quantidade: %{value} (%{percent})<extra></extra>"
    ),
    row=2, col=1
)

# Preparando cores alternadas para a tabela (Dark Mode)
n_rows = len(df_tabela)
fill_colors = [['#374151', '#1F2937'][i % 2] for i in range(n_rows)]

# Linha 3, Col 1: Tabela de Alto Contraste
fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Descrição</b>', '<b>Booleano</b>', '<b>BM25</b>', '<b>Embeddings</b>'],
            fill_color='#1F2937',
            align='left',
            font=dict(size=16, color='#F3F4F6'),
            height=40
        ),
        cells=dict(
            values=[
                df_tabela['Descricao_Vaga'], 
                df_tabela['Score_Booleano'], 
                df_tabela['Score_BM25'].round(4), 
                df_tabela['Score_Embeddings'].round(4)
            ],
            fill_color=[fill_colors * 4], 
            align='left',
            font=dict(size=14, color='#D1D5DB'),
            height=45
        )
    ),
    row=3, col=1
)

# Acabamento Profissional e Dimensionamento (Dark Mode)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='#111827',
    plot_bgcolor='rgba(0,0,0,0)',
    title_text="Dashboard Analítico - Mercado de Dados & NLP",
    title_font_size=28,
    showlegend=False,
    height=1900,
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.update_yaxes(title_text="Quantidade", row=1, col=1)
fig.update_yaxes(title_text="Salário (R$)", tickprefix="R$ ", tickformat=",.2f", row=1, col=2)

# Exportação HTML Autônoma
fig.write_html('Trabalho_Final_Automatizado.html', include_plotlyjs=True)
print("O script analise_final_plotly.py gerou o HTML de alta resolução (Dark Mode) com sucesso.")
