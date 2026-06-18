import pandas as pd
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dados de NLP (Simulação)
data = {
    'titulo_vaga': ['Engenheiro DevOps', 'Analista de SOC pleno', 'Especialista em Pentest e Red Team', 'Cloud Engineer AWS', 'Arquiteto de Segurança'],
    'descricao': [
        'Engenheiro DevOps com foco em AWS, Docker e Kubernetes. Necessário experiência em CI/CD.',
        'Atuação no Security Operations Center analisando incidentes 24/7.',
        'Realizar testes de invasão, simular ataques e elaborar relatórios de vulnerabilidades.',
        'Migração e manutenção de infraestrutura cloud na AWS utilizando Terraform e Kubernetes.',
        'Desenho de soluções seguras e arquitetura zero-trust em ambientes híbridos.'
    ],
    'volume': [150, 200, 80, 120, 90],
    'salario': [12000, 8500, 15000, 13500, 18000]
}
df_vagas = pd.DataFrame(data)

# Limpeza de texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df_vagas['descricao_limpa'] = df_vagas['descricao'].apply(clean_text)

# Modelos de RI
query = "devops aws kubernetes"
query_limpa = clean_text(query)
query_tokens = query_limpa.split()

# Score Booleano
def boolean_score(text, tokens):
    text_tokens = text.split()
    return sum([1 for t in tokens if t in text_tokens])

df_vagas['score_booleano'] = df_vagas['descricao_limpa'].apply(lambda x: boolean_score(x, query_tokens))

# Score BM25
tokenized_corpus = [doc.split() for doc in df_vagas['descricao_limpa']]
bm25 = BM25Okapi(tokenized_corpus)
df_vagas['score_bm25'] = bm25.get_scores(query_tokens)

# Score Embeddings
model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model_embed.encode(df_vagas['descricao_limpa'].tolist(), convert_to_tensor=True)
query_embedding = model_embed.encode(query_limpa, convert_to_tensor=True)
cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
df_vagas['score_embeddings'] = cosine_scores.cpu().numpy()

# Análise de Sentimento (Zero-Shot)
frases_desafios = [
    "A infraestrutura caiu no meio da madrugada e o alerta não funcionou.",
    "Conseguimos automatizar todo o pipeline de deploy com sucesso.",
    "Revisar logs de auditoria do sistema todos os dias é uma tarefa repetitiva.",
    "A nova política de segurança melhorou significativamente a nossa resiliência."
]

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
candidate_labels = ["Positivo", "Neutro", "Negativo"]

resultados_sentimento = classifier(frases_desafios, candidate_labels)
sentimentos = [res['labels'][0] for res in resultados_sentimento]

df_sentimentos = pd.DataFrame({'Sentimento': sentimentos})
contagem_sentimentos = df_sentimentos['Sentimento'].value_counts().reset_index()
contagem_sentimentos.columns = ['Sentimento', 'Contagem']

# Visualização (Grid Assimétrico)
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"type": "bar", "colspan": 2}, None],
           [{"type": "scatter"}, {"type": "domain"}],
           [{"type": "table", "colspan": 2}, None]],
    subplot_titles=("Volume de Vagas por Cargo", "Média Salarial", "Sentimento de Desafios Profissionais", "Scores de Recuperação de Informação (RI)")
)

# Gráfico de Barras (Volume)
fig.add_trace(
    go.Bar(x=df_vagas['titulo_vaga'], y=df_vagas['volume'], marker_color='#00CC96'),
    row=1, col=1
)

# Gráfico de Linhas (Salário)
fig.add_trace(
    go.Scatter(x=df_vagas['titulo_vaga'], y=df_vagas['salario'], mode='lines+markers', line=dict(color='#AB63FA', width=3)),
    row=2, col=1
)

# Donut Chart (Sentimentos)
fig.add_trace(
    go.Pie(labels=contagem_sentimentos['Sentimento'], values=contagem_sentimentos['Contagem'], hole=0.5, marker_colors=['#EF553B', '#00CC96', '#636EFA']),
    row=2, col=2
)

# Tabela (Scores de RI)
fig.add_trace(
    go.Table(
        header=dict(values=["Vaga", "Score Booleano", "Score BM25", "Score Embeddings"],
                    fill_color='#2a2a2a',
                    align='left',
                    font=dict(color='white')),
        cells=dict(values=[df_vagas['titulo_vaga'], 
                           df_vagas['score_booleano'], 
                           df_vagas['score_bm25'].round(3), 
                           df_vagas['score_embeddings'].round(3)],
                   fill_color='#1f1f1f',
                   align='left',
                   font=dict(color='white'))
    ),
    row=3, col=1
)

# Tema e Dimensões
fig.update_layout(
    template='plotly_dark',
    height=1800,
    width=1200,
    showlegend=False,
    title_text="Dashboard Analítico: Cibersegurança & DevOps",
    title_x=0.5
)

# Exportação Estática
fig.write_image("dashboard_cybersec.png", width=1200, height=1800, scale=2)
