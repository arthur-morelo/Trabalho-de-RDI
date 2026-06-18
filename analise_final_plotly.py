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
    'titulo_vaga': [
        'Engenheiro DevOps (CI/CD)', 'Analista de SOC pleno', 'Especialista em Pentest', 
        'Cloud Engineer AWS', 'Arquiteto de GRC', 'SRE Sênior', 
        'Consultor de Cibersegurança', 'Analista DevOps Cloud', 'Especialista em AWS Cloud',
        'Red Team (Pentest)'
    ],
    'descricao': [
        'Engenheiro com foco em CI/CD, Docker e Kubernetes.',
        'Atuação no SOC analisando incidentes e segurança 24/7.',
        'Realizar teste de invasão, simular ataques e relatórios.',
        'Migração cloud na AWS utilizando Terraform e Docker.',
        'Desenho de soluções seguras e GRC em ambientes híbridos.',
        'Engenharia de confiabilidade (SRE) e automação de infra com Kubernetes.',
        'Consultoria focada em arquitetura de segurança e auditoria.',
        'Automação de infra, CI/CD, kubernetes.',
        'Gerenciamento de recursos na AWS e otimização de cloud.',
        'Atuação em red team e teste de invasão avançado.'
    ],
    'salario': [12000, 8500, 15000, 13500, 18000, 16000, 14000, 10000, 14500, 15500],
    'UF': ['SP', 'DF', 'PE', 'MG', 'RS', 'RJ', 'CE', 'PR', 'AM', 'GO']
}
df_vagas = pd.DataFrame(data)

# Categorização de Área (Cyber vs DevOps)
def map_area(descricao):
    descricao_low = descricao.lower()
    if re.search(r'segurança|pentest|soc|cyber|invasão|grc', descricao_low):
        return 'Cibersegurança'
    elif re.search(r'devops|aws|cloud|docker|kubernetes|ci/cd|sre', descricao_low):
        return 'DevOps'
    return 'Outros'

df_vagas['Area'] = df_vagas['descricao'].apply(map_area)

# Extração de Subárea
def get_subarea(desc):
    d = desc.lower()
    if 'invasão' in d or 'pentest' in d or 'red team' in d: return 'Pentest'
    if 'soc' in d or 'incidente' in d: return 'SOC'
    if 'grc' in d or 'arquitetura' in d: return 'GRC/Arq'
    if 'sre' in d: return 'SRE'
    if 'ci/cd' in d: return 'CI/CD'
    if 'cloud' in d or 'aws' in d: return 'Cloud'
    return 'Outros'

df_vagas['Subarea'] = df_vagas['descricao'].apply(get_subarea)

# Mapeamento de Regiões do Brasil
regioes_map = {
    'Norte': ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC'],
    'Nordeste': ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA'],
    'Centro-Oeste': ['MT', 'MS', 'GO', 'DF'],
    'Sudeste': ['SP', 'RJ', 'ES', 'MG'],
    'Sul': ['PR', 'RS', 'SC']
}

def map_regiao(uf):
    for regiao, ufs in regioes_map.items():
        if uf in ufs:
            return regiao
    return 'Indefinido'

df_vagas['Regiao'] = df_vagas['UF'].apply(map_regiao)

# Limpeza de texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df_vagas['descricao_limpa'] = df_vagas['descricao'].apply(clean_text)

# Modelos de RI
query_str = "devops aws kubernetes"
query_limpa = clean_text(query_str)
query_tokens = query_limpa.split()

# Dicionário de Sinônimos (Booleano Inteligente)
sinonimos = {
    "pentest": ["pentest", "teste de invasão", "teste de penetração"],
    "devops": ["devops", "sre", "automação de infra"],
    "aws": ["aws", "amazon web services", "nuvem amazon"],
    "kubernetes": ["kubernetes", "k8s", "orquestração de contêineres"]
}

def intelligent_boolean_score(text, query_terms, sinonimos_dict):
    text_lower = text.lower()
    for term in query_terms:
        found = term in text_lower
        if not found and term in sinonimos_dict:
            for syn in sinonimos_dict[term]:
                if syn in text_lower:
                    found = True
                    break
        if not found:
            return 0
    return 1

df_vagas['score_booleano'] = df_vagas['descricao'].apply(lambda x: intelligent_boolean_score(x, query_tokens, sinonimos))

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

# Preparação de Dados para a Região (Volume de Vagas)
volume_regiao = df_vagas.groupby('Regiao').size().reset_index(name='Volume')
# Ordenar para ficar bonitinho
volume_regiao = volume_regiao.sort_values(by='Volume', ascending=False)

# Geração de Dados de Sub-Carreiras
data_sub = {
    'Sub-carreira': ['Pentest', 'SOC', 'GRC', 'Cloud', 'SRE', 'CI/CD'],
    'Area': ['Cibersegurança', 'Cibersegurança', 'Cibersegurança', 'DevOps', 'DevOps', 'DevOps'],
    'Salario': [8500, 6000, 9500, 10000, 12000, 8000]
}
df_subcarreiras = pd.DataFrame(data_sub)

# Visualização (Layout de 3 Linhas e 2 Colunas)
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}], 
           [{"type": "domain", "colspan": 2}, None], 
           [{"type": "table", "colspan": 2}, None]],
    row_heights=[0.40, 0.35, 0.25],
    vertical_spacing=0.08,
    subplot_titles=("Volume de Vagas por Região", "Média Salarial por Sub-carreira", "Sentimento de Desafios Profissionais", "Scores de Recuperação de Informação (RI)")
)

# Linha 1, Coluna 1 - Gráfico de Barras por Região (substituindo o mapa)
fig.add_trace(
    go.Bar(
        name='Vagas',
        x=volume_regiao['Regiao'],
        y=volume_regiao['Volume'],
        marker_color='#FF5722',
        hovertemplate='Região: %{x}<br>Vagas: %{y}<extra></extra>',
        showlegend=False
    ),
    row=1, col=1
)

# Linha 1, Coluna 2 - Gráfico de Salários (Sub-carreiras)
df_cyber = df_subcarreiras[df_subcarreiras['Area'] == 'Cibersegurança']
fig.add_trace(
    go.Bar(
        name='Cibersegurança',
        x=df_cyber['Sub-carreira'],
        y=df_cyber['Salario'],
        marker_color='#00E5FF',
        hovertemplate='Sub-carreira: %{x}<br>Salário: R$ %{y:,.2f}<extra></extra>',
        legend='legend'
    ),
    row=1, col=2
)

df_devops = df_subcarreiras[df_subcarreiras['Area'] == 'DevOps']
fig.add_trace(
    go.Bar(
        name='DevOps',
        x=df_devops['Sub-carreira'],
        y=df_devops['Salario'],
        marker_color='#B388FF',
        hovertemplate='Sub-carreira: %{x}<br>Salário: R$ %{y:,.2f}<extra></extra>',
        legend='legend'
    ),
    row=1, col=2
)

fig.update_yaxes(tickprefix="R$ ", row=1, col=2)

# Linha 2, Coluna 1 - Gráfico de Rosca (Sentimentos)
color_map = {'Positivo': '#00CC96', 'Negativo': '#EF553B', 'Neutro': '#636EFA'}
pie_colors = [color_map.get(sent, '#888888') for sent in contagem_sentimentos['Sentimento']]

fig.add_trace(
    go.Pie(
        labels=contagem_sentimentos['Sentimento'], 
        values=contagem_sentimentos['Contagem'], 
        hole=0.45, 
        marker_colors=pie_colors,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=15),
        domain=dict(x=[0.38, 0.62]),
        showlegend=True,
        legend='legend2'
    ),
    row=2, col=1
)

fig.update_xaxes(visible=False, showline=False, row=2, col=1)
fig.update_yaxes(visible=False, showline=False, row=2, col=1)

# Linha 3, Coluna 1 - Tabela de RI
cell_colors = [
    ['#1f1f1f'] * len(df_vagas),
    ['#1f1f1f'] * len(df_vagas),
    ['#1f1f1f'] * len(df_vagas),
    ['#1f1f1f'] * len(df_vagas)
]

fig.add_trace(
    go.Table(
        header=dict(values=["Descrição da Vaga", "Booleano", "BM25", "Embeddings"],
                    fill_color='#2a2a2a',
                    align='left',
                    font=dict(color='white', size=14)),
        cells=dict(values=[df_vagas['titulo_vaga'].tolist(), 
                           df_vagas['score_booleano'].tolist(), 
                           df_vagas['score_bm25'].round(3).tolist(), 
                           df_vagas['score_embeddings'].round(3).tolist()],
                   fill_color=cell_colors,
                   align='left',
                   font=dict(color='white', size=14),
                   height=45)
    ),
    row=3, col=1
)

# Tema e Dimensões do Canvas
fig.update_layout(
    template='plotly_dark',
    height=1900,
    width=1200,
    barmode='group',
    margin=dict(l=10, r=10, t=60, b=10),
    title_text="Dashboard Analítico: Cibersegurança & DevOps",
    title_x=0.5,
    legend=dict(orientation='h', x=0.75, y=0.635, xanchor='center', yanchor='top', bgcolor='rgba(0,0,0,0)'),
    legend2=dict(orientation='h', x=0.50, y=0.27, xanchor='center', yanchor='bottom', bgcolor='rgba(0,0,0,0)')
)

# Exportação Estática
fig.write_image("dashboard_cybersec.png", width=1200, height=1900, scale=2)
