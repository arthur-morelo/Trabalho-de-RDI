import plotly.graph_objects as go
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('../../output', exist_ok=True)

# 1. Gráfico 1: Vagas por País
y_paises = ['EUA', 'Reino Unido', 'Canadá', 'Alemanha', 'Brasil']
x_paises = [3500, 2100, 1800, 1200, 850]

fig1 = go.Figure(go.Bar(
    y=y_paises[::-1],
    x=x_paises[::-1],
    orientation='h',
    marker_color='#FF5722'
))
fig1.update_layout(
    title='Volume de Vagas por País',
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#111827')
)
fig1.write_image("../../output/1_vagas_paises.png", width=1200, height=500, scale=2)

# 2. Gráfico 2: Média Salarial (USD)
y_cargos = ['Arquiteto de Segurança', 'Engenheiro de Cloud Sec', 'Especialista em Pentest', 'Analista de GRC', 'Analista de SOC']
x_salarios = [150000, 145000, 120000, 95000, 85000]

fig2 = go.Figure(go.Bar(
    y=y_cargos[::-1],
    x=x_salarios[::-1],
    orientation='h',
    marker_color='#00838F'
))
fig2.update_layout(
    title='Média Salarial Anual (USD)',
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#111827')
)
fig2.update_xaxes(tickprefix="$")
fig2.write_image("../../output/2_salarios_usd.png", width=1200, height=500, scale=2)

# 3. Gráfico 3: Tecnologias Mais Demandadas
y_tech = ['AWS', 'Linux', 'Python', 'Docker', 'Kubernetes', 'SIEM', 'Wireshark', 'Metasploit']
x_tech = [85, 82, 78, 70, 65, 55, 40, 35]

fig3 = go.Figure(go.Bar(
    y=y_tech[::-1],
    x=x_tech[::-1],
    orientation='h',
    marker_color='#2E7D32'
))
fig3.update_layout(
    title='Tecnologias Mais Demandadas',
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#111827')
)
fig3.write_image("../../output/3_tecnologias.png", width=1200, height=500, scale=2)

# 4. Gráfico 4: Sentimento dos Profissionais
labels_sent = ['Positivo', 'Neutro', 'Negativo']
values_sent = [45, 30, 25]

fig4 = go.Figure(go.Pie(
    labels=labels_sent,
    values=values_sent,
    hole=0.45,
    textposition='outside',
    marker=dict(colors=['#2E7D32', '#FFC107', '#FF5722'])
))
fig4.update_layout(
    title='Sentimento de Desafios Profissionais',
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#111827')
)
fig4.write_image("../../output/4_sentimentos.png", width=1200, height=500, scale=2)

# 5. Tabela 5: Scores de RI
cols_tabela = ['Descrição da Vaga', 'Booleano', 'BM25', 'Embeddings']
valores_tabela = [
    ['Engenharia de Detecção e Resposta (SOC)', 'Especialista em Testes de Intrusão Web', 'Arquiteto de Soluções Zero Trust AWS', 'Analista de Governança e Compliance PCI', 'Pesquisador de Ameaças e Malware'],
    ['0', '1', '0', '0', '0'],
    ['0.00', '1.45', '0.00', '0.00', '0.00'],
    ['0.82', '0.91', '0.78', '0.65', '0.88']
]

row_colors = ['#FFFFFF', '#F9FAFB'] * 3
row_colors = row_colors[:len(valores_tabela[0])]

fig5 = go.Figure(data=[go.Table(
    header=dict(
        values=cols_tabela,
        fill_color='#374151',
        font=dict(color='white', size=14),
        align='left'
    ),
    cells=dict(
        values=valores_tabela,
        fill_color=[row_colors] * len(cols_tabela),
        font=dict(color='#111827', size=12),
        align='left'
    )
)])
fig5.update_layout(
    title='Scores de Recuperação da Informação (RI)',
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#111827')
)
fig5.write_image("../../output/5_tabela_ri.png", width=1200, height=500, scale=2)
