import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Configuração Inicial para garantir a compatibilidade do JavaScript
pio.renderers.default = "notebook_connected"

# Criação do DataFrame com os dados geográficos de mercado
dados = {
    'UF': ['SP', 'PR', 'RJ', 'DF', 'MG'],
    'Volume_Vagas': [64, 6, 4, 2, 2],
    'Media_Salarial': [9448.41, 8406.50, 7171.59, 10656.71, 7382.81]
}

# Ordena de forma decrescente pelo volume de vagas
df_join = pd.DataFrame(dados).sort_values(by='Volume_Vagas', ascending=False)

# Criação do layout de subplots (1 linha, 2 colunas)
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Quantidade de Vagas", "Média Salarial (R$)"),
    horizontal_spacing=0.15
)

# Gráfico 1 (Esquerda): Gráfico de barras com a 'Quantidade de Vagas'
fig.add_trace(
    go.Bar(
        x=df_join['UF'],
        y=df_join['Volume_Vagas'],
        name="Vagas",
        marker_color='#2c3e50',
        hovertemplate="<b>UF:</b> %{x}<br><b>Vagas:</b> %{y}<extra></extra>"
    ),
    row=1, col=1
)

# Gráfico 2 (Direita): Gráfico de linhas com marcadores para a 'Média Salarial'
fig.add_trace(
    go.Scatter(
        x=df_join['UF'],
        y=df_join['Media_Salarial'],
        name="Média Salarial",
        mode='lines+markers',
        marker_color='#e74c3c',
        hovertemplate="<b>UF:</b> %{x}<br><b>Média:</b> R$ %{y:,.2f}<extra></extra>"
    ),
    row=1, col=2
)

# Estilização: Template e Formatação dos Eixos
fig.update_layout(
    template="plotly_white",
    title_text="Análise Comparativa Regional: Vagas vs. Compensação",
    showlegend=False,
    margin=dict(l=50, r=50, t=100, b=50)
)

fig.update_yaxes(title_text="Quantidade de Vagas", row=1, col=1)
# Configura o eixo Y do Gráfico 2 para formatar os valores como moeda (R$)
fig.update_yaxes(title_text="Salário Médio", tickprefix="R$ ", tickformat=",.2f", row=1, col=2)
fig.update_xaxes(title_text="Estado (UF)", row=1, col=1)
fig.update_xaxes(title_text="Estado (UF)", row=1, col=2)

# O Bypass de Exportação (MUITO IMPORTANTE)
# Exporta o painel diretamente para um ficheiro estático independente
fig.write_html("Dashboard_Oficial.html")

# Print final informando o sucesso da geração
print("Ficheiro 'Dashboard_Oficial.html' foi gerado com sucesso.")
