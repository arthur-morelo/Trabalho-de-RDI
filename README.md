# Projeto Final de RDI: Dashboard Analítico e Relatório Técnico (Cibersegurança & DevOps)

Este repositório contém a infraestrutura e os scripts necessários para o pipeline de análise de mercado e geração do relatório técnico final sobre a área de **Cibersegurança**. O projeto integra extração de dados, processamento avançado de NLP (Modelos de Recuperação de Informação), visualizações com **Plotly**, e compilação automatizada de relatórios acadêmicos dinâmicos com **Quarto** e **Jupyter**.

---

## 📂 Arquitetura do Projeto e Arquivos Utilizados

A estrutura do repositório está organizada da seguinte forma, cobrindo desde a extração até a geração do PDF:

### 1. Dados (Raw e Processados)
- `data/dados_state_of_data_consolidados.parquet`: Base consolidada da pesquisa State of Data Brazil.
- `data/vagas_infra_cyber.parquet`: Corpus textual de vagas internacionais coletadas via scraping.

### 2. Scripts (Python)
- `src/coleta/consolida_state_of_data.py`: Script responsável por consolidar e limpar os microdados históricos.
- `src/coleta/coleta_externa.py`: Script para coleta programática (Web Scraping) de vagas no exterior.
- `src/visualizacao/analise_final_plotly.py`: Script que processa os dados finais, avalia modelos de RI e exporta os gráficos individuais.

### 3. Artefatos Visuais (Imagens Geradas)
- `output/1_vagas_paises.png`
- `output/2_salarios_usd.png`
- `output/3_tecnologias.png`
- `output/4_sentimentos.png`
- `output/5_tabela_ri.png`

### 4. Documentação e Relatório Final
- `docs/Trabalho_Final_Arthur.qmd`: Arquivo Quarto Markdown contendo a análise teórica e as células executáveis Python para importar as imagens.
- `docs/Trabalho_Final_Arthur.pdf`: O documento final gerado após a compilação acadêmica.

### 5. Configuração
- `requirements.txt`: Dependências e bibliotecas necessárias para rodar o pipeline.

---

## 🛠 Pré-requisitos do Sistema

Antes de iniciar, certifique-se de que os softwares abaixo estão instalados:
- **Python 3.10+**
- **Quarto CLI** (Para interpretar a sintaxe científica `.qmd`).
- **Distribuição LaTeX** (como TeX Live ou TinyTeX) para compilação em PDF.
- **Jupyter** (`pip install jupyter`) para execução das células dinâmicas do Quarto.

---

## 🚀 Como Reproduzir o Projeto (Passo a Passo)

Siga rigorosamente a sequência de execução abaixo na raiz do projeto para obter exatamente o mesmo resultado (`Trabalho_Final_Arthur.pdf`):

### Passo 1: Preparar o Ambiente Virtual e Instalar Dependências

Crie um ambiente virtual isolado para não gerar conflito de bibliotecas:

```bash
# 1. Criação do ambiente virtual
python -m venv .venv

# 2. Ativação do ambiente (Linux/Mac)
source .venv/bin/activate
# No Windows: .venv\Scripts\activate

# 3. Instalação das bibliotecas (pandas, plotly, jupyter, etc.)
pip install -r requirements.txt
```

### Passo 2: Executar as Etapas de Coleta de Dados

Execute os scripts que vão gerar os arquivos `.parquet` na pasta `data/`.

```bash
# Executa a consolidação histórica dos dados
python src/coleta/consolida_state_of_data.py

# Dispara a coleta de vagas internacionais
python src/coleta/coleta_externa.py
```
> **Verificação:** Confira se a pasta `data/` foi populada com os arquivos `.parquet`.

### Passo 3: Gerar os Gráficos Analíticos com Plotly

Execute o script de visualização que aplica os modelos de recuperação de informação e plota os gráficos na paleta de cores clara (`plotly_white`).

```bash
python src/visualizacao/analise_final_plotly.py
```
> **Verificação:** Certifique-se de que 5 imagens (`.png`) foram geradas com sucesso dentro da pasta `output/`.

### Passo 4: Renderização Final do Relatório em PDF (Quarto + Jupyter)

O arquivo `.qmd` agora utiliza a *engine* do Jupyter para executar células de código Python embutidas, formatando o relatório acadêmico de alta qualidade.

```bash
# Entre na pasta da documentação
cd docs

# Execute a renderização do documento utilizando o Quarto
quarto render Trabalho_Final_Arthur.qmd --to pdf
```
> **Conclusão:** O Quarto interpretará as instruções de layout (cores personalizadas, ausência de sumário, margens) e rodará os blocos Python que inserem os 5 gráficos gerados. O arquivo final **`Trabalho_Final_Arthur.pdf`** estará disponível na pasta `docs/`!