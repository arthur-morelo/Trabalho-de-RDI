readme_content = """# Dashboard Analítico & Relatório Técnico: Cibersegurança & DevOps

Este repositório contém a infraestrutura e os scripts necessários para o pipeline de análise de mercado e geração do relatório técnico final sobre as áreas de **Cibersegurança** e **DevOps**. O projeto integra processamento de dados em Python, visualizações avançadas com **Plotly**, e compilação automatizada de relatórios com **Quarto** (Markdown técnico estruturado para PDF).

---

## 1. Organização e Limpeza do Projeto (Arquivos Essenciais)

Para garantir a integridade do pipeline e evitar poluição no diretório de entrega, **mantenha estritamente apenas estes 3 arquivos essenciais**:

1. `analise_final_plotly.py`: Script Python responsável pela engenharia de dados, cálculo de similaridade semântica, classificação de sentimentos e exportação do painel visual estático.
2. `Trabalho_Final_Arthur.qmd`: Arquivo Quarto que estrutura o relatório acadêmico textualmente e renderiza os gráficos gerados.
3. `README.md`: Este guia completo de documentação e operação.

*Nota: Todos os outros arquivos temporários, rascunhos de imagens antigas (`dashboard_cybersec_old.png`) ou scripts experimentais das aulas anteriores (`Aula_15_RDI_Embeddings_Sentimentos.qmd`) podem ser movidos para uma pasta de backup ou deletados permanentemente.*

---

## 2. Pré-requisitos do Ambiente

Antes de executar os scripts, certifique-se de ter as ferramentas estruturais e as bibliotecas de dados devidamente instaladas em seu ambiente local.

### Ferramentas de Sistema
- **Python 3.10 ou superior**
- **Quarto CLI** (Engine de compilação markdown universal)
- **Distribuição LaTeX** (como TeX Live, MiKTeX ou TinyTeX) para permitir que o Quarto converta o código markdown diretamente para o formato PDF compilado.

### Bibliotecas Python
Instale as dependências via terminal executando o comando abaixo: