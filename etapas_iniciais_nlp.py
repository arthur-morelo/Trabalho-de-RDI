import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import warnings

# Suprimir avisos não críticos para manter a saída limpa no Quarto
warnings.filterwarnings('ignore')

# =====================================================================
# ETAPA 1: Importação da Base
# =====================================================================
def carregar_dados(caminho_arquivo: str = 'dados_state_of_data_consolidados.parquet') -> pd.DataFrame:
    """
    Carrega o arquivo parquet e exibe informações básicas usando pandas/pyarrow.
    """
    print(f"--- Etapa 1: Carregando arquivo {caminho_arquivo} ---")
    try:
        # engine='pyarrow' é o padrão, mas deixamos explícito
        df = pd.read_parquet(caminho_arquivo, engine='pyarrow')
        
        print("Dados carregados com sucesso!")
        print(f"Número de linhas: {df.shape[0]}")
        print(f"Número de colunas: {df.shape[1]}")
        
        print("\nInformações das colunas:")
        df.info()
        
        print("\nPrimeiras linhas (Amostra):")
        print(df.head()) # 'print' funciona em qualquer ambiente Python
        
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return pd.DataFrame()

# =====================================================================
# ETAPA 2: Web Scraping Estruturado
# =====================================================================
def raspar_vagas(url: str) -> pd.DataFrame:
    """
    Realiza web scraping genérico de títulos e descrições de vagas.
    """
    print(f"\n--- Etapa 2: Iniciando Web Scraping ---")
    print(f"URL alvo: {url}")
    
    # Header com User-Agent para simular um navegador e evitar bloqueios HTTP (ex: erro 403)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Lança exceção se o status code não for 200 (OK)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ===================================================================
        # LÓGICA GENÉRICA DE EXTRAÇÃO
        # IMPORTANTE: Você precisará inspecionar a página alvo e alterar 
        # os seletores CSS abaixo ('h2.titulo-vaga', 'div.descricao-vaga')
        # ===================================================================
        titulos_html = soup.find_all('h2', class_='titulo-vaga') 
        descricoes_html = soup.find_all('div', class_='descricao-vaga')
        
        vagas = []
        # Se encontrou dados na página (mesmo tamanho para título e descrição)
        if titulos_html and descricoes_html and len(titulos_html) == len(descricoes_html):
            for t, d in zip(titulos_html, descricoes_html):
                vagas.append({
                    'titulo': t.get_text(strip=True),
                    'descricao': d.get_text(strip=True)
                })
        else:
            print("Aviso: Seletores HTML não encontraram dados ou tamanhos não batem. Usando dados simulados para teste.")
            # Dados de simulação caso a página não retorne o formato esperado (placeholder)
            vagas = [
                {'titulo': 'Engenheiro de Dados Especialista', 'descricao': 'Buscamos engenheiro com foco em pipelines de NLP, Python, PyArrow e processamento de grandes volumes.'},
                {'titulo': 'Cientista de Dados - NLP', 'descricao': 'Desenvolvimento de sistemas de recomendação e busca semântica usando Sentence Transformers e similaridade de cosseno.'},
                {'titulo': 'Analista de Dados', 'descricao': 'Criação de dashboards gerenciais, análise exploratória e extração de dados web.'}
            ]
        
        df_vagas = pd.DataFrame(vagas)
        print(f"Foram extraídas {len(df_vagas)} vagas.")
        print(df_vagas.head())
        
        return df_vagas
        
    except requests.exceptions.RequestException as e:
        print(f"Falha na requisição HTTP: {e}")
        return pd.DataFrame()

# =====================================================================
# ETAPA 3: Preparação para Embeddings (Sentence Transformers via CPU)
# =====================================================================
def gerar_embeddings(textos: list):
    """
    Instancia o modelo leve de Sentence Transformers e gera embeddings.
    """
    print("\n--- Etapa 3: Gerando Embeddings (CPU) ---")
    
    # SUGESTÃO DE MODELO LEVE E OTIMIZADO PARA PORTUGUÊS/MULTILÍNGUE
    # O modelo 'paraphrase-multilingual-MiniLM-L12-v2' é excelente para processamento
    # apenas em CPU: ele é compacto (~470MB), rápido e suporta mais de 50 idiomas, incluindo PT-BR.
    nome_modelo = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    print(f"Instanciando o modelo: {nome_modelo}")
    # O parâmetro device='cpu' garante que nenhuma GPU seja exigida
    modelo = SentenceTransformer(nome_modelo, device='cpu')
    
    print(f"Gerando embeddings para {len(textos)} documento(s)...")
    # O método encode transforma os textos em vetores densos (embeddings).
    # O retorno já é um numpy.ndarray, pronto para cálculos matemáticos (ex: sklearn cosine_similarity)
    matriz_embeddings = modelo.encode(textos, show_progress_bar=True)
    
    print(f"Processamento concluído!")
    print(f"Formato (shape) da matriz gerada: {matriz_embeddings.shape}")
    print("-> A matriz de embeddings está pronta para o cálculo de similaridade de cosseno.")
    
    return modelo, matriz_embeddings

# =====================================================================
# BLOCO DE EXECUÇÃO PRINCIPAL
# Pode ser copiado e rodado diretamente em uma célula (chunk) do Quarto
# =====================================================================
if __name__ == "__main__":
    # 1. Carregando a base de dados
    df_state = carregar_dados('dados_state_of_data_consolidados.parquet')
    
    # 2. Raspando vagas (usando URL de exemplo)
    url_placeholder = "https://vagas.com.br/vagas-de-dados" # Altere para a URL real
    df_scraping = raspar_vagas(url_placeholder)
    
    # 3. Gerando Embeddings
    if not df_scraping.empty:
        # Extraindo as descrições como lista para passar ao modelo
        lista_descricoes = df_scraping['descricao'].tolist()
        
        # Gerando os embeddings
        modelo_nlp, matriz_vetores = gerar_embeddings(lista_descricoes)
