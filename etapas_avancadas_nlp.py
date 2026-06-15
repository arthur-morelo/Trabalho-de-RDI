import pandas as pd
import numpy as np
import warnings
from typing import List

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configurações Iniciais
warnings.filterwarnings('ignore')

# Baixar recursos do NLTK (executa apenas na primeira vez)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True) # dependência para tokenização no nltk moderno
except:
    pass

# =====================================================================
# EXPLICAÇÃO TEÓRICA: Busca Lexical vs Busca Semântica
# =====================================================================
"""
Busca Lexical (Ex: Booleana, BM25):
- Baseia-se no "casamento exato" das palavras (keyword matching).
- Procura documentos que contenham as exatas raízes/palavras usadas na query.
- Vantagem: Rápida, excelente para buscas por termos específicos, códigos de produto, siglas ou nomes próprios.
- Desvantagem: Sofre com o 'gap de vocabulário'. Se você buscar "carro", 
  ela não encontra um texto que utilize a palavra "automóvel". 
  * O BM25 é o estado-da-arte na busca lexical pois adiciona pesos baseados na frequência 
    do termo e no tamanho do documento, mas ainda é restrito ao vocabulário exato.

Busca Semântica (Ex: Embeddings + Cosine Similarity):
- Baseia-se no "significado" (contexto) das frases.
- Transforma frases inteiras em vetores numéricos densos (embeddings). Textos com 
  significados parecidos ficam matematicamente próximos no espaço multidimensional.
- Vantagem: Lida perfeitamente com sinônimos e contexto. Se buscar "carro", o modelo 
  entende a intenção e mapeia que "automóvel" ou "veículo" são coisas altamente similares.
- Desvantagem: Mais custosa computacionalmente. Pode trazer resultados conceitualmente 
  "relacionados" quando o usuário queria estritamente uma palavra-chave técnica (ex: 'Python 3.8').
"""

# =====================================================================
# 1. FUNÇÃO DE LIMPEZA DE TEXTO (Remoção de Stopwords)
# =====================================================================
def limpar_texto(texto: str) -> str:
    """
    Limpa o texto convertendo para minúsculas, tokenizando e removendo 
    stopwords e pontuações do idioma português.
    """
    if not isinstance(texto, str):
        return ""
        
    texto = texto.lower()
    
    # Tokenização (dividir a string de texto em uma lista de palavras/tokens)
    tokens = word_tokenize(texto, language='portuguese')
    
    # Lista de stopwords em PT-BR (palavras comuns sem peso semântico, ex: 'o', 'a', 'de', 'que')
    stop_words_pt = set(stopwords.words('portuguese'))
    
    # Filtra mantendo apenas as palavras que NÃO são stopwords e que sejam alfanuméricas (remove vírgulas, pontos)
    tokens_limpos = [word for word in tokens if word not in stop_words_pt and word.isalnum()]
    
    return " ".join(tokens_limpos)

# =====================================================================
# 2. FUNÇÃO DE BUSCA E RANQUEAMENTO (Booleana, BM25, Embeddings)
# =====================================================================
def aplicar_modelos_busca(query: str, df: pd.DataFrame, coluna_texto: str = 'descricao') -> None:
    """
    Aplica 3 modelos diferentes de busca, compara os resultados e exibe o top 3 de cada um.
    """
    print(f"\n{'='*60}\nQUERY DE BUSCA: '{query}'\n{'='*60}")
    
    documentos = df[coluna_texto].fillna("").tolist()
    
    # Para as buscas Lexicais, devemos aplicar a mesma limpeza tanto na Query quanto nos Documentos
    query_limpa = limpar_texto(query)
    query_tokens = query_limpa.split()
    
    docs_limpos = [limpar_texto(doc) for doc in documentos]
    docs_tokens = [doc.split() for doc in docs_limpos]
    
    # ---------------------------------------------------------
    # MODELO 1: Busca Booleana Simples (AND)
    # ---------------------------------------------------------
    print("\n--- 1. BUSCA BOOLEANA (Lexical Estrita) ---")
    # A busca booleana exige que *todas* as palavras da query estejam no documento
    resultados_booleano = []
    for i, tokens in enumerate(docs_tokens):
        # A função 'all' verifica se cada token da query limpa existe dentro dos tokens do documento
        if all(token in tokens for token in query_tokens):
            resultados_booleano.append((i, documentos[i]))
    
    if resultados_booleano:
        for idx, doc in resultados_booleano[:3]: # Limitando ao Top 3
            print(f"[Doc {idx}]: {doc[:150]}...")
    else:
        print("-> Nenhum documento atendeu aos critérios exatos da Busca Booleana.")

    # ---------------------------------------------------------
    # MODELO 2: Busca BM25
    # ---------------------------------------------------------
    print("\n--- 2. BUSCA BM25 (Lexical com Ranqueamento) ---")
    bm25 = BM25Okapi(docs_tokens)
    scores_bm25 = bm25.get_scores(query_tokens)
    
    # Extrai os índices que possuem os maiores scores (ordem decrescente)
    top_n_bm25 = np.argsort(scores_bm25)[::-1][:3]
    
    for idx in top_n_bm25:
        score = scores_bm25[idx]
        if score > 0:
            print(f"[Score: {score:.4f} | Doc {idx}]: {documentos[idx][:150]}...")
        else:
            print("-> Sem mais resultados relevantes no BM25.")
            break

    # ---------------------------------------------------------
    # MODELO 3: Embeddings + Cosine Similarity
    # ---------------------------------------------------------
    print("\n--- 3. BUSCA POR EMBEDDINGS (Semântica) ---")
    # Para embeddings, usamos os textos *originais* (sem remover stopwords), pois 
    # o modelo captura o contexto pelas preposições e estrutura original da frase.
    modelo_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    
    # Transformando a query em vetor
    vetor_query = modelo_embed.encode([query])
    # Transformando os documentos em vetores
    vetores_docs = modelo_embed.encode(documentos, show_progress_bar=False)
    
    # Calculando a Similaridade de Cosseno (retorna valores de 0 a 1, onde 1 é idêntico)
    similaridades = cosine_similarity(vetor_query, vetores_docs)[0]
    
    # Extrai os índices com as maiores similaridades
    top_n_embed = np.argsort(similaridades)[::-1][:3]
    
    for idx in top_n_embed:
        sim = similaridades[idx]
        print(f"[Similaridade: {sim:.4f} | Doc {idx}]: {documentos[idx][:150]}...")

# =====================================================================
# 3. FUNÇÃO DE ZERO-SHOT CLASSIFICATION
# =====================================================================
def classificar_textos_zero_shot(textos: List[str], categorias: List[str]) -> None:
    """
    Usa a pipeline de Zero-Shot do Hugging Face para classificar textos 
    em um conjunto livre de categorias, sem necessidade de treinamento prévio.
    """
    print(f"\n{'='*60}\nCLASSIFICAÇÃO ZERO-SHOT\n{'='*60}")
    print(f"Categorias Candidatas: {categorias}\n")
    
    # Usaremos um modelo muito otimizado para lidar com Zero-Shot Multi-idioma na CPU
    nome_modelo_zs = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    
    print(f"Carregando classificador zero-shot na CPU...")
    # device=-1 obriga a execução nativa na CPU
    classificador = pipeline("zero-shot-classification", model=nome_modelo_zs, device=-1)
    
    resultados = classificador(textos, candidate_labels=categorias)
    
    # Exibindo resultados
    for res in resultados:
        texto_original = res['sequence']
        melhor_label = res['labels'][0]
        melhor_score = res['scores'][0]
        
        print(f"Texto: '{texto_original}'")
        print(f"-> Predição: [{melhor_label}] (Confiança: {melhor_score:.2%})\n")

# =====================================================================
# BLOCO DE TESTES (Exemplo de Execução)
# =====================================================================
if __name__ == "__main__":
    # --- DADOS MOCK (Simulação) ---
    dados_mock = [
        "Vaga para Cientista de Dados. Necessário conhecimentos avançados em Python, Machine Learning e análise preditiva.",
        "Desenvolvedor Java Backend com forte experiência em Spring Boot, APIs e microsserviços.",
        "Analista de Marketing Digital para gerenciar campanhas de SEO, tráfego pago e redes sociais.",
        "Procuramos Especialista em NLP e Deep Learning. Foco total em Sentence Transformers, RAG e Grandes Modelos de Linguagem (LLMs).",
        "Oportunidade para Engenheiro de Dados Pleno. Construção e otimização de pipelines de dados utilizando PySpark, Airflow e AWS."
    ]
    df_teste = pd.DataFrame({'descricao': dados_mock})
    
    # 1. Demonstração de Limpeza
    print(f"{'='*60}\nTESTE DE LIMPEZA DE TEXTO (REMOÇÃO DE STOPWORDS)\n{'='*60}")
    print(f"Original: {dados_mock[0]}")
    print(f"Limpo...: {limpar_texto(dados_mock[0])}")
    
    # 2. Demonstração dos 3 Modelos de Busca
    query_teste = "Cientista de Dados com Python"
    aplicar_modelos_busca(query_teste, df_teste)
    
    # 3. Demonstração Zero-Shot
    textos_curtos = [
        "Estou extremamente satisfeito com a performance e os resultados do novo algoritmo de busca semântica do sistema!",
        "A latência da API subiu para 5000ms e causou timeout no banco. O servidor de produção caiu completamente.",
        "O relatório com a análise mensal de métricas RDI será enviado para aprovação na sexta-feira de manhã."
    ]
    categorias_alvo = ["Positivo", "Neutro", "Negativo"]
    
    classificar_textos_zero_shot(textos_curtos, categorias_alvo)
