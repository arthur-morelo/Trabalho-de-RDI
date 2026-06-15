import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 1. Headers Robustos (Crucial para o LinkedIn)
# Simulando um navegador real para evitar o erro 429 (Too Many Requests)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 2. A URL do LinkedIn (Busca Pública) - geoId=106057199 força o filtro para o BRASIL
url = "https://www.linkedin.com/jobs/search/?keywords=Cientista%20de%20Dados&location=Brasil&geoId=106057199"

print(f"Iniciando coleta em: {url}")
response = requests.get(url, headers=headers)

# 3. Verificação de Status
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Encontra os cartões principais (LinkedIn pode usar div ou li)
    job_cards = soup.find_all(['div', 'li'], class_=lambda c: c and ('base-card' in c or 'job-search-card' in c or 'base-search-card' in c))
    
    if not job_cards:
        print("Aviso: Nenhum cartão de vaga encontrado. O LinkedIn pode ter retornado uma página de Login (AuthWall) ou mudou o layout.")
        
    dados_coletados = []
    
    # 4. Extração Estruturada Robusta
    for card in job_cards:
        try:
            # Busca os elementos dentro do próprio cartão
            title_elem = card.find(['h3', 'span'], class_=lambda c: c and 'title' in c)
            company_elem = card.find(['h4', 'a'], class_=lambda c: c and ('subtitle' in c or 'hidden-nested-link' in c))
            location_elem = card.find('span', class_=lambda c: c and 'location' in c)
            link_elem = card.find('a', class_=lambda c: c and 'link' in c)
            
            # Se encontrar o título, extrai os textos
            if title_elem:
                title = title_elem.get_text(strip=True)
                company = company_elem.get_text(strip=True) if company_elem else "Não Informada"
                location = location_elem.get_text(strip=True) if location_elem else "Não Informada"
                job_link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else "Sem Link"
                
                # Ignora vagas falsas ou mal formadas
                if title:
                    dados_coletados.append({
                        "titulo_vaga": title,
                        "empresa": company,
                        "localizacao": location,
                        "link_vaga": job_link
                    })
        except Exception as e:
            continue
            
    # 5. Transformação e Exportação
    df_vagas = pd.DataFrame(dados_coletados)
    print(f"Coleta concluída! {len(df_vagas)} vagas iniciais estruturadas.")
    
    arquivo_saida = "vagas_linkedin.parquet"
    df_vagas.to_parquet(arquivo_saida, engine='pyarrow', index=False)
    print(f"Arquivo salvo com sucesso: {arquivo_saida}")
    
else:
    print(f"Erro ao acessar a página. Status HTTP: {response.status_code}")
    print("Dica: Se o status for 429, o LinkedIn bloqueou o request temporariamente.")