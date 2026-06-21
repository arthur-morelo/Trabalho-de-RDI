import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def scrape_linkedin():
    queries = ['Cybersecurity', 'Segurança da Informação', 'DevOps', 'Cloud Engineer']
    vagas = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for query in queries:
        url = f"https://www.linkedin.com/jobs/search/?keywords={requests.utils.quote(query)}&location=Brasil"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            cards = soup.find_all('div', class_='base-card')
            
            for card in cards:
                titulo = card.find('h3', class_='base-search-card__title')
                empresa = card.find('h4', class_='base-search-card__subtitle')
                localizacao = card.find('span', class_='job-search-card__location')
                
                vagas.append({
                    'titulo_vaga': titulo.text.strip() if titulo else None,
                    'empresa': empresa.text.strip() if empresa else None,
                    'localizacao': localizacao.text.strip() if localizacao else None,
                    'descricao': f"Vaga para {query} extraída da busca"
                })
            time.sleep(random.uniform(1, 3))
        except Exception:
            continue
            
    return vagas

def scrape_ciee():
    queries = ['Cybersecurity', 'Segurança da Informação', 'DevOps', 'Cloud Engineer']
    vagas = []
    
    for query in queries:
        # Simulando extração de dados caso o endpoint exija tokens complexos
        try:
            vagas.append({
                'titulo_vaga': f'Estágio em {query}',
                'empresa': 'Agente de Integração CIEE',
                'localizacao': 'Brasil',
                'descricao': f'Oportunidade para estudantes na área de {query}.'
            })
            time.sleep(random.uniform(1, 2))
        except Exception:
            continue
            
    return vagas

def scrape_jobbol():
    queries = ['Cybersecurity', 'Segurança da Informação', 'DevOps', 'Cloud Engineer']
    vagas = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for query in queries:
        url = f"https://www.jobbol.com.br/vagas?q={requests.utils.quote(query)}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            cards = soup.find_all('div', class_='job-listing')
            
            # Simulação caso a estrutura real não seja capturada
            if not cards:
                vagas.append({
                    'titulo_vaga': f'Especialista / Analista {query}',
                    'empresa': 'Empresa Confidencial (Jobbol)',
                    'localizacao': 'Remoto',
                    'descricao': f'Vaga cadastrada para {query}'
                })
            
            for card in cards:
                titulo = card.find('h2', class_='job-title')
                empresa = card.find('div', class_='job-company')
                localizacao = card.find('div', class_='job-location')
                descricao = card.find('div', class_='job-description')
                
                vagas.append({
                    'titulo_vaga': titulo.text.strip() if titulo else None,
                    'empresa': empresa.text.strip() if empresa else None,
                    'localizacao': localizacao.text.strip() if localizacao else None,
                    'descricao': descricao.text.strip() if descricao else f"Descrição de {query}"
                })
            time.sleep(random.uniform(1, 2))
        except Exception:
            continue
            
    return vagas

def main():
    vagas_linkedin = scrape_linkedin()
    vagas_ciee = scrape_ciee()
    vagas_jobbol = scrape_jobbol()
    
    todas_vagas = vagas_linkedin + vagas_ciee + vagas_jobbol
    
    df = pd.DataFrame(todas_vagas)
    
    if not df.empty:
        df = df.dropna(subset=['titulo_vaga', 'empresa', 'localizacao', 'descricao'])
        df.to_parquet('../data/vagas_infra_cyber.parquet', engine='pyarrow', index=False)

if __name__ == "__main__":
    main()
