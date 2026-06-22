import pandas as pd
import os

print("--- Iniciando Consolidação: State of Data Brazil (2021-2025) ---")

# Simulando a consolidação das bases anuais do State of Data
dados_consolidados = pd.DataFrame({
    'Ano': [2021, 2022, 2023, 2024, 2025],
    'Cargo': ['Engenheiro DevOps', 'Analista de SOC', 'Arquiteto de Segurança', 'Cloud Engineer', 'Analista de Segurança'],
    'Salario_Medio': [8500, 6000, 18000, 14000, 7500],
    'Regiao': ['Sudeste', 'Sul', 'Sudeste', 'Centro-Oeste', 'Nordeste'],
    'Sentimento_Desafios': ['Positivo', 'Negativo', 'Positivo', 'Positivo', 'Neutro']
})

# Garante que a pasta data existe no nível anterior
os.makedirs('../data', exist_ok=True)

# Salva o arquivo consolidado
caminho_saida = '../data/dados_state_of_data_consolidados.parquet'
dados_consolidados.to_parquet(caminho_saida, index=False)

print(f"Sucesso! Base consolidada salva em: {caminho_saida}")