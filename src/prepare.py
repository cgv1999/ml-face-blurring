"""Подготовка данных для ML-пайплайна"""
import pandas as pd
import yaml
import sys

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Загрузка данных
    df = pd.read_csv("data/raw/iris.csv")
    df.dropna(inplace=True)
    
    # Сохранение обработанных данных
    df.to_csv("data/processed/processed_data.csv", index=False)
    print(f"Данные подготовлены: {len(df)} записей")

if __name__ == "__main__":
    main()
