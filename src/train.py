"""Обучение модели"""
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import pickle

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    train_params = params['training']
    
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("face_blurring_ml")
    
    with mlflow.start_run(run_name="random_forest_baseline"):
        # Генерация данных для демонстрации
        X, y = make_regression(n_features=4, n_informative=2, 
                               random_state=params['preprocessing']['random_state'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['preprocessing']['test_size'], 
            random_state=params['preprocessing']['random_state']
        )
        
        # Обучение модели
        model_params = {k: v for k, v in train_params.items() if k != 'model'}
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        
        # Оценка
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Логирование
        mlflow.log_params(model_params)
        mlflow.log_metrics({"mse": mse})
        mlflow.sklearn.log_model(model, "model")
        
        # Сохранение модели
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        print(f"Модель обучена: MSE = {mse:.4f}")

if __name__ == "__main__":
    main()
