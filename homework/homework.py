import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline as build_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

def cargar_datos(nombre_archivo):
    ruta = Path("files/input")
    return pd.read_csv(ruta / nombre_archivo, compression="zip")

def depurar_datos(data: pd.DataFrame):
    data = data.copy()
    data = data.rename(columns={"default payment next month": "default"})
    data = data.drop(columns="ID")
    data = data.dropna()
    data = data[data["MARRIAGE"] != 0]
    data = data[data["EDUCATION"] != 0]

    data["EDUCATION"] = data["EDUCATION"].map(lambda x: 4 if x > 4 else x)
    return data

def separar_variables(data: pd.DataFrame):
    X = data.drop(columns="default").copy()
    y = data["default"].copy()
    return X, y

def dividir_datos(data: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    X, y = separar_variables(data)
    return train_test_split(X, y, random_state=0)

def construir_pipeline():
    transformador = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ['SEX', 'EDUCATION', 'MARRIAGE'])],
        remainder=StandardScaler()
    )

    modelo = build_pipeline(
        transformador,
        PCA(),
        SelectKBest(k=12),
        SVC(gamma=0.1)
    )

    return modelo

def definir_grid_search(modelo, parametros, cv=10):
    return GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

def guardar_modelo(modelo):
    import pickle
    import gzip

    directorio = Path("files/models")
    directorio.mkdir(exist_ok=True)

    with gzip.open(directorio / "model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo, archivo)

def guardar_metricas(lista_metricas):
    import json

    ruta_salida = Path("files/output")
    ruta_salida.mkdir(exist_ok=True)

    with open(ruta_salida / "metrics.json", "w") as archivo:
        archivo.writelines([json.dumps(m) + "\n" for m in lista_metricas])

def entrenar_modelo(X_train, y_train):
    modelo = construir_pipeline()
    grid = definir_grid_search(modelo, {
        "pca__n_components": [20, 21],
    })

    grid.fit(X_train, y_train)
    guardar_modelo(grid)
    return grid

def calcular_metricas(y_real, y_estimado):
    from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

    return (
        precision_score(y_real, y_estimado),
        balanced_accuracy_score(y_real, y_estimado),
        recall_score(y_real, y_estimado),
        f1_score(y_real, y_estimado)
    )

def matriz_confusion(nombre_set, y_real, y_estimado):
    from sklearn.metrics import confusion_matrix

    matriz = confusion_matrix(y_real, y_estimado)
    return {
        "type": "cm_matrix",
        "dataset": nombre_set,
        "true_0": {
            "predicted_0": int(matriz[0][0]),
            "predicted_1": int(matriz[0][1])
        },
        "true_1": {
            "predicted_0": int(matriz[1][0]),
            "predicted_1": int(matriz[1][1])
        }
    }

def ejecutar_proceso():
    datos_entrenamiento = depurar_datos(cargar_datos("train_data.csv.zip"))
    datos_prueba = depurar_datos(cargar_datos("test_data.csv.zip"))

    X_train, y_train = separar_variables(datos_entrenamiento)
    X_test, y_test = separar_variables(datos_prueba)

    modelo = entrenar_modelo(X_train, y_train)

    resultados = []
    for nombre, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        predicciones = modelo.predict(X)
        precision, bal_acc, sensibilidad, f1 = calcular_metricas(y, predicciones)
        resultados.append({
            "type": "metrics",
            "dataset": nombre,
            "precision": precision,
            "balanced_accuracy": bal_acc,
            "recall": sensibilidad,
            "f1_score": f1
        })

    matrices = [
        matriz_confusion(nombre, y, modelo.predict(X))
        for nombre, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]
    ]

    guardar_metricas(resultados + matrices)

ejecutar_proceso()