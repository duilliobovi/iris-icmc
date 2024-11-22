"""
=========================
Developer: Duíllio Campos Bovi
Instituição de Ensino: UNIVESP (Bacharelado em Ciência de Dados)
Curso: Python para Ciência de Dados (ICMC/USP)
Sistema Operacional (Operational System): Debian Bookworm 12.8 (Linux)
Python (Versão): 3.11.2
Ambiente Virtual (Virtual Environment): Venv
Pacotes instalados via pip (pip-installed packages) - requerimentos/requirements:
    - pandas
    - scikit-learn
    - colorama
    - tabulate
=========================
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union
import os
import time
from colorama import Fore, Style, init
from tabulate import tabulate

# Inicializa o colorama para habilitar cores no terminal
init(autoreset=True)


class Modelo:
    def __init__(self):
        """
        Inicializa os atributos necessários para o funcionamento da classe.
        """
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.model: Optional[Union[SVC, LinearRegression]] = None

    def carregar_dataset(self, path: str) -> None:
        """
        Carrega o dataset Iris a partir de um arquivo CSV.

        :param path: Caminho para o arquivo CSV contendo o dataset.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{Fore.RED}Erro: Arquivo não encontrado no caminho '{path}'. Verifique o caminho fornecido.")

        try:
            names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
            self.df = pd.read_csv(path, names=names)
            print(f"{Fore.GREEN}Dataset carregado com sucesso de: {path}")
        except Exception as e:
            raise RuntimeError(f"{Fore.RED}Erro ao carregar o dataset: {e}")

    def tratamento_de_dados(self) -> None:
        """
        Realiza o pré-processamento dos dados, incluindo o tratamento de valores
        ausentes e codificação das classes.
        """
        if self.df is None:
            raise ValueError(f"{Fore.RED}Erro: O dataset não foi carregado. Use 'carregar_dataset' antes.")

        # Remover possíveis valores ausentes
        self.df.dropna(inplace=True)

        # Codificar a coluna de espécies (classe alvo)
        le = LabelEncoder()
        self.df['Species'] = le.fit_transform(self.df['Species'])

        # Separar as features (X) e o target (y)
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species']

        # Dividir em conjuntos de treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    def exibir_tabela_svm(self, relatorio_classificacao: dict) -> None:
        """
        Exibe o relatório de classificação do SVM formatado em uma tabela ASCII.
        """
        headers = ["Classe", "Precision", "Recall", "F1-Score", "Support"]
        rows = []

        for classe, valores in relatorio_classificacao.items():
            if isinstance(valores, dict):
                rows.append([
                    classe,
                    f"{valores['precision']:.2f}",
                    f"{valores['recall']:.2f}",
                    f"{valores['f1-score']:.2f}",
                    int(valores['support']),
                ])

        # Exibir a tabela no terminal
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def executar_pipeline(self, path: str) -> None:
        """
        Executa o pipeline completo: carregar dados, treinamento do modelo e exibição dos resultados.
        """
        # Carregar dataset
        self.carregar_dataset(path)

        # Preprocessar dados
        self.tratamento_de_dados()

        # Treinamento e resultados para o modelo SVM
        print(f"{Fore.CYAN}Treinando o modelo SVM...")
        start_time = time.time()
        self.model = SVC(kernel="linear")
        self.model.fit(self.X_train, self.y_train)
        tempo_treinamento_svm = time.time() - start_time
        y_pred_svm = self.model.predict(self.X_test)
        acuracia_svm = accuracy_score(self.y_test, y_pred_svm)

        print(f"Treinamento do modelo SVM concluído em {tempo_treinamento_svm:.4f} segundos.")
        print(f"Acurácia do modelo SVM: {acuracia_svm:.2f}")
        relatorio_svm = classification_report(self.y_test, y_pred_svm, output_dict=True)
        self.exibir_tabela_svm(relatorio_svm)

        # Treinamento e resultados para o modelo Linear Regression
        print(f"{Fore.CYAN}Treinando o modelo LinearRegression...")
        start_time = time.time()
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        tempo_treinamento_lr = time.time() - start_time
        y_pred_lr = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred_lr)

        print(f"Treinamento do modelo LinearRegression concluído em {tempo_treinamento_lr:.4f} segundos.")
        print(f"Erro quadrático médio (MSE): {mse:.4f}")


# Exemplo de execução
modelo = Modelo()
dataset_path = '/home/local/iris.data'
modelo.executar_pipeline(dataset_path)
