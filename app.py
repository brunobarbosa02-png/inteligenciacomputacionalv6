# ============================================================
# ANÁLISE DE INTELIGÊNCIA COMPUTACIONAL
# Regressão, Classificação, Agrupamento (incl. Mapa de Kohonen)
# + Pré-análise com recomendações
# + Execução de TODOS os estimadores disponíveis
# + Ajuste manual universal de hiperparâmetros
# + Seção de explicações por algoritmo
# ============================================================

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError as exc:
    raise ImportError(
        "A dependência 'plotly' não está instalada. "
        "Instale com: pip install plotly"
    ) from exc

import streamlit as st

# ----- sklearn -----
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture    
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.discovery import all_estimators

# ----- UCI -----
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

# ----- Bibliotecas externas opcionais -----
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None

try:
    from minisom import MiniSom
except ImportError:
    MiniSom = None


# ============================================================
# CONSTANTES
# ============================================================

DEFAULT_RANDOM_STATE = 42

CURATED_CLASSIFIERS = [
    "LogisticRegression",
    "LinearDiscriminantAnalysis",
    "GaussianNB",
    "KNeighborsClassifier",
    "SVC",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "AdaBoostClassifier",
    "RidgeClassifier",
]

CURATED_REGRESSORS = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "BayesianRidge",
    "SVR",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "AdaBoostRegressor",
]

CURATED_CLUSTERERS = [
    "KMeans",
    "MiniBatchKMeans",
    "AgglomerativeClustering",
    "DBSCAN",
    "GaussianMixture",
    "Birch",
    "MeanShift",
    "SpectralClustering",
    "AffinityPropagation",
    "OPTICS",
    "KohonenSOM",
]


# ============================================================
# DICIONÁRIO DE EXPLICAÇÕES
# ============================================================

ALGORITHM_EXPLANATIONS: dict[str, dict[str, str]] = {
    # ----- Regressão -----
    "LinearRegression": {
        "tipo": "Regressão",
        "descricao": "Ajusta uma reta (ou hiperplano) minimizando a soma dos quadrados dos resíduos. Assume relação linear entre X e y.",
        "como_funciona": "Resolve a equação normal (XᵀX)⁻¹Xᵀy ou usa decomposição SVD para encontrar os coeficientes β₀, β₁, ..., βₚ.",
        "quando_usar": "Baseline rápido, relação aproximadamente linear, baixa dimensionalidade.",
        "limitacoes": "Sensível a outliers e multicolinearidade; não captura não linearidade.",
    },
    "Ridge": {
        "tipo": "Regressão",
        "descricao": "Regressão linear com penalização L2 (α·‖β‖²). Encolhe coeficientes, reduz variância.",
        "como_funciona": "Minimiza ‖y − Xβ‖² + α‖β‖². A solução é (XᵀX + αI)⁻¹Xᵀy.",
        "quando_usar": "Multicolinearidade, p ≈ n ou p > n, quando se quer regularizar sem zerar coeficientes.",
        "limitacoes": "Não faz seleção de variáveis (coeficientes pequenos mas não nulos).",
    },
    "Lasso": {
        "tipo": "Regressão",
        "descricao": "Regressão linear com penalização L1 (α·‖β‖₁). Produz coeficientes exatamente zero (seleção de features).",
        "como_funciona": "Minimiza ‖y − Xβ‖² + α‖β‖₁. Resolvido por coordinate descent.",
        "quando_usar": "Alta dimensionalidade, quando se deseja selecionar um subconjunto de variáveis.",
        "limitacoes": "Pode ser instável quando features são altamente correlacionadas (escolhe uma arbitrariamente).",
    },
    "ElasticNet": {
        "tipo": "Regressão",
        "descricao": "Combina L1 e L2: α(ρ‖β‖₁ + (1−ρ)‖β‖²/2).",
        "como_funciona": "Minimiza ‖y − Xβ‖² + α·ρ·‖β‖₁ + α·(1−ρ)/2·‖β‖². Interpola entre Ridge e Lasso.",
        "quando_usar": "Features correlacionadas + necessidade de seleção.",
        "limitacoes": "Dois hiperparâmetros para ajustar (α e l1_ratio).",
    },
    "BayesianRidge": {
        "tipo": "Regressão",
        "descricao": "Regressão ridge com estimação bayesiana dos hiperparâmetros α e λ.",
        "como_funciona": "Usa evidência para estimar α e λ iterativamente (maximização da verossimilhança marginal).",
        "quando_usar": "Quando não se quer escolher α manualmente.",
        "limitacoes": "Assume ruído gaussiano.",
    },
    "SVR": {
        "tipo": "Regressão",
        "descricao": "Support Vector Regression: encontra um tubo ε ao redor da função, penalizando pontos fora dele.",
        "como_funciona": "Minimiza ½‖w‖² + C·Σ(ξᵢ + ξᵢ*) sujeito a |yᵢ − f(xᵢ)| ≤ ε + ξᵢ. Usa kernel para não linearidade.",
        "quando_usar": "Dados de alta dimensão, não linearidade com poucas amostras.",
        "limitacoes": "Sensível à escala; C, ε e γ precisam de tuning.",
    },
    "KNeighborsRegressor": {
        "tipo": "Regressão",
        "descricao": "Prevê a média (ou mediana) dos k vizinhos mais próximos.",
        "como_funciona": "Para cada ponto, encontra os k vizinhos mais próximos no conjunto de treino e agrega seus y.",
        "quando_usar": "Padrões locais, dados de baixa dimensão.",
        "limitacoes": "Maldição da dimensionalidade; sensível à escala.",
    },
    "DecisionTreeRegressor": {
        "tipo": "Regressão",
        "descricao": "Árvore de decisão que particiona o espaço em regiões e prevê a média em cada folha.",
        "como_funciona": "Escolhe o corte que minimiza o erro quadrático (ou outro critério) em cada nó, recursivamente.",
        "quando_usar": "Interpretabilidade, relações não lineares.",
        "limitacoes": "Alta variância (overfitting) se não podada.",
    },
    "RandomForestRegressor": {
        "tipo": "Regressão",
        "descricao": "Ensemble de árvores treinadas em amostras bootstrap, com seleção aleatória de features.",
        "como_funciona": "Cada árvore é treinada em uma amostra bootstrap; a previsão é a média das árvores.",
        "quando_usar": "Dados tabulares, não linearidade, robustez a outliers.",
        "limitacoes": "Menos interpretável que uma árvore única; mais lento.",
    },
    "ExtraTreesRegressor": {
        "tipo": "Regressão",
        "descricao": "Como Random Forest, mas os cortes são escolhidos aleatoriamente (sem buscar o melhor).",
        "como_funciona": "Similar ao RF, mas cada árvore usa cortes aleatórios nos nós.",
        "quando_usar": "Quando se quer mais aleatoriedade e menos overfitting.",
        "limitacoes": "Pode ter viés maior em alguns datasets.",
    },
    "GradientBoostingRegressor": {
        "tipo": "Regressão",
        "descricao": "Boosting: treina árvores sequencialmente, cada uma corrigindo o erro da anterior.",
        "como_funciona": "F₀ = média(y); Fₘ = Fₘ₋₁ + η·hₘ(x), onde hₘ é uma árvore ajustada aos resíduos.",
        "quando_usar": "Alta acurácia em dados tabulares.",
        "limitacoes": "Sensível a overfitting; tuning cuidadoso de learning_rate e n_estimators.",
    },
    "HistGradientBoostingRegressor": {
        "tipo": "Regressão",
        "descricao": "Boosting com histogramas (inspirado no LightGBM), muito mais rápido em datasets grandes.",
        "como_funciona": "Discretiza features em bins e constrói histogramas para encontrar os melhores cortes.",
        "quando_usar": "Datasets grandes (> 10k amostras).",
        "limitacoes": "Menos preciso em datasets pequenos.",
    },
    "AdaBoostRegressor": {
        "tipo": "Regressão",
        "descricao": "Boosting adaptativo: aumenta o peso dos erros maiores.",
        "como_funciona": "Treina um regressor fraco, atualiza pesos das amostras com base no erro, repete.",
        "quando_usar": "Quando se tem um regressor base simples.",
        "limitacoes": "Sensível a outliers e ruído.",
    },
    "XGBRegressor": {
        "tipo": "Regressão",
        "descricao": "Gradient boosting otimizado da XGBoost, com regularização e paralelismo.",
        "como_funciona": "Similar ao GBR, mas com penalização L1/L2 nos pesos das folhas e histogramas.",
        "quando_usar": "Alta performance em dados tabulares.",
        "limitacoes": "Muitos hiperparâmetros.",
    },
    "LGBMRegressor": {
        "tipo": "Regressão",
        "descricao": "LightGBM: boosting com crescimento leaf-wise e histogramas.",
        "como_funciona": "Cresce a árvore escolhendo a folha com maior ganho, em vez de nível por nível.",
        "quando_usar": "Datasets grandes, quando se precisa de velocidade.",
        "limitacoes": "Pode overfittar em datasets pequenos.",
    },
    "CatBoostRegressor": {
        "tipo": "Regressão",
        "descricao": "Boosting com tratamento nativo de variáveis categóricas e ordered boosting.",
        "como_funciona": "Usa target statistics para categóricas e combina múltiplos modelos com diferentes permutações.",
        "quando_usar": "Dados com muitas variáveis categóricas.",
        "limitacoes": "Mais lento que LightGBM em alguns casos.",
    },
    # ----- Classificação -----
    "LogisticRegression": {
        "tipo": "Classificação",
        "descricao": "Modelo linear que estima a probabilidade de cada classe via função logística (ou softmax).",
        "como_funciona": "P(y=1|x) = 1/(1 + exp(−(β₀ + βᵀx))). Otimiza a log-verossimilhança.",
        "quando_usar": "Baseline linear, probabilidades calibradas, interpretabilidade.",
        "limitacoes": "Fronteira de decisão linear.",
    },
    "LinearDiscriminantAnalysis": {
        "tipo": "Classificação",
        "descricao": "Assume que cada classe segue uma gaussiana com mesma matriz de covariância.",
        "como_funciona": "Projeta os dados em uma direção que maximiza a separação entre classes (Fisher).",
        "quando_usar": "Classes aproximadamente gaussianas, poucas amostras por classe.",
        "limitacoes": "Assume normalidade e homocedasticidade.",
    },
    "GaussianNB": {
        "tipo": "Classificação",
        "descricao": "Naive Bayes com verossimilhança gaussiana para cada feature.",
        "como_funciona": "P(y|x) ∝ P(y)·∏P(xᵢ|y), assumindo independência condicional entre features.",
        "quando_usar": "Baseline rápido, alta dimensionalidade, texto.",
        "limitacoes": "Assume independência entre features (raramente verdade).",
    },
    "KNeighborsClassifier": {
        "tipo": "Classificação",
        "descricao": "Classifica pela votação majoritária dos k vizinhos mais próximos.",
        "como_funciona": "Para cada ponto, encontra os k vizinhos mais próximos no treino e vota.",
        "quando_usar": "Padrões locais, dados de baixa dimensão.",
        "limitacoes": "Maldição da dimensionalidade; sensível à escala.",
    },
    "SVC": {
        "tipo": "Classificação",
        "descricao": "Support Vector Classifier: encontra o hiperplano que maximiza a margem entre classes.",
        "como_funciona": "Maximiza 2/‖w‖ sujeito a yᵢ(wᵀxᵢ + b) ≥ 1. Usa kernel para não linearidade.",
        "quando_usar": "Alta dimensionalidade, fronteiras não lineares.",
        "limitacoes": "Sensível a C e γ; lento para datasets grandes.",
    },
    "DecisionTreeClassifier": {
        "tipo": "Classificação",
        "descricao": "Árvore que particiona o espaço e atribui a classe majoritária em cada folha.",
        "como_funciona": "Escolhe o corte que maximiza a redução de impureza (Gini ou entropia).",
        "quando_usar": "Interpretabilidade, não linearidade.",
        "limitacoes": "Overfitting se não podada.",
    },
    "RandomForestClassifier": {
        "tipo": "Classificação",
        "descricao": "Ensemble de árvores com bootstrap e seleção aleatória de features.",
        "como_funciona": "Cada árvore vota; a classe mais votada é a previsão.",
        "quando_usar": "Dados tabulares, robustez, importância de features.",
        "limitacoes": "Menos interpretável.",
    },
    "ExtraTreesClassifier": {
        "tipo": "Classificação",
        "descricao": "Como Random Forest, mas com cortes aleatórios.",
        "como_funciona": "Similar ao RF, mas os thresholds são escolhidos aleatoriamente.",
        "quando_usar": "Quando se quer mais aleatoriedade.",
        "limitacoes": "Pode ter viés.",
    },
    "GradientBoostingClassifier": {
        "tipo": "Classificação",
        "descricao": "Boosting de árvores para classificação.",
        "como_funciona": "Treina árvores sequencialmente nos resíduos (gradiente da log-loss).",
        "quando_usar": "Alta acurácia.",
        "limitacoes": "Sensível a overfitting.",
    },
    "HistGradientBoostingClassifier": {
        "tipo": "Classificação",
        "descricao": "Boosting com histogramas, rápido.",
        "como_funciona": "Discretiza features e constrói histogramas.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Menos preciso em pequenos.",
    },
    "AdaBoostClassifier": {
        "tipo": "Classificação",
        "descricao": "Boosting adaptativo.",
        "como_funciona": "Aumenta o peso das amostras mal classificadas.",
        "quando_usar": "Com classificadores fracos.",
        "limitacoes": "Sensível a ruído.",
    },
    "RidgeClassifier": {
        "tipo": "Classificação",
        "descricao": "Classificador linear com penalização L2.",
        "como_funciona": "Resolve um problema de regressão ridge para cada classe e escolhe a maior saída.",
        "quando_usar": "Alta dimensionalidade, multiclasse.",
        "limitacoes": "Sem probabilidades.",
    },
    "XGBClassifier": {
        "tipo": "Classificação",
        "descricao": "XGBoost para classificação.",
        "como_funciona": "Boosting com regularização.",
        "quando_usar": "Alta performance.",
        "limitacoes": "Muitos hiperparâmetros.",
    },
    "LGBMClassifier": {
        "tipo": "Classificação",
        "descricao": "LightGBM para classificação.",
        "como_funciona": "Boosting leaf-wise com histogramas.",
        "quando_usar": "Datasets grandes.",
        "limitacoes": "Pode overfittar.",
    },
    "CatBoostClassifier": {
        "tipo": "Classificação",
        "descricao": "CatBoost para classificação.",
        "como_funciona": "Ordered boosting + target statistics.",
        "quando_usar": "Muitas categóricas.",
        "limitacoes": "Lento.",
    },
    # ----- Agrupamento -----
    "KMeans": {
        "tipo": "Agrupamento",
        "descricao": "Particiona os dados em k clusters minimizando a inércia (soma das distâncias ao centroide).",
        "como_funciona": "Itera: atribui cada ponto ao centroide mais próximo; recalcula centroides.",
        "quando_usar": "Clusters esféricos, bem separados, k conhecido.",
        "limitacoes": "Assume clusters esféricos; sensível a outliers; k precisa ser escolhido.",
    },
    "MiniBatchKMeans": {
        "tipo": "Agrupamento",
        "descricao": "Versão do KMeans que usa mini-batches para atualizar centroides.",
        "como_funciona": "Similar ao KMeans, mas atualiza com subamostras.",
        "quando_usar": "Datasets muito grandes.",
        "limitacoes": "Menos preciso que KMeans.",
    },
    "AgglomerativeClustering": {
        "tipo": "Agrupamento",
        "descricao": "Agrupamento hierárquico aglomerativo.",
        "como_funciona": "Começa com cada ponto como cluster e merge os mais próximos (ward, complete, average).",
        "quando_usar": "Quando se quer uma hierarquia.",
        "limitacoes": "O(n²) memória.",
    },
    "DBSCAN": {
        "tipo": "Agrupamento",
        "descricao": "Density-Based Spatial Clustering of Applications with Noise.",
        "como_funciona": "Agrupa pontos densamente conectados; pontos em regiões de baixa densidade são ruído (-1).",
        "quando_usar": "Clusters de forma arbitrária, com ruído.",
        "limitacoes": "Sensível a eps e min_samples.",
    },
    "GaussianMixture": {
        "tipo": "Agrupamento",
        "descricao": "Modelo de mistura gaussiana: assume que os dados vêm de k gaussianas.",
        "como_funciona": "EM: estima médias, covariâncias e pesos.",
        "quando_usar": "Clusters elípticos, probabilidades de pertencimento.",
        "limitacoes": "Assume gaussianidade.",
    },
    "Birch": {
        "tipo": "Agrupamento",
        "descricao": "Balanced Iterative Reducing and Clustering using Hierarchies.",
        "como_funciona": "Constrói uma árvore CF (Clustering Feature) e agrupa as folhas.",
        "quando_usar": "Datasets grandes, memória limitada.",
        "limitacoes": "Menos preciso.",
    },
    "MeanShift": {
        "tipo": "Agrupamento",
        "descricao": "Busca modos da densidade.",
        "como_funciona": "Cada ponto se move para a média dos vizinhos dentro de uma janela (bandwidth).",
        "quando_usar": "Quando não se sabe k.",
        "limitacoes": "Lento; bandwidth crítico.",
    },
    "SpectralClustering": {
        "tipo": "Agrupamento",
        "descricao": "Usa autovetores da matriz Laplaciana para agrupar.",
        "como_funciona": "Constrói grafo de similaridade, calcula autovetores, aplica KMeans.",
        "quando_usar": "Clusters não convexos.",
        "limitacoes": "O(n³).",
    },
    "AffinityPropagation": {
        "tipo": "Agrupamento",
        "descricao": "Troca de mensagens entre pontos para encontrar exemplares.",
        "como_funciona": "Cada ponto envia responsabilidade e disponibilidade; os exemplares emergem.",
        "quando_usar": "Quando não se sabe k.",
        "limitacoes": "Lento para muitos pontos.",
    },
    "OPTICS": {
        "tipo": "Agrupamento",
        "descricao": "Ordering Points To Identify the Clustering Structure.",
        "como_funciona": "Similar ao DBSCAN, mas lida com densidades variáveis.",
        "quando_usar": "Densidades variadas.",
        "limitacoes": "Parâmetros xi e min_samples.",
    },
    "HDBSCAN": {
        "tipo": "Agrupamento",
        "descricao": "Hierarchical DBSCAN: variação do DBSCAN que não exige eps.",
        "como_funciona": "Constrói uma hierarquia de clusters baseada em densidade e extrai o melhor corte.",
        "quando_usar": "Clusters de densidades variadas, com ruído.",
        "limitacoes": "min_cluster_size crítico.",
    },
    "KohonenSOM": {
        "tipo": "Agrupamento",
        "descricao": "Self-Organizing Map (Mapa de Kohonen): rede neural não supervisionada que projeta dados em uma grade 2D preservando topologia.",
        "como_funciona": "Cada neurônio tem um vetor de pesos. Para cada amostra, encontra-se o BMU (Best Matching Unit) e atualiza-se os pesos dos vizinhos com uma função de vizinhança gaussiana.",
        "quando_usar": "Visualização de alta dimensionalidade, agrupamento exploratório, redução de dimensionalidade topológica.",
        "limitacoes": "Número de neurônios e iterações precisam ser escolhidos; não gera clusters diretamente (é preciso agrupar os BMUs).",
    },
}


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Análise IC — Completa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ESTADO
# ============================================================

def init_state() -> None:
    defaults = {
        "df_raw": None,
        "encoding_config": None,
        "encoding_preview": None,
        "results": None,
        "task_type": "regression",
        "encoding_ready": False,
        "prediction_result": None,
        "clustering_result": None,
        "pre_analysis_reg": None,
        "pre_analysis_clf": None,
        "pre_analysis_cluster": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# TRANSFORMERS CUSTOMIZADOS
# ============================================================

class IQRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode: str = "keep", factor: float = 1.5):
        self.mode = mode
        self.factor = factor

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]
        self.lower_ = np.empty(self.n_features_in_, dtype=float)
        self.upper_ = np.empty(self.n_features_in_, dtype=float)
        self.median_ = np.empty(self.n_features_in_, dtype=float)
        for j in range(self.n_features_in_):
            col = X_arr[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                self.lower_[j] = -np.inf
                self.upper_[j] = np.inf
                self.median_[j] = 0.0
                continue
            q1, q3 = np.nanpercentile(finite, [25, 75])
            iqr = q3 - q1
            self.lower_[j] = q1 - self.factor * iqr
            self.upper_[j] = q3 + self.factor * iqr
            self.median_[j] = float(np.nanmedian(finite))
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float).copy()
        if self.mode == "keep":
            return X_arr
        low, up = self.lower_, self.upper_
        mask_low = X_arr < low
        mask_up = X_arr > up
        if self.mode == "winsorize":
            X_arr = np.where(mask_low, low, X_arr)
            X_arr = np.where(mask_up, up, X_arr)
        elif self.mode == "median":
            mask = mask_low | mask_up
            X_arr[mask] = np.broadcast_to(self.median_, X_arr.shape)[mask]
        return X_arr


class ManualMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: dict[str, float] | None = None, unknown_value: float = -1.0):
        self.mapping = mapping or {}
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        out = np.empty(arr.shape, dtype=float)
        for idx, value in np.ndenumerate(arr):
            key = "missing" if pd.isna(value) else str(value)
            out[idx] = float(self.mapping.get(key, self.unknown_value))
        return out


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.10, max_features: int | None = None):
        self.threshold = threshold
        self.max_features = max_features

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        y_sr = pd.Series(y).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)
        y_num = pd.to_numeric(y_sr, errors="coerce")
        if y_num.isna().all():
            y_num = pd.Series(pd.factorize(y_sr.astype(str))[0], dtype=float)
        else:
            y_num = y_num.fillna(y_num.median())
        corr_values = {}
        for col in X_df.columns:
            x = pd.to_numeric(X_df[col], errors="coerce")
            corr_values[col] = float(x.corr(y_num)) if x.notna().sum() > 1 else 0.0
        corr = pd.Series(corr_values).abs().fillna(0).sort_values(ascending=False)
        if corr.empty:
            raise ValueError("Nenhuma variável disponível após o pré-processamento.")
        if self.max_features is not None and int(self.max_features) > 0:
            chosen = list(corr.head(int(self.max_features)).index)
        else:
            chosen = list(corr[corr >= float(self.threshold)].index)
        if not chosen:
            chosen = [corr.index[0]]
        self.selected_indices_ = [X_df.columns.get_loc(c) for c in chosen]
        self.selected_features_ = [str(c) for c in chosen]
        self.n_features_in_ = X_df.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X)
        return arr[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


# ============================================================
# WRAPPER KOHONEN SOM
# ============================================================

class KohonenClusterer(BaseEstimator):
    """Wrapper para MiniSom que expõe fit_predict compatível com sklearn."""

    def __init__(
        self,
        som_x: int = 5,
        som_y: int = 5,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        num_iteration: int = 1000,
        random_state: int | None = None,
        neighborhood_function: str = "gaussian",
    ):
        self.som_x = som_x
        self.som_y = som_y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.random_state = random_state
        self.neighborhood_function = neighborhood_function
        self.som_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        if MiniSom is None:
            raise ImportError(
                "MiniSom não está instalado. Execute: pip install minisom"
            )
        X = np.asarray(X, dtype=float)
        self.som_ = MiniSom(
            self.som_x,
            self.som_y,
            X.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function=self.neighborhood_function,
            random_seed=self.random_state,
        )
        self.som_.random_weights_init(X)
        self.som_.train_random(X, self.num_iteration)
        self.labels_ = self._assign_labels(X)
        return self

    def _assign_labels(self, X):
        labels = []
        for x in X:
            bmu = self.som_.winner(x)
            labels.append(bmu[0] * self.som_y + bmu[1])
        return np.asarray(labels)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def get_params(self, deep=True):
        return {
            "som_x": self.som_x,
            "som_y": self.som_y,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "num_iteration": self.num_iteration,
            "random_state": self.random_state,
            "neighborhood_function": self.neighborhood_function,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ============================================================
# UTILITÁRIOS DE DADOS
# ============================================================

def make_onehot_encoder():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=float)
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore", dtype=float)


def detect_categorical_columns(df: pd.DataFrame, max_unique_int: int = 20) -> list[str]:
    cats: list[str] = []
    for col in df.columns:
        try:
            s = df[col]
            if (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(s)
            ):
                cats.append(col)
                continue
            if pd.api.types.is_integer_dtype(s):
                nun = s.nunique(dropna=True)
                if 1 < nun <= max_unique_int:
                    cats.append(col)
        except Exception:
            continue
    return cats


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or str(series.dtype) == "string"
        ):
            continue
        s = series.astype(str).str.strip()
        s_num = (
            s.str.replace(r"R\$\s*", "", regex=True)
            .str.replace(r"%\s*$", "", regex=True)
            .str.replace(" ", "", regex=False)
        )
        direct = pd.to_numeric(s_num, errors="coerce")
        brazilian = pd.to_numeric(
            s_num.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        non_null = series.notna().sum()
        if non_null == 0:
            continue
        if direct.notna().sum() / non_null >= 0.90:
            df[col] = direct
        elif brazilian.notna().sum() / non_null >= 0.90:
            df[col] = brazilian
    return df


def load_dataframe(uploaded_file, source_type: str, uci_id: int | None = None) -> pd.DataFrame:
    if source_type == "UCI":
        if fetch_ucirepo is None:
            raise ImportError("Pacote 'ucimlrepo' não está instalado.")
        ds = fetch_ucirepo(id=int(uci_id))
        if getattr(ds.data, "features", None) is not None and getattr(ds.data, "targets", None) is not None:
            df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        else:
            df = ds.data.original
        return coerce_numeric(df)
    if uploaded_file is None:
        raise ValueError("Nenhum arquivo foi enviado.")
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = uploaded_file.read().decode("utf-8", errors="replace").lstrip("\ufeff")
        first = raw.splitlines()[0] if raw else ""
        counts = {
            ";": first.count(";"),
            ",": first.count(","),
            "\t": first.count("\t"),
            "|": first.count("|"),
        }
        sep = max(counts, key=counts.get) if max(counts.values()) else ","
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError("Formato não suportado. Use CSV, XLS/XLSX ou JSON.")
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    return coerce_numeric(df)


def parse_list(text: str) -> list[Any]:
    values: list[Any] = []
    for raw_token in str(text).split(","):
        token = raw_token.strip()
        if not token:
            continue
        low = token.lower()
        if low in {"scale", "auto"}:
            values.append(low)
            continue
        try:
            value = float(token)
            if value.is_integer() and "." not in token and "e" not in low:
                value = int(value)
            values.append(value)
        except ValueError as exc:
            raise ValueError(
                f"Valor inválido na lista: '{token}'. Use números ou 'scale'/'auto'."
            ) from exc
    if not values:
        raise ValueError("A lista de hiperparâmetros não pode ficar vazia.")
    return values


def validate_feature_selection(df: pd.DataFrame, features: list[str], target: str) -> None:
    if target not in df.columns:
        raise ValueError("A variável alvo selecionada não existe no dataset.")
    if not features:
        raise ValueError("Selecione pelo menos uma variável preditora.")
    if target in features:
        raise ValueError("A variável alvo não pode estar entre as preditoras.")
    if len(df[features].columns) != len(set(features)):
        raise ValueError("Existem variáveis preditoras duplicadas.")


def iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < low) | (s > up)).sum())
        rows.append([col, q1, q3, iqr, low, up, n, 100 * n / len(s)])
    return pd.DataFrame(
        rows,
        columns=["Variável", "Q1", "Q3", "IQR", "Limite inf.", "Limite sup.", "Outliers", "%"],
    )


# ============================================================
# CODIFICAÇÃO
# ============================================================

def apply_categorical_encoding_preview(
    df: pd.DataFrame,
    encoding_map: dict[str, str],
    ordinal_orders: dict[str, list[str]],
    manual_maps: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    out = df.copy()
    report: list[tuple[str, str, str]] = []
    for col, method in encoding_map.items():
        if col not in out.columns:
            continue
        if method == "drop":
            out = out.drop(columns=[col])
            report.append((col, "drop", "coluna removida"))
            continue
        s = out[col].astype("string").fillna("missing")
        if method == "onehot":
            enc = make_onehot_encoder()
            arr = enc.fit_transform(s.to_frame())
            names = [f"{col}__{cat}" for cat in enc.categories_[0]]
            out = pd.concat(
                [out.drop(columns=[col]), pd.DataFrame(arr, columns=names, index=out.index)],
                axis=1,
            )
            report.append((col, "onehot", f"{len(names)} colunas"))
        elif method == "label":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[col] = enc.fit_transform(s.to_frame()).ravel()
            report.append((col, "label", "1 coluna"))
        elif method == "ordinal":
            order = ordinal_orders.get(col, sorted(s.unique().tolist()))
            enc = OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)
            out[col] = enc.fit_transform(s.to_frame()).ravel()
            report.append((col, "ordinal", "1 coluna"))
        elif method == "manual":
            mapping = manual_maps.get(col, {})
            out[col] = s.map(mapping).fillna(-1).astype(float)
            report.append((col, "manual", f"{len(mapping)} regras"))
    return out, report


def build_preprocessor(
    X: pd.DataFrame,
    encoding_map: dict[str, str],
    ordinal_orders: dict[str, list[str]],
    manual_maps: dict[str, dict[str, float]],
    outlier_mode: str,
) -> ColumnTransformer:
    categorical_columns, onehot_columns, label_columns = [], [], []
    ordinal_columns, manual_columns, dropped_columns = [], [], []
    for col, method in encoding_map.items():
        if col not in X.columns:
            continue
        if method == "drop":
            dropped_columns.append(col)
        elif method == "onehot":
            onehot_columns.append(col)
            categorical_columns.append(col)
        elif method == "label":
            label_columns.append(col)
            categorical_columns.append(col)
        elif method == "ordinal":
            ordinal_columns.append(col)
            categorical_columns.append(col)
        elif method == "manual":
            manual_columns.append(col)
            categorical_columns.append(col)
    numeric_columns = [c for c in X.columns if c not in categorical_columns and c not in dropped_columns]
    transformers = []
    if numeric_columns:
        numeric_pipe = Pipeline([
            ("outliers", IQRTransformer(mode=outlier_mode)),
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", numeric_pipe, numeric_columns))
    if onehot_columns:
        onehot_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_onehot_encoder()),
        ])
        transformers.append(("onehot", onehot_pipe, onehot_columns))
    if label_columns:
        label_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("label", label_pipe, label_columns))
    for col in ordinal_columns:
        order = ordinal_orders.get(col)
        if not order:
            raise ValueError(f"A ordem ordinal da coluna '{col}' não foi informada.")
        ordinal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append((f"ordinal_{col}", ordinal_pipe, [col]))
    for col in manual_columns:
        mapping = manual_maps.get(col, {})
        manual_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("mapper", ManualMapper(mapping=mapping, unknown_value=-1.0)),
        ])
        transformers.append((f"manual_{col}", manual_pipe, [col]))
    if not transformers:
        raise ValueError("Nenhuma variável ficou disponível para o modelo.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


# ============================================================
# DESCOBERTA DE ESTIMADORES
# ============================================================

@st.cache_data(show_spinner=False)
def get_all_classifiers() -> list[str]:
    valid: list[str] = []
    for name, klass in all_estimators(type_filter="classifier"):
        try:
            klass()
            valid.append(name)
        except Exception:
            continue
    if xgb is not None and hasattr(xgb, "XGBClassifier"):
        valid.append("XGBClassifier")
    if lgb is not None and hasattr(lgb, "LGBMClassifier"):
        valid.append("LGBMClassifier")
    if cb is not None and hasattr(cb, "CatBoostClassifier"):
        valid.append("CatBoostClassifier")
    return sorted(set(valid))


@st.cache_resource(show_spinner=False)
def get_classifier_discovery_registry() -> dict[str, Any]:
    reg = {}
    try:
        reg.update(dict(all_estimators(type_filter="classifier")))
    except Exception:
        pass
    if xgb is not None:
        reg["XGBClassifier"] = xgb.XGBClassifier
    if lgb is not None:
        reg["LGBMClassifier"] = lgb.LGBMClassifier
    if cb is not None:
        reg["CatBoostClassifier"] = cb.CatBoostClassifier
    return reg


@st.cache_data(show_spinner=False)
def get_all_regressors() -> list[str]:
    valid: list[str] = []
    for name, klass in all_estimators(type_filter="regressor"):
        try:
            klass()
            valid.append(name)
        except Exception:
            continue
    if xgb is not None and hasattr(xgb, "XGBRegressor"):
        valid.append("XGBRegressor")
    if lgb is not None and hasattr(lgb, "LGBMRegressor"):
        valid.append("LGBMRegressor")
    if cb is not None and hasattr(cb, "CatBoostRegressor"):
        valid.append("CatBoostRegressor")
    return sorted(set(valid))


@st.cache_resource(show_spinner=False)
def get_regressor_discovery_registry() -> dict[str, Any]:
    reg = {}
    try:
        reg.update(dict(all_estimators(type_filter="regressor")))
    except Exception:
        pass
    if xgb is not None:
        reg["XGBRegressor"] = xgb.XGBRegressor
    if lgb is not None:
        reg["LGBMRegressor"] = lgb.LGBMRegressor
    if cb is not None:
        reg["CatBoostRegressor"] = cb.CatBoostRegressor
    return reg


@st.cache_data(show_spinner=False)
def get_all_clusterers() -> list[str]:
    valid: list[str] = []
    try:
        for name, klass in all_estimators(type_filter="cluster"):
            try:
                klass()
                valid.append(name)
            except Exception:
                continue
    except Exception:
        pass
    for name in CURATED_CLUSTERERS:
        if name not in valid:
            valid.append(name)
    if hdbscan_lib is not None:
        valid.append("HDBSCAN")
    if MiniSom is not None:
        valid.append("KohonenSOM")
    return sorted(set(valid))


@st.cache_resource(show_spinner=False)
def get_clusterer_discovery_registry() -> dict[str, Any]:
    reg = {}
    try:
        reg.update(dict(all_estimators(type_filter="cluster")))
    except Exception:
        pass
    if hdbscan_lib is not None:
        reg["HDBSCAN"] = hdbscan_lib.HDBSCAN
    if MiniSom is not None:
        reg["KohonenSOM"] = KohonenClusterer
    return reg


# ============================================================
# SEED
# ============================================================

def _set_random_state(estimator, random_state):
    if random_state is None:
        return estimator
    try:
        params = estimator.get_params()
    except Exception:
        return estimator
    if "random_state" in params:
        try:
            estimator.set_params(random_state=random_state)
        except Exception:
            pass
    return estimator


def resolve_random_state(choice: str, custom_value: int | None = None):
    if choice == "fixa":
        return DEFAULT_RANDOM_STATE
    if choice == "aleatoria":
        return None
    return int(custom_value)


# ============================================================
# PRÉ-ANÁLISE E RECOMENDAÇÕES
# ============================================================

def analyze_dataset(df: pd.DataFrame, features: list[str], target: str | None, task: str) -> dict[str, Any]:
    X = df[features]
    n = len(X)
    p = len(features)
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in numeric_cols]
    analysis: dict[str, Any] = {
        "n_samples": int(n),
        "n_features": int(p),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(cat_cols),
        "numeric_features": numeric_cols,
        "categorical_features": cat_cols,
        "missing_pct": float(100 * X.isna().sum().sum() / (n * p)) if n and p else 0.0,
        "warnings": [],
        "notes": [],
    }
    if n < 100:
        analysis["warnings"].append("Poucas amostras (< 100). Prefira modelos simples.")
    if n > 100_000:
        analysis["notes"].append("Muitas amostras. Modelos escaláveis (SGD, HistGradientBoosting, MiniBatchKMeans) tendem a ser melhores.")
    if p == 0:
        analysis["warnings"].append("Nenhuma variável preditora selecionada.")
    if p > 0 and n > 0 and p > n:
        analysis["warnings"].append("p > n (mais features que amostras). Modelos regularizados são indicados.")
    if p > 50:
        analysis["warnings"].append("Alta dimensionalidade. Modelos lineares regularizados ou redução de dimensionalidade podem ajudar.")
    if analysis["missing_pct"] > 20:
        analysis["warnings"].append(f"{analysis['missing_pct']:.1f}% de valores ausentes. Verifique a imputação.")
    if cat_cols:
        analysis["notes"].append(f"{len(cat_cols)} variável(is) categórica(s) — codificação já é tratada pelo pipeline.")

    if task == "regression" and target and target in df.columns:
        y = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(y):
            analysis["target_mean"] = float(y.mean())
            analysis["target_std"] = float(y.std())
            try:
                analysis["target_skew"] = float(y.skew())
            except Exception:
                analysis["target_skew"] = 0.0
            if abs(analysis["target_skew"]) > 2:
                analysis["warnings"].append("Distribuição do alvo muito assimétrica — considere transformação (log, sqrt).")
            if numeric_cols and len(y) > 5:
                try:
                    X_num = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
                    y_aligned = y.reindex(X_num.index).dropna()
                    if len(y_aligned) > 5:
                        corrs = X_num.loc[y_aligned.index].corrwith(y_aligned).abs().mean()
                        analysis["mean_abs_corr"] = float(corrs) if pd.notna(corrs) else 0.0
                        if analysis["mean_abs_corr"] < 0.15:
                            analysis["notes"].append("Correlação linear média baixa — modelos não lineares podem se sair melhor.")
                except Exception:
                    pass

    elif task == "classification" and target and target in df.columns:
        y = df[target].astype("string").fillna("missing")
        counts = y.value_counts()
        analysis["n_classes"] = int(len(counts))
        analysis["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        analysis["is_binary"] = len(counts) == 2
        if len(counts) > 1 and counts.min() > 0:
            ratio = float(counts.max() / counts.min())
            analysis["imbalance_ratio"] = ratio
            if ratio > 5:
                analysis["warnings"].append(f"Classes desbalanceadas (razão {ratio:.1f}). Use class_weight='balanced' e F1 macro.")
        if len(counts) >= 2 and counts.min() < 5:
            analysis["warnings"].append("Alguma classe possui menos de 5 amostras — a CV estratificada pode falhar.")
        if len(counts) > 10:
            analysis["warnings"].append(f"Muitas classes ({len(counts)}). Modelos lineares podem ter desempenho limitado.")

    elif task == "clustering":
        if p > 20:
            analysis["notes"].append("Alta dimensionalidade tende a degradar o agrupamento — considere PCA.")
        try:
            if len(numeric_cols) > 1:
                corr = X[numeric_cols].corr().abs().values
                np.fill_diagonal(corr, 0)
                if corr.size:
                    analysis["max_feature_corr"] = float(np.nanmax(corr))
                    if analysis["max_feature_corr"] > 0.95:
                        analysis["notes"].append("Existem features altamente correlacionadas — podem ser redundantes.")
        except Exception:
            pass
    return analysis


def recommend_algorithms(analysis: dict[str, Any], task: str) -> list[tuple[str, str]]:
    n = analysis["n_samples"]
    p = analysis["n_features"]
    recs: list[tuple[str, str]] = []

    if task == "regression":
        recs.append(("LinearRegression", "Baseline linear, rápido e interpretável."))
        recs.append(("Ridge", "Regularização L2 — robusto em alta dimensionalidade."))
        if p >= 3 and n >= 30:
            recs.append(("Lasso", "Regularização L1 — faz seleção de features."))
            recs.append(("ElasticNet", "Combina L1 e L2 — útil com features correlacionadas."))
        if n <= 5000:
            recs.append(("KNeighborsRegressor", "Bom para amostras pequenas com padrões locais."))
            recs.append(("SVR", "SVM para regressão — eficaz em alta dimensão."))
        recs.append(("DecisionTreeRegressor", "Árvore simples e interpretável."))
        if n >= 200:
            recs.append(("RandomForestRegressor", "Ensemble robusto para não linearidade."))
            recs.append(("ExtraTreesRegressor", "Ensemble mais aleatório, menos overfitting."))
            recs.append(("GradientBoostingRegressor", "Boosting, geralmente alta acurácia."))
        if n >= 1000:
            recs.append(("HistGradientBoostingRegressor", "Boosting rápido para datasets maiores."))
        if analysis.get("mean_abs_corr", 1.0) < 0.2:
            recs.append(("RandomForestRegressor", "Correlações lineares baixas — árvores capturam não linearidade."))
        if p > n and n > 0:
            recs.insert(0, ("ElasticNet", "Recomendado quando p > n."))
            recs.insert(0, ("Ridge", "Recomendado quando p > n."))

    elif task == "classification":
        recs.append(("LogisticRegression", "Baseline linear, rápido e interpretável."))
        if analysis.get("is_binary", False):
            recs.append(("LinearDiscriminantAnalysis", "Bom para binário com classes gaussianas."))
            recs.append(("RidgeClassifier", "Regularização L2, robusto."))
        if n <= 5000:
            recs.append(("KNeighborsClassifier", "Bom para amostras pequenas."))
            recs.append(("SVC", "SVM kernelizado, eficaz com poucas amostras."))
            recs.append(("GaussianNB", "Rápido, bom baseline probabilístico."))
        recs.append(("DecisionTreeClassifier", "Árvore interpretável."))
        if n >= 200:
            recs.append(("RandomForestClassifier", "Ensemble robusto."))
            recs.append(("ExtraTreesClassifier", "Ensemble aleatório."))
        if n >= 500:
            recs.append(("GradientBoostingClassifier", "Boosting, alta acurácia."))
            recs.append(("HistGradientBoostingClassifier", "Boosting rápido e escalável."))
            recs.append(("AdaBoostClassifier", "Boosting adaptativo."))
        if analysis.get("imbalance_ratio", 1.0) > 5:
            recs.append(("RandomForestClassifier", "Use class_weight='balanced' para classes desbalanceadas."))
            recs.append(("LogisticRegression", "Use class_weight='balanced'."))
        if p > 50:
            recs.insert(0, ("LogisticRegression", "Alta dimensionalidade — modelos lineares regularizados são indicados."))
            recs.append(("SVC", "Kernel lida bem com alta dimensionalidade."))

    elif task == "clustering":
        recs.append(("KMeans", "Rápido, assume clusters esféricos e bem separados."))
        if n >= 10000:
            recs.append(("MiniBatchKMeans", "Versão escalável do KMeans."))
            recs.append(("Birch", "Escalável — boa para dados grandes."))
        if n <= 10000:
            recs.append(("AgglomerativeClustering", "Hierárquico — dispensa número fixo de clusters."))
            recs.append(("GaussianMixture", "Probabilístico — clusters elípticos."))
        recs.append(("DBSCAN", "Baseado em densidade — detecta outliers como ruído."))
        recs.append(("MeanShift", "Baseado em densidade — sem número fixo de clusters."))
        recs.append(("AffinityPropagation", "Sem número fixo — escolhe automaticamente."))
        if n <= 5000:
            recs.append(("SpectralClustering", "Bom para clusters não convexos."))
        recs.append(("OPTICS", "Variação de DBSCAN — robusto a densidades variadas."))
        if hdbscan_lib is not None:
            recs.append(("HDBSCAN", "DBSCAN hierárquico — não exige eps."))
        if MiniSom is not None:
            recs.append(("KohonenSOM", "Mapa auto-organizável — visualização topológica e agrupamento exploratório."))

    seen = set()
    unique_recs = []
    for name, why in recs:
        if name not in seen:
            seen.add(name)
            unique_recs.append((name, why))
    return unique_recs


def _render_pre_analysis(analysis: dict[str, Any], recommendations: list[tuple[str, str]]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", analysis["n_samples"])
    c2.metric("Features", analysis["n_features"])
    c3.metric("Numéricas", analysis["n_numeric"])
    c4.metric("Categóricas", analysis["n_categorical"])
    st.caption(f"Valores ausentes: {analysis['missing_pct']:.2f}%")
    if analysis.get("target_skew") is not None:
        st.caption(
            f"Alvo: média={analysis.get('target_mean'):.3f}, "
            f"desvio={analysis.get('target_std'):.3f}, "
            f"assimetria={analysis.get('target_skew'):.3f}"
        )
    if analysis.get("n_classes"):
        st.caption(
            f"Classes: {analysis['n_classes']} "
            f"(desbalanceamento ≈ {analysis.get('imbalance_ratio', 1.0):.2f})"
        )
    if analysis["warnings"]:
        for w in analysis["warnings"]:
            st.warning(w)
    if analysis["notes"]:
        for note in analysis["notes"]:
            st.info(note)
    if recommendations:
        st.markdown("#### ✅ Algoritmos recomendados")
        st.dataframe(
            pd.DataFrame(recommendations, columns=["Algoritmo", "Justificativa"]),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# GRADES DE HIPERPARÂMETROS
# ============================================================

def get_regressor_param_grid(name: str):
    grids = {
        "Ridge": {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"regressor__alpha": [0.001, 0.01, 0.1, 1.0]},
        "ElasticNet": {
            "regressor__alpha": [0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.2, 0.5, 0.8],
        },
        "SVR": {
            "regressor__C": [0.1, 1.0, 10.0],
            "regressor__kernel": ["rbf", "linear"],
            "regressor__gamma": ["scale", "auto"],
            "regressor__epsilon": [0.01, 0.1, 0.5],
        },
        "KNeighborsRegressor": {
            "regressor__n_neighbors": [3, 5, 7, 9, 11],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],
        },
        "DecisionTreeRegressor": {
            "regressor__max_depth": [None, 3, 5, 10],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        "RandomForestRegressor": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 5, 10],
            "regressor__min_samples_split": [2, 5],
        },
        "ExtraTreesRegressor": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 5, 10],
        },
        "GradientBoostingRegressor": {
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
            "regressor__max_depth": [2, 3],
        },
        "HistGradientBoostingRegressor": {
            "regressor__learning_rate": [0.05, 0.1, 0.2],
            "regressor__max_iter": [100, 200],
        },
        "AdaBoostRegressor": {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__learning_rate": [0.05, 0.1, 0.5],
        },
        "BayesianRidge": {
            "regressor__alpha_1": [1e-6, 1e-5, 1e-4],
            "regressor__alpha_2": [1e-6, 1e-5, 1e-4],
        },
        "XGBRegressor": {
            "regressor__n_estimators": [100, 300],
            "regressor__max_depth": [3, 6, 10],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
        },
        "LGBMRegressor": {
            "regressor__n_estimators": [100, 300],
            "regressor__num_leaves": [15, 31, 63],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
        },
        "CatBoostRegressor": {
            "regressor__iterations": [200, 500],
            "regressor__depth": [4, 6, 8],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
            "regressor__verbose": [False],
        },
    }
    return grids.get(name)


def get_classifier_param_grid(name: str):
    grids = {
        "LogisticRegression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__max_iter": [200, 500],
        },
        "LinearDiscriminantAnalysis": [
            {"classifier__solver": ["svd"]},
            {"classifier__solver": ["lsqr"], "classifier__shrinkage": [None, "auto"]},
        ],
        "GaussianNB": {"classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
        "KNeighborsClassifier": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "SVC": {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        },
        "DecisionTreeClassifier": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 2, 3, 4, 5, 7, 10],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "RandomForestClassifier": {
            "classifier__n_estimators": [100, 200, 400],
            "classifier__max_depth": [None, 3, 5, 10],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "ExtraTreesClassifier": {
            "classifier__n_estimators": [100, 200, 400],
            "classifier__max_depth": [None, 3, 5, 10],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "GradientBoostingClassifier": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_depth": [1, 2, 3],
            "classifier__subsample": [0.8, 1.0],
        },
        "HistGradientBoostingClassifier": {
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_iter": [100, 200, 300],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__l2_regularization": [0.0, 0.1, 1.0],
        },
        "AdaBoostClassifier": {
            "classifier__n_estimators": [50, 100, 200, 400],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        "RidgeClassifier": {
            "classifier__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "XGBClassifier": {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [3, 6, 10],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
        },
        "LGBMClassifier": {
            "classifier__n_estimators": [100, 300],
            "classifier__num_leaves": [15, 31, 63],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
        },
        "CatBoostClassifier": {
            "classifier__iterations": [200, 500],
            "classifier__depth": [4, 6, 8],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__verbose": [False],
        },
    }
    return grids.get(name)


def get_clusterer_param_grid(name: str):
    grids = {
        "KMeans": {
            "n_clusters": list(range(2, 11)),
            "init": ["k-means++", "random"],
            "n_init": [10, 20],
        },
        "MiniBatchKMeans": {
            "n_clusters": list(range(2, 11)),
            "n_init": [10, 20],
        },
        "AgglomerativeClustering": {
            "n_clusters": list(range(2, 11)),
            "linkage": ["ward", "complete", "average"],
        },
        "DBSCAN": {
            "eps": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],
            "min_samples": [3, 5, 7, 10],
        },
        "GaussianMixture": {
            "n_components": list(range(2, 11)),
            "covariance_type": ["full", "tied", "diag", "spherical"],
            "reg_covar": [1e-6, 1e-4, 1e-2],
        },
        "Birch": {
            "n_clusters": list(range(2, 11)),
            "threshold": [0.3, 0.5, 0.7, 1.0],
            "branching_factor": [25, 50],
        },
        "SpectralClustering": {
            "n_clusters": list(range(2, 8)),
            "affinity": ["rbf", "nearest_neighbors"],
        },
        "MeanShift": {"bandwidth": [None, 0.5, 1.0, 2.0]},
        "AffinityPropagation": {"damping": [0.5, 0.7, 0.9]},
        "OPTICS": {"min_samples": [3, 5, 10], "xi": [0.05, 0.1, 0.2]},
        "HDBSCAN": {
            "min_cluster_size": [5, 10, 20],
            "min_samples": [None, 3, 5, 10],
        },
        "KohonenSOM": {
            "som_x": [3, 5, 7, 10],
            "som_y": [3, 5, 7, 10],
            "sigma": [0.5, 1.0, 2.0],
            "learning_rate": [0.1, 0.5, 1.0],
            "num_iteration": [500, 1000, 2000],
        },
    }
    return grids.get(name)


# ============================================================
# AJUSTE MANUAL UNIVERSAL
# ============================================================

def parse_param_string(value_str: str):
    s = str(value_str).strip()
    low = s.lower()
    if low in {"none", "null"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if s == "":
        return s
    try:
        if "." not in s and "e" not in low:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def format_param_value(value) -> str:
    if value is None:
        return "None"
    return str(value)


def extract_editable_params(estimator, prefix: str) -> list[str]:
    """Extrai todos os parâmetros numéricos/booleanos/string de um estimador."""
    try:
        params = estimator.get_params(deep=True)
    except Exception:
        return []
    editable = []
    skip_keywords = {
        "n_jobs", "random_state", "verbose", "warm_start",
        "copy_X", "fit_intercept", "positive", "normalize",
        "precompute", "selection", "tol", "max_iter",
    }
    for key, value in params.items():
        if not key.startswith(prefix):
            continue
        short = key.split("__", 1)[-1]
        if short in skip_keywords:
            continue
        if isinstance(value, (int, float, bool, str, type(None))):
            editable.append(key)
    return sorted(editable)


def render_manual_params_universal(
    estimator,
    prefix: str,
    current_params: dict,
    key_prefix: str,
) -> dict[str, str]:
    keys = extract_editable_params(estimator, prefix)
    if not keys:
        return {}
    edited: dict[str, str] = {}
    cols = st.columns(min(3, max(1, len(keys))))
    for i, key in enumerate(keys):
        short = key.split("__", 1)[-1] if "__" in key else key
        cur_val = current_params.get(key, "")
        with cols[i % len(cols)]:
            edited[key] = st.text_input(
                short,
                value=format_param_value(cur_val),
                key=f"{key_prefix}_{key}",
            )
    return edited


# ============================================================
# PIPELINES
# ============================================================

def make_regression_pipeline(estimator, X_train, encoding_map, ordinal_orders, manual_maps, outlier_mode, threshold, max_features):
    preprocessor = build_preprocessor(X_train, encoding_map, ordinal_orders, manual_maps, outlier_mode)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("selector", CorrelationSelector(threshold, max_features)),
        ("scaler", StandardScaler()),
        ("regressor", estimator),
    ])


def make_classification_pipeline(estimator, X_train, encoding_map, ordinal_orders, manual_maps):
    preprocessor = build_preprocessor(X_train, encoding_map, ordinal_orders, manual_maps, outlier_mode="keep")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def make_clustering_preprocessor(X, encoding_map, ordinal_orders, manual_maps, outlier_mode="keep"):
    preprocessor = build_preprocessor(X, encoding_map, ordinal_orders, manual_maps, outlier_mode=outlier_mode)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
    ])


# ============================================================
# AVALIAÇÃO REGRESSÃO
# ============================================================

def evaluate_regression(X_train, y_train, X_test, y_test, models, encoding_map, ordinal_orders, manual_maps, outlier_mode, threshold=0.10, max_features=None, cv=5, search_method="grid", n_iter=20, random_state=42):
    results = {}
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    progress = st.progress(0)
    status = st.empty()
    items = list(models.items())
    for idx_model, (name, cfg) in enumerate(items, start=1):
        status.info(f"Treinando **{name}** ({idx_model}/{len(items)})...")
        try:
            pipe = make_regression_pipeline(clone(cfg["estimator"]), X_train, encoding_map, ordinal_orders, manual_maps, outlier_mode, threshold, max_features)
            cv_rows = None
            if cfg["params"]:
                if search_method == "grid":
                    search = GridSearchCV(pipe, cfg["params"], scoring="r2", cv=cv_strategy, n_jobs=-1, return_train_score=True, refit=True, error_score="raise")
                elif search_method == "random":
                    n_random = randomized_iteration_limit(cfg["params"], n_iter)
                    search = RandomizedSearchCV(pipe, cfg["params"], n_iter=n_random, scoring="r2", cv=cv_strategy, n_jobs=-1, return_train_score=True, refit=True, random_state=random_state, error_score="raise")
                else:
                    manual_params = {key: [values[0]] for key, values in cfg["params"].items()}
                    search = GridSearchCV(pipe, manual_params, scoring="r2", cv=cv_strategy, n_jobs=-1, return_train_score=True, refit=True, error_score="raise")
                search.fit(X_train, y_train)
                best = search.best_estimator_
                best_params = search.best_params_
                best_cv = float(search.best_score_)
                best_std = float(search.cv_results_["std_test_score"][search.best_index_])
                idx_best = search.best_index_
                cv_scores = np.asarray([search.cv_results_[f"split{k}_test_score"][idx_best] for k in range(cv)], dtype=float)
                cv_rows = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(15)
            else:
                cv_res = cross_validate(pipe, X_train, y_train, scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}, cv=cv_strategy, n_jobs=-1, error_score="raise")
                best = pipe.fit(X_train, y_train)
                best_params = {}
                best_cv = float(cv_res["test_r2"].mean())
                best_std = float(cv_res["test_r2"].std())
                cv_scores = np.asarray(cv_res["test_r2"], dtype=float)
                cv_rows = None
            prediction = best.predict(X_test)
            mse = float(mean_squared_error(y_test, prediction))
            results[name] = {
                "best_model": best, "best_params": best_params, "cv_mean": best_cv, "cv_std": best_std,
                "cv_scores": cv_scores, "test_r2": float(r2_score(y_test, prediction)),
                "test_rmse": float(np.sqrt(mse)), "test_mae": float(mean_absolute_error(y_test, prediction)),
                "test_mse": mse, "y_pred": prediction, "search_table": cv_rows,
            }
        except Exception as exc:
            results[name] = {"error": str(exc)}
        progress.progress(idx_model / len(items))
    status.success("✅ Regressão concluída!")
    progress.empty()
    return results


def randomized_iteration_limit(params: dict[str, list[Any]], requested: int) -> int:
    if not params:
        return 1
    combinations = 1
    for values in params.values():
        if isinstance(values, list):
            combinations *= max(len(values), 1)
    return max(1, min(int(requested), combinations))


# ============================================================
# AVALIAÇÃO CLASSIFICAÇÃO
# ============================================================

def _format_best_params(params: dict[str, Any]) -> str:
    if not params:
        return "—"
    return ", ".join(f"{k.replace('classifier__', '')}={v}" for k, v in params.items())


def evaluate_classifiers(X_train, y_train, X_test, y_test, classifier_names, encoding_map, ordinal_orders, manual_maps, cv=5, random_state=42, tune=False, search_method="grid", n_iter=20, tuning_metric="f1_weighted"):
    registry = dict(get_classifier_discovery_registry())
    results = {}
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy", "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted", "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy",
    }
    tuning_scorer = scoring[tuning_metric]
    progress = st.progress(0)
    status = st.empty()
    names = list(dict.fromkeys(classifier_names))
    for idx_model, name in enumerate(names, start=1):
        status.info(f"Avaliando **{name}** ({idx_model}/{len(names)})...")
        try:
            if name not in registry:
                raise ValueError(f"Classificador '{name}' não encontrado.")
            estimator = _set_random_state(registry[name](), random_state)
            pipe = make_classification_pipeline(estimator, X_train, encoding_map, ordinal_orders, manual_maps)
            param_grid = get_classifier_param_grid(name) if tune else None
            best_params = {}
            tuning_table = None
            tuned = False
            if tune and param_grid:
                if search_method == "grid":
                    search = GridSearchCV(pipe, param_grid, scoring=tuning_scorer, cv=cv_strategy, n_jobs=-1, refit=True, error_score="raise")
                else:
                    combinations = list(ParameterGrid(param_grid))
                    n_to_draw = min(int(n_iter), len(combinations))
                    search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_to_draw, scoring=tuning_scorer, cv=cv_strategy, n_jobs=-1, refit=True, random_state=random_state, error_score="raise")
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
                tuned = True
                tuning_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(15)
                cv_best = float(search.best_score_)
                cv_best_std = float(search.cv_results_["std_test_score"][search.best_index_])
            else:
                model = pipe.fit(X_train, y_train)
                cv_res = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1, error_score="raise")
                cv_best = float(cv_res[f"test_{tuning_metric}"].mean())
                cv_best_std = float(cv_res[f"test_{tuning_metric}"].std())
            if tuned:
                cv_res = cross_validate(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1, error_score="raise")
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            test_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            test_balanced = balanced_accuracy_score(y_test, y_pred)
            results[name] = {
                "best_model": model, "tuned": tuned, "tuning_method": search_method if tuned else "none",
                "tuning_metric": tuning_metric, "best_params": best_params, "best_params_text": _format_best_params(best_params),
                "tuning_table": tuning_table, "cv_tuning_mean": cv_best, "cv_tuning_std": cv_best_std,
                "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
                "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
                "cv_f1_weighted_mean": float(cv_res["test_f1_weighted"].mean()),
                "cv_f1_weighted_std": float(cv_res["test_f1_weighted"].std()),
                "cv_f1_macro_mean": float(cv_res["test_f1_macro"].mean()),
                "cv_f1_macro_std": float(cv_res["test_f1_macro"].std()),
                "cv_balanced_accuracy_mean": float(cv_res["test_balanced_accuracy"].mean()),
                "cv_acc_scores": cv_res["test_accuracy"].tolist(),
                "cv_f1_scores": cv_res["test_f1_weighted"].tolist(),
                "test_accuracy": float(test_accuracy), "test_precision": float(test_precision),
                "test_recall": float(test_recall), "test_f1": float(test_f1),
                "test_f1_macro": float(test_f1_macro), "test_balanced_accuracy": float(test_balanced),
                "y_pred": y_pred,
            }
            try:
                if hasattr(model, "predict_proba"):
                    score_values = model.predict_proba(X_test)
                    if score_values.shape[1] == 2:
                        results[name]["test_roc_auc"] = float(roc_auc_score(y_test, score_values[:, 1]))
                    else:
                        results[name]["test_roc_auc"] = float(roc_auc_score(y_test, score_values, multi_class="ovr", average="weighted"))
                elif hasattr(model, "decision_function"):
                    score_values = model.decision_function(X_test)
                    if np.ndim(score_values) == 1:
                        results[name]["test_roc_auc"] = float(roc_auc_score(y_test, score_values))
                    else:
                        results[name]["test_roc_auc"] = float(roc_auc_score(y_test, score_values, multi_class="ovr", average="weighted"))
            except Exception:
                results[name]["test_roc_auc"] = None
        except Exception as exc:
            results[name] = {"error": str(exc)}
        progress.progress(idx_model / len(names))
    status.success("✅ Classificação concluída!")
    progress.empty()
    return results


# ============================================================
# AVALIAÇÃO AGRUPAMENTO
# ============================================================

def calculate_external_cluster_metrics(reference, labels):
    reference = pd.Series(reference).astype("string").fillna("missing").reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    if len(reference) != len(labels):
        return {"adjusted_rand": None, "normalized_mutual_info": None}
    try:
        ari = float(adjusted_rand_score(reference, labels))
    except Exception:
        ari = None
    try:
        nmi = float(normalized_mutual_info_score(reference, labels))
    except Exception:
        nmi = None
    return {"adjusted_rand": ari, "normalized_mutual_info": nmi}


def calculate_cluster_metrics(X_array, labels):
    X_array = np.asarray(X_array)
    labels = np.asarray(labels)
    noise = labels == -1
    valid = ~noise
    X_valid = X_array[valid]
    labels_valid = labels[valid]
    unique = np.unique(labels_valid)
    result = {
        "n_clusters": int(len(unique)),
        "n_noise": int(noise.sum()),
        "noise_pct": float(noise.mean() * 100) if len(labels) else 0.0,
        "silhouette": None, "calinski_harabasz": None, "davies_bouldin": None,
    }
    if len(unique) < 2 or len(labels_valid) <= len(unique):
        return result
    for key, fn in [("silhouette", silhouette_score), ("calinski_harabasz", calinski_harabasz_score), ("davies_bouldin", davies_bouldin_score)]:
        try:
            result[key] = float(fn(X_valid, labels_valid))
        except Exception:
            pass
    return result


def cluster_objective(metrics: dict, metric: str):
    value = metrics.get(metric)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def make_cluster_estimator(name, params, random_state=None):
    registry = dict(get_clusterer_discovery_registry())
    registry.setdefault("KMeans", KMeans)
    registry.setdefault("MiniBatchKMeans", MiniBatchKMeans)
    registry.setdefault("AgglomerativeClustering", AgglomerativeClustering)
    registry.setdefault("DBSCAN", DBSCAN)
    registry.setdefault("GaussianMixture", GaussianMixture)
    registry.setdefault("Birch", Birch)
    registry.setdefault("MeanShift", MeanShift)
    registry.setdefault("SpectralClustering", SpectralClustering)
    registry.setdefault("AffinityPropagation", AffinityPropagation)
    registry.setdefault("OPTICS", OPTICS)
    if hdbscan_lib is not None:
        registry.setdefault("HDBSCAN", hdbscan_lib.HDBSCAN)
    if MiniSom is not None:
        registry.setdefault("KohonenSOM", KohonenClusterer)
    if name not in registry:
        raise ValueError(f"Algoritmo de agrupamento desconhecido: {name}")
    try:
        est = registry[name]()
    except Exception as exc:
        raise ValueError(f"Não foi possível instanciar {name}: {exc}")
    try:
        valid = set(est.get_params().keys())
    except Exception:
        valid = set(params.keys())
    filtered = {k: v for k, v in params.items() if k in valid}
    if random_state is not None and "random_state" in valid:
        filtered.setdefault("random_state", random_state)
    try:
        est.set_params(**filtered)
    except Exception as exc:
        raise ValueError(f"Erro ao configurar {name}: {exc}")
    return est


def tune_single_clustering_algorithm(name, X_transformed, tune=True, search_method="grid", n_iter=20, objective="silhouette", random_state=42):
    space = get_clusterer_param_grid(name)
    defaults = {
        "KMeans": {"n_clusters": 3, "init": "k-means++", "n_init": 10},
        "MiniBatchKMeans": {"n_clusters": 3, "n_init": 10},
        "AgglomerativeClustering": {"n_clusters": 3, "linkage": "ward"},
        "DBSCAN": {"eps": 0.5, "min_samples": 5},
        "GaussianMixture": {"n_components": 3, "covariance_type": "full", "reg_covar": 1e-6},
        "Birch": {"n_clusters": 3, "threshold": 0.5, "branching_factor": 50},
        "MeanShift": {},
        "SpectralClustering": {"n_clusters": 3},
        "AffinityPropagation": {"damping": 0.9},
        "OPTICS": {"min_samples": 5},
        "HDBSCAN": {"min_cluster_size": 5},
        "KohonenSOM": {"som_x": 5, "som_y": 5, "sigma": 1.0, "learning_rate": 0.5, "num_iteration": 1000},
    }
    if not tune or space is None:
        candidates = [defaults.get(name, {})]
    elif search_method == "grid":
        try:
            candidates = list(ParameterGrid(space))
        except Exception:
            candidates = [defaults.get(name, {})]
    else:
        try:
            total = max(1, len(list(ParameterGrid(space))))
        except Exception:
            total = 1
        candidates = list(ParameterSampler(space, n_iter=min(int(n_iter), total), random_state=random_state))
    rows = []
    best = None
    best_value = np.inf if objective == "davies_bouldin" else -np.inf
    for params in candidates:
        try:
            model = make_cluster_estimator(name, params, random_state)
            labels = model.fit_predict(X_transformed)
            metrics = calculate_cluster_metrics(X_transformed, labels)
            value = cluster_objective(metrics, objective)
            if value is None:
                continue
            row = dict(params)
            row.update({
                "n_clusters_real": metrics["n_clusters"], "noise_pct": metrics["noise_pct"],
                "silhouette": metrics["silhouette"], "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"], "objective": value,
            })
            rows.append(row)
            improved = value < best_value if objective == "davies_bouldin" else value > best_value
            if improved:
                best_value = value
                best = {"estimator": model, "labels": np.asarray(labels), "metrics": metrics, "best_params": params}
        except Exception:
            continue
    if best is None:
        params = defaults.get(name, {})
        model = make_cluster_estimator(name, params, random_state)
        labels = model.fit_predict(X_transformed)
        best = {"estimator": model, "labels": np.asarray(labels), "metrics": calculate_cluster_metrics(X_transformed, labels), "best_params": params}
    tuning_table = pd.DataFrame(rows)
    if not tuning_table.empty:
        tuning_table = tuning_table.sort_values("objective", ascending=(objective == "davies_bouldin")).head(20)
    best["tuned"] = bool(tune)
    best["tuning_method"] = search_method if tune else "none"
    best["tuning_metric"] = objective
    best["tuning_table"] = tuning_table
    return best


def fit_clustering_models(X, encoding_map, ordinal_orders, manual_maps, selected_algorithms, n_clusters=3, eps=0.5, min_samples=5, outlier_mode="keep", random_state=42, reference=None, tune=False, search_method="grid", n_iter=20, objective="silhouette"):
    pipeline = make_clustering_preprocessor(X, encoding_map, ordinal_orders, manual_maps, outlier_mode)
    X_transformed = pipeline.fit_transform(X)
    results = {}
    for name in selected_algorithms:
        try:
            if tune:
                result = tune_single_clustering_algorithm(name, X_transformed, tune=True, search_method=search_method, n_iter=n_iter, objective=objective, random_state=random_state)
            else:
                defaults = {
                    "KMeans": {"n_clusters": n_clusters, "init": "k-means++", "n_init": 10},
                    "MiniBatchKMeans": {"n_clusters": n_clusters, "n_init": 10},
                    "AgglomerativeClustering": {"n_clusters": n_clusters, "linkage": "ward"},
                    "DBSCAN": {"eps": eps, "min_samples": min_samples},
                    "GaussianMixture": {"n_components": n_clusters, "covariance_type": "full", "reg_covar": 1e-6},
                    "Birch": {"n_clusters": n_clusters, "threshold": 0.5, "branching_factor": 50},
                    "MeanShift": {},
                    "SpectralClustering": {"n_clusters": n_clusters},
                    "AffinityPropagation": {"damping": 0.9},
                    "OPTICS": {"min_samples": min_samples},
                    "HDBSCAN": {"min_cluster_size": min_samples},
                    "KohonenSOM": {"som_x": 5, "som_y": 5, "sigma": 1.0, "learning_rate": 0.5, "num_iteration": 1000},
                }
                params = defaults.get(name, {})
                model = make_cluster_estimator(name, params, random_state)
                labels = model.fit_predict(X_transformed)
                result = {
                    "estimator": model, "labels": np.asarray(labels),
                    "metrics": calculate_cluster_metrics(X_transformed, labels),
                    "best_params": params, "tuned": False, "tuning_method": "none",
                    "tuning_metric": objective, "tuning_table": None,
                }
            if reference is not None:
                result["metrics"].update(calculate_external_cluster_metrics(reference, result["labels"]))
            results[name] = result
        except Exception as exc:
            results[name] = {"error": str(exc)}
    return X_transformed, results


def clustering_scatter_plot(X_transformed, labels, title):
    X_transformed = np.asarray(X_transformed)
    labels = np.asarray(labels)
    if X_transformed.shape[1] == 1:
        coords = np.column_stack([X_transformed[:, 0], np.zeros(len(X_transformed))])
    elif X_transformed.shape[1] == 2:
        coords = X_transformed[:, :2]
    else:
        coords = PCA(n_components=2, random_state=0).fit_transform(X_transformed)
    plot_df = pd.DataFrame({"Componente 1": coords[:, 0], "Componente 2": coords[:, 1], "Cluster": labels.astype(str)})
    return px.scatter(plot_df, x="Componente 1", y="Componente 2", color="Cluster", title=title)


# ============================================================
# GRÁFICOS
# ============================================================

def regression_boxplot(valid):
    rows = []
    for name, result in valid.items():
        cv_scores = result.get("cv_scores")
        if cv_scores is None:
            continue
        cv_scores = np.asarray(cv_scores).ravel()
        if cv_scores.size == 0:
            continue
        for fold, score in enumerate(cv_scores, start=1):
            rows.append({"Modelo": name, "Fold": fold, "R²": float(score)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df
    fig = px.box(df, x="Modelo", y="R²", color="Modelo", points="all", hover_data=["Fold"], title="📦 R² por dobra da validação cruzada")
    fig.update_layout(height=500, showlegend=False)
    return fig, df


def regression_scatter_plot(y_test, valid):
    rows = []
    for name, result in valid.items():
        pred = np.asarray(result["y_pred"])
        real = np.asarray(y_test)
        resid = real - pred
        for y_real, y_pred, residual in zip(real, pred, resid):
            rows.append({"Modelo": name, "Real": y_real, "Predito": y_pred, "Resíduo": residual})
    df = pd.DataFrame(rows)
    if df.empty:
        return None, None
    fig_real = px.scatter(df, x="Real", y="Predito", color="Modelo", facet_col="Modelo", facet_col_wrap=2, title="Real × Predito", hover_data=["Resíduo"])
    fig_resid = px.scatter(df, x="Predito", y="Resíduo", color="Modelo", facet_col="Modelo", facet_col_wrap=2, title="Resíduos")
    fig_resid.add_hline(y=0, line_dash="dash")
    return fig_real, fig_resid


def classification_cv_plot(valid):
    rows = []
    for name, result in valid.items():
        cv_scores = result.get("cv_f1_scores")
        if cv_scores is None:
            continue
        cv_scores = np.asarray(cv_scores).ravel()
        if cv_scores.size == 0:
            continue
        for fold, score in enumerate(cv_scores, start=1):
            rows.append({"Classificador": name, "Fold": fold, "F1 ponderado": float(score)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df
    fig = px.box(df, x="Classificador", y="F1 ponderado", color="Classificador", points="all", hover_data=["Fold"], title="📦 F1 ponderado por dobra da CV")
    fig.update_layout(height=550, showlegend=False, xaxis_tickangle=-40)
    return fig, df


def confusion_matrix_plot(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, x=classes, y=classes, labels={"x": "Predito", "y": "Real", "color": "Quantidade"}, title=title, aspect="auto")
    return fig


# ============================================================
# INFERÊNCIA
# ============================================================

def predict_new_classification(model, new_df, features, classes):
    missing = [col for col in features if col not in new_df.columns]
    if missing:
        raise ValueError("O novo dataset não possui as variáveis necessárias: " + ", ".join(missing))
    X_new = new_df[features].copy()
    prediction = model.predict(X_new)
    labels = []
    for value in prediction:
        try:
            idx = int(value)
            labels.append(classes[idx] if 0 <= idx < len(classes) else str(value))
        except (TypeError, ValueError):
            labels.append(str(value))
    result = new_df.copy()
    result["Classe_Prevista"] = labels
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)
            result["Confianca"] = proba.max(axis=1)
    except Exception:
        pass
    return result


# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_regression_target(y):
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().all():
        raise ValueError("A variável alvo escolhida para regressão não é numérica.")
    if y_num.isna().any():
        st.warning("Há valores inválidos na variável alvo. As linhas com alvo ausente ou não numérico serão removidas antes do treino.")
    y_num = y_num.dropna()
    if len(y_num) < 10:
        raise ValueError("A regressão requer pelo menos 10 observações válidas.")
    return y_num


def validate_classification_target(y, cv):
    y_clean = y.astype("string").fillna("missing")
    counts = y_clean.value_counts()
    if len(counts) < 2:
        raise ValueError("A classificação requer pelo menos 2 classes.")
    if counts.min() < cv:
        raise ValueError(f"A classe com menos observações possui {int(counts.min())} amostra(s), mas a CV exige pelo menos {cv}.")
    return y_clean


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0;">🧠 Análise de Inteligência Computacional — Completa</h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">
            Regressão, classificação, agrupamento (incl. Mapa de Kohonen),
            pré-análise, recomendações, todos os algoritmos e ajuste manual universal.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuração")
    task_options = {"Regressão": "regression", "Classificação": "classification", "Agrupamento": "clustering"}
    task_label = st.radio("Tipo de tarefa", list(task_options.keys()), horizontal=False)
    st.session_state["task_type"] = task_options[task_label]
    st.divider()
    st.subheader("📁 Fonte de dados")
    source = st.radio("Origem", ["Arquivo Local", "UCI Repository"])
    uploaded_file = None
    uci_id = None
    if source == "Arquivo Local":
        uploaded_file = st.file_uploader("Envie o arquivo", type=["csv", "xlsx", "xls", "json"])
    else:
        uci_id = st.number_input("ID do Dataset (UCI)", min_value=1, value=186, step=1)
    if st.button("📥 Carregar Dados", width="stretch"):
        with st.spinner("Carregando dados..."):
            try:
                df = load_dataframe(uploaded_file, "file") if source == "Arquivo Local" else load_dataframe(None, "UCI", int(uci_id))
                if df.empty:
                    raise ValueError("O dataset carregado está vazio.")
                st.session_state["df_raw"] = df
                st.session_state["encoding_config"] = None
                st.session_state["encoding_preview"] = None
                st.session_state["encoding_ready"] = False
                st.session_state["results"] = None
                st.session_state["prediction_result"] = None
                st.session_state["clustering_result"] = None
                st.session_state["pre_analysis_reg"] = None
                st.session_state["pre_analysis_clf"] = None
                st.session_state["pre_analysis_cluster"] = None
                st.success(f"✅ Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas.")
            except Exception as exc:
                st.error(f"Erro ao carregar dados: {exc}")

if st.session_state["df_raw"] is None:
    st.info("👈 Use a barra lateral para carregar um conjunto de dados.")
    st.stop()

df_raw = st.session_state["df_raw"]

tabs = st.tabs(["📊 Diagnóstico", "🏷️ Codificação", "🎯 Modelagem", "📈 Resultados", "📚 Explicações"])

# TAB 1 — DIAGNÓSTICO
with tabs[0]:
    st.header("📊 Descrição dos Dados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", df_raw.shape[0])
    c2.metric("Colunas", df_raw.shape[1])
    c3.metric("Numéricas", len(df_raw.select_dtypes(include=np.number).columns))
    c4.metric("Categóricas", len(detect_categorical_columns(df_raw)))
    info_rows = []
    for col in df_raw.columns:
        info_rows.append({
            "Coluna": col, "dtype": str(df_raw[col].dtype),
            "Não-nulos": int(df_raw[col].notna().sum()), "Nulos": int(df_raw[col].isna().sum()),
            "Únicos": int(df_raw[col].nunique(dropna=True)),
            "Amostra": " | ".join(df_raw[col].dropna().astype(str).head(3).tolist())[:80],
        })
    st.subheader("Tipos de dados por coluna")
    st.dataframe(pd.DataFrame(info_rows), width="stretch", height=320)
    with st.expander("📋 Estatísticas descritivas", expanded=True):
        st.dataframe(df_raw.describe(include="all").T, width="stretch")
    with st.expander("🔍 Outliers IQR — diagnóstico"):
        outlier_df = iqr_outlier_summary(df_raw)
        if outlier_df.empty:
            st.info("Nenhuma coluna numérica disponível.")
        else:
            st.dataframe(outlier_df.round(4), width="stretch")
    st.subheader("📈 Visualizações")
    numeric_cols = list(df_raw.select_dtypes(include=np.number).columns)
    if numeric_cols:
        selected_hist = st.multiselect("Colunas para histogramas", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        if selected_hist:
            hist_long = df_raw[selected_hist].melt(var_name="Variável", value_name="Valor")
            fig = px.histogram(hist_long, x="Valor", facet_col="Variável", facet_col_wrap=2, marginal="box", title="Distribuições numéricas")
            st.plotly_chart(fig, width="stretch")
    if len(numeric_cols) >= 2:
        corr = df_raw[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=len(numeric_cols) <= 12, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matriz de correlação")
        st.plotly_chart(fig_corr, width="stretch")

# TAB 2 — CODIFICAÇÃO
with tabs[1]:
    st.header("🏷️ Configuração de Variáveis Categóricas")
    st.caption("A tabela abaixo é apenas uma prévia. O treinamento real usa ColumnTransformer/Pipeline.")
    categorical_cols = detect_categorical_columns(df_raw)
    if not categorical_cols:
        st.success("Nenhuma coluna categórica detectada. Todas as colunas serão tratadas como numéricas.")
        st.session_state["encoding_config"] = {"encoding_map": {}, "ordinal_orders": {}, "manual_maps": {}}
        st.session_state["encoding_preview"] = df_raw.copy()
        st.session_state["encoding_ready"] = True
    else:
        encoding_map, ordinal_orders, manual_maps = {}, {}, {}
        st.write(f"**{len(categorical_cols)} coluna(s) categórica(s) detectada(s).**")
        for col in categorical_cols:
            with st.expander(f"🔸 {col}", expanded=False):
                values = df_raw[col].dropna().astype(str).drop_duplicates().tolist()
                values_sorted = sorted(values)
                a, b, c = st.columns(3)
                a.metric("Únicos", len(values_sorted))
                b.metric("Ausentes", int(df_raw[col].isna().sum()))
                c.metric("dtype", str(df_raw[col].dtype))
                st.caption("Primeiros valores")
                st.code(" | ".join(values_sorted[:50]) or "(sem valores)", language=None)
                method = st.radio(
                    "Método", ["onehot", "label", "ordinal", "manual", "drop"],
                    format_func=lambda x: {
                        "onehot": "One-Hot — recomendado para categorias nominais",
                        "label": "Label — compatibilidade; impõe ordem numérica",
                        "ordinal": "Ordinal — use quando existe ordem real",
                        "manual": "Manual — mapeamento próprio",
                        "drop": "Descartar coluna",
                    }[x], key=f"encoding_{col}",
                )
                encoding_map[col] = method
                if method == "ordinal":
                    order_text = st.text_input("Ordem (separada por vírgula)", value=", ".join(values_sorted), key=f"ordinal_{col}")
                    ordinal_orders[col] = [v.strip() for v in order_text.split(",") if v.strip()]
                elif method == "manual":
                    suggestion = ", ".join(f"{v}={i}" for i, v in enumerate(values_sorted))
                    map_text = st.text_input("Mapeamento categoria=código", value=suggestion, key=f"manual_{col}")
                    mapping, invalid_tokens = {}, []
                    for token in map_text.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        if "=" not in token:
                            invalid_tokens.append(token)
                            continue
                        key, value = token.split("=", 1)
                        try:
                            mapping[key.strip()] = float(value.strip())
                        except ValueError:
                            invalid_tokens.append(token)
                    if invalid_tokens:
                        st.warning("Entradas ignoradas no mapeamento: " + ", ".join(invalid_tokens))
                    manual_maps[col] = mapping
        if st.button("✅ Salvar configuração de codificação", width="stretch", type="primary"):
            try:
                preview, report = apply_categorical_encoding_preview(df_raw, encoding_map, ordinal_orders, manual_maps)
                st.session_state["encoding_config"] = {"encoding_map": encoding_map, "ordinal_orders": ordinal_orders, "manual_maps": manual_maps}
                st.session_state["encoding_preview"] = preview
                st.session_state["encoding_ready"] = True
                st.success("✅ Configuração salva. As transformações serão ajustadas automaticamente em cada treino/CV.")
                st.dataframe(pd.DataFrame(report, columns=["Coluna", "Método", "Resultado"]), width="stretch")
                with st.expander("🔎 Prévia codificada"):
                    st.dataframe(preview.head(100), width="stretch")
            except Exception as exc:
                st.error(f"Erro na configuração: {exc}")

# TAB 3 — MODELAGEM
with tabs[2]:
    st.header("🎯 Configuração e Execução")
    if not st.session_state["encoding_ready"]:
        st.warning("Configure a codificação na aba anterior.")
    else:
        encoding_config = st.session_state["encoding_config"] or {}
        encoding_map = encoding_config.get("encoding_map", {})
        ordinal_orders = encoding_config.get("ordinal_orders", {})
        manual_maps = encoding_config.get("manual_maps", {})
        all_columns = list(df_raw.columns)

        # REGRESSÃO
        if st.session_state["task_type"] == "regression":
            c1, c2 = st.columns([1, 2])
            with c1:
                target = st.selectbox("🎯 Variável alvo", all_columns, index=len(all_columns) - 1, key="target_selection_reg")
            with c2:
                features = st.multiselect("📊 Variáveis preditoras", [c for c in all_columns if c != target], default=[c for c in all_columns if c != target][:10], key="feature_selection_reg")
            if not features:
                st.info("Selecione pelo menos uma variável preditora.")
            else:
                with st.expander("🔍 Pré-análise e recomendações", expanded=False):
                    if st.button("Analisar dados", key="analyze_reg"):
                        try:
                            analysis = analyze_dataset(df_raw, features, target, "regression")
                            recs = recommend_algorithms(analysis, "regression")
                            st.session_state["pre_analysis_reg"] = {"analysis": analysis, "recommendations": recs}
                        except Exception as exc:
                            st.error(f"Erro na pré-análise: {exc}")
                    pre_reg = st.session_state.get("pre_analysis_reg")
                    if pre_reg:
                        _render_pre_analysis(pre_reg["analysis"], pre_reg["recommendations"])
                st.subheader("⚙️ Parâmetros de Regressão")
                c1, c2, c3 = st.columns(3)
                with c1:
                    outlier_mode = st.selectbox("Tratamento de outliers numéricos", ["keep", "winsorize", "median"], format_func=lambda x: {"keep": "Manter", "winsorize": "Winsorizar", "median": "Substituir por mediana"}[x])
                with c2:
                    threshold = st.number_input("Threshold de correlação", 0.0, 1.0, 0.10, 0.05)
                with c3:
                    max_feat_input = st.number_input("Máx. features (0 = todas)", 0, 200, 0)
                st.markdown("**Catálogo de modelos**")
                c1, c2 = st.columns([1, 1])
                with c1:
                    catalog_reg = st.selectbox("Catálogo", ["Recomendados (pré-análise)", "Lista curada", "Todos os regressores"], key="catalog_reg")
                with c2:
                    tune_reg = st.checkbox("🔧 Otimizar hiperparâmetros", value=True, key="tune_reg")
                if catalog_reg == "Recomendados (pré-análise)":
                    pre_reg = st.session_state.get("pre_analysis_reg")
                    available_reg = [name for name, _ in pre_reg["recommendations"]] if pre_reg else CURATED_REGRESSORS
                elif catalog_reg == "Todos os regressores":
                    available_reg = get_all_regressors()
                else:
                    available_reg = CURATED_REGRESSORS
                chosen_reg = st.multiselect("Regressores para avaliar", available_reg, default=available_reg[:min(8, len(available_reg))], key="chosen_regressors")
                st.markdown("**Busca de hiperparâmetros**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    search_method = st.selectbox("Método", ["grid", "random", "manual"], format_func=lambda x: {"grid": "Grid Search", "random": "Random Search", "manual": "Manual — primeiro valor"}[x])
                    n_iter = st.number_input("n_iter (random)", 2, 500, 20)
                    cv_reg = st.number_input("Folds da CV", 3, 10, 5)
                with c2:
                    seed_choice = st.radio("Semente", ["fixa", "aleatoria", "custom"], format_func=lambda x: {"fixa": "Fixa (42)", "aleatoria": "Aleatória", "custom": "Customizada"}[x], horizontal=True, key="seed_reg_v3")
                    custom_seed = st.number_input("Valor", 0, 999999, 42, disabled=seed_choice != "custom", key="custom_seed_reg_v3")
                if st.button("▶️ Executar Análise de Regressão", width="stretch", type="primary"):
                    try:
                        validate_feature_selection(df_raw, features, target)
                        if not chosen_reg:
                            raise ValueError("Selecione pelo menos um regressor.")
                        data = df_raw[features + [target]].copy()
                        y = pd.to_numeric(data[target], errors="coerce")
                        valid_mask = y.notna()
                        X = data.loc[valid_mask, features].copy()
                        y = y.loc[valid_mask].copy()
                        if len(y) < 10:
                            raise ValueError("Poucas observações válidas para regressão.")
                        registry = get_regressor_discovery_registry()
                        models = {}
                        for name in chosen_reg:
                            if name not in registry:
                                continue
                            try:
                                est = registry[name]()
                            except Exception:
                                continue
                            est = _set_random_state(est, DEFAULT_RANDOM_STATE)
                            grid = get_regressor_param_grid(name) if tune_reg else None
                            models[name] = {"estimator": est, "params": grid or {}}
                        if not models:
                            raise ValueError("Nenhum regressor pôde ser instanciado.")
                        seed = resolve_random_state(seed_choice, int(custom_seed))
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
                        results = evaluate_regression(X_train, y_train, X_test, y_test, models, encoding_map, ordinal_orders, manual_maps, outlier_mode, threshold=threshold, max_features=int(max_feat_input) if max_feat_input > 0 else None, cv=int(cv_reg), search_method=search_method, n_iter=int(n_iter), random_state=seed)
                        st.session_state["results"] = {
                            "task": "regression", "results": results, "y_test": y_test,
                            "X_train": X_train, "X_test": X_test, "y_train": y_train,
                            "features": features, "target": target, "train_n": len(X_train),
                            "test_n": len(X_test), "seed": seed, "cv": int(cv_reg),
                            "encoding_config": encoding_config, "outlier_mode": outlier_mode,
                            "threshold": float(threshold),
                            "max_features": int(max_feat_input) if max_feat_input > 0 else None,
                        }
                        st.success("🎉 Regressão concluída. Abra a aba **Resultados**.")
                    except Exception as exc:
                        st.error(f"Não foi possível executar a regressão: {exc}")

        # CLASSIFICAÇÃO
        elif st.session_state["task_type"] == "classification":
            c1, c2 = st.columns([1, 2])
            with c1:
                target = st.selectbox("🎯 Variável alvo", all_columns, index=len(all_columns) - 1, key="target_selection_clf")
            with c2:
                features = st.multiselect("📊 Variáveis preditoras", [c for c in all_columns if c != target], default=[c for c in all_columns if c != target][:10], key="feature_selection_clf")
            if not features:
                st.info("Selecione pelo menos uma variável preditora.")
            else:
                with st.expander("🔍 Pré-análise e recomendações", expanded=False):
                    if st.button("Analisar dados", key="analyze_clf"):
                        try:
                            analysis = analyze_dataset(df_raw, features, target, "classification")
                            recs = recommend_algorithms(analysis, "classification")
                            st.session_state["pre_analysis_clf"] = {"analysis": analysis, "recommendations": recs}
                        except Exception as exc:
                            st.error(f"Erro na pré-análise: {exc}")
                    pre_clf = st.session_state.get("pre_analysis_clf")
                    if pre_clf:
                        _render_pre_analysis(pre_clf["analysis"], pre_clf["recommendations"])
                st.subheader("⚙️ Parâmetros de Classificação")
                c1, c2, c3 = st.columns(3)
                with c1:
                    cv_clf = st.number_input("Folds da CV", 3, 10, 5)
                with c2:
                    test_size_clf = st.number_input("Fração de teste", 0.10, 0.40, 0.20, 0.05)
                with c3:
                    catalog = st.selectbox("Catálogo", ["Recomendados (pré-análise)", "Lista recomendada", "Todos os classificadores"], key="catalog_clf")
                if catalog == "Recomendados (pré-análise)":
                    pre_clf = st.session_state.get("pre_analysis_clf")
                    classifiers = [name for name, _ in pre_clf["recommendations"]] if pre_clf else CURATED_CLASSIFIERS
                elif catalog == "Lista recomendada":
                    classifiers = CURATED_CLASSIFIERS
                else:
                    classifiers = get_all_classifiers()
                h1, h2, h3 = st.columns(3)
                with h1:
                    tune_clf = st.checkbox("🔧 Otimizar hiperparâmetros", value=True, key="tune_clf_v4")
                with h2:
                    tuning_method_clf = st.selectbox("Busca", ["grid", "random"], format_func=lambda x: "Grid Search" if x == "grid" else "Random Search", disabled=not tune_clf, key="tuning_method_clf_v4")
                with h3:
                    n_iter_clf = st.number_input("n_iter (Random)", 2, 100, 20, 1, disabled=(not tune_clf or tuning_method_clf != "random"), key="n_iter_clf_v4")
                tuning_metric_clf = st.selectbox("Métrica de seleção", ["f1_weighted", "f1_macro", "accuracy", "balanced_accuracy"], format_func=lambda x: {"f1_weighted": "F1 ponderado", "f1_macro": "F1 macro", "accuracy": "Acurácia", "balanced_accuracy": "Balanced accuracy"}[x], disabled=not tune_clf, key="tuning_metric_clf_v4")
                seed_choice = st.radio("Semente", ["fixa", "aleatoria", "custom"], format_func=lambda x: {"fixa": "Fixa (42)", "aleatoria": "Aleatória", "custom": "Customizada"}[x], horizontal=True, key="seed_clf_v3")
                custom_seed_clf = st.number_input("Semente customizada", 0, 999999, 42, disabled=seed_choice != "custom", key="custom_seed_clf_v3")
                chosen = st.multiselect("Classificadores para avaliar", classifiers, default=classifiers[:min(12, len(classifiers))], key="chosen_classifiers_v3")
                if st.button("▶️ Executar Classificação", width="stretch", type="primary"):
                    try:
                        validate_feature_selection(df_raw, features, target)
                        data = df_raw[features + [target]].copy()
                        y_raw = validate_classification_target(data[target], int(cv_clf))
                        data = data.loc[y_raw.index]
                        X, y = data[features], y_raw
                        if not chosen:
                            raise ValueError("Selecione pelo menos um classificador.")
                        class_counts = y.value_counts()
                        if class_counts.min() < 2:
                            raise ValueError("Cada classe precisa ter pelo menos 2 observações.")
                        seed = resolve_random_state(seed_choice, int(custom_seed_clf))
                        X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=float(test_size_clf), random_state=seed, stratify=y)
                        classes = sorted(y_train_raw.astype(str).unique().tolist())
                        if not set(y_test_raw.astype(str)).issubset(set(classes)):
                            raise ValueError("Há uma classe presente no teste que não aparece no treino.")
                        label_to_int = {label: i for i, label in enumerate(classes)}
                        y_train = y_train_raw.astype(str).map(label_to_int).to_numpy()
                        y_test = y_test_raw.astype(str).map(label_to_int).to_numpy()
                        results = evaluate_classifiers(X_train, y_train, X_test, y_test, chosen, encoding_map, ordinal_orders, manual_maps, cv=int(cv_clf), random_state=seed, tune=bool(tune_clf), search_method=tuning_method_clf, n_iter=int(n_iter_clf), tuning_metric=tuning_metric_clf)
                        st.session_state["results"] = {
                            "task": "classification", "results": results, "y_test": y_test,
                            "X_train": X_train, "X_test": X_test, "y_train": y_train,
                            "features": features, "target": target, "classes": classes,
                            "train_n": len(X_train), "test_n": len(X_test), "seed": seed,
                            "cv": int(cv_clf), "encoding_config": encoding_config,
                        }
                        st.session_state["prediction_result"] = None
                        st.success("🎉 Classificação concluída. Abra a aba **Resultados**.")
                    except Exception as exc:
                        st.error(f"Não foi possível executar a classificação: {exc}")

                # Inferência
                trained = st.session_state.get("results")
                if trained and trained.get("task") == "classification":
                    st.divider()
                    st.subheader("🔮 Classificar novos dados")
                    valid_models = {name: res for name, res in trained["results"].items() if "error" not in res}
                    if valid_models:
                        model_name = st.selectbox("Modelo treinado", list(valid_models.keys()), key="prediction_model_v3")
                        new_file = st.file_uploader("Novo dataset", type=["csv", "xlsx", "xls", "json"], key="new_classification_file_v3")
                        if new_file is not None:
                            try:
                                new_data = load_dataframe(new_file, "file")
                                missing = [col for col in trained["features"] if col not in new_data.columns]
                                if missing:
                                    st.error("Variáveis obrigatórias ausentes: " + ", ".join(missing))
                                else:
                                    st.dataframe(new_data[trained["features"]].head(20), width="stretch")
                                    if st.button("🔮 Classificar arquivo", type="primary", width="stretch", key="classify_new_file_v3"):
                                        predicted = predict_new_classification(valid_models[model_name]["best_model"], new_data, trained["features"], trained["classes"])
                                        st.session_state["prediction_result"] = predicted
                                        st.success(f"✅ {len(predicted)} registro(s) classificados usando {model_name}.")
                            except Exception as exc:
                                st.error(f"Não foi possível classificar o novo arquivo: {exc}")
                        if st.session_state.get("prediction_result") is not None:
                            st.markdown("### Resultado da previsão")
                            st.dataframe(st.session_state["prediction_result"], width="stretch", height=350)
                            st.download_button("📥 Baixar previsões (.csv)", st.session_state["prediction_result"].to_csv(index=False).encode("utf-8"), file_name=f"classificacoes_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv", width="stretch", key="download_new_classification_v3")

        # AGRUPAMENTO
        else:
            st.subheader("🧩 Agrupamento não supervisionado")
            cluster_features = st.multiselect("📊 Variáveis para agrupamento", all_columns, default=all_columns, key="cluster_features_v3")
            reference_options = ["Nenhuma (sem rótulo)"] + all_columns
            reference_target = st.selectbox("🏷️ Rótulo de referência (opcional)", reference_options, key="cluster_reference_v3")
            if cluster_features:
                with st.expander("🔍 Pré-análise e recomendações", expanded=False):
                    if st.button("Analisar dados", key="analyze_cluster"):
                        try:
                            analysis = analyze_dataset(df_raw, cluster_features, None, "clustering")
                            recs = recommend_algorithms(analysis, "clustering")
                            st.session_state["pre_analysis_cluster"] = {"analysis": analysis, "recommendations": recs}
                        except Exception as exc:
                            st.error(f"Erro na pré-análise: {exc}")
                    pre_clu = st.session_state.get("pre_analysis_cluster")
                    if pre_clu:
                        _render_pre_analysis(pre_clu["analysis"], pre_clu["recommendations"])
            c1, c2, c3 = st.columns(3)
            with c1:
                n_clusters = st.number_input("Número de clusters", 2, 15, 3, 1, key="cluster_n_v4")
                cluster_outlier = st.selectbox("Tratamento de outliers numéricos", ["keep", "winsorize", "median"], format_func=lambda x: {"keep": "Manter", "winsorize": "Winsorizar", "median": "Substituir por mediana"}[x], key="cluster_outlier_v4")
            with c2:
                eps = st.number_input("DBSCAN — eps", 0.01, 10.0, 0.50, 0.05, key="dbscan_eps_v4")
                min_samples = st.number_input("DBSCAN — min_samples", 2, 50, 5, 1, key="dbscan_min_samples_v4")
            with c3:
                seed_choice = st.radio("Semente", ["fixa", "aleatoria", "custom"], format_func=lambda x: {"fixa": "Fixa (42)", "aleatoria": "Aleatória", "custom": "Customizada"}[x], horizontal=True, key="cluster_seed_v4")
                custom_seed = st.number_input("Semente customizada", 0, 999999, 42, disabled=seed_choice != "custom", key="cluster_custom_seed_v4")
            st.markdown("**Catálogo de algoritmos**")
            c1, c2 = st.columns([1, 1])
            with c1:
                catalog_cluster = st.selectbox("Catálogo", ["Recomendados (pré-análise)", "Lista curada", "Todos os clusterers"], key="catalog_cluster")
            with c2:
                tune_cluster = st.checkbox("🔧 Otimizar hiperparâmetros", value=True, key="tune_cluster_v4")
            if catalog_cluster == "Recomendados (pré-análise)":
                pre_clu = st.session_state.get("pre_analysis_cluster")
                available_cluster = [name for name, _ in pre_clu["recommendations"]] if pre_clu else CURATED_CLUSTERERS
            elif catalog_cluster == "Todos os clusterers":
                available_cluster = get_all_clusterers()
            else:
                available_cluster = CURATED_CLUSTERERS
            algorithms = st.multiselect("Algoritmos de agrupamento", available_cluster, default=available_cluster[:min(5, len(available_cluster))], key="cluster_algorithms_v4")
            t1, t2, t3 = st.columns(3)
            with t1:
                tuning_method_cluster = st.selectbox("Busca", ["grid", "random"], format_func=lambda x: "Grid Search" if x == "grid" else "Random Search", disabled=not tune_cluster, key="tuning_method_cluster_v4")
            with t2:
                n_iter_cluster = st.number_input("n_iter (Random)", 2, 100, 20, 1, disabled=(not tune_cluster or tuning_method_cluster != "random"), key="n_iter_cluster_v4")
            with t3:
                tuning_metric_cluster = st.selectbox("Métrica de seleção", ["silhouette", "calinski_harabasz", "davies_bouldin"], format_func=lambda x: {"silhouette": "Silhouette (maior melhor)", "calinski_harabasz": "Calinski-Harabasz (maior melhor)", "davies_bouldin": "Davies-Bouldin (menor melhor)"}[x], disabled=not tune_cluster, key="tuning_metric_cluster_v4")
            if st.button("▶️ Executar Agrupamento", width="stretch", type="primary", key="run_cluster_v3"):
                try:
                    if len(cluster_features) < 2:
                        raise ValueError("Selecione pelo menos duas variáveis.")
                    if not algorithms:
                        raise ValueError("Selecione pelo menos um algoritmo.")
                    reference = None if reference_target == "Nenhuma (sem rótulo)" else df_raw[reference_target]
                    seed = resolve_random_state(seed_choice, int(custom_seed))
                    X_cluster = df_raw[cluster_features].copy()
                    X_transformed, cluster_results = fit_clustering_models(X_cluster, encoding_map, ordinal_orders, manual_maps, algorithms, n_clusters=int(n_clusters), eps=float(eps), min_samples=int(min_samples), outlier_mode=cluster_outlier, random_state=seed, reference=reference, tune=bool(tune_cluster), search_method=tuning_method_cluster, n_iter=int(n_iter_cluster), objective=tuning_metric_cluster)
                    state = {
                        "task": "clustering", "results": cluster_results, "features": cluster_features,
                        "transformed": X_transformed, "source_data": X_cluster, "seed": seed,
                        "n_clusters": int(n_clusters),
                        "reference_target": None if reference_target == "Nenhuma (sem rótulo)" else reference_target,
                    }
                    st.session_state["clustering_result"] = state
                    st.session_state["results"] = state
                    st.success("🎉 Agrupamento concluído. Veja também a aba **Resultados**.")
                except Exception as exc:
                    st.error(f"Não foi possível executar o agrupamento: {exc}")
            cluster_state = st.session_state.get("clustering_result")
            if cluster_state:
                valid_cluster = {name: res for name, res in cluster_state["results"].items() if "error" not in res}
                if valid_cluster:
                    rows = []
                    for name, res in valid_cluster.items():
                        m = res["metrics"]
                        rows.append({"Algoritmo": name, "Clusters": m["n_clusters"], "Silhouette": m["silhouette"], "Calinski-Harabasz": m["calinski_harabasz"], "Davies-Bouldin": m["davies_bouldin"], "Ruído (%)": m["noise_pct"], "ARI": m.get("adjusted_rand"), "NMI": m.get("normalized_mutual_info")})
                    st.subheader("📋 Métricas")
                    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)

# TAB 4 — RESULTADOS
with tabs[3]:
    st.header("📈 Resultados e avaliação")
    result_state = st.session_state.get("results")
    if result_state is None:
        st.info("Execute uma análise na aba **Modelagem** para visualizar os resultados.")
    else:
        task = result_state["task"]
        results = result_state["results"]
        valid = {name: value for name, value in results.items() if "error" not in value}
        errors = {name: value["error"] for name, value in results.items() if "error" in value}
        if errors:
            with st.expander("⚠️ Modelos/algoritmos que falharam"):
                st.dataframe(pd.DataFrame([{"Modelo": name, "Erro": error} for name, error in errors.items()]), width="stretch")

        if task == "regression":
            if not valid:
                st.error("Nenhum modelo de regressão executado com sucesso.")
            else:
                rows = []
                for name, res in valid.items():
                    params = ", ".join(f"{k.replace('regressor__','')}={v}" for k, v in res["best_params"].items()) or "—"
                    cv_mean_val = res.get("cv_mean")
                    cv_std_val = res.get("cv_std")
                    rows.append({
                        "Modelo": name,
                        "CV R² médio": round(cv_mean_val, 4) if cv_mean_val is not None else None,
                        "CV R² desvio": round(cv_std_val, 4) if cv_std_val is not None else None,
                        "R² teste": round(res["test_r2"], 4),
                        "RMSE teste": round(res["test_rmse"], 4),
                        "MAE teste": round(res["test_mae"], 4),
                        "Hiperparâmetros": params,
                        "_sort_cv": cv_mean_val if cv_mean_val is not None else -np.inf,
                    })
                result_table = pd.DataFrame(rows).sort_values("_sort_cv", ascending=False).drop(columns=["_sort_cv"]).reset_index(drop=True)
                st.dataframe(result_table, width="stretch", hide_index=True)
                with_cv = [r for r in rows if r["_sort_cv"] != -np.inf]
                selected = max(with_cv, key=lambda r: r["_sort_cv"])["Modelo"] if with_cv else rows[0]["Modelo"]
                st.info(f"Modelo selecionado pela média de R² na CV: **{selected}**.")
                fig_real, fig_resid = regression_scatter_plot(result_state["y_test"], valid)
                if fig_real is not None:
                    st.plotly_chart(fig_real, width="stretch")
                    st.plotly_chart(fig_resid, width="stretch")
                fig_box, box_df = regression_boxplot(valid)
                if fig_box is not None:
                    st.plotly_chart(fig_box, width="stretch")

                # Ajuste manual universal — Regressão
                st.divider()
                st.subheader("🔧 Ajuste manual de hiperparâmetros (Regressão)")
                manual_candidates = [m for m in valid.keys() if not m.endswith("[manual]")]
                if not manual_candidates:
                    st.info("Nenhum modelo disponível.")
                else:
                    manual_reg_model = st.selectbox("Modelo", manual_candidates, key="manual_reg_model_select")
                    base_est = valid[manual_reg_model]["best_model"].named_steps["regressor"]
                    current_params = valid[manual_reg_model]["best_params"]
                    edited_vals = render_manual_params_universal(base_est, "regressor__", current_params, "manual_reg_univ")
                    if edited_vals:
                        if st.button("🔄 Recalcular com hiperparâmetros manuais", type="primary", width="stretch", key="manual_reg_recalc_univ"):
                            try:
                                parsed_params = {k: parse_param_string(v) for k, v in edited_vals.items()}
                                enc_cfg = result_state["encoding_config"] or {}
                                base_est2 = clone(base_est)
                                pipe = make_regression_pipeline(base_est2, result_state["X_train"], enc_cfg.get("encoding_map", {}), enc_cfg.get("ordinal_orders", {}), enc_cfg.get("manual_maps", {}), result_state["outlier_mode"], result_state["threshold"], result_state["max_features"])
                                pipe.set_params(**parsed_params)
                                pipe.fit(result_state["X_train"], result_state["y_train"])
                                pred = pipe.predict(result_state["X_test"])
                                mse = float(mean_squared_error(result_state["y_test"], pred))
                                new_name = f"{manual_reg_model} [manual]"
                                result_state["results"][new_name] = {
                                    "best_model": pipe, "best_params": parsed_params,
                                    "cv_mean": None, "cv_std": None, "cv_scores": np.array([]),
                                    "test_r2": float(r2_score(result_state["y_test"], pred)),
                                    "test_rmse": float(np.sqrt(mse)),
                                    "test_mae": float(mean_absolute_error(result_state["y_test"], pred)),
                                    "test_mse": mse, "y_pred": pred, "search_table": None, "manual": True,
                                }
                                st.session_state["results"] = result_state
                                st.success(f"✅ '{new_name}' recalculado.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Erro ao recalcular: {exc}")
                if any(m.endswith("[manual]") for m in valid.keys()):
                    if st.button("🗑️ Limpar ajustes manuais", key="clear_manual_reg_univ", width="stretch"):
                        result_state["results"] = {k: v for k, v in result_state["results"].items() if not k.endswith("[manual]")}
                        st.session_state["results"] = result_state
                        st.rerun()

        elif task == "classification":
            if not valid:
                st.error("Nenhum classificador executado com sucesso.")
            else:
                rows = []
                for name, res in valid.items():
                    cv_tuning = res.get("cv_tuning_mean")
                    rows.append({
                        "Classificador": name,
                        "_score_selection": float(cv_tuning) if cv_tuning is not None else -np.inf,
                        "Média CV (tuning)": round(cv_tuning, 4) if cv_tuning is not None else None,
                        "Hiperparâmetros": res.get("best_params_text", "—"),
                        "Acurácia teste": round(res["test_accuracy"], 4),
                        "F1 ponderado teste": round(res["test_f1"], 4),
                        "F1 macro teste": round(res["test_f1_macro"], 4),
                        "Balanced accuracy teste": round(res["test_balanced_accuracy"], 4),
                        "ROC-AUC teste": None if res.get("test_roc_auc") is None else round(res["test_roc_auc"], 4),
                    })
                result_table = pd.DataFrame(rows).sort_values("_score_selection", ascending=False).reset_index(drop=True)
                st.dataframe(result_table.drop(columns=["_score_selection"]), width="stretch", height=450, hide_index=True)
                with_score = [r for r in rows if r["_score_selection"] != -np.inf]
                selected = with_score[0]["Classificador"] if with_score else rows[0]["Classificador"]
                st.info(f"Classificador selecionado: **{selected}**.")
                selected_result = valid[selected]
                cm_fig = confusion_matrix_plot(result_state["y_test"], selected_result["y_pred"], result_state["classes"], f"Matriz de confusão — {selected}")
                st.plotly_chart(cm_fig, width="stretch")
                fig_cv, cv_df = classification_cv_plot(valid)
                if fig_cv is not None:
                    st.plotly_chart(fig_cv, width="stretch")

                # Ajuste manual universal — Classificação
                st.divider()
                st.subheader("🔧 Ajuste manual de hiperparâmetros (Classificação)")
                manual_clf_candidates = [m for m in valid.keys() if not m.endswith("[manual]")]
                if not manual_clf_candidates:
                    st.info("Nenhum classificador disponível.")
                else:
                    manual_clf_model = st.selectbox("Classificador", manual_clf_candidates, key="manual_clf_model_select_univ")
                    base_est = valid[manual_clf_model]["best_model"].named_steps["classifier"]
                    current_params = valid[manual_clf_model].get("best_params", {})
                    edited_clf = render_manual_params_universal(base_est, "classifier__", current_params, "manual_clf_univ")
                    if edited_clf:
                        if st.button("🔄 Recalcular com hiperparâmetros manuais", type="primary", width="stretch", key="manual_clf_recalc_univ"):
                            try:
                                parsed_params = {k: parse_param_string(v) for k, v in edited_clf.items()}
                                enc_cfg = result_state["encoding_config"] or {}
                                base_est2 = clone(base_est)
                                pipe = make_classification_pipeline(base_est2, result_state["X_train"], enc_cfg.get("encoding_map", {}), enc_cfg.get("ordinal_orders", {}), enc_cfg.get("manual_maps", {}))
                                pipe.set_params(**parsed_params)
                                pipe.fit(result_state["X_train"], result_state["y_train"])
                                y_pred = pipe.predict(result_state["X_test"])
                                new_name = f"{manual_clf_model} [manual]"
                                result = {
                                    "best_model": pipe, "tuned": False, "tuning_method": "manual", "tuning_metric": "—",
                                    "best_params": parsed_params,
                                    "best_params_text": ", ".join(f"{k.replace('classifier__','')}={v}" for k, v in parsed_params.items()) or "—",
                                    "tuning_table": None, "cv_tuning_mean": None, "cv_tuning_std": None,
                                    "cv_accuracy_mean": None, "cv_accuracy_std": None,
                                    "cv_f1_weighted_mean": None, "cv_f1_weighted_std": None,
                                    "cv_f1_macro_mean": None, "cv_f1_macro_std": None,
                                    "cv_balanced_accuracy_mean": None, "cv_acc_scores": [], "cv_f1_scores": [],
                                    "test_accuracy": float(accuracy_score(result_state["y_test"], y_pred)),
                                    "test_precision": float(precision_score(result_state["y_test"], y_pred, average="weighted", zero_division=0)),
                                    "test_recall": float(recall_score(result_state["y_test"], y_pred, average="weighted", zero_division=0)),
                                    "test_f1": float(f1_score(result_state["y_test"], y_pred, average="weighted", zero_division=0)),
                                    "test_f1_macro": float(f1_score(result_state["y_test"], y_pred, average="macro", zero_division=0)),
                                    "test_balanced_accuracy": float(balanced_accuracy_score(result_state["y_test"], y_pred)),
                                    "y_pred": y_pred, "manual": True,
                                }
                                try:
                                    if hasattr(pipe, "predict_proba"):
                                        probs = pipe.predict_proba(result_state["X_test"])
                                        if probs.shape[1] == 2:
                                            result["test_roc_auc"] = float(roc_auc_score(result_state["y_test"], probs[:, 1]))
                                        else:
                                            result["test_roc_auc"] = float(roc_auc_score(result_state["y_test"], probs, multi_class="ovr", average="weighted"))
                                    elif hasattr(pipe, "decision_function"):
                                        scores = pipe.decision_function(result_state["X_test"])
                                        if np.ndim(scores) == 1:
                                            result["test_roc_auc"] = float(roc_auc_score(result_state["y_test"], scores))
                                        else:
                                            result["test_roc_auc"] = float(roc_auc_score(result_state["y_test"], scores, multi_class="ovr", average="weighted"))
                                except Exception:
                                    result["test_roc_auc"] = None
                                result_state["results"][new_name] = result
                                st.session_state["results"] = result_state
                                st.success(f"✅ '{new_name}' recalculado.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Erro ao recalcular: {exc}")
                if any(m.endswith("[manual]") for m in valid.keys()):
                    if st.button("🗑️ Limpar ajustes manuais", key="clear_manual_clf_univ", width="stretch"):
                        result_state["results"] = {k: v for k, v in result_state["results"].items() if not k.endswith("[manual]")}
                        st.session_state["results"] = result_state
                        st.rerun()

                st.divider()
                st.subheader("🔮 Classificar novos dados")
                prediction_models = list(valid.keys())
                pred_model_name = st.selectbox("Modelo treinado", prediction_models, key="prediction_result_model_v3")
                prediction_file = st.file_uploader("Novo arquivo para previsão", type=["csv", "xlsx", "xls", "json"], key="prediction_result_file_v3")
                if prediction_file is not None:
                    try:
                        new_data = load_dataframe(prediction_file, "file")
                        missing = [c for c in result_state["features"] if c not in new_data.columns]
                        if missing:
                            st.error("Variáveis ausentes: " + ", ".join(missing))
                        elif st.button("🔮 Executar previsão", type="primary", width="stretch", key="predict_result_v3"):
                            predicted = predict_new_classification(valid[pred_model_name]["best_model"], new_data, result_state["features"], result_state["classes"])
                            st.session_state["prediction_result"] = predicted
                    except Exception as exc:
                        st.error(f"Erro na previsão: {exc}")
                if st.session_state.get("prediction_result") is not None:
                    st.dataframe(st.session_state["prediction_result"], width="stretch", height=350)
                    st.download_button("📥 Baixar previsões (.csv)", st.session_state["prediction_result"].to_csv(index=False).encode("utf-8"), file_name=f"classificacoes_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv", width="stretch", key="download_prediction_result_v3")

        else:  # clustering
            if not valid:
                st.error("Nenhum algoritmo de agrupamento executado com sucesso.")
            else:
                rows = []
                for name, res in valid.items():
                    m = res["metrics"]
                    objective_key = res.get("tuning_metric", "silhouette")
                    objective_value = m.get(objective_key)
                    rows.append({
                        "Algoritmo": name, "_score_selection": objective_value,
                        "Clusters": m["n_clusters"], "Silhouette": m["silhouette"],
                        "Calinski-Harabasz": m["calinski_harabasz"],
                        "Davies-Bouldin": m["davies_bouldin"],
                        "Ruído (%)": m["noise_pct"],
                        "ARI": m.get("adjusted_rand"), "NMI": m.get("normalized_mutual_info"),
                        "Hiperparâmetros": str(res.get("best_params", {})),
                    })
                sort_ascending = True if valid[next(iter(valid))].get("tuning_metric") == "davies_bouldin" else False
                result_table = pd.DataFrame(rows).sort_values("_score_selection", ascending=sort_ascending, na_position="last")
                st.dataframe(result_table.drop(columns=["_score_selection"]).round(4), width="stretch", hide_index=True)
                st.info("Silhouette e Calinski-Harabasz: maiores = melhor. Davies-Bouldin: menor = melhor. ARI/NMI requerem rótulo de referência.")
                for name, res in valid.items():
                    fig = clustering_scatter_plot(result_state["transformed"], res["labels"], f"{name} — clusters")
                    st.plotly_chart(fig, width="stretch")
                    assignment = result_state["source_data"].copy()
                    assignment["Cluster"] = res["labels"]
                    with st.expander(f"🔎 Atribuições — {name}"):
                        st.dataframe(assignment, width="stretch")

                # Ajuste manual universal — Agrupamento
                st.divider()
                st.subheader("🔧 Ajuste manual de hiperparâmetros (Agrupamento)")
                manual_cluster_candidates = [m for m in valid.keys() if not m.endswith("[manual]")]
                if not manual_cluster_candidates:
                    st.info("Nenhum algoritmo disponível.")
                else:
                    manual_cluster_name = st.selectbox("Algoritmo", manual_cluster_candidates, key="manual_cluster_select_univ")
                    base_est = valid[manual_cluster_name]["estimator"]
                    current_params = valid[manual_cluster_name].get("best_params", {})
                    edited_cluster = render_manual_params_universal(base_est, "", current_params, "manual_cluster_univ")
                    if edited_cluster:
                        if st.button("🔄 Recalcular agrupamento", type="primary", width="stretch", key="manual_cluster_recalc_univ"):
                            try:
                                parsed = {k: parse_param_string(v) for k, v in edited_cluster.items()}
                                X_transformed = result_state["transformed"]
                                model = make_cluster_estimator(manual_cluster_name, parsed, result_state.get("seed"))
                                labels = model.fit_predict(X_transformed)
                                metrics = calculate_cluster_metrics(X_transformed, labels)
                                ref_target = result_state.get("reference_target")
                                if ref_target:
                                    metrics.update(calculate_external_cluster_metrics(df_raw[ref_target], labels))
                                new_name = f"{manual_cluster_name} [manual]"
                                result_state["results"][new_name] = {
                                    "estimator": model, "labels": np.asarray(labels), "metrics": metrics,
                                    "best_params": parsed, "tuned": False, "tuning_method": "manual",
                                    "tuning_metric": valid[manual_cluster_name].get("tuning_metric", "silhouette"),
                                    "tuning_table": None, "manual": True,
                                }
                                st.session_state["results"] = result_state
                                st.session_state["clustering_result"] = result_state
                                st.success(f"✅ '{new_name}' recalculado.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Erro ao recalcular: {exc}")
                if any(m.endswith("[manual]") for m in valid.keys()):
                    if st.button("🗑️ Limpar ajustes manuais", key="clear_manual_cluster_univ", width="stretch"):
                        result_state["results"] = {k: v for k, v in result_state["results"].items() if not k.endswith("[manual]")}
                        st.session_state["results"] = result_state
                        st.session_state["clustering_result"] = result_state
                        st.rerun()

        # Exportação
        st.divider()
        st.subheader("💾 Exportar resultados")
        if task == "clustering":
            export_rows = [{"algoritmo": name, **res["metrics"]} for name, res in valid.items()]
        else:
            export_rows = []
            for name, res in valid.items():
                clean = {}
                for key, item in res.items():
                    if key in {"best_model", "y_pred", "cv_scores", "cv_acc_scores", "cv_f1_scores", "search_table", "tuning_table"}:
                        continue
                    if isinstance(item, (np.floating, np.integer)):
                        clean[key] = item.item()
                    else:
                        clean[key] = item
                clean["modelo"] = name
                export_rows.append(clean)
        csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar resultados (.csv)", csv_bytes, file_name=f"resultados_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv", width="stretch")
        report = {
            "tarefa": task, "features": result_state.get("features", []),
            "seed": result_state.get("seed"), "modelos": export_rows,
            "config": {k: result_state.get(k) for k in ["outlier_mode", "threshold", "max_features", "cv", "n_clusters", "reference_target"]},
        }
        if task != "clustering":
            report.update({"alvo": result_state.get("target"), "train_n": result_state.get("train_n"), "test_n": result_state.get("test_n")})
        st.download_button("📥 Baixar relatório (.json)", json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8"), file_name=f"relatorio_{datetime.now():%Y%m%d_%H%M%S}.json", mime="application/json", width="stretch")

# TAB 5 — EXPLICAÇÕES
with tabs[4]:
    st.header("📚 Explicações dos Algoritmos")
    st.caption(
        "Descrições, funcionamento, quando usar e limitações de cada algoritmo "
        "disponível no sistema. Consulte esta seção para escolher o algoritmo mais adequado."
    )
    task_filter = st.radio(
        "Filtrar por tipo",
        ["Todos", "Regressão", "Classificação", "Agrupamento"],
        horizontal=True,
        key="explanation_filter",
    )
    search_term = st.text_input("🔍 Buscar por nome", "", key="explanation_search")
    items = sorted(ALGORITHM_EXPLANATIONS.items())
    filtered = []
    for name, info in items:
        if task_filter != "Todos" and info["tipo"] != task_filter:
            continue
        if search_term and search_term.lower() not in name.lower():
            continue
        filtered.append((name, info))
    if not filtered:
        st.info("Nenhum algoritmo corresponde aos filtros.")
    else:
        for name, info in filtered:
            with st.expander(f"**{name}** — {info['tipo']}", expanded=False):
                st.markdown(f"**O que faz:** {info['descricao']}")
                st.markdown(f"**Como funciona:** {info['como_funciona']}")
                st.markdown(f"**Quando usar:** {info['quando_usar']}")
                st.markdown(f"**Limitações:** {info['limitacoes']}")
    st.divider()
    st.caption(
        "💡 **Nota:** O dicionário cobre os algoritmos mais usados do scikit-learn "
        "e das bibliotecas externas (XGBoost, LightGBM, CatBoost, HDBSCAN, MiniSom). "
        "Algoritmos não listados ainda podem ser executados, mas não têm explicação detalhada."
    )

# RODAPÉ
st.divider()
st.caption(
    "🧠 Análise de Inteligência Computacional — Streamlit + Plotly | "
    "Pré-análise, recomendações, todos os algoritmos do sklearn + XGBoost/LightGBM/CatBoost/HDBSCAN/MiniSom, "
    "ajuste manual universal de hiperparâmetros e seção de explicações."
)
