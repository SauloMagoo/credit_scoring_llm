"""Caminhos, semente e listas de colunas do projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "credit_scoring.csv"
DATA_PATH_FALLBACK = ROOT / "credit_scoring.csv"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

TEST_START = "2016-01-01"
N_TRAIN = 50_000
N_TEST = 15_000

TARGET = "mau"
GROUP_COL = "index"
DATE_COL = "data_ref"
ID_COLS = ["Unnamed: 0", GROUP_COL, DATE_COL]

NUMERIC_FEATURES = [
    "qtd_filhos",
    "idade",
    "tempo_emprego",
    "qt_pessoas_residencia",
    "renda",
]
CATEGORICAL_FEATURES = [
    "sexo",
    "posse_de_veiculo",
    "posse_de_imovel",
    "tipo_renda",
    "educacao",
    "estado_civil",
    "tipo_residencia",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Colunas que nunca podem ser usadas como entrada do modelo.
FORBIDDEN_FEATURES = ID_COLS + [TARGET]

REQUIRED_COLUMNS = ID_COLS + FEATURES + [TARGET]
