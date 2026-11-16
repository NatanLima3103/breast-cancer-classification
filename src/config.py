"""Configurações centralizadas do projeto.

Todos os caminhos são relativos à raiz do projeto, determinada a partir da
localização deste arquivo, de modo que os comandos funcionem em qualquer máquina.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
EXAMPLES_DIR: Final[Path] = PROJECT_ROOT / "examples"

DATA_PATH: Final[Path] = DATA_DIR / "breast-cancer.csv"
MODEL_PATH: Final[Path] = MODELS_DIR / "best_model.joblib"
EXAMPLE_PATIENT_PATH: Final[Path] = EXAMPLES_DIR / "nova_paciente.json"

# Nomes dos arquivos gerados em reports/
EDA_REPORT_FILE: Final[str] = "exploratory_analysis.txt"
CLASS_DISTRIBUTION_PLOT_FILE: Final[str] = "class_distribution.png"
RECURRENCE_BY_ATTRIBUTE_PLOT_FILE: Final[str] = "recurrence_rate_by_attribute.png"
COMPARISON_CSV_FILE: Final[str] = "model_comparison.csv"
SMOTE_COMPARISON_FILE: Final[str] = "smote_comparison.csv"
CLASSIFICATION_REPORT_FILE: Final[str] = "classification_report.txt"
CONFUSION_MATRIX_PLOT_FILE: Final[str] = "confusion_matrix.png"
ACCURACY_PLOT_FILE: Final[str] = "accuracy_comparison.png"
TRAINING_SUMMARY_FILE: Final[str] = "training_summary.json"

# ---------------------------------------------------------------------------
# Reprodutibilidade e validação
# ---------------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_N_SPLITS: Final[int] = 5
CV_N_REPEATS: Final[int] = 10
# Diferença máxima (em pontos absolutos de acurácia média de validação cruzada)
# para que dois modelos sejam considerados "empatados". Dentro dessa faixa o
# critério de desempate é o recall médio da classe positiva na validação cruzada.
ACCURACY_TIE_TOLERANCE: Final[float] = 0.01
N_JOBS: Final[int] = -1

# ---------------------------------------------------------------------------
# Esquema dos dados
# ---------------------------------------------------------------------------
TARGET: Final[str] = "Class"
POSITIVE_CLASS: Final[str] = "recurrence-events"
NEGATIVE_CLASS: Final[str] = "no-recurrence-events"
CLASS_LABELS: Final[tuple[str, str]] = (NEGATIVE_CLASS, POSITIVE_CLASS)

# Marcador de valor desconhecido usado no arquivo original.
MISSING_MARKER: Final[str] = "?"

ORDINAL_FEATURES: Final[tuple[str, ...]] = ("deg-malig",)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "breast",
    "breast-quad",
    "irradiat",
)
# Ordem original das colunas no arquivo CSV.
FEATURES: Final[tuple[str, ...]] = (
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
)
ALL_COLUMNS: Final[tuple[str, ...]] = (*FEATURES, TARGET)

# Domínio documentado de cada atributo (arquivo breast-cancer.names da UCI).
# A UCI documenta "left-up", "left-low" etc.; o CSV fornecido usa sublinhado
# ("left_up"), por isso o domínio segue a grafia efetivamente presente na base.
# Os intervalos estão em ordem crescente, o que também é usado nos relatórios.
DOCUMENTED_DOMAIN: Final[dict[str, tuple]] = {
    "age": ("10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"),
    "menopause": ("lt40", "ge40", "premeno"),
    "tumor-size": (
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
        "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    ),
    "inv-nodes": (
        "0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20",
        "21-23", "24-26", "27-29", "30-32", "33-35", "36-39",
    ),
    "node-caps": ("no", "yes"),
    "deg-malig": (1, 2, 3),
    "breast": ("left", "right"),
    "breast-quad": ("central", "left_low", "left_up", "right_low", "right_up"),
    "irradiat": ("no", "yes"),
    TARGET: CLASS_LABELS,
}

# Descrição dos atributos (usada no relatório exploratório).
FEATURE_DESCRIPTIONS: Final[dict[str, str]] = {
    "age": "Faixa etária da paciente no diagnóstico (intervalos de 10 anos)",
    "menopause": "Status menopausal: premeno, lt40 (antes dos 40) ou ge40 (40 ou mais)",
    "tumor-size": "Diâmetro do tumor em mm (intervalos de 5 mm)",
    "inv-nodes": "Número de linfonodos axilares com metástase (intervalos)",
    "node-caps": "Penetração do tumor na cápsula do linfonodo (yes/no)",
    "deg-malig": "Grau histológico de malignidade (1, 2 ou 3; ordinal)",
    "breast": "Mama acometida (left/right)",
    "breast-quad": "Quadrante da mama acometido",
    "irradiat": "Paciente submetida à radioterapia (yes/no)",
    TARGET: "Ocorrência de recidiva (recurrence-events) ou não (no-recurrence-events)",
}

# ---------------------------------------------------------------------------
# Cores dos gráficos (paleta categórica validada; texto em tons neutros)
# ---------------------------------------------------------------------------
COLOR_NEGATIVE: Final[str] = "#2a78d6"   # no-recurrence-events
COLOR_POSITIVE: Final[str] = "#eb6834"   # recurrence-events
COLOR_BASELINE: Final[str] = "#a3a29c"   # classificador de referência
COLOR_TEXT: Final[str] = "#0b0b0b"
COLOR_TEXT_SECONDARY: Final[str] = "#52514e"
COLOR_GRID: Final[str] = "#e4e3df"
COLOR_SURFACE: Final[str] = "#fcfcfb"
