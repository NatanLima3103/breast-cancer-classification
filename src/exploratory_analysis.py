"""Análise exploratória da base Breast Cancer (UCI).

Uso:
    python -m src.exploratory_analysis

Gera ``reports/exploratory_analysis.txt`` e dois gráficos: a distribuição das
classes e a taxa de recorrência por categoria de cada atributo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from src import config  # noqa: E402
from src.data import coerce_types, read_raw_dataset, replace_missing_marker, validate_domain  # noqa: E402


def _ordered_categories(col: str, observed: pd.Index) -> list:
    """Ordena as categorias observadas segundo o domínio documentado."""
    domain = list(config.DOCUMENTED_DOMAIN[col])
    if col in config.ORDINAL_FEATURES:
        domain = [float(v) for v in domain]
    return [v for v in domain if v in set(observed)]


def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().reindex(list(config.CLASS_LABELS), fill_value=0)
    return pd.DataFrame({"registros": counts, "percentual": counts / counts.sum() * 100})


def duplicate_summary(raw: pd.DataFrame) -> dict[str, object]:
    """Conta duplicações completas, duplicações apenas nos preditores e conflitos."""
    features = list(config.FEATURES)
    conflict_groups = (
        raw.groupby(features)[config.TARGET].nunique().loc[lambda s: s > 1].reset_index()
    )
    conflicting_rows = raw.merge(conflict_groups[features], on=features, how="inner")
    return {
        "full_duplicates": int(raw.duplicated().sum()),
        "feature_duplicates": int(raw[features].duplicated().sum()),
        "rows_in_full_duplicate_groups": int(raw.duplicated(keep=False).sum()),
        "conflicting_combinations": int(len(conflict_groups)),
        "rows_in_conflicting_combinations": int(len(conflicting_rows)),
        "conflicting_table": (
            conflicting_rows.groupby(features + [config.TARGET]).size()
            .rename("registros").reset_index()
        ),
    }


def association_table(df: pd.DataFrame) -> pd.DataFrame:
    """Teste qui-quadrado de independência e V de Cramér de cada atributo com a classe.

    Valores ausentes são ignorados apenas neste teste descritivo.
    """
    rows = []
    for col in config.FEATURES:
        table = pd.crosstab(df[col], df[config.TARGET])
        chi2, p_value, dof, expected = chi2_contingency(table)
        n = table.to_numpy().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        rows.append({
            "atributo": col,
            "qui2": chi2,
            "gl": dof,
            "p_valor": p_value,
            "v_cramer": cramers_v,
            "celulas_esperado_<5_%": (expected < 5).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("v_cramer", ascending=False)


def recurrence_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    table = pd.crosstab(df[col], df[config.TARGET])
    table = table.reindex(columns=list(config.CLASS_LABELS), fill_value=0)
    table = table.loc[_ordered_categories(col, table.index)]
    table["total"] = table.sum(axis=1)
    table["%_total"] = table["total"] / table["total"].sum() * 100
    table["taxa_recorrencia_%"] = table[config.POSITIVE_CLASS] / table["total"] * 100
    return table


def plot_class_distribution(dist: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=config.COLOR_SURFACE)
    colors = [config.COLOR_NEGATIVE, config.COLOR_POSITIVE]
    ax.barh(dist.index, dist["registros"], color=colors, height=0.55)
    for i, (n, pct) in enumerate(zip(dist["registros"], dist["percentual"])):
        ax.text(n + 3, i, f"{n} ({pct:.1f}%)", va="center", fontsize=10, color=config.COLOR_TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, dist["registros"].max() * 1.3)
    ax.set_xlabel("Registros")
    ax.set_title("Distribuição da variável-alvo Class", loc="left", fontsize=11,
                 color=config.COLOR_TEXT)
    ax.set_facecolor(config.COLOR_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(config.COLOR_GRID)
    ax.tick_params(colors=config.COLOR_TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_recurrence_by_attribute(df: pd.DataFrame, overall_rate: float, path: Path) -> None:
    """Pequenos múltiplos: taxa de recorrência por categoria (n acima de cada barra)."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), facecolor=config.COLOR_SURFACE)
    for ax, col in zip(axes.flat, config.FEATURES):
        table = recurrence_by_category(df, col)
        labels = [str(int(v)) if isinstance(v, float) else str(v) for v in table.index]
        x = np.arange(len(table))
        ax.bar(x, table["taxa_recorrencia_%"], color=config.COLOR_POSITIVE, width=0.7)
        for xi, (rate, n) in enumerate(zip(table["taxa_recorrencia_%"], table["total"])):
            ax.text(xi, rate + 2, f"n={n}", ha="center", fontsize=7,
                    color=config.COLOR_TEXT_SECONDARY, zorder=3,
                    bbox={"facecolor": config.COLOR_SURFACE, "edgecolor": "none", "pad": 0.6})
        ax.axhline(overall_rate, color=config.COLOR_TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.set_xticks(x, labels=labels, rotation=45 if len(labels) > 4 else 0,
                      ha="right" if len(labels) > 4 else "center", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(col, loc="left", fontsize=10, color=config.COLOR_TEXT)
        ax.set_facecolor(config.COLOR_SURFACE)
        ax.grid(axis="y", color=config.COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(config.COLOR_GRID)
        ax.tick_params(colors=config.COLOR_TEXT_SECONDARY, labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("% recurrence-events", fontsize=9, color=config.COLOR_TEXT_SECONDARY)
    fig.suptitle(
        "Taxa de recorrência por categoria de cada atributo "
        f"(linha tracejada = taxa geral de {overall_rate:.1f}%; valores '?' omitidos)",
        x=0.01, ha="left", fontsize=12, color=config.COLOR_TEXT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(raw: pd.DataFrame, source_name: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Monta o texto do relatório e retorna também os dados usados nos gráficos."""
    df = coerce_types(replace_missing_marker(raw))
    validate_domain(df)
    y = df[config.TARGET]
    dist = class_distribution(y)
    dup = duplicate_summary(raw)
    unknown = (raw == config.MISSING_MARKER).sum()
    empty = (raw.apply(lambda s: s.astype(str).str.strip() == "")).sum()
    majority_rate = dist["percentual"].max()

    sep = "=" * 78
    out: list[str] = [
        "ANÁLISE EXPLORATÓRIA — BREAST CANCER (UCI Machine Learning Repository)",
        sep,
        f"Arquivo analisado: {source_name}",
        "Referência: Zwitter, M. & Soklic, M. (1988). Breast Cancer [Dataset]. UCI Machine "
        "Learning Repository. https://doi.org/10.24432/C51P4M",
        "",
        "1. DIMENSÕES",
        f"   Registros: {raw.shape[0]}",
        f"   Colunas: {raw.shape[1]} ({len(config.FEATURES)} atributos preditores + "
        f"variável-alvo '{config.TARGET}')",
        "",
        "2. TIPOS DE DADOS",
        "   Tipos lidos do CSV (todas as colunas como texto) e tipo adotado no modelo:",
    ]
    for col in config.ALL_COLUMNS:
        if col == config.TARGET:
            adopted = "alvo binário (categórico)"
        elif col in config.ORDINAL_FEATURES:
            adopted = "ordinal numérico (1 < 2 < 3)"
        else:
            adopted = "categórico nominal (one-hot)"
        out.append(f"   {col:<12} lido={str(raw[col].dtype):<7} adotado={adopted:<30} "
                   f"| {config.FEATURE_DESCRIPTIONS[col]}")
    out.append("   Observação: age, tumor-size e inv-nodes são intervalos ordenados, mas "
               "foram tratados como nominais (one-hot), conforme a especificação do projeto.")

    out += ["", "3. DISTRIBUIÇÃO DAS CLASSES", dist.to_string(float_format=lambda v: f"{v:.2f}"),
            "",
            f"   Razão de desbalanceamento: {dist['registros'].max() / dist['registros'].min():.2f}"
            " negativos para cada positivo.",
            f"   Consequência: um classificador trivial que sempre prevê "
            f"'{config.NEGATIVE_CLASS}' atinge {majority_rate:.2f}% de acurácia com recall 0 "
            "para recurrence-events.",
            "   Por isso a acurácia é comparada com essa referência e acompanhada de "
            "precision, recall, F1 e matriz de confusão da classe minoritária. As divisões "
            "treino/teste e da validação cruzada são estratificadas para preservar a proporção."]

    out += ["", "4. VALORES AUSENTES E DESCONHECIDOS"]
    out.append(f"   Células vazias / NaN no arquivo: {int(empty.sum())}")
    out.append(f"   Valores desconhecidos '?': {int(unknown.sum())} no total")
    for col, n in unknown[unknown > 0].items():
        out.append(f"     - {col}: {n} ({n / len(raw) * 100:.2f}% dos registros)")
    rows_with_unknown = int((raw == config.MISSING_MARKER).any(axis=1).sum())
    out.append(f"   Registros com ao menos um '?': {rows_with_unknown}")
    unk_rows = raw.loc[(raw == config.MISSING_MARKER).any(axis=1)]
    out.append("   Distribuição das classes nesses registros: "
               + ", ".join(f"{k}={v}" for k, v in unk_rows[config.TARGET].value_counts().items()))
    out.append("   Tratamento: '?' é convertido em NaN e imputado pela moda dentro do pipeline "
               "(ajustado só no treino); nenhum registro é excluído.")

    out += ["", "5. REGISTROS DUPLICADOS",
            f"   Duplicações completas (todas as 10 colunas): {dup['full_duplicates']}",
            f"   Registros envolvidos em grupos completamente duplicados: "
            f"{dup['rows_in_full_duplicate_groups']}",
            f"   Duplicações considerando apenas os 9 preditores: {dup['feature_duplicates']}",
            f"   Combinações de preditores associadas a classes diferentes: "
            f"{dup['conflicting_combinations']} ({dup['rows_in_conflicting_combinations']} "
            "registros)",
            "   Combinações conflitantes:",
            dup["conflicting_table"].to_string(index=False),
            "",
            "   Decisão: os duplicados foram MANTIDOS. Os atributos são faixas discretas, o que",
            "   torna plausível que pacientes distintas tenham perfis idênticos; removê-los",
            "   alteraria a distribuição original sem evidência de erro de registro. As",
            "   combinações conflitantes mostram que parte do erro é irredutível: nenhum",
            "   classificador baseado apenas nesses atributos pode acertar todos esses casos.",
            "   Ressalva: cópias idênticas podem cair no treino e no teste ao mesmo tempo, o",
            "   que pode tornar as estimativas ligeiramente otimistas."]

    out += ["", "6. FREQUÊNCIA DE CADA CATEGORIA E RELAÇÃO COM A VARIÁVEL-ALVO",
            "   (contagens por classe, % do total e taxa de recorrência por categoria; "
            "'?' omitido)"]
    for col in config.FEATURES:
        table = recurrence_by_category(df, col)
        table.index = [str(int(v)) if isinstance(v, float) else str(v) for v in table.index]
        out += ["", f"   [{col}] — {config.FEATURE_DESCRIPTIONS[col]}",
                table.to_string(float_format=lambda v: f"{v:.1f}")]
        unseen = [str(v) for v in config.DOCUMENTED_DOMAIN[col] if
                  (float(v) if col in config.ORDINAL_FEATURES else v) not in set(df[col].dropna())]
        if unseen:
            out.append(f"   Categorias documentadas pela UCI ausentes na base: {unseen}")
        rare = table.loc[table["total"] < 10]
        if not rare.empty:
            out.append(f"   Categorias raras (< 10 registros): {list(rare.index)}")

    assoc = association_table(df)
    out += ["", "7. ASSOCIAÇÃO DE CADA ATRIBUTO COM A CLASSE (qui-quadrado de independência)",
            assoc.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
            "   Atenção: com categorias raras, várias tabelas têm células com frequência",
            "   esperada < 5, o que torna o p-valor aproximado. O teste é descritivo e não foi",
            "   usado para selecionar atributos (todos os nove são mantidos nos modelos)."]

    out += ["", "8. PRINCIPAIS CONCLUSÕES",
            f"   * Base pequena ({raw.shape[0]} registros) e desbalanceada "
            f"({dist.loc[config.NEGATIVE_CLASS, 'percentual']:.2f}% / "
            f"{dist.loc[config.POSITIVE_CLASS, 'percentual']:.2f}%).",
            f"   * A referência majoritária já obtém {majority_rate:.2f}% de acurácia: ganhos "
            "reais precisam superá-la e ter recall não nulo na classe minoritária.",
            "   * Atributos com associação mais forte com recorrência (maior V de Cramér): "
            + ", ".join(assoc["atributo"].head(4)) + ".",
            f"   * Poucos valores desconhecidos ({int(unknown.sum())} células em {rows_with_unknown} "
            "registros) — imputação pela moda é adequada.",
            "   * Duplicados e combinações conflitantes limitam o desempenho alcançável.",
            ""]
    return "\n".join(out), df, dist


def run_eda(
    data_path: Path | str = config.DATA_PATH, reports_dir: Path | str = config.REPORTS_DIR
) -> Path:
    """Executa a análise exploratória e grava relatório e gráficos em ``reports_dir``."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw_dataset(data_path)
    report, df, dist = build_report(raw, Path(data_path).name)
    report_path = reports_dir / config.EDA_REPORT_FILE
    report_path.write_text(report, encoding="utf-8")
    plot_class_distribution(dist, reports_dir / config.CLASS_DISTRIBUTION_PLOT_FILE)
    overall = float(dist.loc[config.POSITIVE_CLASS, "percentual"])
    plot_recurrence_by_attribute(df, overall, reports_dir / config.RECURRENCE_BY_ATTRIBUTE_PLOT_FILE)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.exploratory_analysis",
                                     description="Análise exploratória da base Breast Cancer.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args(argv)
    path = run_eda(args.data, args.reports_dir)
    print(path.read_text(encoding="utf-8"))
    print(f"Relatório salvo em: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
