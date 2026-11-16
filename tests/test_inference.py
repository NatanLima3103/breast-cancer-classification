"""Testes do módulo de inferência."""

from __future__ import annotations

import json

import joblib
import pytest
from sklearn.svm import SVC

from src import config
from src.data import load_dataset, split_features_target
from src.inference import (
    DISCLAIMER,
    InputValidationError,
    ModelLoadError,
    classify_patient,
    format_result,
    load_model,
    load_patient_json,
    main,
    predict,
    validate_patient,
)
from src.preprocessing import build_pipeline, get_training_categories


@pytest.fixture
def model_path(trained_artifacts):
    return trained_artifacts["model_path"]


# --- Validação do JSON --------------------------------------------------------
def test_load_patient_json_reads_example_file():
    data = load_patient_json(config.EXAMPLE_PATIENT_PATH)
    assert set(data) == set(config.FEATURES)
    assert config.TARGET not in data


def test_load_patient_json_accepts_utf8_bom(example_patient, tmp_path):
    path = tmp_path / "com_bom.json"
    path.write_text("﻿" + json.dumps(example_patient), encoding="utf-8")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert load_patient_json(path) == example_patient


def test_load_patient_json_missing_file(tmp_path):
    with pytest.raises(InputValidationError, match="não encontrado"):
        load_patient_json(tmp_path / "nada.json")


def test_load_patient_json_invalid_syntax(tmp_path):
    path = tmp_path / "ruim.json"
    path.write_text("{ \"age\": ", encoding="utf-8")
    with pytest.raises(InputValidationError, match="JSON inválido"):
        load_patient_json(path)


def test_load_patient_json_must_be_object(tmp_path):
    path = tmp_path / "lista.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(InputValidationError, match="único objeto"):
        load_patient_json(path)


# --- Validação dos atributos --------------------------------------------------
def test_valid_patient_produces_single_row_frame(example_patient):
    frame, warnings = validate_patient(example_patient)
    assert frame.shape == (1, 9)
    assert list(frame.columns) == list(config.FEATURES)
    assert frame.loc[0, "deg-malig"] == 1.0
    assert warnings == []


def test_missing_attributes_are_reported(example_patient):
    del example_patient["age"]
    del example_patient["irradiat"]
    with pytest.raises(InputValidationError) as exc:
        validate_patient(example_patient)
    assert "age" in str(exc.value) and "irradiat" in str(exc.value)


def test_invalid_categories_are_reported_together(example_patient):
    example_patient["menopause"] = "pos-menopausa"
    example_patient["breast"] = "ambas"
    example_patient["deg-malig"] = 4
    with pytest.raises(InputValidationError) as exc:
        validate_patient(example_patient)
    message = str(exc.value)
    assert "menopause" in message and "breast" in message and "deg-malig" in message


@pytest.mark.parametrize("value", [0, 4, 2.5, "alto", "1.5", True, [2]])
def test_invalid_deg_malig_values(example_patient, value):
    example_patient["deg-malig"] = value
    with pytest.raises(InputValidationError, match="deg-malig"):
        validate_patient(example_patient)


@pytest.mark.parametrize("value", [2, "2", 2.0])
def test_valid_deg_malig_representations(example_patient, value):
    example_patient["deg-malig"] = value
    frame, _ = validate_patient(example_patient)
    assert frame.loc[0, "deg-malig"] == 2.0


def test_non_string_category_is_rejected(example_patient):
    example_patient["breast"] = 1
    with pytest.raises(InputValidationError, match="breast"):
        validate_patient(example_patient)


def test_class_column_is_rejected(example_patient):
    example_patient[config.TARGET] = config.POSITIVE_CLASS
    with pytest.raises(InputValidationError, match="não deve ser enviada"):
        validate_patient(example_patient)


def test_unexpected_attribute_is_rejected(example_patient):
    example_patient["peso"] = 70
    with pytest.raises(InputValidationError, match="não reconhecidos"):
        validate_patient(example_patient)


def test_unknown_values_are_imputed_with_warning(example_patient):
    example_patient["node-caps"] = "?"
    example_patient["breast-quad"] = None
    frame, warnings = validate_patient(example_patient)
    assert frame[["node-caps", "breast-quad"]].isna().all(axis=None)
    assert len(warnings) == 2


def test_category_unseen_in_training_requires_opt_in(example_patient, model_path):
    categories = get_training_categories(load_model(model_path))
    example_patient["age"] = "80-89"  # documentado pela UCI, ausente na base
    with pytest.raises(InputValidationError, match="não ocorreu no treinamento"):
        validate_patient(example_patient, categories)
    frame, warnings = validate_patient(example_patient, categories, allow_unseen_categories=True)
    assert frame.loc[0, "age"] == "80-89"
    assert any("80-89" in w for w in warnings)


# --- Modelo e previsão --------------------------------------------------------
def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="python -m src.training"):
        load_model(tmp_path / "best_model.joblib")


def test_invalid_model_file_is_rejected(tmp_path):
    path = tmp_path / "modelo.joblib"
    joblib.dump({"nao": "e um pipeline"}, path)
    with pytest.raises(ModelLoadError, match="Pipeline"):
        load_model(path)


def test_inference_returns_valid_class_and_probabilities(example_patient, model_path):
    result, warnings = classify_patient(example_patient, model_path)
    assert result.predicted_class in config.CLASS_LABELS
    assert warnings == []
    if result.probabilities is not None:
        assert set(result.probabilities) == set(config.CLASS_LABELS)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        assert max(result.probabilities, key=result.probabilities.get) == result.predicted_class


def test_load_model_accepts_imblearn_pipeline(example_patient, model_path):
    from imblearn.pipeline import Pipeline as ImbPipeline

    model = load_model(model_path)
    assert isinstance(model, ImbPipeline)
    assert set(get_training_categories(model)) == set(config.CATEGORICAL_FEATURES)
    result, _ = classify_patient(example_patient, model_path)
    assert result.predicted_class in config.CLASS_LABELS


def test_inference_does_not_require_class(example_patient, model_path):
    assert config.TARGET not in example_patient
    result, _ = classify_patient(example_patient, model_path)
    assert result.predicted_class in config.CLASS_LABELS


def test_saved_pipeline_matches_training_data_predictions(model_path):
    """Compatibilidade: registros da base, validados pela inferência, devem gerar
    a mesma previsão que o pipeline aplicado diretamente ao DataFrame de treino."""
    model = load_model(model_path)
    X, _ = split_features_target(load_dataset())
    sample = X.dropna().head(20)
    direct = model.predict(sample)
    for (_, row), expected in zip(sample.iterrows(), direct):
        record = {k: (int(v) if k == "deg-malig" else v) for k, v in row.items()}
        frame, _ = validate_patient(record, get_training_categories(model))
        assert predict(model, frame).predicted_class == expected


def test_probabilities_hidden_when_estimator_lacks_predict_proba(example_patient, tmp_path):
    X, y = split_features_target(load_dataset())
    pipe = build_pipeline(SVC(random_state=config.RANDOM_STATE), scale=True).fit(X, y)
    frame, _ = validate_patient(example_patient)
    result = predict(pipe, frame)
    assert result.probabilities is None
    assert "não disponíveis" in format_result(result)


def test_format_result_contains_class_and_disclaimer(example_patient, model_path):
    result, _ = classify_patient(example_patient, model_path)
    text = format_result(result)
    assert "Resultado da inferência" in text
    assert f"Classe prevista: {result.predicted_class}" in text
    assert DISCLAIMER in text


# --- Linha de comando ---------------------------------------------------------
def test_cli_with_json(example_patient, model_path, tmp_path, capsys):
    path = tmp_path / "paciente.json"
    path.write_text(json.dumps(example_patient), encoding="utf-8")
    assert main(["--json", str(path), "--model", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "Classe prevista:" in out


def test_cli_with_individual_arguments(model_path, capsys):
    args = ["--age", "40-49", "--menopause", "premeno", "--tumor-size", "30-34",
            "--inv-nodes", "3-5", "--node-caps", "yes", "--deg-malig", "3",
            "--breast", "left", "--breast-quad", "left_low", "--irradiat", "yes",
            "--model", str(model_path)]
    assert main(args) == 0
    assert "Classe prevista:" in capsys.readouterr().out


def test_cli_reports_missing_attribute(model_path, capsys):
    assert main(["--age", "40-49", "--model", str(model_path)]) == 1
    assert "Atributos ausentes" in capsys.readouterr().err


def test_cli_reports_missing_model(example_patient, tmp_path, capsys):
    path = tmp_path / "paciente.json"
    path.write_text(json.dumps(example_patient), encoding="utf-8")
    assert main(["--json", str(path), "--model", str(tmp_path / "x.joblib")]) == 1
    assert "Arquivo do modelo não encontrado" in capsys.readouterr().err


def test_cli_without_input_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "nenhuma paciente informada" in capsys.readouterr().err
