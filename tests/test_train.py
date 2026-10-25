"""Pruebas del parseo de hiperparámetros que llegan desde mlflow run."""

from MiBici.modeling.train import parse_hyperparams


def test_declarados_vacios_se_descartan():
  args = ["--n_estimators=300", "--max_depth=", "--class_weight="]
  assert parse_hyperparams(args) == {"n_estimators": "300"}


def test_no_declarados_con_espacio():
  args = ["--learning_rate=0.05", "--extra", "7"]
  assert parse_hyperparams(args) == {"learning_rate": "0.05", "extra": "7"}


def test_valor_none_se_conserva_como_texto():
  assert parse_hyperparams(["--class_weight=None"]) == {"class_weight": "None"}
