"""Importa el champion original al Model Registry local."""

import mlflow
import mlflow.sklearn
import numpy as np

from mlflow.models import Model

from MiBici.config import PROJ_ROOT
from MiBici.modeling.helper import setup_mlflow, SKOPS_TRUSTED_TYPES
from MiBici.modeling.register import Alias, Question, register, verify


def main():
  model_path = PROJ_ROOT / "models" / "mibici-equipo37-p2-riesgo"

  if not (model_path / "MLmodel").is_file():
    raise FileNotFoundError(f"No existe el champion en {model_path}")

  setup_mlflow()

  # Recuperar el modelo original y su firma.
  metadata = Model.load(str(model_path))
  example = metadata.load_input_example(str(model_path))
  champion = mlflow.sklearn.load_model(str(model_path))

  pred_original = champion.predict(example)

  # Importar el modelo original sin volver a entrenarlo.
  with mlflow.start_run(run_name="P2 - Champion original importado") as run:
    mlflow.log_param("tau", 0.15)

    mlflow.set_tags({
      "pregunta": "P2",
      "equipo": "37",
      "model_family": "hist_gradient_boosting",
      "stage_of_work": "import",
      "model_role": "champion",
      "source_run_id": metadata.run_id or "desconocido",
      "tau_source": "recalculado_en_validacion_local",
    })

    logged = mlflow.sklearn.log_model(
      sk_model=champion,
      name="model",
      signature=metadata.signature,
      input_example=example,
      skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

    # Verificar que la importación conserve las predicciones.
    recuperado = mlflow.sklearn.load_model(logged.model_uri)
    np.testing.assert_array_equal(
      pred_original,
      recuperado.predict(example),
    )

    run_id = run.info.run_id

  version = register(Question.p2, run_id, Alias.champion)
  verify(Question.p2, Alias.champion)

  print(f"Champion importado correctamente: versión {version.version}")
  print(f"Run ID local: {run_id}")


if __name__ == "__main__":
  main()