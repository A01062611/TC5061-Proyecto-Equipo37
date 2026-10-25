"""Registro del modelo elegido en el Model Registry y su ciclo de vida con alias.

Se ejecuta con el MLflow Project desde la raíz del repo:
  mlflow run . -e register -P question=p2 -P run_id=<run del modelo elegido> \
    --experiment-name mibici-equipo37 --env-manager local
  mlflow run . -e register -P question=p2 -P run_id=<run del siguiente mejor> -P alias=challenger \
    --experiment-name mibici-equipo37 --env-manager local

El modelo se registra desde el run donde se entrenó, así que cada versión queda ligada a su
corrida de origen y a su experimento. El commit y la versión de los datos son etiquetas del run
y MLflow no los copia a la versión, así que se copian aquí como etiquetas de la versión.

Ciclo de vida: se usan aliases (champion para la versión en producción, challenger para la
siguiente mejor). Al final se verifica que el modelo se puede cargar por
 models:/<nombre>@<alias> y que predice con su ejemplo de entrada.
"""

from enum import Enum

from loguru import logger
import mlflow
import typer

from MiBici.config import MLFLOW_EXPERIMENT
from MiBici.modeling.helper import setup_mlflow


class Question(str, Enum):
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"


class Alias(str, Enum):
    champion = "champion"
    challenger = "challenger"


MODEL_NAMES = {
    Question.p1: "mibici-equipo37-p1-demanda",
    Question.p2: "mibici-equipo37-p2-riesgo",
    Question.p3: "mibici-equipo37-p3-grupos",
}

MODEL_DESCRIPTIONS = {
    Question.p1: (
        "P1 · Demanda diaria por estación. Pronostica cuántos viajes se iniciarán en una estación "
        "el día siguiente a partir del calendario y de los viajes de días anteriores (lags de 1 y 7 días "
        "y promedio de 7 días). Métrica de negocio: MAE de validación, en viajes por estación-día."
    ),
    Question.p2: (
        "P2 · Riesgo de vaciado o llenado. Clasifica cada estación-hora en normal, vaciado o llenado "
        "para las próximas 2 horas, con la ocupación reconstruida a partir de los viajes. Métrica de "
        "negocio: recall@20, la fracción de eventos de riesgo que se cubren visitando 20 estaciones por hora."
    ),
    Question.p3: (
        "P3 · Grupos de estaciones. Asigna cada estación a un grupo según su perfil horario de salidas y "
        "llegadas entre semana y en fin de semana. Recibe el perfil por estación sin estandarizar. "
        "Métricas: silueta, Davies-Bouldin y concordancia (ARI) con Ward."
    ),
}

# Métricas que resumen cada versión en su descripción (las que existan en el run)
SUMMARY_METRICS = {
    Question.p1: ["val_mae", "val_rmse", "test_mae"],
    Question.p2: ["val_recall_at_20", "val_ap_vaciado", "val_ap_llenado", "val_f2_vaciado", "val_f2_llenado"],
    Question.p3: ["train_silhouette", "train_davies_bouldin", "train_ari_vs_ward"],
}


def find_logged_model(run):
    """El modelo registrado con log_model en el run."""
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format="list",
    )
    if not models:
        raise typer.BadParameter(
            f"El run {run.info.run_id} ({run.info.run_name}) no tiene modelo registrado con log_model. "
            "Los baselines de P1 y P2, el run de comparación de P1 y el clustering jerárquico de P3 no tienen modelo."
        )
    return models[0]


def lineage_tags(run, question: Question) -> dict:
    """Linaje de la versión: experimento, corrida de origen, commit y versión de los datos."""
    tags = run.data.tags
    return {
        "experimento": mlflow.MlflowClient().get_experiment(run.info.experiment_id).name,
        "run_origen": run.info.run_id,
        "run_origen_nombre": run.info.run_name,
        "git_commit": tags.get("git_commit", "desconocido"),
        "git_worktree_clean": tags.get("git_worktree_clean", "desconocido"),
        # P1 guarda la versión de los datos como data_version; P2 y P3 como version_datos
        "version_datos": tags.get("version_datos") or tags.get("data_version") or "desconocida",
        "pregunta": question.value.upper(),
        "model_family": tags.get("model_family", "desconocida"),
        "equipo": tags.get("equipo", "37"),
    }


def version_description(run, question: Question) -> str:
    metrics = run.data.metrics
    summary = ", ".join(f"{m}={metrics[m]:.4f}" for m in SUMMARY_METRICS[question] if m in metrics)
    tags = run.data.tags
    return (
        f"{tags.get('model_family', 'modelo')} del run {run.info.run_name} ({run.info.run_id}), "
        f"commit {tags.get('git_commit', 'desconocido')[:7]}. {summary}"
    )


def register(question: Question, run_id: str, alias: Alias):
    """Registra el modelo del run (o reusa su versión si ya existe), le pone el alias y lo verifica."""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    name = MODEL_NAMES[question]
    logged_model = find_logged_model(run)

    # Si el run ya se registró antes, se reusa su versión: cambiar de alias no crea versiones repetidas
    existing = client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    if existing:
        version = max(existing, key=lambda v: int(v.version))
        logger.info(f"El run {run_id} ya es la versión {version.version} de {name}; se reusa.")
    else:
        version = mlflow.register_model(logged_model.model_uri, name)
        logger.info(f"Registrada la versión {version.version} de {name} desde {logged_model.model_uri}.")

    for key, value in lineage_tags(run, question).items():
        client.set_model_version_tag(name, version.version, key, value)
    client.update_model_version(name, version.version, description=version_description(run, question))
    client.update_registered_model(name, description=MODEL_DESCRIPTIONS[question])
    client.set_registered_model_tag(name, "pregunta", question.value.upper())
    client.set_registered_model_tag(name, "equipo", "37")

    client.set_registered_model_alias(name, alias.value, version.version)
    logger.info(f"Alias {alias.value} → versión {version.version} de {name}.")
    return version


def verify(question: Question, alias: Alias) -> dict:
    """Carga el modelo por nombre y alias y predice con el ejemplo de entrada guardado junto al modelo."""
    uri = f"models:/{MODEL_NAMES[question]}@{alias.value}"
    model = mlflow.pyfunc.load_model(uri)
    example = mlflow.models.Model.load(uri).load_input_example(mlflow.artifacts.download_artifacts(uri))
    predictions = model.predict(example)
    logger.success(f"{uri} carga y predice: {list(predictions)[:5]}")
    return {"uri": uri, "filas_ejemplo": len(example), "predicciones": [str(p) for p in list(predictions)[:5]]}


def main(
    question: Question = typer.Option(..., help="Pregunta del modelo (p1, p2 o p3)"),
    run_id: str = typer.Option(..., help="Run donde se entrenó el modelo elegido"),
    alias: Alias = typer.Option(Alias.champion, help="champion (producción) o challenger (siguiente mejor)"),
):
    setup_mlflow()
    version = register(question, run_id, alias)
    check = verify(question, alias)

    # El run que crea mlflow run para este entry point deja constancia del registro
    with mlflow.start_run(run_name=f"{question.value.upper()} - registro {alias.value}"):
        mlflow.set_tags({
            "pregunta": question.value.upper(),
            "stage_of_work": "registry",
            "equipo": "37",
            "modelo_registrado": version.name,
            "version_registrada": version.version,
            "alias": alias.value,
            "run_origen": run_id,
        })
        mlflow.log_dict(check, "results/verificacion_carga.json")
    logger.success(f"{version.name} v{version.version} @{alias.value} en el experimento {MLFLOW_EXPERIMENT}.")


if __name__ == "__main__":
    typer.run(main)
