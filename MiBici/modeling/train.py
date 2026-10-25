import typer

from enum import Enum
from loguru import logger

from MiBici.modeling import train_p1, train_p2, train_p3

app = typer.Typer()

class Question(str, Enum):
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"
    all = "all"

TRAINING_MODELS = {
    Question.p2: train_p2,
    Question.p3: train_p3,
    Question.all: [train_p2, train_p3]
}

# Hiperparámetros que acepta run_p1 de P1, con su tipo
P1_HYPERPARAMS = {"n_estimators": int, "max_depth": int, "min_samples_leaf": int, "n_jobs": int}


def run_p1(model: str, hyperparams: dict, search: bool) -> None:
    """Adapta el contrato de train.py a train_p1.run_p1 (que usa all/rf/hgb/ridge y un flag de búsqueda)."""
    desconocidos = set(hyperparams) - set(P1_HYPERPARAMS)
    if desconocidos:
        raise typer.BadParameter(f"P1 solo acepta {sorted(P1_HYPERPARAMS)}; recibió {sorted(desconocidos)}")
    kwargs = {k: P1_HYPERPARAMS[k](v) for k, v in hyperparams.items()}
    train_p1.run_p1(model="all" if model == "none" else model, search=search, **kwargs)



@app.command()
def search(
    question: Question = typer.Option(Question.p1, help="Pregunta a entrenar (p1, p2, p3). Default: p1"),
    model: str = typer.Option("none", help="Nombre del modelo a entrenar")
):
    """Realiza una búsqueda de hiperparámetros para la pregunta y modelo especificados."""
    if question == Question.all:
        raise typer.BadParameter("La búsqueda se hace para una sola pregunta (p1, p2 o p3)")

    if question == Question.p1:
        # solo busqueda con RF
        logger.info("P1 solo tiene búsqueda de RF; se ignora model=...")
        run_p1("rf", {}, search=True)
    else:
        logger.info(f"Realizando búsqueda de hiperparámetros para {question.value}")
        TRAINING_MODELS[question].search(model)
    

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    question: Question = typer.Option(Question.all, help="Pregunta a entrenar (p1, p2, p3, all). Default: all"),
    model: str = typer.Option("none", help="Nombre del modelo a entrenar")
):
    # Extraer los nombres de los hiperparametros recibidos desde CLI
    param_names = [arg.lstrip("-") for arg in ctx.args[::2]]
    hyperparams = dict(zip(param_names, ctx.args[1::2]))

    if question == Question.all:
        logger.info("Entrenando modelos para todas las preguntas con hiperparámetros default")

        ## Correr p1
        run_p1(model, {}, search=True)

        for train_func in TRAINING_MODELS[question]:
            train_func.main(model, {})
    else:
        if question == Question.p1:
            run_p1(model, hyperparams, search=False)
        else:
            logger.info(f"Entrenando modelo para {question.value} con hiperparámetros: {hyperparams}")
            TRAINING_MODELS[question].main(model, hyperparams)

    logger.info("Entrenamiento finalizado")


if __name__ == "__main__":
    app()
