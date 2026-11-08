FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  MODEL_NAME=mibici-equipo37-p2-riesgo \
  MODEL_DIR=/app/models/mibici-equipo37-p2-riesgo

WORKDIR /app

# Instalar dependencias de la API
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt

# Crear usuario sin privilegios de root
RUN groupadd --gid 10001 appgroup \
  && useradd --uid 10001 --gid 10001 --create-home appuser

# Copiar la API y el modelo exportado
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup models/mibici-equipo37-p2-riesgo/ ./models/mibici-equipo37-p2-riesgo/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import json, sys, urllib.request; status = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)).get('status'); sys.exit(0 if status == 'ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]