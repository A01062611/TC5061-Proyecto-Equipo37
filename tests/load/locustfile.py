import random
from locust import HttpUser, between, task


def payload():
    ocup = round(random.uniform(0, 1), 3)
    dia = random.randint(0, 6)
    return {
        "hora": random.randint(0, 23),
        "ocup_est": ocup,
        "estado_est": 1 if ocup <= 0.05 else 2 if ocup >= 0.95 else 0,
        "tend_1h": random.randint(-5, 5),
        "sal_1h": random.randint(0, 10),
        "sal_2h": random.randint(0, 10),
        "sal_ayer_2h": random.choice([None, random.randint(0, 15)]),
        "sal_sem_2h": random.randint(0, 15),
        "sal_media7_2h": round(random.uniform(0, 10), 2),
        "lle_1h": random.randint(0, 10),
        "lle_2h": random.randint(0, 10),
        "lle_ayer_2h": random.choice([None, random.randint(0, 15)]),
        "lle_sem_2h": random.randint(0, 15),
        "lle_media7_2h": round(random.uniform(0, 10), 2),
        "frac_vacio_7d": round(random.uniform(0, 1), 2),
        "frac_lleno_7d": round(random.uniform(0, 1), 2),
        "riesgo_ayer": random.choice([None, 0, 1, 2]),
        "dia_semana": dia,
        "fin_semana": int(dia >= 5),
        "festivo": int(random.random() < 0.05),
        "C": random.randint(10, 40),
        "lat": round(random.uniform(20.60, 20.72), 4),
        "lon": round(random.uniform(-103.42, -103.30), 4),
    }


class OperadorMiBici(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predecir(self):
        with self.client.post("/predict_p2/", json=payload(), catch_response=True, timeout=8) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif r.json().get("prediction") not in (0, 1, 2):
                r.failure("Predicción inválida")