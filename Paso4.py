import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def main():
    print("✅ Inicio del proceso...")

    # ===============================
    # 📥 1. Carga y Preprocesamiento
    # ===============================
    df = pd.read_excel("data_split_personalizado.xlsx")
    df = df[df['AÑO'].isin([2020, 2021])]

    for col in ['COMPUTADOR EN VIVIENDA', 'INTERNET EN VIVIENDA']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df = df[df['COMPUTADOR EN VIVIENDA'].isin(['si', 'no'])]
    df = df[df['INTERNET EN VIVIENDA'].isin(['si', 'no'])]

    df = df[['PUNTAJE GLOBAL', 'COMPUTADOR EN VIVIENDA', 'INTERNET EN VIVIENDA', 'ESTRATO DE VIVIENDA']].dropna()

    df['compu_bin'] = (df['COMPUTADOR EN VIVIENDA'] == 'si').astype(int)
    df['internet_bin'] = (df['INTERNET EN VIVIENDA'] == 'si').astype(int)

    df['estrato'] = df['ESTRATO DE VIVIENDA'].astype(str).str.extract(r'(\d+)')
    df['estrato'] = pd.to_numeric(df['estrato'], errors='coerce')
    df['PUNTAJE GLOBAL'] = pd.to_numeric(df['PUNTAJE GLOBAL'], errors='coerce')
    df = df.dropna(subset=['estrato', 'PUNTAJE GLOBAL'])

    y_data = df['PUNTAJE GLOBAL'].values.astype(float)
    x_compu = df['compu_bin'].values
    x_internet = df['internet_bin'].values
    x_estrato = df['estrato'].values

    print(f"\n🔢 Número de observaciones utilizadas: {len(y_data)}")

    # ===============================
    # 📊 2. Modelo de Regresión Bayesiana
    # ===============================
    with pm.Model() as modelo_regresion:
        alpha = pm.Normal("alpha", mu=300, sigma=100)
        beta_compu = pm.Normal("beta_compu", mu=0, sigma=50)
        beta_internet = pm.Normal("beta_internet", mu=0, sigma=50)
        beta_estrato = pm.Normal("beta_estrato", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=50)

        mu = (alpha +
              beta_compu * x_compu +
              beta_internet * x_internet +
              beta_estrato * x_estrato)

        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

        trace = pm.sample(1000, tune=500, return_inferencedata=True, random_seed=42)

    # ===============================
    # 📈 3. Resultados Posteriores
    # ===============================
    resumen = az.summary(trace, var_names=["alpha", "beta_compu", "beta_internet", "beta_estrato", "sigma"], hdi_prob=0.80)
    print("\n📌 Resumen de la distribución posterior (80% HDI):")
    print(resumen)

    az.plot_posterior(trace, var_names=["alpha", "beta_compu", "beta_internet", "beta_estrato"], hdi_prob=0.80)
    plt.suptitle("Distribuciones Posteriores de los Coeficientes (80% HDI)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ===============================
    # 🎯 4. Predicción Futura con 80% HDI
    # ===============================
    with modelo_regresion:
        pred = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)

    predicciones = pred.posterior_predictive["y_obs"].values
    print("📏 Forma de predicciones:", predicciones.shape)

    hdi_80 = az.hdi(predicciones, hdi_prob=0.80)

    print(f"\n🎯 Intervalo de credibilidad del 80% para predicciones futuras de PUNTAJE GLOBAL:")
    print(f" ▸ Límite inferior promedio: {hdi_80[:, 0].mean():.2f}")
    print(f" ▸ Límite superior promedio: {hdi_80[:, 1].mean():.2f}")

    # Subconjunto pequeño para evitar errores de memoria
    subset = predicciones[:, :100].flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(subset, bins=30, color='lightblue', edgecolor='black', density=True)
    plt.axvline(hdi_80[:, 0].mean(), color='red', linestyle='--', label='Límite inferior 80%')
    plt.axvline(hdi_80[:, 1].mean(), color='red', linestyle='--', label='Límite superior 80%')
    plt.title("Predicción futura de PUNTAJE GLOBAL (80% HDI)")
    plt.xlabel("PUNTAJE GLOBAL")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===============================
# 🚀 Ejecutar
# ===============================
if __name__ == "__main__":
    main()
