import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("data.csv")

# Filtrar las filas donde 'periodo' sea igual a 2020
df = df[df["PERIODO"] == 2020]

# Guardar el archivo modificado si lo necesitas
df.to_csv("dat_filtrado.csv", index=False)

print(df.head())  # Opcional, para visualizar las primeras filas
