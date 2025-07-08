import pandas as pd

# Leer el archivo CSV en lugar de Excel
df = pd.read_csv("data.csv")

# Suponiendo que la columna que contiene los datos es la primera
nombre_columna = df.columns[0]
print(f"Nombre de la columna con los datos: {nombre_columna}")

# Aplicamos str.split para separar la cadena en columnas según la coma
df_split = df[nombre_columna].str.split(",", expand=True)

# Define una lista con los nombres de cada columna según corresponda.
nombres_columnas = [
    "GÉNERO", "AÑO", "PAÍS DE RESIDENCIA", "DEPARTAMENTO DE RESIDENCIA", 
    "MUNICIPIO DE RESIDENCIA", "VALOR DE LA MATRÍCULA UNIVERSITARIA", 
    "PAGO PROPIO DE MATRÍCULA", "NIVEL DE EDUCACIÓN DEL PADRE", 
    "NIVEL DE EDUCACIÓN DE LA MADRE", "ESTRATO DE VIVIENDA", 
    "INTERNET EN VIVIENDA", "COMPUTADOR EN VIVIENDA", "LAVADORA EN VIVIENDA", 
    "HORNO MICROHONDAS EN VIVIENDA", "TELEVISIÓN EN VIVIENDA", 
    "AUTOMÓVIL EN VIVIENDA", "MOTOCICLETA EN VIVIENDA", "CONSOLA EN VIVIENDA", 
    "BAÑO COMPARTIDO EN VIVIENDA", "HORAS DE TRABAJO SEMANALES DEL ESTUDIANTE", 
    "NOMBRE DE LA INSTITUCIÓN EDUCATIVA", "PROGRAMA ACADÉMICO DEL ESTUDIANTE", 
    "PUNTAJE GLOBAL", "--", "CANTIDAD DE DATOS", "--", "--"
]

# Asignar los nombres personalizados a las nuevas columnas
df_split.columns = nombres_columnas[:df_split.shape[1]]  # Para evitar errores si hay más o menos columnas

# Opcional: eliminar espacios en blanco adicionales en cada celda
df_split = df_split.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Visualizar el DataFrame resultante
print("\nDataFrame con columnas separadas y renombradas:")
print(df_split.head())

# Guardar el resultado en un nuevo archivo CSV
df_split.to_csv("data_split_personalizado.csv", index=False)
