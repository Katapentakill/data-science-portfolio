df = pd.DataFrame(df)

# Convertir la notación científica a decimal y manejar valores no numéricos
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir de string a float, reemplaza errores con NaN

# Manejo de valores negativos y cero
offsets = {}  # Diccionario para almacenar los offsets usados en cada columna

def log_transform(column):
    # Calcular el valor mínimo de la columna
    column_min = column.min()
    
    if column_min <= 0:
        # Calcular el offset necesario para que todos los valores sean positivos
        offset = abs(column_min) + column.max() + 1  # Sumar el valor máximo también para asegurar un rango amplio
        column_adjusted = column + offset  # Ajustar la columna
        offsets[column.name] = offset  # Guardar el offset
    else:
        column_adjusted = column
        offsets[column.name] = 0  # Sin ajuste si todos los valores ya son positivos
    
    # Aplicar logaritmo solo a los valores positivos
    return np.log10(column_adjusted)

# Aplicar la transformación logarítmica a las columnas deseadas
log_transformed_columns = ['Mrr', 'Mtt', 'Mpp', 'Mrt']
for col in log_transformed_columns:
    df[col + '_log'] = log_transform(df[col])

# Revertir la transformación logarítmica y el ajuste de offset
def inverse_log_transform(column_log, column_name):
    offset = offsets[column_name]  # Obtener el offset que se usó en la transformación
    # Revertir la transformación logarítmica y restar el offset
    return (10 ** column_log) - offset

# Aplicar la operación inversa para revertir los valores a su forma original
for col in log_transformed_columns:
    df[col + '_reverted'] = inverse_log_transform(df[col + '_log'], col)

# Mostrar los resultados de las primeras tres filas usando tabulate
print("\nResultados de las primeras tres filas:")
print(tabulate(df.head(3), headers='keys', tablefmt='psql', showindex=False))

# Guardar el nuevo archivo con las nuevas características
df.to_csv('archivo_transformado_normalizado.csv', index=False)

print("Ingeniería de características completada, columnas 'date' y 'time' eliminadas, y archivo guardado.")


# Convertir la columna de fecha a un formato de fecha para facilitar la búsqueda de terremotos
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])

# Encontrar la magnitud máxima de los terremotos por año
max_magnitude_per_year = df.groupby('year')['magnitude'].max().reset_index()
max_magnitude_per_year.columns = ['year', 'max_magnitude']

# Encontrar la magnitud anterior al terremoto más grande
def get_previous_magnitude(group):
    # Ordenar por fecha
    group = group.sort_values(by='date')
    # Obtener el índice de la magnitud máxima
    max_idx = group['magnitude'].idxmax()
    # Si hay un terremoto anterior, devolver su magnitud
    if max_idx > 0:
        return group.loc[max_idx - 1, 'magnitude']
    return np.nan

# Aplicar la función para obtener la magnitud anterior por año
previous_magnitude_per_year = df.groupby('year').apply(get_previous_magnitude).reset_index()
previous_magnitude_per_year.columns = ['year', 'previous_magnitude']

# Calcular la cantidad de terremotos por año
earthquake_count_per_year = df['year'].value_counts().sort_index().reset_index()
earthquake_count_per_year.columns = ['year', 'count']

# Unir las tablas por año
combined_data = pd.merge(max_magnitude_per_year, previous_magnitude_per_year, on='year')
combined_data = pd.merge(combined_data, earthquake_count_per_year, on='year')

# Crear un gráfico combinado
plt.figure(figsize=(12, 6))

# Gráfico de líneas para la magnitud máxima
plt.subplot(2, 1, 1)
sns.lineplot(data=combined_data, x='year', y='max_magnitude', marker='o', color='red', label='Magnitud Máxima')
sns.lineplot(data=combined_data, x='year', y='previous_magnitude', marker='o', color='blue', label='Magnitud Anterior')
plt.title('Magnitud Máxima y Anterior de Terremotos por Año', fontsize=16)
plt.xlabel('Año', fontsize=14)
plt.ylabel('Magnitud', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.grid()


plt.show()