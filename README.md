# Laboratorio de distribuciones · Estadística I

## Ejecutar
`pip install -r requirements.txt` y luego `streamlit run app.py`.

## Publicar
Sube `app.py`, `requirements.txt` y `supabase_schema.sql` a GitHub. En Streamlit Community Cloud, Main file path: `app.py`.

## Ranking compartido con Supabase
El modo sin configuración es local/de prueba. Para un ranking común entre estudiantes:
1. Crea un proyecto en https://supabase.com/.
2. Ejecuta `supabase_schema.sql` en SQL Editor.
3. En Streamlit Cloud > App settings > Secrets, configura:
```toml
SUPABASE_URL = "https://TU-PROYECTO.supabase.co"
SUPABASE_KEY = "TU_SERVICE_ROLE_KEY"
```
4. Reinicia la aplicación. La clave service-role es privilegiada: nunca la subas a GitHub ni la expongas en el código cliente.

## Diseño
- 5 casos de Ingeniería Industrial, tamaños n=300, 500, 400, 350 y 450.
- Asignación reproducible de 2 casos por identificador.
- Datos redondeados a 2 decimales; parámetros con formato de 1 cifra significativa.
- KDE, curva teórica y QQ-plot se actualizan al editar parámetros.
- Envío conjunto de dos casos; 100 puntos por caso (40 familia + 60 parámetros), máximo 200. Error relativo ≤5% da puntaje completo, 5–35% disminuye linealmente y ≥35% no da puntos para ese parámetro. Familia incorrecta = 0 en el caso. El tiempo no se considera.
