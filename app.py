import streamlit as st, numpy as np, pandas as pd, matplotlib.pyplot as plt, requests, json, hashlib
from scipy import stats
from scipy.stats import gaussian_kde
from datetime import datetime, timezone

st.set_page_config(page_title='Laboratorio de distribuciones | Estadística I', layout='wide')

# Banco de 5 casos estructurados de Ingeniería Industrial con contextos amplios, rigurosos y detallados
CASES = {
    'C01': dict(
        title='Tiempo de ensamble de una unidad electrónica',
        area='Ingeniería de Métodos y Tiempos',
        context='En una planta automatizada de manufactura de equipos de precisión, la división de Ingeniería de Métodos lleva a cabo un estudio analítico sobre el tiempo de ciclo operativo. Se mide el tiempo (en minutos) que toma a un técnico completar el ensamble completo de una unidad de control tras la estandarización del proceso. Las observaciones se recolectaron durante condiciones de estabilidad operativa sin interrupciones técnicas.',
        variable='Tiempo de ensamble (min)',
        kind='normal',
        n=300,
        p={'mu': 18.0, 'sigma': 2.4},
        seed=8121
    ),
    'C02': dict(
        title='Duración de reparaciones en equipos críticos',
        area='Gestión del Mantenimiento Industrial',
        context='El departamento de Gestión de Mantenimiento registra la duración (en horas) de las intervenciones correctivas no programadas efectuadas sobre el sistema principal de bombeo industrial. La variabilidad observada presenta una concentración inicial en fallas menores de rápida solución, junto con una cola superior extendida originada por diagnósticos complejos y sustitución de piezas de gran envergadura.',
        variable='Duración de reparación (h)',
        kind='lognormal',
        n=500,
        p={'mu_log': 1.15, 'sigma_log': 0.55},
        seed=9284
    ),
    'C03': dict(
        title='Vida útil de componentes mecánicos',
        area='Ingeniería de Confiabilidad y Calidad',
        context='En la unidad de Ensayos Destructivos y Confiabilidad, se evalúa el tiempo transcurrido hasta el fallo (en horas de trabajo continuo) de rodamientos de alta precisión sometidos a un protocolo de fatiga acelerada. La recolección empírica busca caracterizar la tasa de falla del componente crítico previa a su implementación masiva.',
        variable='Tiempo hasta la falla (h)',
        kind='weibull',
        n=400,
        p={'shape': 2.4, 'scale': 1200.0},
        seed=10473
    ),
    'C04': dict(
        title='Intervalo entre llegadas a la estación de control',
        area='Control Estadístico de Procesos',
        context='Dentro del flujo logístico interno de una planta metalmecánica, se evalúa el tiempo (en minutos) transcurrido entre arribos consecutivos de piezas moldeadas a la celda de inspección dimensional automatizada. Se asume una tasa constante de flujo operacional y eventos aleatorios independientes en el tiempo.',
        variable='Intervalo de llegada (min)',
        kind='exponential',
        n=350,
        p={'scale': 4.8},
        seed=11762
    ),
    'C05': dict(
        title='Consumo diario de insumos industriales',
        area='Administración de Inventarios y Logística',
        context='La sección de Control de Inventarios realiza el seguimiento del consumo diario (en kilogramos) de polímero granulado consumido en la máquina de moldeo por inyección. Las variaciones registradas se encuentran acotadas dentro de un rango estrictamente definido por los límites de capacidad teórica y lotes de producción estándar.',
        variable='Consumo diario (kg)',
        kind='uniform',
        n=450,
        p={'low': 22.0, 'high': 38.0},
        seed=12991
    )
}

LABEL = {
    'normal': 'Normal',
    'lognormal': 'Lognormal',
    'weibull': 'Weibull',
    'exponential': 'Exponencial',
    'uniform': 'Uniforme',
    'triangular': 'Triangular'
}

def data(c):
    r = np.random.default_rng(c['seed'])
    p = c['p']
    n = c['n']
    k = c['kind']
    if k == 'normal':
        x = r.normal(p['mu'], p['sigma'], n)
    elif k == 'lognormal':
        x = r.lognormal(p['mu_log'], p['sigma_log'], n)
    elif k == 'weibull':
        x = p['scale'] * r.weibull(p['shape'], n)
    elif k == 'exponential':
        x = r.exponential(p['scale'], n)
    else:
        x = r.uniform(p['low'], p['high'], n)
    return np.round(x, 2)

def assigned(s):
    d = hashlib.sha256(('estadistica1-2026' + s.lower()).encode()).digest()
    r = np.random.default_rng(int.from_bytes(d[:8], 'big'))
    return sorted(r.choice(list(CASES), 2, replace=False).tolist())

def pack(dist, v):
    if dist == 'normal':
        return dict(mu=v[0] if len(v) > 0 else 0.0, sigma=v[1] if len(v) > 1 else 1.0)
    elif dist == 'lognormal':
        return dict(mu_log=v[0] if len(v) > 0 else 0.0, sigma_log=v[1] if len(v) > 1 else 1.0)
    elif dist == 'weibull':
        return dict(shape=v[0] if len(v) > 0 else 1.0, scale=v[1] if len(v) > 1 else 1.0)
    elif dist == 'exponential':
        return dict(scale=v[0] if len(v) > 0 else 1.0)
    elif dist == 'uniform':
        return dict(low=v[0] if len(v) > 0 else 0.0, high=v[1] if len(v) > 1 else 1.0)
    elif dist == 'triangular':
        return dict(low=v[0] if len(v) > 0 else 0.0, mode=v[1] if len(v) > 1 else 0.5, high=v[2] if len(v) > 2 else 1.0)
    return {}

def pdf(d, p, x):
    try:
        if d == 'normal':
            return stats.norm.pdf(x, loc=p['mu'], scale=p['sigma'])
        if d == 'lognormal':
            return stats.lognorm.pdf(x, s=p['sigma_log'], scale=np.exp(p['mu_log']))
        if d == 'weibull':
            return stats.weibull_min.pdf(x, c=p['shape'], scale=p['scale'])
        if d == 'exponential':
            return stats.expon.pdf(x, scale=p['scale'])
        if d == 'uniform':
            return stats.uniform.pdf(x, loc=p['low'], scale=p['high'] - p['low'])
        if d == 'triangular':
            return stats.triang.pdf(x, c=(p['mode'] - p['low']) / (p['high'] - p['low']), loc=p['low'], scale=p['high'] - p['low'])
    except:
        pass
    return np.zeros_like(x)

def qtheory(d, p, q):
    if d == 'normal':
        return stats.norm.ppf(q, loc=p['mu'], scale=p['sigma'])
    if d == 'lognormal':
        return stats.lognorm.ppf(q, s=p['sigma_log'], scale=np.exp(p['mu_log']))
    if d == 'weibull':
        return stats.weibull_min.ppf(q, c=p['shape'], scale=p['scale'])
    if d == 'exponential':
        return stats.expon.ppf(q, scale=p['scale'])
    if d == 'uniform':
        return stats.uniform.ppf(q, loc=p['low'], scale=p['high'] - p['low'])
    if d == 'triangular':
        return stats.triang.ppf(q, c=(p['mode'] - p['low']) / (p['high'] - p['low']), loc=p['low'], scale=p['high'] - p['low'])

def score(cid, d, p):
    c = CASES[cid]
    truth = c['p']
    if d != c['kind']:
        return 0, 0
    q = []
    for key, t in truth.items():
        e = abs(p[key] - t) / abs(t)
        q.append(1.0 if e <= 0.05 else (0.0 if e >= 0.35 else (0.35 - e) / 0.30))
    pts = round(40 + 60 * np.mean(q))
    return pts, pts

def cloud():
    return bool(st.secrets.get('SUPABASE_URL', '') and st.secrets.get('SUPABASE_KEY', ''))

def headers():
    k = st.secrets['SUPABASE_KEY']
    return {'apikey': k, 'Authorization': 'Bearer ' + k, 'Content-Type': 'application/json', 'Prefer': 'return=representation'}

def submit(row):
    if not cloud():
        st.session_state.setdefault('rank_local', [])
        st.session_state.rank_local = [x for x in st.session_state.rank_local if x['student_id'] != row['student_id']]
        st.session_state.rank_local.append(row)
        return 'local'
    try:
        u = st.secrets['SUPABASE_URL'].rstrip('/') + '/rest/v1/ranking'
        h = headers()
        h['Prefer'] = 'resolution=merge-duplicates,return=representation'
        r = requests.post(u, headers=h, params={'on_conflict': 'student_id'}, json=row, timeout=15)
        return 'cloud' if r.status_code in (200, 201, 204) else 'error ' + r.text[:200]
    except Exception as e:
        return 'error ' + str(e)

def ranking():
    if not cloud():
        return pd.DataFrame(st.session_state.get('rank_local', []))
    try:
        u = st.secrets['SUPABASE_URL'].rstrip('/') + '/rest/v1/ranking'
        r = requests.get(u, headers=headers(), params={'select': 'student_id,score,submitted_at,case_1,case_2,detail', 'order': 'score.desc,submitted_at.asc', 'limit': '1000'}, timeout=15)
        return pd.DataFrame(r.json()) if r.ok else pd.DataFrame()
    except:
        return pd.DataFrame()

# Banner principal
st.markdown('''<style>.block-container{max-width:1250px;padding-top:1.5rem}.hero{padding:1.5rem 1.8rem;border-radius:16px;background:linear-gradient(115deg,#142b4a,#245b75);color:white;margin-bottom:1rem}.hero h1{font-size:2rem;margin:0}.hero p{color:#dceaf2;margin:.4rem 0 0}.stTabs [data-baseweb=tab]{height:46px;padding:0 18px}</style><div class="hero"><h1>Laboratorio de distribuciones 📊</h1><p>Estadística I · Ingeniería Industrial · Modelación Teórica y Ajuste Empírico</p></div>''', unsafe_allow_html=True)

# Barra lateral (Sidebar)
with st.sidebar:
    st.markdown('### ⚙️ Identificación')
    student = st.text_input('Nombre y Código', placeholder='Ej. Carlos Ramírez - 2194852').strip()
    st.caption('Conserva la estructura formal (ej. Carlos Ramírez - 2194852).')
    
    st.divider()
    with st.expander('📋 Guía Metodológica y Reglas', expanded=False):
        st.markdown('''**Objetivo Institucional:**  
Identificar la función de probabilidad teórica óptima para variables continuas en Ingeniería Industrial y estimar sus parámetros poblacionales mediante diagnóstico gráfico dual (KDE y Q-Q Plot).

**Secuencia de Evaluación:**
1. **Análisis de Contexto:** Examine la definición técnica de la variable y las restricciones del proceso operacional.
2. **Revisión Muestral:** Inspeccione el conjunto de observaciones cuantitativas en la tabla o descargue el archivo CSV.
3. **Selección de Familia:** Formule su hipótesis seleccionando una distribución entre Normal, Lognormal, Weibull, Exponencial, Uniforme o Triangular.
4. **Estimación Paramétrica:** Digite los valores numéricos correspondientes a una cifra decimal. Observe la concordancia gráfica en tiempo real.
5. **Evaluación Gráfica:** Verifique la superposición de la densidad teórica (línea punteada) sobre la estimación no paramétrica (KDE) y analice la linealidad de la trayectoria en el gráfico Q-Q.
6. **Consolidación y Registro:** Remita la respuesta integrada de ambos casos para su evaluación en la plataforma.

**Criterios de Puntuación:**
* Cada caso asignado otorga hasta un máximo de **100 puntos** (máximo acumulado de **200 puntos**).
* **40 puntos:** Identificación correcta de la familia probabilística.
* **60 puntos:** Precisión de los parámetros ajustados.
* **Margen de Tolerancia Paramétrica:**
  - Error relativo $\le 5\%$ : Puntuación máxima del parámetro.
  - Error relativo entre $5\%$ y $35\%$ : Penalización lineal proporcional.
  - Error relativo $\ge 35\%$ : 0 puntos en el parámetro.
  - En caso de seleccionar una familia incorrecta, la puntuación total del caso será de 0 puntos.''')

t1, t2 = st.tabs(['Actividad', 'Ranking'])

if student:
    ids = assigned(student)
    st.session_state['student'] = student
    st.session_state['ids'] = ids
    st.session_state.setdefault('answers', {})
else:
    ids = []
    st.info('👈 Por favor, ingrese su nombre y código en el panel lateral para habilitar los casos asignados.')

with t1:
    if not student:
        st.write('La plataforma habilitará la actividad una vez registrado su identificador.')
    else:
        st.subheader('Casos de Estudio Asignados')
        st.caption('Las representaciones gráficas responden de manera inmediata a la parametrización ingresada.')
        
        for ix, cid in enumerate(ids, 1):
            c = CASES[cid]
            x = data(c)
            with st.container(border=True):
                st.markdown(f'#### Caso {ix} · {c["title"]}')
                st.caption(f'{c["area"]} · Muestra n = {c["n"]}')
                st.write(c['context'])
                
                with st.expander('Consultar o descargar los datos de la muestra'):
                    st.dataframe(pd.DataFrame({c['variable']: x}), height=180, use_container_width=True)
                    st.download_button('Descargar Muestra (CSV)', pd.DataFrame({c['variable']: x}).to_csv(index=False).encode(), f'{cid}_muestra.csv', 'text/csv', key='dl' + cid)
                
                l, r = st.columns([0.9, 1.4], gap='large')
                with l:
                    d = st.selectbox('Distribución candidata', list(LABEL), format_func=lambda z: LABEL[z], key='d' + cid)
                    st.markdown('**Parámetros (1 cifra decimal)**')
                    
                    if d == 'normal':
                        v = [
                            st.number_input('Media μ', value=round(float(np.mean(x)), 1), format='%.1f', step=0.1, key='m' + cid),
                            st.number_input('Desviación σ', value=round(float(np.std(x, ddof=1)), 1), min_value=0.1, format='%.1f', step=0.1, key='s' + cid)
                        ]
                    elif d == 'lognormal':
                        v = [
                            st.number_input('Media logarítmica μ_log', value=1.0, format='%.1f', step=0.1, key='ml' + cid),
                            st.number_input('Desviación logarítmica σ_log', value=0.5, min_value=0.1, format='%.1f', step=0.1, key='sl' + cid)
                        ]
                    elif d == 'weibull':
                        v = [
                            st.number_input('Parámetro de forma k', value=2.0, min_value=0.1, format='%.1f', step=0.1, key='k' + cid),
                            st.number_input('Parámetro de escala λ', value=round(float(np.mean(x)), 1), min_value=0.1, format='%.1f', step=0.1, key='w' + cid)
                        ]
                    elif d == 'exponential':
                        v = [
                            st.number_input('Parámetro de escala / media λ', value=round(float(np.mean(x)), 1), min_value=0.1, format='%.1f', step=0.1, key='e' + cid)
                        ]
                    elif d == 'uniform':
                        v = [
                            st.number_input('Límite inferior a', value=round(float(np.min(x)), 1), format='%.1f', step=0.1, key='lo' + cid),
                            st.number_input('Límite superior b', value=round(float(np.max(x)), 1), format='%.1f', step=0.1, key='hi' + cid)
                        ]
                    else: # triangular
                        v = [
                            st.number_input('Límite inferior a', value=round(float(np.min(x)), 1), format='%.1f', step=0.1, key='tl' + cid),
                            st.number_input('Moda c', value=round(float(np.median(x)), 1), format='%.1f', step=0.1, key='tm' + cid),
                            st.number_input('Límite superior b', value=round(float(np.max(x)), 1), format='%.1f', step=0.1, key='th' + cid)
                        ]
                    
                    p = pack(d, v)
                    valid = all(np.isfinite(vv) for vv in v)
                    if d == 'normal' and v[1] <= 0: valid = False
                    if d == 'lognormal' and v[1] <= 0: valid = False
                    if d == 'weibull' and (v[0] <= 0 or v[1] <= 0): valid = False
                    if d == 'exponential' and v[0] <= 0: valid = False
                    if d == 'uniform' and v[1] <= v[0]: valid = False
                    if d == 'triangular' and not (v[0] < v[1] < v[2]): valid = False
                    
                    st.session_state.answers[cid] = {'dist': d, 'params': p, 'valid': valid}
                
                with r:
                    st.markdown('**Previsualización Gráfica Dinámica**')
                    if valid:
                        xx = np.linspace(float(x.min()), float(x.max()), 350)
                        
                        # Gráfico 1: Densidad teórica vs KDE observada
                        fig1, ax1 = plt.subplots(figsize=(6.5, 3.0))
                        ax1.hist(x, bins='fd', density=True, alpha=0.22, color='#245b75', edgecolor='white', label='Histrograma de Datos')
                        try:
                            kde = gaussian_kde(x)
                            ax1.plot(xx, kde(xx), color='#142b4a', lw=2.0, label='KDE Muestral')
                        except:
                            pass
                        # Modelo candidato con línea PUNTEADA según requerimiento
                        ax1.plot(xx, pdf(d, p, xx), color='#d97735', lw=2.2, ls='--', label='Modelo Teórico Candidato')
                        ax1.set_title('Densidad Teórica vs KDE Observada', fontsize=11, fontweight='bold', pad=8)
                        ax1.set_xlabel(c['variable'], fontsize=9)
                        ax1.set_ylabel('Densidad', fontsize=9)
                        ax1.grid(True, alpha=0.2)
                        ax1.legend(frameon=False, fontsize=8)
                        fig1.tight_layout()
                        st.pyplot(fig1, use_container_width=True)
                        plt.close(fig1)
                        
                        # Gráfico 2: Q-Q Plot
                        fig2, ax2 = plt.subplots(figsize=(6.5, 2.7))
                        q = (np.arange(1, len(x) + 1) - 0.5) / len(x)
                        sx = np.sort(x)
                        tq = qtheory(d, p, q)
                        mask = np.isfinite(tq)
                        ax2.scatter(tq[mask], sx[mask], s=12, alpha=0.5, color='#245b75')
                        if mask.sum() > 1:
                            lo_v = min(tq[mask].min(), sx.min())
                            hi_v = max(tq[mask].max(), sx.max())
                            ax2.plot([lo_v, hi_v], [lo_v, hi_v], ls='--', color='#d97735', lw=1.8)
                        ax2.set_title('Gráfico Q-Q (Cuantiles Teóricos vs Observados)', fontsize=11, fontweight='bold', pad=8)
                        ax2.set_xlabel('Cuantiles Teóricos', fontsize=9)
                        ax2.set_ylabel('Cuantiles Observados', fontsize=9)
                        ax2.grid(True, alpha=0.2)
                        fig2.tight_layout()
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                    else:
                        st.warning('Verifique las restricciones matemáticas de los parámetros ingresados.')
        
        st.divider()
        st.subheader('Envío Integrado de Evaluaciones')
        st.write('Pulse el botón para consolidar y calificar sus respuestas de ambos casos.')
        
        if st.button('Enviar mis dos casos', type='primary', use_container_width=True):
            if any(not st.session_state.answers.get(cid, {}).get('valid', False) for cid in ids):
                st.error('Existen parámetros inválidos en uno o ambos casos. Verifique antes de continuar.')
            else:
                ss_details = {}
                total_score = 0
                for cid in ids:
                    pts, _ = score(cid, st.session_state.answers[cid]['dist'], st.session_state.answers[cid]['params'])
                    ss_details[cid] = pts
                    total_score += pts
                
                row = {
                    'student_id': student,
                    'score': total_score,
                    'submitted_at': datetime.now(timezone.utc).isoformat(),
                    'case_1': ids[0],
                    'case_2': ids[1],
                    'detail': json.dumps(ss_details)
                }
                res = submit(row)
                
                # Mensaje de confirmación específico solicitado
                st.success('¡Tus datos han sido enviados! Consulta el ranking para ver tu clasificación.')
                st.balloons()
                st.rerun()

with t2:
    st.subheader('Clasificación General')
    df = ranking()
    if not df.empty:
        df = df.sort_values(['score', 'submitted_at'], ascending=[False, True]).reset_index(drop=True)
        
        # Procesar posiciones con emojis integrados (🏆 1 🏆, ⭐ 2 ⭐, ⭐ 3 ⭐)
        def fmt_pos(idx):
            pos = idx + 1
            if pos == 1:
                return '🏆 1 🏆'
            elif pos in (2, 3):
                return f'⭐ {pos} ⭐'
            return str(pos)
        
        df['Posición'] = [fmt_pos(i) for i in range(len(df))]
        
        # Extraer puntajes detallados de cada caso si están disponibles
        p_c1 = []
        p_c2 = []
        for idx, r_row in df.iterrows():
            det = r_row.get('detail', {})
            if isinstance(det, str):
                try:
                    det = json.loads(det)
                except:
                    det = {}
            if isinstance(det, dict):
                p_c1.append(det.get(r_row['case_1'], '-'))
                p_c2.append(det.get(r_row['case_2'], '-'))
            else:
                p_c1.append('-')
                p_c2.append('-')
                
        df['Primer caso asignado'] = df['case_1'].map(lambda k: CASES.get(k, {}).get('title', k))
        df['Puntuación del primer caso'] = p_c1
        df['Segundo caso asignado'] = df['case_2'].map(lambda k: CASES.get(k, {}).get('title', k))
        df['Puntuación del segundo caso asignado'] = p_c2
        df['Puntaje Total'] = df['score']
        df['Estudiante'] = df['student_id']
        
        cols_display = [
            'Posición',
            'Estudiante',
            'Primer caso asignado',
            'Puntuación del primer caso',
            'Segundo caso asignado',
            'Puntuación del segundo caso asignado',
            'Puntaje Total'
        ]
        
        st.dataframe(df[cols_display], hide_index=True, use_container_width=True)
    else:
        st.info('Aún no existen registros de envío en la tabla de posiciones.')
    
    if st.button('Actualizar ranking'):
        st.rerun()

st.divider()
st.caption('Laboratorio de Modelación Probabilística · Estadística I · Ingeniería Industrial')
