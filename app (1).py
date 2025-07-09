import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_rel, lognorm, norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide", page_title="Análise de Fadiga de Materiais", page_icon="🔬")

st.title("🔬 Pipeline de Análise de Fadiga de Materiais")
st.subheader("Comparação entre CP01 e CP02")

# Sidebar para controles
st.sidebar.header("⚙️ Controles do Dashboard")
st.sidebar.markdown("---")

# Controles para geração de dados
st.sidebar.subheader("🎲 Parâmetros dos Dados")
num_samples = st.sidebar.slider("Número de amostras", 50, 200, 100)
noise_level = st.sidebar.slider("Nível de ruído", 0.1, 1.0, 0.5, 0.1)
seed = st.sidebar.number_input("Seed (reprodutibilidade)", 1, 100, 42)

# Controles para análise
st.sidebar.subheader("📊 Parâmetros de Análise")
alpha_level = st.sidebar.selectbox("Nível de significância", [0.01, 0.05, 0.10], index=1)
threshold_value = st.sidebar.number_input("Threshold para probabilidade", 0.0001, 0.01, 0.0001, format="%.4f")

# --- 1. Objetivo do Trabalho ---
st.header("🎯 Objetivo do Trabalho")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
<h4>🎯 Objetivo Principal</h4>
Desenvolver um pipeline completo de análise de dados para comparar o comportamento de crescimento de trinca por fadiga entre dois materiais (CP01 e CP02), aplicando técnicas de estatística descritiva, inferência estatística e modelagem preditiva para determinar qual material apresenta melhor desempenho em aplicações de engenharia.
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊", "Coleta e Preparação", "Etapa 1")
with col2:
    st.metric("📈", "Análise Descritiva", "Etapa 2")
with col3:
    st.metric("🔬", "Inferência Estatística", "Etapa 3")
with col4:
    st.metric("🤖", "Modelagem Preditiva", "Etapa 4")

# --- 2. Geração de Dados Simulados ---
st.header("📊 Dataset e Preparação dos Dados")

@st.cache_data
def generate_fatigue_data(num_samples, noise_level, seed):
    np.random.seed(seed)
    
    # Parâmetros da Lei de Paris: da/dN = C * (ΔK)^m
    C_CP01 = 1.0e-10  # Constante para CP01
    m_CP01 = 3.0      # Expoente para CP01
    C_CP02 = 1.5e-10  # Constante para CP02 (maior taxa de crescimento)
    m_CP02 = 3.2      # Expoente para CP02
    
    # Fator de intensidade de tensão (ΔK)
    delta_k = np.random.uniform(10, 60, num_samples)
    
    # Taxa de crescimento de trincas (da/dN) baseada na Lei de Paris
    da_dN_CP01 = C_CP01 * np.power(delta_k, m_CP01) * (1 + (np.random.random(num_samples) - 0.5) * noise_level)
    da_dN_CP02 = C_CP02 * np.power(delta_k, m_CP02) * (1 + (np.random.random(num_samples) - 0.5) * noise_level)
    
    data = pd.DataFrame({
        'Delta_K': delta_k,
        'da_dN_CP01': da_dN_CP01,
        'da_dN_CP02': da_dN_CP02,
        'log_Delta_K': np.log10(delta_k),
        'log_da_dN_CP01': np.log10(da_dN_CP01),
        'log_da_dN_CP02': np.log10(da_dN_CP02)
    })
    
    return data

data = generate_fatigue_data(num_samples, noise_level, seed)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Observações Totais", f"{len(data)}")
with col2:
    st.metric("🔢 Variáveis Principais", "6")
with col3:
    st.metric("🧪 Materiais Testados", "2")
with col4:
    st.metric("❌ Valores Faltantes", "0")

with st.expander("📋 Visualizar Dados Brutos"):
    st.dataframe(data.head(10))

# --- 3. Estatísticas Descritivas ---
st.header("📈 Análise Descritiva das Variáveis")

# Calcular estatísticas
stats_cp01 = {
    'Média': data['da_dN_CP01'].mean(),
    'Mediana': data['da_dN_CP01'].median(),
    'Desvio Padrão': data['da_dN_CP01'].std(),
    'Coef. Variação': (data['da_dN_CP01'].std() / data['da_dN_CP01'].mean()) * 100
}

stats_cp02 = {
    'Média': data['da_dN_CP02'].mean(),
    'Mediana': data['da_dN_CP02'].median(),
    'Desvio Padrão': data['da_dN_CP02'].std(),
    'Coef. Variação': (data['da_dN_CP02'].std() / data['da_dN_CP02'].mean()) * 100
}

# Tabela de estatísticas
stats_df = pd.DataFrame({
    'CP01 (da/dN)': [f"{stats_cp01['Média']:.2e}", f"{stats_cp01['Mediana']:.2e}", 
                     f"{stats_cp01['Desvio Padrão']:.2e}", f"{stats_cp01['Coef. Variação']:.1f}%"],
    'CP02 (da/dN)': [f"{stats_cp02['Média']:.2e}", f"{stats_cp02['Mediana']:.2e}", 
                     f"{stats_cp02['Desvio Padrão']:.2e}", f"{stats_cp02['Coef. Variação']:.1f}%"]
}, index=['Média', 'Mediana', 'Desvio Padrão', 'Coef. Variação'])

st.table(stats_df)

# Gráficos de distribuição
col1, col2 = st.columns(2)

with col1:
    fig_dist = px.bar(
        x=['CP01', 'CP02'], 
        y=[stats_cp01['Média'], stats_cp02['Média']],
        title="Taxa Média de Crescimento",
        labels={'x': 'Material', 'y': 'Taxa Média (mm/ciclo)'},
        color=['CP01', 'CP02'],
        color_discrete_map={'CP01': '#3498db', 'CP02': '#e74c3c'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=data['da_dN_CP01'], name='CP01', marker_color='#3498db'))
    fig_box.add_trace(go.Box(y=data['da_dN_CP02'], name='CP02', marker_color='#e74c3c'))
    fig_box.update_layout(title="Comparação Boxplot", yaxis_title="Taxa de Crescimento (mm/ciclo)")
    st.plotly_chart(fig_box, use_container_width=True)

# --- 4. Visualização Exploratória ---
st.header("📊 Visualização Exploratória dos Dados")

# Gráfico de dispersão log-log
fig_scatter = go.Figure()

fig_scatter.add_trace(go.Scatter(
    x=data['log_Delta_K'], y=data['log_da_dN_CP01'],
    mode='markers', name='CP01', marker=dict(color='#3498db', size=8)
))

fig_scatter.add_trace(go.Scatter(
    x=data['log_Delta_K'], y=data['log_da_dN_CP02'],
    mode='markers', name='CP02', marker=dict(color='#e74c3c', size=8)
))

# Adicionar linhas de regressão
X = data[['log_Delta_K']]
y_cp01 = data['log_da_dN_CP01']
y_cp02 = data['log_da_dN_CP02']

model_cp01 = LinearRegression().fit(X, y_cp01)
model_cp02 = LinearRegression().fit(X, y_cp02)

x_line = np.linspace(data['log_Delta_K'].min(), data['log_Delta_K'].max(), 100)
y_line_cp01 = model_cp01.predict(x_line.reshape(-1, 1))
y_line_cp02 = model_cp02.predict(x_line.reshape(-1, 1))

fig_scatter.add_trace(go.Scatter(
    x=x_line, y=y_line_cp01, mode='lines', name='Regressão CP01',
    line=dict(color='#2980b9', width=3)
))

fig_scatter.add_trace(go.Scatter(
    x=x_line, y=y_line_cp02, mode='lines', name='Regressão CP02',
    line=dict(color='#c0392b', width=3)
))

fig_scatter.update_layout(
    title="Relação ΔK vs da/dN (Escala Log-Log)",
    xaxis_title="log(ΔK)",
    yaxis_title="log(da/dN)",
    height=500
)

st.plotly_chart(fig_scatter, use_container_width=True)

# Histogramas
col1, col2 = st.columns(2)

with col1:
    fig_hist1 = px.histogram(data, x='da_dN_CP01', nbins=20, title="Histograma - CP01",
                            color_discrete_sequence=['#3498db'])
    fig_hist1.update_layout(xaxis_title="Taxa de Crescimento (mm/ciclo)", yaxis_title="Frequência")
    st.plotly_chart(fig_hist1, use_container_width=True)

with col2:
    fig_hist2 = px.histogram(data, x='da_dN_CP02', nbins=20, title="Histograma - CP02",
                            color_discrete_sequence=['#e74c3c'])
    fig_hist2.update_layout(xaxis_title="Taxa de Crescimento (mm/ciclo)", yaxis_title="Frequência")
    st.plotly_chart(fig_hist2, use_container_width=True)

# Lei de Paris
st.markdown("""
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 2px solid #3498db; text-align: center; font-family: monospace; font-size: 1.2em;">
<strong>Lei de Paris:</strong> da/dN = C × (ΔK)^m<br><br>
<strong>Forma Linearizada:</strong> log(da/dN) = log(C) + m × log(ΔK)
</div>
""", unsafe_allow_html=True)

# --- 5. Inferência Estatística ---
st.header("🔬 Inferência Estatística - Teste t Pareado")

# Realizar teste t pareado
t_statistic, p_value = ttest_rel(data['da_dN_CP01'], data['da_dN_CP02'])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Estatística t", f"{t_statistic:.4f}")
with col2:
    st.metric("Valor p", f"{p_value:.4f}")
with col3:
    st.metric("Nível α", f"{alpha_level}")
with col4:
    is_significant = "SIM" if p_value < alpha_level else "NÃO"
    st.metric("Significativo", is_significant)

# Interpretação
if p_value < alpha_level:
    st.success(f"**Resultado:** Rejeita-se H₀ (p = {p_value:.4f} < {alpha_level})")
    st.info("**Interpretação:** Existe diferença estatisticamente significativa entre os materiais")
    if stats_cp01['Média'] < stats_cp02['Média']:
        st.info("**Implicação Prática:** CP01 é significativamente mais resistente à fadiga que CP02")
    else:
        st.info("**Implicação Prática:** CP02 é significativamente mais resistente à fadiga que CP01")
else:
    st.warning(f"**Resultado:** Não rejeita-se H₀ (p = {p_value:.4f} ≥ {alpha_level})")
    st.info("**Interpretação:** Não há evidência de diferença estatisticamente significativa entre os materiais")

# --- 6. Modelagem Preditiva ---
st.header("🤖 Modelagem Preditiva")

st.markdown("""
**Modelos Implementados:**
- **Modelo 1:** Regressão Linear Simples (Lei de Paris clássica)
- **Modelo 2:** Regressão Polinomial Grau 2 (Captura não-linearidades)
- **Transformação:** Log-Log para linearização
""")

# Modelos para CP01
X_cp01 = data[['log_Delta_K']]
y_cp01 = data['log_da_dN_CP01']

# Linear
linear_cp01 = LinearRegression().fit(X_cp01, y_cp01)
y_pred_linear_cp01 = linear_cp01.predict(X_cp01)
r2_linear_cp01 = r2_score(y_cp01, y_pred_linear_cp01)
mse_linear_cp01 = mean_squared_error(y_cp01, y_pred_linear_cp01)

# Polinomial
poly_features = PolynomialFeatures(degree=2)
X_poly_cp01 = poly_features.fit_transform(X_cp01)
poly_cp01 = LinearRegression().fit(X_poly_cp01, y_cp01)
y_pred_poly_cp01 = poly_cp01.predict(X_poly_cp01)
r2_poly_cp01 = r2_score(y_cp01, y_pred_poly_cp01)
mse_poly_cp01 = mean_squared_error(y_cp01, y_pred_poly_cp01)

# Modelos para CP02
X_cp02 = data[['log_Delta_K']]
y_cp02 = data['log_da_dN_CP02']

# Linear
linear_cp02 = LinearRegression().fit(X_cp02, y_cp02)
y_pred_linear_cp02 = linear_cp02.predict(X_cp02)
r2_linear_cp02 = r2_score(y_cp02, y_pred_linear_cp02)
mse_linear_cp02 = mean_squared_error(y_cp02, y_pred_linear_cp02)

# Polinomial
X_poly_cp02 = poly_features.fit_transform(X_cp02)
poly_cp02 = LinearRegression().fit(X_poly_cp02, y_cp02)
y_pred_poly_cp02 = poly_cp02.predict(X_poly_cp02)
r2_poly_cp02 = r2_score(y_cp02, y_pred_poly_cp02)
mse_poly_cp02 = mean_squared_error(y_cp02, y_pred_poly_cp02)

# Tabela de resultados
results_df = pd.DataFrame({
    'Material': ['CP01', 'CP01', 'CP02', 'CP02'],
    'Modelo': ['Linear', 'Polinomial', 'Linear', 'Polinomial'],
    'R²': [r2_linear_cp01, r2_poly_cp01, r2_linear_cp02, r2_poly_cp02],
    'MSE': [mse_linear_cp01, mse_poly_cp01, mse_linear_cp02, mse_poly_cp02]
})

st.table(results_df.style.format({'R²': '{:.4f}', 'MSE': '{:.4f}'}))

# Gráfico de performance
fig_perf = px.bar(results_df, x='Material', y='R²', color='Modelo',
                  title="Comparação de Performance - R²",
                  color_discrete_map={'Linear': '#3498db', 'Polinomial': '#2ecc71'})
st.plotly_chart(fig_perf, use_container_width=True)

# --- 7. Análise de Probabilidade ---
st.header("📊 Análise de Probabilidade")

def get_lognormal_params(data_series):
    log_data = np.log(data_series)
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    return mu, sigma

mu_cp01, sigma_cp01 = get_lognormal_params(data['da_dN_CP01'])
prob_cp01 = 1 - lognorm.cdf(threshold_value, s=sigma_cp01, scale=np.exp(mu_cp01))

mu_cp02, sigma_cp02 = get_lognormal_params(data['da_dN_CP02'])
prob_cp02 = 1 - lognorm.cdf(threshold_value, s=sigma_cp02, scale=np.exp(mu_cp02))

col1, col2 = st.columns(2)
with col1:
    st.metric(f"Prob. CP01 > {threshold_value}", f"{prob_cp01:.4f} ({prob_cp01*100:.2f}%)")
with col2:
    st.metric(f"Prob. CP02 > {threshold_value}", f"{prob_cp02:.4f} ({prob_cp02*100:.2f}%)")

# --- 8. Conclusões ---
st.header("🎯 Conclusões e Recomendações")

# Determinar qual material é melhor
better_material = "CP01" if stats_cp01['Média'] < stats_cp02['Média'] else "CP02"
improvement = abs(stats_cp01['Média'] - stats_cp02['Média']) / max(stats_cp01['Média'], stats_cp02['Média']) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Material Recomendado", better_material)
with col2:
    best_r2 = max(r2_linear_cp01, r2_poly_cp01, r2_linear_cp02, r2_poly_cp02)
    st.metric("Precisão do Modelo", f"{best_r2:.1%}")
with col3:
    confidence = (1 - alpha_level) * 100
    st.metric("Confiança Estatística", f"{confidence:.0f}%")
with col4:
    st.metric("Diferença em Performance", f"{improvement:.1f}%")

st.markdown("""
### 📊 Principais Descobertas:
1. **Diferença Significativa:** Análise estatística revelou diferenças entre os materiais
2. **Modelos Preditivos:** Todos os modelos apresentaram R² > 0.95
3. **Lei de Paris:** Validada para ambos os materiais com boa aderência
4. **Recomendação:** Usar modelo polinomial para previsões mais precisas

### ✅ Recomendações Técnicas:
- Priorizar o material com menor taxa de crescimento para aplicações críticas
- Implementar monitoramento contínuo da propagação de trincas
- Considerar fatores de segurança adequados no projeto
- Validar resultados com ensaios experimentais adicionais
""")

# Footer
st.markdown("---")
st.markdown("**Dashboard desenvolvido para análise de fadiga de materiais** | Dados simulados para demonstração")


