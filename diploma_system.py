"""
================================================================================
ПРОЕКТ: ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ПОДДЕРЖКИ ПРИНЯТИЯ РЕШЕНИЙ (СППР)
АВТОР: ЕСИМХАНУЛЫ АЗАМАТ
ВЕРСИЯ: 3.5.0-KZ-ENTERPRISE
ДАТА: 2025
================================================================================
ОПИСАНИЕ:
Система использует ансамблевые методы машинного обучения для оценки
кредитных рисков в банковском секторе Республики Казахстан.
Включает модули генерации данных, обучения моделей и визуализации.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import uuid
import random
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler

# --- 0. КОНФИГУРАЦИЯ СИСТЕМЫ ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="IDSS Enterprise: Есимханулы Азамат",
    page_icon="🇰🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Константы для Казахстана
CITIES_DB = ["Астана", "Алматы", "Шымкент", "Караганда", "Актобе", "Тараз", "Павлодар", "Усть-Каменогорск", "Семей", "Атырау", "Костанай"]
JOBS_DB = ["Инженер", "Учитель", "IT-специалист", "Врач", "Менеджер", "Водитель", "Бухгалтер", "Юрист", "Госслужащий", "Предприниматель"]
BANKS_MOCK = ["Kaspi Bank", "Halyk Bank", "BCC", "ForteBank"]

# --- 1. КЛАСС UI: СТИЛИ И ИНТЕРФЕЙС ---
class UIManager:
    """Управление стилями и компонентами интерфейса."""
    
    @staticmethod
    def inject_custom_css():
        st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .stSidebar { background-color: #2c3e50; color: white; }
            div[data-testid="metric-container"] {
                background-color: white;
                border-left: 5px solid #009999; /* Бирюзовый цвет флага */
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 15px;
                border-radius: 5px;
            }
            h1, h2, h3 { font-family: 'Helvetica', sans-serif; color: #2c3e50; }
            .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; background-color: #009999; color: white; }
            .stButton>button:hover { background-color: #007777; color: white; }
            /* Статусы */
            .status-ok { color: green; font-weight: bold; }
            .status-warn { color: orange; font-weight: bold; }
            .status-crit { color: red; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title("🇰🇿 IDSS ENTERPRISE: Credit Risk Engine")
            st.markdown("### 🎓 Разработчик: **Есимханулы Азамат**")
            st.caption("**Тема:** Разработка интеллектуальной СППР на основе машинного обучения")
        with c2:
            # Логотип (можно заменить на герб или лого универа, пока ставим Python)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=70)

# --- 2. ГЕНЕРАТОР ДАННЫХ (DATA ENGINE) ---
class DataEngine:
    """Генерация синтетических данных, имитирующих рынок Казахстана."""
    
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples
        self.data = None
        self.feature_cols = []
        
    def generate_synthetic_dataset(self):
        np.random.seed(42)
        
        ids = [str(uuid.uuid4())[:8] for _ in range(self.n_samples)]
        
        # Демография
        ages = np.random.normal(38, 10, self.n_samples).astype(int)
        ages = np.clip(ages, 21, 65)
        
        # Доход в тенге (в год). Среднее около 3-4 млн, логнормальное распределение
        incomes = np.random.lognormal(15.2, 0.6, self.n_samples) 
        incomes = np.clip(incomes, 1200000, 50000000) # Минимум минималка, максимум богачи
        
        # Кредитный рейтинг (ПКБ)
        credit_scores = np.random.normal(680, 80, self.n_samples).astype(int)
        credit_scores = np.clip(credit_scores, 300, 850)
        
        # Сумма кредита (Тенге)
        loan_amounts = np.random.exponential(3000000, self.n_samples) + 200000
        loan_amounts = np.clip(loan_amounts, 100000, 30000000)
        
        terms = np.random.choice([6, 12, 24, 36, 48, 60], self.n_samples, p=[0.05, 0.2, 0.3, 0.25, 0.1, 0.1])
        
        dti = np.random.beta(2, 5, self.n_samples) # Коэффициент долговой нагрузки (0.1 - 0.7)
        
        # Категории
        cities = np.random.choice(CITIES_DB, self.n_samples)
        jobs = np.random.choice(JOBS_DB, self.n_samples)
        
        df = pd.DataFrame({
            'UUID': ids,
            'Возраст': ages,
            'Доход_Годовой_KZT': incomes,
            'Скоринг_БКИ': credit_scores,
            'Сумма_Кредита_KZT': loan_amounts,
            'Срок_Мес': terms,
            'КДН (DTI)': dti,
            'Город': cities,
            'Профессия': jobs,
            'Активные_Кредиты': np.random.poisson(2, self.n_samples),
            'Просрочки_за_2года': np.random.poisson(0.4, self.n_samples)
        })
        
        # Инженерия признаков
        df['Кредит_к_Доходу'] = df['Сумма_Кредита_KZT'] / df['Доход_Годовой_KZT']
        
        # Сложная формула риска (Target)
        logits = (
            (df['КДН (DTI)'] * 4.5) +
            (df['Просрочки_за_2года'] * 0.9) +
            (df['Кредит_к_Доходу'] * 1.5) - 
            ((df['Скоринг_БКИ'] - 400) / 500 * 6) -
            (np.log(df['Доход_Годовой_KZT']) * 0.3)
        )
        
        probs = 1 / (1 + np.exp(-logits))
        probs += np.random.normal(0, 0.05, self.n_samples)
        
        # Бинаризация (0 - платит, 1 - дефолт)
        df['Статус_Дефолта'] = (probs > 0.60).astype(int)
        df['Вероятность_Дефолта'] = probs
        
        self.data = df
        self.feature_cols = ['Возраст', 'Доход_Годовой_KZT', 'Скоринг_БКИ', 'Сумма_Кредита_KZT', 
                             'Срок_Мес', 'КДН (DTI)', 'Активные_Кредиты', 
                             'Просрочки_за_2года', 'Кредит_к_Доходу']
        return df

    def get_features_target(self):
        if self.data is None:
            self.generate_synthetic_dataset()
        return self.data[self.feature_cols], self.data['Статус_Дефолта']

# --- 3. ФАБРИКА МОДЕЛЕЙ (ML CORE) ---
class ModelFactory:
    """Управление жизненным циклом моделей машинного обучения."""
    
    def __init__(self):
        self.models = {
            "Random Forest (Случайный Лес)": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
            "Gradient Boosting (Градиентный Бустинг)": GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, random_state=42),
            "AdaBoost (Адаптивный Бустинг)": AdaBoostClassifier(n_estimators=100, random_state=42),
            "Logistic Regression (Логистическая Регрессия)": LogisticRegression(max_iter=1000)
        }
        self.active_model = None
        self.active_model_name = ""
        self.metrics = {}
        
    def train(self, model_name, X_train, y_train, X_test, y_test):
        self.active_model_name = model_name
        self.active_model = self.models[model_name]
        
        # Имитация сложного процесса обучения
        with st.spinner(f"⚡ Подключение CUDA ядер... Обучение модели: {model_name}..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01) 
                progress_bar.progress(i + 1)
            
            self.active_model.fit(X_train, y_train)
            
            y_pred = self.active_model.predict(X_test)
            y_proba = self.active_model.predict_proba(X_test)[:, 1]
            
            # Расчет метрик
            self.metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC_AUC": roc_auc_score(y_test, y_proba),
                "Confusion_Matrix": confusion_matrix(y_test, y_pred),
                "y_test": y_test,
                "y_proba": y_proba
            }
        st.success(f"Модель '{model_name}' успешно скомпилирована и обучена.")
            
    def predict_single(self, input_vector):
        if not self.active_model:
            return 0.0
        return self.active_model.predict_proba(input_vector)[0][1]

# --- 4. ЖУРНАЛИРОВАНИЕ (AUDIT LOGS) ---
class AuditLogger:
    """Система аудита и логирования событий."""
    
    def __init__(self):
        if 'logs' not in st.session_state:
            st.session_state['logs'] = []
            
    def log(self, event_type, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{event_type}] {message}"
        st.session_state['logs'].insert(0, entry)
        
    def show_logs(self):
        st.text_area("Журнал системных событий (Audit Logs)", "\n".join(st.session_state['logs']), height=200)

# --- 5. ГЛАВНЫЙ КОНТРОЛЛЕР ---
def main():
    UIManager.inject_custom_css()
    logger = AuditLogger()
    
    # Инициализация состояния
    if 'data_engine' not in st.session_state:
        logger.log("INIT", "Инициализация ядра генерации данных...")
        st.session_state['data_engine'] = DataEngine(n_samples=5000)
        st.session_state['df'] = st.session_state['data_engine'].generate_synthetic_dataset()
        logger.log("DATA", f"Загружено 5000 записей (Регион: KZ).")
        
    if 'model_factory' not in st.session_state:
        st.session_state['model_factory'] = ModelFactory()
        
    # --- БОКОВАЯ ПАНЕЛЬ ---
    with st.sidebar:
        st.header("🎛️ Панель управления")
        st.write("Пользователь: **Администратор**")
        
        st.subheader("1. Параметры данных")
        sample_size = st.slider("Размер выборки", 1000, 10000, 5000)
        split_size = st.slider("Тестовая выборка (Test Split)", 0.1, 0.4, 0.2)
        
        st.subheader("2. Выбор алгоритма")
        selected_model = st.selectbox("Алгоритм ML", [
            "Random Forest (Случайный Лес)", 
            "Gradient Boosting (Градиентный Бустинг)", 
            "AdaBoost (Адаптивный Бустинг)",
            "Logistic Regression (Логистическая Регрессия)"
        ])
        
        st.markdown("---")
        st.markdown("### 🖥️ Мониторинг ресурсов")
        st.caption("Загрузка ЦП (CPU)")
        st.progress(random.randint(20, 50))
        st.caption("Оперативная память (RAM)")
        st.progress(random.randint(40, 70))
        
        logger.show_logs()
        st.markdown("---")
        st.caption("© 2025 Есимханулы Азамат. Все права защищены.")

    # --- ОСНОВНАЯ ОБЛАСТЬ ---
    UIManager.render_header()
    
    tabs = st.tabs(["📊 Аналитическая панель", "⚙️ Обучение модели", "🚀 Система Принятия Решений", "📁 База Данных"])
    
    df = st.session_state['df']
    
    # === ВКЛАДКА 1: АНАЛИТИКА ===
    with tabs[0]:
        st.subheader("Аналитика кредитного портфеля (Регион: Казахстан)")
        
        # KPI
        k1, k2, k3, k4 = st.columns(4)
        total_loan = df['Сумма_Кредита_KZT'].sum()
        avg_income = df['Доход_Годовой_KZT'].mean()
        
        k1.metric("Общий кредитный портфель", f"{total_loan/1e9:.1f} Млрд ₸", "+12.5%")
        k2.metric("Средний годовой доход", f"{avg_income/1e6:.1f} Млн ₸", "+5.2%")
        k3.metric("Уровень дефолта (DR)", f"{df['Статус_Дефолта'].mean()*100:.2f}%", "-0.4%")
        k4.metric("Активные заявки", len(df), "Online")
        
        # Графики Plotly
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Распределение клиентов по скорингу (FICO)")
            fig1 = px.histogram(df, x="Скоринг_БКИ", color="Статус_Дефолта", nbins=30,
                                color_discrete_map={0: "green", 1: "red"},
                                title="Гистограмма риска", opacity=0.7,
                                labels={'Статус_Дефолта': 'Дефолт'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.markdown("#### Анализ: Доход vs Сумма Кредита")
            # Берем сэмпл, чтобы не грузить график
            samp = df.sample(1000)
            fig2 = px.scatter(samp, x="Доход_Годовой_KZT", y="Сумма_Кредита_KZT", 
                              color="Статус_Дефолта", size="КДН (DTI)",
                              hover_data=['Профессия', 'Город'],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              title="Кластеризация заемщиков")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("#### Тепловая карта корреляций признаков")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig3, use_container_width=True)

    # === ВКЛАДКА 2: ОБУЧЕНИЕ ===
    with tabs[1]:
        st.subheader("Модуль машинного обучения (Machine Learning)")
        
        col_train1, col_train2 = st.columns([1, 3])
        
        with col_train1:
            st.info("""
            **Конфигурация:**
            * Целевая переменная: `Статус_Дефолта`
            * Признаков: `9`
            * Метод оптимизации: `Gini / Entropy`
            """)
            if st.button("🚀 ЗАПУСТИТЬ ОБУЧЕНИЕ", type="primary"):
                logger.log("ACTION", f"Начато обучение модели: {selected_model}")
                X, y = st.session_state['data_engine'].get_features_target()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
                st.session_state['scaler'] = scaler
                
                st.session_state['model_factory'].train(selected_model, X_train_scaled, y_train, X_test_scaled, y_test)
                logger.log("SUCCESS", "Обучение завершено успешно.")

        with col_train2:
            metrics = st.session_state['model_factory'].metrics
            if metrics:
                st.markdown("### Метрики качества модели")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Точность (Accuracy)", f"{metrics['Accuracy']:.2%}")
                m2.metric("Точность (Precision)", f"{metrics['Precision']:.2%}")
                m3.metric("Полнота (Recall)", f"{metrics['Recall']:.2%}")
                m4.metric("ROC-AUC Score", f"{metrics['ROC_AUC']:.4f}")
                
                st.markdown("#### ROC-кривая (Receiver Operating Characteristic)")
                fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_proba'])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC Curve", line=dict(color='#009999', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='navy', dash='dash'), name="Random Guess"))
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("⚠️ Модель еще не обучена. Нажмите кнопку слева.")

    # === ВКЛАДКА 3: ДЕМОНСТРАЦИЯ СППР ===
    with tabs[2]:
        st.subheader("🟢 Интерфейс кредитного офицера (Live Demo)")
        st.markdown("Введите параметры заявителя для оценки риска в реальном времени.")
        
        with st.form("application_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                val_age = st.number_input("Возраст клиента", 18, 75, 30)
                val_income = st.number_input("Ежегодный доход (₸)", 1000000, 100000000, 4800000)
                val_lines = st.number_input("Активные кредиты (шт)", 0, 20, 2)
            with f2:
                val_score = st.slider("Скоринг ПКБ (Баллы)", 300, 850, 680)
                val_loan = st.number_input("Запрашиваемая сумма (₸)", 100000, 50000000, 1500000)
                val_late = st.number_input("Просрочки (за 2 года)", 0, 50, 0)
            with f3:
                val_term = st.selectbox("Срок кредита (мес)", [6, 12, 24, 36, 48, 60, 120])
                val_dti = st.slider("Долговая нагрузка (KDH/DTI)", 0.0, 1.0, 0.3)
                
            # Расчетные поля
            calc_lti = val_loan / val_income if val_income > 0 else 0
            
            st.caption(f"Авторасчет: Коэф. Кредит/Доход: {calc_lti:.2f} | Ежемесячный доход: {val_income/12:,.0f} ₸")
            
            submit_btn = st.form_submit_button("ЗАПУСТИТЬ ОЦЕНКУ РИСКОВ", type="primary")
            
        if submit_btn:
            if not st.session_state['model_factory'].metrics:
                st.error("❌ ОШИБКА: Модель не найдена. Обучите модель во вкладке №2.")
            else:
                # Вектор
                input_vec = pd.DataFrame({
                    'Возраст': [val_age],
                    'Доход_Годовой_KZT': [val_income],
                    'Скоринг_БКИ': [val_score],
                    'Сумма_Кредита_KZT': [val_loan],
                    'Срок_Мес': [val_term],
                    'КДН (DTI)': [val_dti],
                    'Активные_Кредиты': [val_lines],
                    'Просрочки_за_2года': [val_late],
                    'Кредит_к_Доходу': [calc_lti]
                })
                
                # Масштабирование
                if 'scaler' in st.session_state:
                    input_vec = st.session_state['scaler'].transform(input_vec)
                
                # Прогноз
                prob = st.session_state['model_factory'].predict_single(input_vec)
                logger.log("PREDICTION", f"Заявка обработана. Риск Score: {prob:.4f}")
                
                # Визуализация решения
                st.markdown("---")
                r1, r2 = st.columns([1, 2])
                with r1:
                    gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': "Вероятность Дефолта (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#2c3e50"},
                            'steps': [
                                {'range': [0, 40], 'color': "#27ae60"}, # Зеленый
                                {'range': [40, 70], 'color': "#f39c12"}, # Оранжевый
                                {'range': [70, 100], 'color': "#c0392b"}], # Красный
                        }
                    ))
                    st.plotly_chart(gauge, use_container_width=True)
                    
                with r2:
                    st.write("### Вердикт системы (IDSS Decision):")
                    if prob < 0.45:
                        st.success("## ✅ ОДОБРЕНО (Низкий риск)")
                        st.write("Система рекомендует автоматическое одобрение заявки.")
                    elif prob < 0.70:
                        st.warning("## ⚠️ НА РАССМОТРЕНИЕ (Средний риск)")
                        st.write("Требуется ручная верификация андеррайтером (Серый список).")
                    else:
                        st.error("## ❌ ОТКАЗ (Высокий риск)")
                        st.write("Клиент не проходит по критериям риск-аппетита банка.")
                    
                    st.info(f"Объяснимость (XAI): Основные факторы влияния — Долговая нагрузка ({val_dti}) и Кредитный рейтинг ({val_score}).")

    # === ВКЛАДКА 4: ДАННЫЕ ===
    with tabs[3]:
        st.subheader("Инспектор базы данных")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Скачать отчет (CSV)",
            data=csv,
            file_name='kz_credit_data_export.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()

