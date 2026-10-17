import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
# Add src directory to python path to import bearing_module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from bearing_module import (
    BearingCNN,
    predict_bearing_fault_real_time,
    class_names,
    severity_weights
)

st.set_page_config(page_title="Bearing Diagnosis", page_icon="⚙️", layout="wide")

# ==========================================
# BILINGUAL SETUP (English & Persian)
# ==========================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'
lang = st.session_state.lang

texts = {
    'EN': {
        'title': "Bearing Fault Diagnosis Dashboard",
        'subtitle': "Real-time analysis using CNN and Mamdani Fuzzy Inference System",
        'sidebar_title': "⚙️ CWRU Bearing Monitoring",
        'sidebar_desc': "This panel acts as the core of the fuzzy inference system and parallel vibration processing connected to 48K bearing sensors.",
        'data_missing': "⚠️ Binary model files not found!",
        'data_instr': "To run real processing, please ensure the following files are in their correct directories:",
        'data_instr_1': "Compressed Dataset: `data/CWRU_48k_load_1_CNN_data.npz`",
        'data_instr_2': "Neural Network Weights: `src/bearing_cnn_model.pth`",
        'data_info': "💡 The sample pattern in the code is prepared so the system structure runs without errors after copying to local.",
        'success_load': "✅ CNN model and test dataset loaded successfully. Available evaluation samples:",
        'input_label': "Enter vibration test index (between 0 and {max}):",
        'btn_analyze': "🔍 Detailed Fuzzy Frequency Analysis",
        'ready_msg': "📥 System is ready for analysis. Please specify the target index and press the <b>Analyze</b> button.",
        'results_title': "### ⚡ Real-time Fault Diagnosis Results",
        'metric_hi': "Health Index",
        'metric_conf': "Model Confidence",
        'metric_sev': "Fuzzy Severity",
        'metric_risk': "Risk Level",
        'metric_delta': "{:.1f}% vs previous cycle",
        'state': "Equipment State",
        'action': "Recommended Action",
        'chart1_title': "#### 📡 Raw Frequency Spectrum Matrix (32x32 Input Block)",
        'chart1_x': "Spectrogram of sample index #{}",
        'chart2_title': "#### 📊 Model Output Class Probabilities (Softmax)",
        'chart2_x': "Probability of Fault Occurrence (%)",
        'chart2_y': "Prediction probability by CNN Block",
        'expander': "📝 Fuzzy Inference Details and Active Variables",
        'exp_class': "Identified Concept Class",
        'exp_sev': "Base Damage Weight",
        'exp_trend': "Health Trend (Differential)",
        'exp_urg': "Defuzzified Urgency",
        'err_load': "Error loading data structure: "
    },
    'FA': {
        'title': "داشبورد داده‌محور ارزیابی و تشخیص نقص بلبرینگ",
        'subtitle': "دسترسی به برد با فرکانس داده‌های تستی CWRU از دوج فشرده مدل ریال‌تایم",
        'sidebar_title': "⚙️ پایش تخصصی برینگ CWRU",
        'sidebar_desc': "این پانل به عنوان هسته سیستم استنتاج فازی و پردازش موازی ارتعاشات متصل به شبکه 48K سنسورهای بلبرینگ عمل می‌کند.",
        'data_missing': "⚠️ پرونده‌های باینری مدل یافت نشدند!",
        'data_instr': "برای اجرای پردازش واقعی، لطفاً از قرارگیری فایل‌های زیر در مسیرهای صحیح مطمئن شوید:",
        'data_instr_1': "فایل دیتاست فشرده: `data/CWRU_48k_load_1_CNN_data.npz`",
        'data_instr_2': "وزن‌های شبکه عصبی مدل: `src/bearing_cnn_model.pth`",
        'data_info': "💡 الگوی نمونه در کدهای Streamlit زیر تدارک دیده شده است تا ساختار سیستم پس از کپی شدن به لوکال، بدون خطا اجرا گردد.",
        'success_load': "✅ مدل CNN و دیتاست تست با موفقیت بارگذاری شدند. تعداد نمونه‌های ارزیابی موجود:",
        'input_label': "نمایه تست ارتعاشات را وارد کنید (بین 0 تا {max}):",
        'btn_analyze': "🔍 تحلیل دقیق فرکانس فازی",
        'ready_msg': "📥 سیستم آماده تحلیل داده است. لطفاً شناسه ایندکس مورد نظر را مشخص کرده و کلید <b>تحلیل دقیق</b> را بفشارید.",
        'results_title': "### ⚡ نتایج تحلیل و خروجی عیب‌یابی بیدرنگ",
        'metric_hi': "شاخص سلامت (Health Index)",
        'metric_conf': "ضریب اطمینان مدل (Confidence)",
        'metric_sev': "شدت ذاتی نقص فازی (Severity)",
        'metric_risk': "سطح کلی ریسک (Risk Level)",
        'metric_delta': "{:.1f}% نسبت به دور قبل",
        'state': "وضعیت تجهیز",
        'action': "پیشنهاد مانیتورینگ",
        'chart1_title': "#### 📡 ماتریس طیف فرکانسی خام (32x32 Input Block)",
        'chart1_x': "Spectrogram of sample index #{}",
        'chart2_title': "#### 📊 توزیع احتمالات کلاسی فازی خروجی مدل (Softmax)",
        'chart2_x': "امکان سنجی وقوع خطا (%)",
        'chart2_y': "Prediction probability by CNN Block",
        'expander': "📝 جزئیات استنتاج فازی و متغیرهای فعال",
        'exp_class': "کلاس شناسایی‌شده توسط شبکه کانسپت",
        'exp_sev': "شدت وزن آسیب پایه",
        'exp_trend': "روند تغییرات سلامت",
        'exp_urg': "مقدار فوریتی خروجی فازی (Defuzzified Urgency)",
        'err_load': "خطا در لود کردن ساختار داده‌ها: "
    }
}

t = texts[lang]

# ==========================================
# DYNAMIC STYLING
# ==========================================
dir_style = 'rtl' if lang == 'FA' else 'ltr'
align_style = 'right' if lang == 'FA' else 'left'

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Vazirmatn', sans-serif !important;
        text-align: {align_style};
        direction: {dir_style};
    }}
    .main-title {{ color: #f59e0b; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }}
    .custom-alert {{ padding: 15px; border-radius: 8px; color: white; margin-bottom: 15px; font-weight: bold; }}
    .alert-success {{ background-color: #065f46; border: 1px solid #047857; }}
    .alert-warning {{ background-color: #92400e; border: 1px solid #b45309; }}
    .alert-danger {{ background-color: #991b1b; border: 1px solid #b91c1c; }}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown(f"# {t['sidebar_title']}")
st.sidebar.markdown(t['sidebar_desc'])

# ==========================================
# DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_assets():
    # Paths relative to the dashboard folder
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'CWRU_48k_load_1_CNN_data.npz'))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'bearing_cnn_model.pth'))

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        return None, None, None, False

    try:
        data = np.load(data_path)
        X = data['data']
        y = data['labels']

        X_normalized = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_normalized[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        _, X_test, _, y_test = train_test_split(
            X_normalized, y_encoded,
            test_size=0.2, random_state=42, stratify=y_encoded
        )

        model = BearingCNN(num_classes=len(le.classes_))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        return X_test, y_test, model, True
    except Exception as e:
        return None, None, None, str(e)

X_test, y_test, model, status_or_error = load_assets()

st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f"<p style='color: #94a3b8;'>{t['subtitle']}</p>", unsafe_allow_html=True)
st.write("---")

if status_or_error is False:
    st.warning(t['data_missing'])
    st.markdown(f"""
    {t['data_instr']}
    1. {t['data_instr_1']}
    2. {t['data_instr_2']}
    """)
    st.info(t['data_info'])

elif isinstance(status_or_error, str):
    st.error(f"{t['err_load']} {status_or_error}")
else:
    st.success(f"{t['success_load']} {len(X_test)}")

    # Input Panel
    col_input, col_action = st.columns([3, 1])
    with col_input:
        target_idx = st.number_input(
            t['input_label'].format(max=len(X_test)-1),
            min_value=0, max_value=len(X_test)-1, value=0, step=1,
            key="vibe_index_input"
        )
    with col_action:
        st.write(" ")
        st.write(" ")
        trigger_analyser = st.button(t['btn_analyze'], use_container_width=True)

    if "analysed_trigger" not in st.session_state:
        st.session_state["analysed_trigger"] = False

    if trigger_analyser:
        st.session_state["analysed_trigger"] = True

    if not st.session_state["analysed_trigger"]:
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 30px; border-radius: 12px; border: 1px dashed #475569; text-align: center; margin-top: 20px;">
            <p style="color: #94a3b8; font-size: 1.1rem; margin: 0;">{t['ready_msg']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        test_sample = X_test[target_idx]

        # Simulated historical trend for fuzzy logic context
        hi_history = [95.0, 92.5, 88.0, 82.5]

        with st.spinner("Processing... / در حال پردازش..."):
            result, probs = predict_bearing_fault_real_time(model, test_sample, hi_history)

        st.write(t['results_title'])

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(t['metric_hi'], f"{result['Health_Index']} / 100", t['metric_delta'].format(result['Health_Index'] - hi_history[-1]))
        with m2:
            st.metric(t['metric_conf'], f"{result['Confidence']}%")
        with m3:
            st.metric(t['metric_sev'], f"{result['Fault_Severity']}")
        with m4:
            st.metric(t['metric_risk'], f"{result['Risk_Level']}")

        # Alert Box
        style_class = result["badge_style"]
        state_text = result["State"]
        action_text = result["Recommended_Action"]

        # Translate state and action for UI if needed (optional, but good for FA)
        if lang == 'FA':
            if state_text == "Healthy":
                state_text, action_text = "سالم", "بهره‌برداری عادی - پایش دوره‌ای تجهیز"
            elif state_text == "Warning":
                state_text, action_text = "هشدار", "برنامه‌ریزی کنترل کیفیت و روانکاری مجدد در کوتاه مدت"
            else:
                state_text, action_text = "بحرانی", "توقف فوری سیستم و تعویض بلبرینگ جهت پیشگیری از فاجعه مکانیکی"

        if style_class == "success":
            st.markdown(f'<div class="custom-alert alert-success">🟢 {t["state"]}: {state_text} | {t["action"]}: {action_text}</div>', unsafe_allow_html=True)
        elif style_class == "warning":
            st.markdown(f'<div class="custom-alert alert-warning">🟡 {t["state"]}: {state_text} | {t["action"]}: {action_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="custom-alert alert-danger">🔴 {t["state"]}: {state_text} | {t["action"]}: {action_text}</div>', unsafe_allow_html=True)

        st.write("---")
        g1, g2 = st.columns(2)

        # Charts
        with g1:
            st.write(t['chart1_title'])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(test_sample, cmap="viridis", cbar=True, ax=ax, xticklabels=False, yticklabels=False)
            ax.set_title(t['chart1_x'].format(target_idx), color="white")
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            st.pyplot(fig)

        with g2:
            st.write(t['chart2_title'])
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            y_pos = np.arange(len(class_names))
            ax2.barh(y_pos, probs * 100, align='center', color='#6366f1', edgecolor='#4338ca')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(class_names, color="white")
            ax2.invert_yaxis()
            ax2.set_xlabel(t['chart2_x'], color="white")
            ax2.set_title(t['chart2_y'], color="white")
            fig2.patch.set_facecolor('#0f172a')
            ax2.set_facecolor('#1e293b')
            ax2.xaxis.label.set_color('white')
            ax2.tick_params(colors='white')
            st.pyplot(fig2)

        # Details
        with st.expander(t['expander']):
            st.markdown(f"""
            - **{t['exp_class']}**: `{result['Fault_Type']}`
            - **{t['exp_sev']}**: `{result['Fault_Severity']}`
            - **{t['exp_trend']}**: `{hi_history[-2] - hi_history[-1]:.1f}`
            - **{t['exp_urg']}**: `{result['Urgency']} / 100`
            """)