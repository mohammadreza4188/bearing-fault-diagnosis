import streamlit as st

st.set_page_config(
    page_title="AI-CM | Industrial Condition Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BILINGUAL SETUP (English & Persian)
# ==========================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

def toggle_lang():
    st.session_state.lang = 'FA' if st.session_state.lang == 'EN' else 'EN'

lang = st.session_state.lang

texts = {
    'EN': {
        'title': "Industrial Condition Monitoring System (AI-CM)",
        'subtitle': "Powered by Deep CNN and Mamdani Fuzzy Logic",
        'sidebar_title': "AI-CM Industrial",
        'sidebar_desc': "Integrated Machine Fault Diagnosis System",
        'sidebar_info': "👈 Please select an independent diagnosis page from the navigation menu.",
        'lang_btn': "تغییر به فارسی",
        'arch_title': "🛠️ System Architecture Overview",
        'arch_desc': "This industrial diagnosis framework is a distributed system with multi-module capabilities. The Convolutional Neural Network (CNN) analyzes and extracts hidden features from 32x32 vibration matrices. Subsequently, the probabilistic output is fused with fuzzy logic to evaluate the final Risk Level and Urgency without presenting any fake data.",
        'plat_title': "🌐 Modular and Extensible Platform",
        'plat_desc': "Built with standard software engineering design patterns. Sub-pages in the pages/ directory ensure scalability:",
        'plat_list1': "Bearing Monitoring: Full analysis with CWRU sensor data and fuzzy logic [Active]",
        'plat_list2': "Real-time API: Ready for integration with industrial IoT sensors [Future]",
        'plat_list3': "Predictive Maintenance: RUL estimation and anomaly detection [Future]",
        'specs_title': "📊 CNN Model Specs",
        'specs_shape': "Input Shape:",
        'specs_classes': "Classes:",
        'specs_acc': "Test Accuracy:",
        'specs_mode': "Training Mode:",
        'specs_device': "Device:",
        'guide_title': "📂 Quick Start Guide",
        'footer': "Designed as a core real-time condition monitoring for smart industrial boards • 2026",
        'dir': 'ltr', 'align': 'left'
    },
    'FA': {
        'title': "سامانه هوشمند مانیتورینگ وضعیت صنعتی (AI-CM)",
        'subtitle': "برپایه استنتاج هوش مصنوعی (Deep CNN) و منطق منسجم فازی (Mamdani Logic)",
        'sidebar_title': "AI-CM Industrial",
        'sidebar_desc': "سامانه یکپارچه عیب‌یابی ماشین‌آلات",
        'sidebar_info': "👈 لطفا از منوی ناوبری کناری صفحات مستقل عیب‌یابی را انتخاب فرمایید.",
        'lang_btn': "Switch to English",
        'arch_title': "🛠️ معرفی معماری سامانه ناظر بر سلامت تجهیز",
        'arch_desc': "این فریم‌ورک عیب‌یابی صنعتی، یک سیستم توزیع‌شده با قابلیت پذیرش فیوچرهای چندماژوله است. شبکه عصبی کانولوشنال (CNN) به‌صورت بیدرنگ وظیفه تحلیل و استخراج ویژگی‌های پنهان درون ماتریس‌های ارتعاشی ۳۲×۳۲ بلبرینگ را به عهده دارد. در مرحله بعد خروجی احتمالاتی مدل، با منطق فازی تلاقی یافته و سطح ریسک نهایی و کلاس فوریتی عیب را بدون نشان دادن هیچ‌گونه دیتای فیک، ارزیابی می‌کند.",
        'plat_title': "🌐 پلتفرم ماژولار و توسعه‌پذیر",
        'plat_desc': "این پروژه با توجه به الگوهای طراحی استاندارد مهندسی نرم‌افزار توسعه یافته است. صفحات فرعی در دایرکتوری pages/ میزان مقیاس‌پذیری زیرساخت را تضمین می‌کنند:",
        'plat_list1': "پایش سیستم برینگ (Bearing Monitoring): تحلیل کامل با دیتای سنسورهای CWRU [فعال]",
        'plat_list2': "ای پی آی بیدرنگ (Real-time API): آماده اتصال به سنسورهای IoT صنعتی [آینده]",
        'plat_list3': "نگهداری پیشگویانه (Predictive Maintenance): تخمین RUL و تشخیص آنومالی [آینده]",
        'specs_title': "📊 مشخصات فنی مدل CNN رولر",
        'specs_shape': "فرمت فضا:",
        'specs_classes': "تعداد کلاس‌ها:",
        'specs_acc': "دقت نهایی تست:",
        'specs_mode': "حالت آموزش:",
        'specs_device': "دستگاه بهینه:",
        'guide_title': "📂 راهنمای گام اول راه‌اندازی",
        'footer': "طراحی شده به عنوان هسته مانیتورینگ وضعیت بیدرنگ بردهای هوشمند صنعتی • ۲۰۲۶",
        'dir': 'rtl', 'align': 'right'
    }
}

t = texts[lang]

# Deep Custom Styling based on selected language
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Vazirmatn', sans-serif !important;
        text-align: {t['align']};
        direction: {t['dir']};
    }}
    .main-title {{ color: #6366f1; font-size: 2.3rem; font-weight: 700; margin-bottom: 0.2rem; }}
    .sub-title {{ color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }}
    .card-box {{
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: {t['align']};
        direction: {t['dir']};
    }}
    .stButton>button {{
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: #4338ca;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.4);
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown(f"""
<div style='text-align: center; padding: 10px;'>
    <h3 style='color: #6366f1; margin: 0;'>{t['sidebar_title']}</h3>
    <small style='color: #64748b;'>{t['sidebar_desc']}</small>
</div>
<hr style='border: 1px solid #334155; margin: 10px 0;' />
""", unsafe_allow_html=True)

# Language Toggle Button
st.sidebar.button(t['lang_btn'], on_click=toggle_lang)

st.sidebar.info(t['sidebar_info'])

# Main Content
st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{t["subtitle"]}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="card-box">
        <h3 style="color: #6366f1; margin-top: 0;">{t['arch_title']}</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">{t['arch_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-box">
        <h3 style="color: #10b981; margin-top: 0;">{t['plat_title']}</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">{t['plat_desc']}</p>
        <ul style="color: #cbd5e1; direction: {t['dir']}; text-align: {t['align']}; line-height: 1.8;">
            <li><b>{t['plat_list1']}</b></li>
            <li><b>{t['plat_list2']}</b></li>
            <li><b>{t['plat_list3']}</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card-box" style="text-align: center;">
        <h4 style="color: #f59e0b; margin-top: 0;">{t['specs_title']}</h4>
        <table style="width: 100%; color: #94a3b8; border-collapse: collapse; text-align: {t['align']}; direction: {t['dir']};">
            <tr style="border-bottom: 1px solid #334155;"><td style="padding: 8px 0;">{t['specs_shape']}</td><td style="color: white; font-weight: bold;">(1, 32, 32)</td></tr>
            <tr style="border-bottom: 1px solid #334155;"><td style="padding: 8px 0;">{t['specs_classes']}</td><td style="color: white; font-weight: bold;">10</td></tr>
            <tr style="border-bottom: 1px solid #334155;"><td style="padding: 8px 0;">{t['specs_acc']}</td><td style="color: #10b981; font-weight: bold;">97.07%</td></tr>
            <tr style="border-bottom: 1px solid #334155;"><td style="padding: 8px 0;">{t['specs_mode']}</td><td style="color: white; font-weight: bold;">Early Stopping</td></tr>
            <tr><td style="padding: 8px 0;">{t['specs_device']}</td><td style="color: #a78bfa; font-weight: bold;">CPU / CUDA</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-box" style="text-align: center;">
        <h4 style="color: #3b82f6; margin-top:0;">{t['guide_title']}</h4>
        <code style="color: #38bdf8; display: block; margin-bottom: 15px; font-family: monospace;">pip install -r requirements.txt</code>
        <code style="color: #38bdf8; display: block; font-family: monospace;">streamlit run main.py</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
***
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 20px;">
    {t['footer']}
</div>
""", unsafe_allow_html=True)