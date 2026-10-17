import numpy as np
import torch
import torch.nn as nn
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==================== 1. CLASS DEFINITIONS & CONFIG ====================
# Note: The order strictly matches the LabelEncoder output from the dataset
class_names = ['Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'Normal', 'OR_007', 'OR_014', 'OR_021']

severity_weights = {
    'Normal': 0,
    'Ball_007': 20, 'Ball_014': 35, 'Ball_021': 50,
    'IR_007': 35, 'IR_014': 60, 'IR_021': 80,
    'OR_007': 45, 'OR_014': 70, 'OR_021': 90
}

# ==================== 2. CNN PYTORCH MODEL ====================
class BearingCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BearingCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== 3. FUZZY LOGIC CONTROLLER SYSTEM ====================
# Antecedents & Consequents
hi_var = ctrl.Antecedent(np.arange(0, 101, 1), 'Health_Index')
conf = ctrl.Antecedent(np.arange(0, 101, 1), 'Confidence')
sev = ctrl.Antecedent(np.arange(0, 101, 1), 'Fault_Severity')
trend = ctrl.Antecedent(np.arange(-50, 51, 1), 'HI_Trend')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'Risk_Level')
urgency = ctrl.Consequent(np.arange(0, 101, 1), 'Urgency')

# --- Membership Functions Setup ---
# Health Index (Higher is better)
hi_var['Critical'] = fuzz.trapmf(hi_var.universe, [0, 0, 20, 35])
hi_var['Poor'] = fuzz.trimf(hi_var.universe, [25, 40, 55])
hi_var['Fair'] = fuzz.trimf(hi_var.universe, [50, 65, 80])
hi_var['Good'] = fuzz.trimf(hi_var.universe, [75, 85, 95])
hi_var['Excellent'] = fuzz.trapmf(hi_var.universe, [90, 95, 100, 100])

# Confidence (Model certainty)
conf['Low'] = fuzz.trimf(conf.universe, [0, 0, 65])
conf['Medium'] = fuzz.trimf(conf.universe, [55, 75, 88])
conf['High'] = fuzz.trimf(conf.universe, [82, 95, 100])

# Severity (Inherent fault weight)
sev['Low'] = fuzz.trimf(sev.universe, [0, 0, 35])
sev['Medium'] = fuzz.trimf(sev.universe, [30, 50, 70])
sev['High'] = fuzz.trimf(sev.universe, [65, 82, 100])

# Trend (Worsening: +, Improving: -)
trend['Improving'] = fuzz.trimf(trend.universe, [-50, -20, 0])
trend['Stable'] = fuzz.trimf(trend.universe, [-5, 0, 5])
trend['Worsening'] = fuzz.trimf(trend.universe, [0, 20, 50])

# Outputs: Risk & Urgency
risk['Healthy'] = fuzz.trimf(risk.universe, [0, 12, 32])
risk['Warning'] = fuzz.trimf(risk.universe, [28, 48, 68])
risk['Critical'] = fuzz.trimf(risk.universe, [65, 85, 100])

urgency['Low'] = fuzz.trimf(urgency.universe, [0, 22, 42])
urgency['Medium'] = fuzz.trimf(urgency.universe, [38, 58, 78])
urgency['High'] = fuzz.trimf(urgency.universe, [75, 92, 100])

# ==================== Comprehensive Fuzzy Rules Base ====================
rules = [
    # ---------------------------------------------------------
    # 1. HEALTHY STATES (Risk is Low)
    # ---------------------------------------------------------
    ctrl.Rule(hi_var['Excellent'], risk['Healthy']),
    ctrl.Rule(hi_var['Good'] & sev['Low'] & conf['High'], risk['Healthy']),
    ctrl.Rule(hi_var['Good'] & sev['Low'] & conf['Medium'], risk['Healthy']),
    ctrl.Rule(hi_var['Excellent'] & trend['Stable'], risk['Healthy']),
    ctrl.Rule(hi_var['Excellent'] & trend['Improving'], risk['Healthy']),
    ctrl.Rule(hi_var['Good'] & trend['Improving'], risk['Healthy']),
    ctrl.Rule(hi_var['Fair'] & sev['Low'] & trend['Improving'], risk['Healthy']),

    # ---------------------------------------------------------
    # 2. WARNING STATES (Risk is Medium)
    # ---------------------------------------------------------
    ctrl.Rule(hi_var['Fair'] & sev['Low'] & trend['Stable'], risk['Warning']),
    ctrl.Rule(hi_var['Fair'] & sev['Medium'] & trend['Stable'], risk['Warning']),
    ctrl.Rule(hi_var['Good'] & sev['Medium'], risk['Warning']),
    ctrl.Rule(hi_var['Excellent'] & conf['Low'], risk['Warning']),
    ctrl.Rule(hi_var['Good'] & conf['Low'], risk['Warning']),
    ctrl.Rule(hi_var['Fair'] & conf['Low'], risk['Warning']),
    ctrl.Rule(hi_var['Poor'] & sev['Low'] & trend['Improving'], risk['Warning']),
    ctrl.Rule(hi_var['Excellent'] & trend['Worsening'], risk['Warning']),
    ctrl.Rule(hi_var['Good'] & trend['Worsening'] & conf['Low'], risk['Warning']),

    # ---------------------------------------------------------
    # 3. CRITICAL STATES (Risk is High)
    # ---------------------------------------------------------
    ctrl.Rule(hi_var['Critical'], risk['Critical']),
    ctrl.Rule(hi_var['Poor'] & sev['High'], risk['Critical']),
    ctrl.Rule(hi_var['Poor'] & sev['Medium'], risk['Critical']),
    ctrl.Rule(hi_var['Poor'] & trend['Worsening'], risk['Critical']),
    ctrl.Rule(hi_var['Poor'] & trend['Stable'], risk['Critical']),
    ctrl.Rule(hi_var['Fair'] & sev['High'], risk['Critical']),
    ctrl.Rule(hi_var['Fair'] & trend['Worsening'], risk['Critical']),
    ctrl.Rule(hi_var['Fair'] & sev['Medium'] & trend['Worsening'], risk['Critical']),
    ctrl.Rule(hi_var['Good'] & sev['High'], risk['Critical']),
    ctrl.Rule(hi_var['Good'] & trend['Worsening'] & conf['High'], risk['Critical']),
    ctrl.Rule(hi_var['Excellent'] & sev['High'], risk['Critical']),
    ctrl.Rule(sev['High'] & conf['High'], risk['Critical']),

    # ---------------------------------------------------------
    # 4. URGENCY CALCULATION (Dependent on Risk & Confidence/Trend)
    # ---------------------------------------------------------
    # Critical Risk Urgency
    ctrl.Rule(risk['Critical'] & conf['High'], urgency['High']),
    ctrl.Rule(risk['Critical'] & conf['Medium'], urgency['High']),
    ctrl.Rule(risk['Critical'] & conf['Low'], urgency['Medium']),
    ctrl.Rule(risk['Critical'] & trend['Worsening'], urgency['High']),

    # Warning Risk Urgency
    ctrl.Rule(risk['Warning'] & trend['Worsening'], urgency['High']),
    ctrl.Rule(risk['Warning'] & conf['High'], urgency['Medium']),
    ctrl.Rule(risk['Warning'] & conf['Medium'], urgency['Medium']),
    ctrl.Rule(risk['Warning'] & conf['Low'], urgency['Medium']),
    ctrl.Rule(risk['Warning'] & trend['Improving'], urgency['Low']),

    # Healthy Risk Urgency
    ctrl.Rule(risk['Healthy'] & trend['Worsening'], urgency['Medium']),
    ctrl.Rule(risk['Healthy'] & conf['High'], urgency['Low']),
    ctrl.Rule(risk['Healthy'] & conf['Low'], urgency['Low']),
]

risk_ctrl = ctrl.ControlSystem(rules)

# ==================== 4. SYSTEM CALCULATION FUNCTIONS ====================
def compute_health_index(probs_np):
    damage_index = 0.0
    for i, prob in enumerate(probs_np):
        cls = class_names[i]
        damage_index += prob * severity_weights.get(cls, 50)

    health_index = max(0, 100 - damage_index)
    return health_index

def get_fault_severity(predicted_class):
    return severity_weights.get(predicted_class, 50)

def compute_hi_trend(hi_history):
    if hi_history is None or len(hi_history) < 2:
        return 0.0
    return hi_history[-2] - hi_history[-1]

def fuzzy_decision(probs_np, predicted_class, hi_history=None):
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    hi_score = compute_health_index(probs_np)
    confidence_score = float(np.max(probs_np)) * 100
    fault_sev = get_fault_severity(predicted_class)
    trend_value = compute_hi_trend(hi_history)

    risk_sim.input['Health_Index'] = hi_score
    risk_sim.input['Confidence'] = confidence_score
    risk_sim.input['Fault_Severity'] = fault_sev
    risk_sim.input['HI_Trend'] = trend_value

    risk_sim.compute()

    final_risk = risk_sim.output['Risk_Level']
    final_urgency = risk_sim.output['Urgency']

    # Outputs are kept in English for core logic (Translations applied in the UI)
    if final_risk < 35:
        state = "Healthy"
        action = "Normal operation - Periodic equipment monitoring"
        badge_style = "success"
    elif final_risk < 65:
        state = "Warning"
        action = "Schedule quality control and re-lubrication in the short term"
        badge_style = "warning"
    else:
        state = "Critical"
        action = "Immediate system shutdown and bearing replacement to prevent mechanical failure"
        badge_style = "danger"

    return {
        "Fault_Type": predicted_class,
        "Health_Index": round(hi_score, 1),
        "Confidence": round(confidence_score, 1),
        "Risk_Level": round(final_risk, 1),
        "Urgency": round(final_urgency, 1),
        "State": state,
        "Recommended_Action": action,
        "Fault_Severity": round(fault_sev, 1),
        "badge_style": badge_style
    }

def predict_bearing_fault_real_time(model, signal_window, hi_history=None, device='cpu'):
    model.eval()
    with torch.no_grad():
        if len(signal_window.shape) == 2:  # (32, 32)
            x = torch.FloatTensor(signal_window).unsqueeze(0).unsqueeze(0).to(device)
        elif len(signal_window.shape) == 3:  # (1, 32, 32)
            x = torch.FloatTensor(signal_window).unsqueeze(0).to(device)
        else:
            raise ValueError("Input dimensions must be (32,32) or (1,32,32)")

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        probs_np = probs[0].cpu().numpy()

        predicted_idx = probs.argmax().item()
        predicted_class = class_names[predicted_idx]

    result = fuzzy_decision(probs_np, predicted_class, hi_history)
    return result, probs_np