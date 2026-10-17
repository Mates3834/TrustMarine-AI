"""Synthetic shipboard condition monitoring and auditable diagnostic agent.
Research prototype only: not validated for operational ship safety decisions.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

FEATURES = ['generator_load_kw','coolant_temp_c','bearing_vibration_mm_s','oil_pressure_bar','bus_frequency_hz']
FAULTS = ['normal','cooling_degradation','bearing_wear','lubrication_loss','load_instability']

def generate_data(n_per_class=400, seed=42):
    rng = np.random.default_rng(seed)
    frames=[]
    for label in FAULTS:
        n=n_per_class
        load=rng.normal(480,75,n).clip(180,780)
        temp=64+0.026*load+rng.normal(0,2.2,n)
        vibration=1.35+0.0015*load+rng.normal(0,.22,n)
        pressure=4.8-0.0008*load+rng.normal(0,.13,n)
        frequency=50+rng.normal(0,.065,n)
        severity=rng.uniform(.5,1.0,n)
        if label=='cooling_degradation': temp+=10+13*severity
        if label=='bearing_wear': vibration+=2.1+2.7*severity; temp+=2*severity
        if label=='lubrication_loss': pressure-=1.3+1.25*severity; vibration+=.55*severity
        if label=='load_instability': frequency+=rng.choice([-1,1],n)*(0.4+0.8*severity); load+=rng.normal(0,45,n)
        f=pd.DataFrame(dict(generator_load_kw=load,coolant_temp_c=temp,bearing_vibration_mm_s=vibration,oil_pressure_bar=pressure,bus_frequency_hz=frequency, fault_type=label))
        frames.append(f)
    return pd.concat(frames,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)

def fit_detector(train_normal, seed=42):
    model=IsolationForest(n_estimators=200,contamination=.02,random_state=seed)
    model.fit(train_normal[FEATURES])
    return model

@dataclass
class Decision:
    suspected_fault: str
    anomalous: bool
    confidence_level: str
    evidence: list
    recommended_action: str
    requires_human_review: bool
    tools_used: list
    limitations: str

class DiagnosticAgent:
    """Auditable tool-use workflow, not an LLM. Advisory only."""
    def __init__(self, detector, baseline):
        self.detector=detector
        self.baseline=baseline
        self.means=baseline[FEATURES].mean()
        self.stds=baseline[FEATURES].std().clip(lower=.01)

    def inspect_sensors(self, sample):
        return {f:float(sample[f]) for f in FEATURES}

    def detect_anomaly(self, sample):
        x=pd.DataFrame([self.inspect_sensors(sample)])
        return bool(self.detector.predict(x)[0]==-1),float(self.detector.decision_function(x)[0])

    def assess(self, sample):
        vals=self.inspect_sensors(sample)
        anomaly,score=self.detect_anomaly(sample)
        # Independent engineering review trigger: catches extreme sensor readings even if the ML detector misses them.
        hard_limit = (vals['bearing_vibration_mm_s'] > 5.0 or vals['coolant_temp_c'] > 105.0 or vals['oil_pressure_bar'] < 2.5 or abs(vals['bus_frequency_hz']-50) > 0.8)
        anomaly = anomaly or hard_limit
        z={f:(vals[f]-self.means[f])/self.stds[f] for f in FEATURES}
        # Rule-based diagnosis: transparent baseline; not a validated physical diagnostic model.
        candidates={
            'cooling_degradation':max(0,z['coolant_temp_c']),
            'bearing_wear':max(0,z['bearing_vibration_mm_s']),
            'lubrication_loss':max(0,-z['oil_pressure_bar']),
            'load_instability':max(0,abs(vals['bus_frequency_hz']-50)/.10),
        }
        fault=max(candidates,key=candidates.get)
        magnitude=candidates[fault]
        if not anomaly:
            fault='normal'
        evidence=[]
        if anomaly:
            for f in FEATURES:
                if abs(z[f])>=2.5:
                    evidence.append(f'{f}={vals[f]:.2f} (baseline z={z[f]:+.1f})')
            if not evidence: evidence=['IsolationForest flagged multivariate deviation without a single >=2.5 sigma feature']
        else: evidence=['No anomaly detected by the fitted baseline model']
        ambiguous=anomaly and (magnitude<3 or sorted(candidates.values(),reverse=True)[0]-sorted(candidates.values(),reverse=True)[1]<1.0)
        confidence='low' if ambiguous else ('moderate' if anomaly else 'baseline-consistent')
        action='Request engineer inspection and confirm with independent measurements; do not automate maintenance actions.' if anomaly else 'Continue routine monitoring; no automated safety decision.'
        return Decision(fault,anomaly,confidence,evidence,action,anomaly or ambiguous,
                        ['inspect_sensors','detect_anomaly','rule_based_diagnosis','human_review_gate'],
                        'Synthetic training data; heuristic diagnosis; anomaly score is not a calibrated probability. Not for operational use.')

def evaluate(y_true,y_pred):
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[False,True]).ravel()
    return dict(precision=float(precision),recall=float(recall),f1=float(f1),true_negative=int(tn),false_positive=int(fp),false_negative=int(fn),true_positive=int(tp))
