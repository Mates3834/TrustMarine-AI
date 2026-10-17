from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.core import generate_data, fit_detector, DiagnosticAgent, FEATURES, evaluate

out=Path('results'); out.mkdir(exist_ok=True)
data=generate_data(n_per_class=400,seed=42)
normal=data[data.fault_type=='normal']
train_normal,test_normal=train_test_split(normal,test_size=.35,random_state=42)
faults=data[data.fault_type!='normal']
_,test_faults=train_test_split(faults,test_size=.35,random_state=42,stratify=faults.fault_type)
test=pd.concat([test_normal,test_faults],ignore_index=True).sample(frac=1,random_state=42)
model=fit_detector(train_normal)
agent=DiagnosticAgent(model,train_normal)
true=(test.fault_type!='normal').to_numpy()
model_pred=model.predict(test[FEATURES])==-1
thresholds={'coolant_temp_c':100,'bearing_vibration_mm_s':3.2,'oil_pressure_bar':3.5}
threshold_pred=np.zeros(len(test),dtype=bool)
for f,t in thresholds.items(): threshold_pred |= (test[f].to_numpy()<t if f=='oil_pressure_bar' else test[f].to_numpy()>t)
threshold_pred |= np.abs(test.bus_frequency_hz.to_numpy()-50)>.4
rows=[]
for name,pred in [('fixed_threshold',threshold_pred),('isolation_forest',model_pred)]:
    rows.append(dict(method=name,**evaluate(true,pred)))
report=pd.DataFrame(rows)
report.to_csv(out/'metrics.csv',index=False)
# Agent does not improve anomaly detection: same model, with explainable diagnosis and review gate.
examples=[]
for label in ['normal','cooling_degradation','bearing_wear','lubrication_loss','load_instability']:
    row=test[test.fault_type==label].iloc[0]
    d=agent.assess(row)
    examples.append({'ground_truth':label,'decision':d.__dict__})
(out/'agent_cases.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False))
fig,ax=plt.subplots(figsize=(7,4))
x=np.arange(len(report)); w=.25
for i,k in enumerate(['precision','recall','f1']): ax.bar(x+(i-1)*w,report[k],w,label=k)
ax.set_xticks(x,report.method);ax.set_ylim(0,1);ax.set_ylabel('Score');ax.legend();ax.set_title('Synthetic holdout: anomaly detection')
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
print(report.to_string(index=False))
print('Saved results/metrics.csv, results/agent_cases.json, results/comparison.png')
