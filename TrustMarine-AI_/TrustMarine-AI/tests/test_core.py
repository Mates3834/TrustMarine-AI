from src.core import generate_data, fit_detector, DiagnosticAgent, FEATURES

def test_reproducible():
    assert generate_data(10,42).equals(generate_data(10,42))

def test_review_gate_on_anomaly():
    data=generate_data(80,42)
    baseline=data[data.fault_type=='normal']
    agent=DiagnosticAgent(fit_detector(baseline),baseline)
    extreme={f:float(baseline[f].median()) for f in FEATURES}
    extreme['bearing_vibration_mm_s']=20
    decision=agent.assess(extreme)
    assert decision.anomalous and decision.requires_human_review
    assert decision.evidence
