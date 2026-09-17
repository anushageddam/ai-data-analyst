import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("Testing live FastAPI endpoints on", BASE_URL)

    # 1. Test index page
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Index failed: {r.status_code}"
    assert "YOO PROJECT" in r.text
    print(" [PASS] GET / (Serves Single Page Application)")

    # 2. Test sample loading (SAP Procurement)
    r = requests.post(f"{BASE_URL}/api/load-sample", json={"sample_id": "sap_procurement"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    print(" [PASS] POST /api/load-sample (SAP Procurement)")

    # 3. Test status
    r = requests.get(f"{BASE_URL}/api/status")
    assert r.status_code == 200
    st = r.json()
    assert st["loaded"] is True
    assert "Procurement" in st["domain"]
    print(f" [PASS] GET /api/status (Domain: {st['domain']}, Health: {st['health_score']}%)")

    # 4. Test dashboard endpoint
    r = requests.post(f"{BASE_URL}/api/dashboard", json={"filters": {}})
    assert r.status_code == 200
    dash = r.json()
    assert len(dash["kpis"]) >= 3
    assert len(dash["visuals"]) >= 2
    assert len(dash["insights"]) >= 2
    print(f" [PASS] POST /api/dashboard ({len(dash['kpis'])} KPIs, {len(dash['visuals'])} Visuals, {len(dash['insights'])} Insights)")

    # 5. Test dynamic cross-filtering
    plant_filter = {"Plant": ["Plant 1010"]}
    r = requests.post(f"{BASE_URL}/api/dashboard", json={"filters": plant_filter})
    assert r.status_code == 200
    filtered = r.json()
    assert filtered["active_rows"] < dash["total_rows"]
    print(f" [PASS] POST /api/dashboard Cross-Filter (Filtered to {filtered['active_rows']} of {dash['total_rows']} rows)")

    # 6. Test profiler details
    r = requests.get(f"{BASE_URL}/api/profiler")
    assert r.status_code == 200
    prof = r.json()
    assert "summary" in prof
    print(f" [PASS] GET /api/profiler ({len(prof['columns'])} columns profiled, Outliers: {prof['summary']['total_outliers_detected']})")

    # 7. Test forecasting
    r = requests.post(f"{BASE_URL}/api/forecast", json={"measure": "Net_Value", "horizon": 3})
    assert r.status_code == 200
    fc = r.json()
    assert fc["is_suitable"] is True
    print(f" [PASS] POST /api/forecast (Predicted Mean: {fc['metrics']['forecast_mean']:,.2f}, MAPE: {fc['metrics']['mape']}%)")

    # 8. Test chat
    r = requests.post(f"{BASE_URL}/api/chat", json={"query": "Which vendor has the highest spend?"})
    assert r.status_code == 200
    chat_resp = r.json()
    assert "answer" in chat_resp
    assert chat_resp["chart"] is not None
    print(" [PASS] POST /api/chat (Grounded answer + inline chart generated)")

    # 9. Test explorer
    r = requests.get(f"{BASE_URL}/api/explorer?page=1&page_size=10")
    assert r.status_code == 200
    exp = r.json()
    assert len(exp["data"]) <= 10
    print(f" [PASS] GET /api/explorer (Page 1 of records returned)")

    # 10. Test export
    r = requests.get(f"{BASE_URL}/api/export")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    print(" [PASS] GET /api/export (CSV export streamed)")

    print("\n>>> ALL 10 API INTEGRATION TESTS PASSED! <<<")

if __name__ == "__main__":
    test_api()
