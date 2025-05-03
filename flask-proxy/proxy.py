from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib3
import ssl
import json
from urllib.parse import urlencode
import shap
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://heroic-flan-80b6ca.netlify.app"}})

@app.route("/api/proxy", methods=["GET"])
def proxy():
    service_key = "rA86OMjx7TmsRL+UAjovPORxHyyDJZxd6dIPJyKlqbPZzNo5fetvxLXhZ/MPki0fWIgUPGXq0thGIvFG5BmTZg=="
    base_url = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"

    response_data = {"message": "응답 내용"}
    return jsonify(response_data)

    params = {
        "serviceKey": service_key,
        "radius": request.args.get("radius"),
        "cx": request.args.get("cx"),
        "cy": request.args.get("cy"),
        "indsLclsCd": "I2",  # 전체 음식점
        "indsMclsCd": request.args.get("indsMclsCd"),  # 필터링용 중분류 업종코드
        "numOfRows": 100,
        "pageNo": 1,
        "type": "json"
    }

    try:
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")

        encoded_params = urlencode(params)
        full_url = f"{base_url}?{encoded_params}"
        print("[요청 URL]", full_url)

        http = urllib3.PoolManager(ssl_context=ctx)
        response = http.request("GET", full_url)

        print("[응답 상태]", response.status)
        data = json.loads(response.data.decode("utf-8"))
        return jsonify(data)

    except Exception as e:
        print("예외 발생:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict_sales():
    from geopy.distance import geodesic
    import pandas as pd
    import joblib
    import shap

    data = request.get_json()
    lat = float(data["lat"])
    lon = float(data["lon"])
    indsMclsCd = data["indsMclsCd"]
    radius_m = float(data["radius"])
    radius_km = radius_m / 1000

    # 데이터 로드
    model = joblib.load("xgboost_market_model.pkl")
    df_info = pd.read_csv("merged_market_with_competition.csv")
    df_store = pd.read_csv("광진구_한식중식카페_완전수집.csv", encoding="cp949")

    features = [
        "lat", "lon",
        "총_유동인구_수", "연령대_20_유동인구_수", "연령대_30_유동인구_수", "연령대_40_유동인구_수",
        "시간대_14_17_유동인구_수", "시간대_17_21_유동인구_수", "시간대_21_24_유동인구_수",
        "경쟁_점포_수"
    ]

    # 가장 가까운 상권 찾기
    df_info["거리"] = df_info.apply(
        lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km, axis=1
    )
    nearest = df_info.loc[df_info["거리"].idxmin()].copy()

    # 경쟁 점포 수 계산
    subset = df_store[df_store["indsMclsCd"] == indsMclsCd]
    경쟁수 = sum(
        geodesic((lat, lon), (row["lat"], row["lon"])).km <= radius_km
        for _, row in subset.iterrows()
    )
    nearest["경쟁_점포_수"] = 경쟁수

    # 모델 입력 생성
    sample = nearest[features].to_frame().T.astype(float)
    예측매출 = model.predict(sample)[0]

    # SHAP 계산
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # 주요 기여도 테아블
    shap_impact = pd.DataFrame({
        'Feature': sample.columns,
        'Feature Value': sample.values.flatten(),
        'SHAP Value': shap_values.flatten()
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    # force_plot HTML 생성
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],  # 첫 번째 샘플만
        sample.iloc[0],
        matplotlib=False,
        show=False
    )

    # force_plot을 HTML string으로 저장
    shap_html = f"""
        <html><head>{shap.getjs()}</head><body>{force_plot.html()}</body></html>
        """

    # 결과 전달
    return jsonify({
        "상권명": nearest["상권_코드_명"],
        "경쟁수": int(경쟁수),
        "예측매출": int(예측매출),
        "SHAP": shap_impact.head(5).to_dict(orient="records"),  # 상위 5개만 전달
        "shap_html": shap_html
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
