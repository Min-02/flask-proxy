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
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/proxy", methods=["GET"])
def proxy():
    service_key = "rA86OMjx7TmsRL+UAjovPORxHyyDJZxd6dIPJyKlqbPZzNo5fetvxLXhZ/MPki0fWIgUPGXq0thGIvFG5BmTZg=="
    base_url = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"

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

        if response.status != 200:
            return jsonify({"error": "외부 API 요청에 실패했습니다."}), 500

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

    start_time, end_time = data["time_range"]
    selected_days = data["day_of_week"]

    # 데이터 로드
    model = joblib.load("0504_xgboost_market_model.pkl")
    df = pd.read_csv("0504_광진구 상권 데이터 통합 완성본.csv", encoding="cp949")

    # ✅ 시간대 정의 (언더바 기반)
    defined_times = {
        "시간대_00_06": (0, 6),
        "시간대_06_11": (6, 11),
        "시간대_11_14": (11, 14),
        "시간대_14_17": (14, 17),
        "시간대_17_21": (17, 21),
        "시간대_21_24": (21, 24)
    }

    # 가장 가까운 상권 찾기
    df["거리"] = df.apply(
        lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km, axis=1
    )
    nearest = df.loc[df["거리"].idxmin()].copy()

    # ✅ 상권 + 업종 기준으로 경쟁 업종 수 추출
    competition_row = df[
        (df['상권_코드_명'] == nearest['상권_코드_명']) &
        (df['서비스_업종_코드_명'] == indsMclsCd)
        ].iloc[0]
    num_competitors = competition_row['300m내_경쟁_업종_수']

    # ✅ 300m 내 매출 데이터가 있는 점포 수
    nearby_with_sales = df[
        (df['서비스_업종_코드_명'] == indsMclsCd) &
        (df['거리'] <= 300) &
        (df['당월_매출_금액'].notna())
        ]
    num_with_sales = len(nearby_with_sales)

    print()
    if num_competitors == 0:
        print("⚠️ 해당 상권에 해당 업종 점포가 없어 예측이 불가합니다.")
        exit()
    elif num_competitors < 3 or num_with_sales == 0:
        print("⚠️ 이 상권의 해당 업종 혹은 매출 데이터가 부족하여 신뢰도가 낮습니다.")

    # ✅ 입력 벡터 구성
    features = pd.DataFrame([{
        '총_유동인구_수': nearest['총_유동인구_수'],
        '남성_유동인구_수': nearest['남성_유동인구_수'],
        '여성_유동인구_수': nearest['여성_유동인구_수'],
        '연령대_10_유동인구_수': nearest.get('연령대_10_유동인구_수', 0),
        '연령대_20_유동인구_수': nearest.get('연령대_20_유동인구_수', 0),
        '연령대_30_유동인구_수': nearest.get('연령대_30_유동인구_수', 0),
        '연령대_40_유동인구_수': nearest.get('연령대_40_유동인구_수', 0),
        '연령대_50_유동인구_수': nearest.get('연령대_50_유동인구_수', 0),
        '연령대_60_이상_유동인구_수': nearest.get('연령대_60_이상_유동인구_수', 0),
        '시간대_00_06_유동인구_수': nearest.get('시간대_00_06_유동인구_수', 0),
        '시간대_06_11_유동인구_수': nearest.get('시간대_06_11_유동인구_수', 0),
        '시간대_11_14_유동인구_수': nearest.get('시간대_11_14_유동인구_수', 0),
        '시간대_14_17_유동인구_수': nearest.get('시간대_14_17_유동인구_수', 0),
        '시간대_17_21_유동인구_수': nearest.get('시간대_17_21_유동인구_수', 0),
        '시간대_21_24_유동인구_수': nearest.get('시간대_21_24_유동인구_수', 0),
        '월요일_유동인구_수': nearest.get('월요일_유동인구_수', 0),
        '화요일_유동인구_수': nearest.get('화요일_유동인구_수', 0),
        '수요일_유동인구_수': nearest.get('수요일_유동인구_수', 0),
        '목요일_유동인구_수': nearest.get('목요일_유동인구_수', 0),
        '금요일_유동인구_수': nearest.get('금요일_유동인구_수', 0),
        '토요일_유동인구_수': nearest.get('토요일_유동인구_수', 0),
        '일요일_유동인구_수': nearest.get('일요일_유동인구_수', 0),
        '서비스_업종_코드_명': indsMclsCd,
        '상권_변화_지표_명': nearest.get('상권_변화_지표_명'),
        '300m내_경쟁_업종_수': num_competitors
    }])

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

    # 결과 전달
    return jsonify({
        "상권명": nearest["상권_코드_명"],
        "경쟁수": int(num_competitors),
        "예측매출": int(예측매출),
        "SHAP": shap_impact.head(5).to_dict(orient="records"),  # 상위 5개만 전달
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
