from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from urllib.parse import urlencode
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import urllib3
import ssl
import json
import joblib
import shap
import io
import os
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
def predicted_sales():

    # 데이터 로드
    model = joblib.load("0504_xgboost_market_model.pkl")
    label_encoders = joblib.load("0504_label_encoders.pkl")
    df = pd.read_csv("0504_광진구 상권 데이터 통합 완성본.csv", encoding="cp949")

    data = request.get_json()
    lat = float(data["lat"])
    lon = float(data["lon"])
    indsMclsCd = data["indsMclsCd"]
    radius_m = float(data["radius"])
    radius_km = radius_m / 1000

    time_range = data["time_range"]  # 예: "6-14"
    start_time_str, end_time_str = time_range.split("-")
    start_time = int(start_time_str)
    end_time = int(end_time_str)
    selected_days = data["day_of_week"]

    # 코드 → 업종명 매핑
    industry_code_map = {
        "I212": "커피-음료",
        "I201": "한식음식점",
        "I202": "중식음식점",
        # 필요 시 계속 추가
    }
    # ✅ 업종 코드 → 업종명 변환
    category = industry_code_map.get(indsMclsCd)

    try:
        encoded_category = label_encoders['서비스_업종_코드_명'].transform([category])[0]
        # 가장 가까운 상권 찾기
        df['거리'] = df.apply(
            lambda row: geodesic((lat, lon), (row['위도'], row['경도'])).meters, axis=1)
        nearest = df.loc[df['거리'].idxmin()]
        change_encoded = label_encoders['상권_변화_지표_명'].transform([nearest['상권_변화_지표_명']])[0]

        # ✅ 상권 + 업종 기준으로 경쟁 업종 수 추출
        competition_row = df[
            (df['상권_코드_명'] == nearest['상권_코드_명']) &
            (df['서비스_업종_코드_명'] == category)
        ].iloc[0]
        num_competitors = competition_row['300m내_경쟁_업종_수']

        # ✅ 300m 내 매출 데이터가 있는 점포 수
        nearby_with_sales = df[
            (df['서비스_업종_코드_명'] == category) &
            (df['거리'] <= 300) &
            (df['당월_매출_금액'].notna())
        ]
        num_with_sales = len(nearby_with_sales)

        if num_competitors == 0:
            return jsonify({'message': "⚠️ 해당 상권에 해당 업종 점포가 없어 예측이 불가합니다."})
        elif num_competitors < 3 or len(nearby_with_sales) == 0:
            confidence = "⚠️ 이 상권의 해당 업종 혹은 매출 데이터가 부족하여 신뢰도가 낮습니다."
        else:
            confidence = "✅ 신뢰도 양호"

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
            '서비스_업종_코드_명': encoded_category,
            '상권_변화_지표_명': change_encoded,
            '300m내_경쟁_업종_수': num_competitors
        }])

        # 모델 입력 생성
        prediction = model.predict(features)[0]

        # ✅ 시간대 정의 (언더바 기반)
        defined_times = {
            "시간대_00_06": (0, 6),
            "시간대_06_11": (6, 11),
            "시간대_11_14": (11, 14),
            "시간대_14_17": (14, 17),
            "시간대_17_21": (17, 21),
            "시간대_21_24": (21, 24)
        }

        # ✅ 요일 보정
        total_weekly_sales = sum([nearest.get(f"{day}요일_매출_금액", 0) for day in ['월', '화', '수', '목', '금', '토', '일']])
        selected_sales = sum([nearest.get(f"{day}요일_매출_금액", 0) for day in selected_days])
        if total_weekly_sales > 0:
            prediction *= (selected_sales / total_weekly_sales)

        # ✅ 시간대 보정
        total_time_sales = 0
        selected_time_sales = 0
        for col, (t_start, t_end) in defined_times.items():
            overlap = max(0, min(end_time, t_end) - max(start_time, t_start))
            duration = t_end - t_start
            sale_amt = nearest.get(f"{col}_매출_금액", 0)
            total_time_sales += sale_amt
            if overlap > 0:
                selected_time_sales += sale_amt * (overlap / duration)
        if total_time_sales > 0:
            prediction *= (selected_time_sales / total_time_sales)

        # ✅ 경쟁 점포 수로 나누기
        if num_competitors > 0:         # 0 나눗셈 방지
            prediction /= num_competitors

        print("📤 예측 결과 응답:", {
            "위치": {lat, lon},
            "상권명": nearest["상권_코드_명"],
            "경쟁수": int(num_competitors),
            "predicted_sales": int(prediction),
            "신뢰도": confidence
        })

        # 결과 전달
        return jsonify({
            "상권명": nearest["상권_코드_명"],
            "경쟁수": int(num_competitors),
            "predicted_sales": int(prediction),
            "신뢰도": confidence,
        })
    except Exception as e:
        return jsonify({'message': f"❌ 예측 중 오류 발생: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
