from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from urllib.parse import urlencode
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import importlib.util
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import urllib3
import math
import ssl
import json
import joblib
import os
import io

matplotlib.use('Agg')
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
    # 📦 보정 로직 불러오기
    spec = importlib.util.spec_from_file_location("bojeong", "보정로직_서비스구조_정합버전.py")
    bojeong = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bojeong)
    get_sales_distribution_basis = bojeong.get_sales_distribution_basis
    apply_temporal_corrections = bojeong.apply_temporal_corrections

    # 📂 모델 및 입력 피처 예측값 불러오기
    model_paths = {
        "한식음식점": "0520_model_Korean_Chinese.pkl",
        "중식음식점": "0520_model_Korean_Chinese.pkl",
        "커피-음료": "0520_model_Cafe_Beverage.pkl"
    }
    label_encoders = joblib.load("0520_encoders.pkl")
    feature_df = pd.read_csv("20252_input_vector_0521.csv")

    # 📂 데이터셋 로드 함수 정의
    def load_dataframe(path):
        try:
            return pd.read_csv(path, encoding='cp949')
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='utf-8-sig')

    # 📂 데이터 로딩
    df = load_dataframe("0510_광진구 상권, 지하철 통합 완성본.csv")
    df_subway = load_dataframe("광진구 지하철 평균 승하차 인원 수.csv").dropna(subset=["위도", "경도"])

    # 📌 기준분기 추가 (예: 20244)
    df["기준분기"] = df["기준_년분기_코드"].astype(str).str[:4].astype(int) * 10 + df["기준_년분기_코드"].astype(str).str[-1].astype(int)

    # 📍 BallTree 구축 (위치 기반 최근 상권 탐색용)
    coords_rad = np.radians(df[["위도", "경도"]])
    tree = BallTree(coords_rad, metric="haversine")

    # 🔹 위도/경도 기준 거리 이동 보정 함수
    def offset_latlon(lat, lon, dy_m, dx_m):
        delta_lat = dy_m / 111000
        delta_lon = dx_m / (111000 * math.cos(math.radians(lat)))
        return lat + delta_lat, lon + delta_lon

    # 🔹 가장 가까운 지하철역 찾기
    def find_nearest_station(lat, lon):
        best_row = df_subway.iloc[((df_subway["위도"] - lat) ** 2 + (df_subway["경도"] - lon) ** 2).idxmin()]
        dist = geodesic((lat, lon), (best_row["위도"], best_row["경도"])).meters
        name = f"{best_row['역명']} ({best_row['노선명']})"
        traffic = best_row["일일_평균_승하차_인원_수"]
        return name, dist, traffic

    # 🔹 가장 가까운 상권 탐색
    def find_nearest_area(lat, lon):
        query_rad = np.radians([[lat, lon]])
        dist, idx = tree.query(query_rad, k=1)
        nearest = df.iloc[idx[0][0]]
        return nearest, dist[0][0] * 6371000

    # 🔹 상권 코드로 입력 피처 예측 벡터 불러오기
    def load_predicted_vector(area_code: int) -> dict:
        row = feature_df[feature_df["상권_코드"] == area_code]
        if row.empty:
            raise ValueError(f"❌ 상권 코드 {area_code} 에 해당하는 예측 피처 없음")
        return row.iloc[0].to_dict()

    # ✅ 파생 피처 생성 함수 (안전 버전)
    def add_derived_features(df):
        df = df.copy()
        df["남성_비율"] = df["남성_유동인구_수"] / (df["총_유동인구_수"] + 1)
        df["여성_비율"] = df["여성_유동인구_수"] / (df["총_유동인구_수"] + 1)
        df["연령대_중심값"] = (
            df["연령대_10_유동인구_수"] * 10 +
            df["연령대_20_유동인구_수"] * 20 +
            df["연령대_30_유동인구_수"] * 30 +
            df["연령대_40_유동인구_수"] * 40 +
            df["연령대_50_유동인구_수"] * 50 +
            df["연령대_60_이상_유동인구_수"] * 65
        ) / (df["총_유동인구_수"] + 1)
        df["상주대비_유동비"] = df["총_유동인구_수"] / (df["총_상주인구_수"] + 1)
        df["직장대비_유동비"] = df["총_유동인구_수"] / (df["총_직장_인구_수"] + 1)
        if "운영_영업_개월_평균" in df.columns and "서울_운영_영업_개월_평균" in df.columns:
            df["상권_vs_서울_운영차"] = df["운영_영업_개월_평균"] - df["서울_운영_영업_개월_평균"]
        else:
            df["상권_vs_서울_운영차"] = np.nan

        if "폐업_영업_개월_평균" in df.columns and "서울_폐업_영업_개월_평균" in df.columns:
            df["상권_vs_서울_폐업차"] = df["폐업_영업_개월_평균"] - df["서울_폐업_영업_개월_평균"]
        else:
            df["상권_vs_서울_폐업차"] = np.nan
        df["경쟁_밀집도"] = df["300m내_경쟁_업종_수"] / (df["총_유동인구_수"] + 1)
        df["역_접근성"] = df["가장_가까운_역_승하차_인원_수"] / (df["역까지_거리_m"] + 1)
        return df

    # 입력된 데이터
    data = request.get_json()
    lat = float(data["lat"])
    lon = float(data["lon"])
    indsMclsCd = data["indsMclsCd"]

    time_range = data["time_range"]  # 예: "6-14"
    start_time_str, end_time_str = time_range.split("-")
    start_time = int(start_time_str)
    end_time = int(end_time_str)
    selected_days = data["day_of_week"]
    store_count = int(data["store_count"])

    # 코드 → 업종명 매핑
    industry_code_map = {
        "I212": "커피-음료",
        "I201": "한식음식점",
        "I202": "중식음식점",
        # 필요 시 계속 추가
    }
    # ✅ 업종 코드 → 업종명 변환
    category = industry_code_map.get(indsMclsCd)

    print(lat, lon, start_time, end_time, selected_days, category, indsMclsCd)

    try:
        # 📍 입력 위치 기준 상권/지하철 분석
        nearest, _ = find_nearest_area(lat, lon)
        station_name, station_dist, station_traffic = find_nearest_station(lat, lon)

        change_encoder = (
            label_encoders["상권_변화_지표_명"]["커피_음료"]
            if "커피" in category else label_encoders["상권_변화_지표_명"]["한식중식_통합"]
        )
        change_encoded = change_encoder.transform([nearest["상권_변화_지표_명"]])[0]

        model = joblib.load(model_paths[category])
        df_basis = get_sales_distribution_basis(df, nearest["상권_코드_명"], category)
        df_basis = df_basis.dropna(subset=["점포_당_매출_금액"])

        # 📌 예측 불가능한 조건 처리
        if len(df_basis) == 0:
            print(f"\n📍 가장 가까운 상권: {nearest['상권_코드_명']}")
            print(f"🧾 업종: {category}")
            print("\n❌ 예측 불가: 해당 상권+업종 조합에 대한 매출 데이터가 존재하지 않아 예측할 수 없습니다.")
            print("\n📍 신뢰할 수 있는 주변 위치 분석 중...")
            base_sales = None
            return jsonify({
                "error": "해당 상권+업종 조합에 대한 매출 데이터가 없어 예측할 수 없습니다."
            }), 400

        else:
            if len(df_basis) <= 3:
                print(f"\n📍 가장 가까운 상권: {nearest['상권_코드_명']}")
                print(f"🧾 업종: {category}")
                print("\n⚠️ 참고: 해당 상권+업종 조합은 매출 데이터가 3개 이하로, 예측 결과의 신뢰도가 낮을 수 있습니다.")

            # ✅ 입력 피처 구성
            input_vec = load_predicted_vector(nearest["상권_코드"])
            input_vec["역까지_거리_m"] = station_dist
            input_vec["가장_가까운_역_승하차_인원_수"] = station_traffic
            input_vec["상권_변화_지표_명"] = int(change_encoded)
            if store_count is not None:
                input_vec["300m내_경쟁_업종_수"] = store_count
            else:
                input_vec["300m내_경쟁_업종_수"] = nearest["300m내_경쟁_업종_수"]  # fallback

            # ✅ 누락 피처 보완
            needed_cols = [
                "운영_영업_개월_평균", "폐업_영업_개월_평균",
                "서울_운영_영업_개월_평균", "서울_폐업_영업_개월_평균"
            ]
            recent_row = df[
                (df["기준분기"] == 20244) &
                (df["상권_코드"].astype(int) == int(nearest["상권_코드"])) &
                (df["서비스_업종_코드_명"] == category)
                ]
            if not recent_row.empty:
                for col in needed_cols:
                    if col not in input_vec:
                        input_vec[col] = recent_row.iloc[0].get(col, np.nan)
            else:
                for col in needed_cols:
                    input_vec[col] = np.nan

            # ✅ 파생 피처 포함 입력 데이터프레임 구성 및 예측
            input_df = pd.DataFrame([input_vec])
            input_df = add_derived_features(input_df)
            input_df = input_df[model.feature_names_in_]

            predicted_sales = model.predict(input_df)[0]
            predicted_sales = apply_temporal_corrections(predicted_sales, df_basis, selected_days, start_time, end_time)
            base_sales = predicted_sales

            base_result = {"lat": lat, "lon": lon, "sales": int(predicted_sales),
                           "상권명": nearest["상권_코드_명"],
                           "지하철역": station_name,
                           "지하철역거리": int(station_dist),
                           "승하차": int(station_traffic)
                           }

            print(f"\n📍 가장 가까운 상권: {nearest['상권_코드_명']}")
            print(f"🚇 가장 가까운 지하철역: {station_name} (거리: {station_dist:.1f}m, 일일 승하차: {int(station_traffic):,}명)")
            print(f"🕒 영업 시간: {start_time}시 ~ {end_time}시")
            print(f"📆 영업 요일: {', '.join(selected_days)}")
            print(f"💰 예측 월 매출: 약 {int(predicted_sales):,}원 (기준 100%)")
            print("\n📍 신뢰할 수 있는 주변 위치 분석 중...")

        results = []

        if base_sales is not None:
            for dy in range(-300, 301, 30):
                for dx in range(-300, 301, 30):
                    adj_lat, adj_lon  = offset_latlon(lat, lon, dy, dx)
                    dist = geodesic((lat, lon), (adj_lat, adj_lon )).meters
                    if dist <= 300:
                        if abs(adj_lat - lat) < 1e-6 and abs(adj_lon - lon) < 1e-6:
                            continue

                        near, _ = find_nearest_area(adj_lat, adj_lon)
                        try:
                            input_vec = load_predicted_vector(near["상권_코드"])
                        except Exception:
                            continue

                        df_basis_near = get_sales_distribution_basis(df, near["상권_코드_명"], category)
                        df_basis_near = df_basis_near.dropna(subset=["점포_당_매출_금액"])
                        if len(df_basis_near) < 4:
                            continue

                        stat_name, stat_d, stat_t = find_nearest_station(adj_lat, adj_lon)
                        chg_enc = change_encoder.transform([near["상권_변화_지표_명"]])[0]

                        input_vec["역까지_거리_m"] = stat_d
                        input_vec["가장_가까운_역_승하차_인원_수"] = stat_t
                        input_vec["상권_변화_지표_명"] = int(chg_enc)
                        input_vec["300m내_경쟁_업종_수"] = near["300m내_경쟁_업종_수"]

                        # ✅ 누락 피처 보완
                        recent_row = df[
                            (df["기준분기"] == 20244) &
                            (df["상권_코드"].astype(int) == int(near["상권_코드"])) &
                            (df["서비스_업종_코드_명"] == category)
                            ]
                        if not recent_row.empty:
                            for col in needed_cols:
                                if col not in input_vec:
                                    input_vec[col] = recent_row.iloc[0].get(col, np.nan)
                        else:
                            for col in needed_cols:
                                input_vec[col] = np.nan

                        # 🔹 계절성 피처 삽입
                        input_vec["연도"] = 2025
                        input_vec["분기"] = 2
                        input_vec["분기_sin"] = np.sin(2 * np.pi * 2 / 4)
                        input_vec["분기_cos"] = np.cos(2 * np.pi * 2 / 4)

                        input_df = pd.DataFrame([input_vec])
                        input_df = add_derived_features(input_df)
                        input_df = input_df[model.feature_names_in_]

                        sales = model.predict(input_df)[0]
                        sales = apply_temporal_corrections(sales, df_basis_near, selected_days, start_time, end_time)
                        percent = round(sales / base_sales * 100) if base_sales else None

                        results.append({
                            "lat": adj_lat, "lon": adj_lon, "dist": int(dist), "sales": int(sales),
                            "percent": percent, "상권명": near["상권_코드_명"],
                            "지하철역": stat_name, "지하철역거리": int(stat_d), "승하차": int(stat_t)
                        })

        # 🔹 추천 결과 출력
        final_recommendations = []
        ranked_output = []
        if results:
            results_sorted = sorted(results, key=lambda x: -x["sales"])
            rank = 1
            i = 0
            printed_ranks = 0

            while i < len(results_sorted) and printed_ranks < 3:
                current_group = [results_sorted[i]]
                current_sales = results_sorted[i]["sales"]
                i += 1
                while i < len(results_sorted) and abs(results_sorted[i]["sales"] - current_sales) <= 1_000_000:
                    current_group.append(results_sorted[i])
                    i += 1

                group_percent = (
                    f"(입력 위치 대비: {round(current_sales / base_sales * 100)}%)"
                    if base_sales else ""
                )
                title = (
                    f"🔸 공동 {rank}위 (약 {int(current_sales):,}원 {group_percent})"
                    if len(current_group) > 1 else
                    f"{rank}위 (약 {int(current_sales):,}원 {group_percent})"
                )
                print(f"\n{title}\n")

                group_result = {
                    "순위": rank,
                    "매출": int(current_sales),
                    "퍼센트": group_percent,
                    "공동": len(current_group) > 1,
                    "추천지": []
                }

                for loc in current_group[:3]:
                    print(f"🛍️ 상권: {loc['상권명']}")
                    print(f"📍 위치: 위도 {loc['lat']:.6f}, 경도 {loc['lon']:.6f}, 거리 {loc['dist']}m")
                    print(f"🚇 지하철: {loc['지하철역']} / 거리: {loc['지하철역거리']}m / 승하차: {loc['승하차']:,}명\n")

                    group_result["추천지"].append({
                        "상권명": loc["상권명"],
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                        "거리": loc["dist"],
                        "지하철역": loc["지하철역"],
                        "지하철역거리": loc["지하철역거리"],
                        "승하차": loc["승하차"],
                        "예상매출": loc["sales"]
                    })
                final_recommendations.append(loc)
                ranked_output.append(group_result)
                printed_ranks += 1
                rank += 1
        else:
            print("\n✅ 주변에 더 나은 위치는 없습니다.")

        print("입력위치", base_result)
        print("추천위치", len(final_recommendations), ranked_output)
        print("추천순위", len(ranked_output), ranked_output)

        return jsonify({
            "입력위치": base_result,
            "추천위치": final_recommendations,
            "추천순위": ranked_output
            })
    except Exception as e:
        return jsonify({'message': f"❌ 예측 중 오류 발생: {str(e)}"})

@app.route("/api/population_chart", methods=["GET"])
def population_chart():
    # 한글 폰트 설정
    font_path = 'nanumgothic-regular.ttf'  # 서버 환경에 맞게 변경하세요
    font_prop = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_prop)
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

    # 🔹 데이터 미리 로딩
    df = pd.read_csv("0510_광진구 상권, 지하철 통합 완성본.csv", encoding="utf-8")
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))

        # 📍 가장 가까운 상권 찾기
        df["거리"] = df.apply(lambda row: geodesic((lat, lon), (row["위도"], row["경도"])).meters, axis=1)
        nearest_row = df.loc[df["거리"].idxmin()]

        # 🔸 유동인구 데이터 추출
        gender_data = nearest_row[[6, 7]].tolist()
        age_data = nearest_row[8:14].tolist()
        time_data = pd.to_numeric(nearest_row[14:20], errors='coerce').fillna(0).tolist()
        day_data = pd.to_numeric(nearest_row[20:27], errors='coerce').fillna(0).tolist()

        # 🔸 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        labels = [
            (gender_data, ['남성', '여성'], '성별 유동인구 비율'),
            (age_data, ['10대', '20대', '30대', '40대', '50대', '60대'], '연령대별 유동인구 비율'),
            (time_data, ['0~6시', '6~11시', '11~14시', '14~17시', '17~21시', '21~24시'], '시간대별 유동인구 비율'),
            (day_data, ['월', '화', '수', '목', '금', '토', '일'], '요일별 유동인구 비율')
        ]

        for ax, (data, lbls, title) in zip(axes.flat, labels):
            if sum(data) > 0:
                ax.pie(data, labels=lbls, autopct='%1.1f%%', shadow=True, startangle=140)
            else:
                ax.text(0, 0, "데이터 없음", ha='center', va='center')
            ax.set_title(title)
            ax.axis('equal')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
