
def get_sales_distribution_basis(df, area_name: str, category: str):
    """
    특정 상권+업종 조합에 대해 요일/시간대별 매출 분포 계산에 사용할 데이터 반환
    """
    df_filtered = df[
        (df["상권_코드_명"] == area_name) &
        (df["서비스_업종_코드_명"] == category)
    ].copy()
    return df_filtered


def apply_temporal_corrections(predicted_sales, df_basis, selected_days, start_time, end_time):
    """
    선택된 요일/시간대에 따라 매출 예측값을 보정
    """
    defined_times = {
        "시간대_00_06": (0, 6), "시간대_06_11": (6, 11),
        "시간대_11_14": (11, 14), "시간대_14_17": (14, 17),
        "시간대_17_21": (17, 21), "시간대_21_24": (21, 24)
    }

    def safe_sum(df, col):
        return df[col].fillna(0).sum() if col in df.columns else 0

    # 요일 기반 보정
    total_weekly_sales = sum([safe_sum(df_basis, f"{day}요일_매출_금액") for day in ['월', '화', '수', '목', '금', '토', '일']])
    selected_sales = sum([safe_sum(df_basis, f"{day}요일_매출_금액") for day in selected_days])
    if total_weekly_sales > 0:
        ratio = selected_sales / total_weekly_sales
        predicted_sales *= ratio

    # 시간대 기반 보정
    total_time_sales = 0
    selected_time_sales = 0
    for col, (t_start, t_end) in defined_times.items():
        overlap = max(0, min(end_time, t_end) - max(start_time, t_start))
        duration = t_end - t_start
        amt = safe_sum(df_basis, f"{col}_매출_금액")
        total_time_sales += amt
        if overlap > 0:
            selected_time_sales += amt * (overlap / duration)

    if total_time_sales > 0:
        ratio = selected_time_sales / total_time_sales
        predicted_sales *= ratio

    return predicted_sales
