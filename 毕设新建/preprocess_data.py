import pandas as pd
import os

# 配置区
INPUT_FILE = './dataset/EV_Data/EV_Load.csv'
OUTPUT_FILE = './dataset/EV_Data/EV_Load_Cleaned_all.csv'

# 1. 站点筛选配置
ZERO_THRESHOLD = 0.3  # 如果某列 0 的比例超过 30%，则丢弃该列
MAX_STATIONS = 100    # 最大保留站点数 (建议 50-100 以获得极速体验)

# 2. 时间筛选配置 (格式: 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD')
# 根据你的要求：初始时间是 2023/4/1 0:00:00
# 这里默认截取到 7月31日 (约4个月数据)，如果你想只到 6月30日，请修改 END_DATE
START_DATE = '2023-05-15 00:00:00'
END_DATE = '2023-9-30 23:00:00'

def clean_data():
    print(f"正在读取数据: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    # 读取 CSV
    df = pd.read_csv(INPUT_FILE)
    
    # 确保 date 列是 datetime 类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        print("错误：找不到 'date' 列！")
        return

    original_rows = len(df)
    original_cols = len(df.columns)
    
    print(f"原始数据范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"原始大小: {df.shape}")

    # --- 1. 时间裁剪 ---
    print(f"\n正在进行时间裁剪 ({START_DATE} ~ {END_DATE})...")
    mask = (df['date'] >= pd.to_datetime(START_DATE)) & (df['date'] <= pd.to_datetime(END_DATE))
    df_time_filtered = df.loc[mask]
    
    if len(df_time_filtered) == 0:
        print("警告：时间筛选后数据为空！请检查数据中的年份是否真的是 2023。")
        print(f"数据中实际的年份范围是: {df['date'].dt.year.unique()}")
        return # 停止执行，因为数据为空会导致后续报错
    else:
        print(f"时间筛选后行数: {len(df_time_filtered)} (原: {original_rows})")

    # --- 2. 站点筛选 ---
    # 假设前 8 列是 date 和天气/价格特征
    fixed_columns = df.columns[:8] 
    station_columns = df.columns[8:]
    
    print("\n正在分析站点数据质量...")
    
    valid_stations = []
    dropped_zeros = 0
    # 注意：要在时间筛选后的数据上计算零值比例
    for col in station_columns:
        zero_ratio = (df_time_filtered[col] == 0).mean()
        if zero_ratio <= ZERO_THRESHOLD:
            valid_stations.append(col)
        else:
            dropped_zeros += 1
            
    print(f"质量筛选: 移除了 {dropped_zeros} 个含零过多的站点")

    # --- 3. 站点数量限制 ---
    if MAX_STATIONS is not None and len(valid_stations) > MAX_STATIONS:
        print(f"站点数量限制: 截取前 {MAX_STATIONS} 个站点...")
        valid_stations = valid_stations[:MAX_STATIONS]
    
    # 合并列
    final_columns = list(fixed_columns) + valid_stations
    df_final = df_time_filtered[final_columns]
    
    print("-" * 30)
    print(f"处理完成！")
    print(f"最终数据范围: {df_final['date'].min()} 到 {df_final['date'].max()}")
    print(f"最终大小: {df_final.shape}")
    print(f"最终特征数 (Input Vars): {len(final_columns) - 1} (扣除 date 列)")
    print("-" * 30)
    
    # 保存文件
    print(f"正在保存清洗后的数据到: {OUTPUT_FILE} ...")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("保存成功！")
    
    # 输出下一步建议
    new_input_vars = len(final_columns) - 1
    print("\n" + "="*40)
    print("【下一步操作指南】")
    print(f"1. 请修改启动脚本 (scripts/Run_EV_Multi.bat):")
    print(f"   set DATA_PATH=EV_Load_Cleaned.csv")
    print(f"   set INPUT_VARS={new_input_vars}")
    print("="*40)

if __name__ == "__main__":
    clean_data()