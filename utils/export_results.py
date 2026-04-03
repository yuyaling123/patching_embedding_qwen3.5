import numpy as np
import pandas as pd
import os
import glob

# ================= 配置区域 =================
# 结果文件夹根目录
RESULT_ROOT = './results/'

# 这里必须和 Run_EV_Multi.bat 里的 --model_id 保持一致
MODEL_ID_KEYWORD = 'EV_Test_Run' 

# 原始数据路径 (用于读取列名以匹配站点)
DATA_PATH = './dataset/EV_Data/EV_Load_Cleaned.csv'
# ===========================================

def find_latest_result_dir():
    """寻找最新的结果文件夹"""
    # 搜索路径类似于 ./results/EV_All_Stations_Run*
    search_path = os.path.join(RESULT_ROOT, f"*{MODEL_ID_KEYWORD}*")
    dirs = glob.glob(search_path)
    
    if not dirs:
        print(f"【错误】未找到包含 '{MODEL_ID_KEYWORD}' 的结果文件夹。")
        print("请检查 results 目录下是否有文件夹，或是否已运行 Run_EV_Multi.bat。")
        return None
        
    # 按修改时间排序，找最新的一个
    latest_dir = max(dirs, key=os.path.getmtime)
    print(f"【系统】锁定最新结果文件夹: {latest_dir}")
    return latest_dir

def export_station_data():
    print("="*60)
    print("正在启动站点数据提取工具...")
    print("="*60)

    # 1. 获取列名
    if not os.path.exists(DATA_PATH):
        print(f"【错误】找不到原始数据文件: {DATA_PATH}")
        return
    
    df_raw = pd.read_csv(DATA_PATH)
    # Time-LLM 的 DataLoader 通常会丢弃第一列(date)，所以模型特征从第二列开始
    # 这里的 feature_columns 应该对应模型输出的 95 个维度
    feature_columns = list(df_raw.columns[1:])
    
    print(f"【数据】原始CSV特征共 {len(feature_columns)} 列 (不含Date)")
    
    # 2. 加载预测结果
    result_dir = find_latest_result_dir()
    if not result_dir:
        return

    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')
    
    # 优先使用真实刻度结果
    pred_orig_path = os.path.join(result_dir, 'pred_original.npy')
    true_orig_path = os.path.join(result_dir, 'true_original.npy')
    
    if os.path.exists(pred_orig_path) and os.path.exists(true_orig_path):
        print(f"【系统】检测到真实刻度结果 (Original Scale)，将使用此文件进行导出。")
        pred_path = pred_orig_path
        true_path = true_orig_path
    else:
        print(f"【提示】未找到真实刻度结果，将使用标准化结果 (Standardized Scale)。")
    
    if not os.path.exists(pred_path):
        print("【错误】找不到 pred.npy，请确保训练已正常完成。")
        return
        
    # 加载 npy 文件
    # Shape 通常是: (Samples, Pred_Len, Features)
    # 例如: (256, 96, 95)
    preds = np.load(pred_path) 
    trues = np.load(true_path)
    
    print(f"【结果】加载预测矩阵成功，形状: {preds.shape}")
    
    # 3. 校验维度
    # preds.shape[2] 是模型输出的特征数
    if preds.shape[2] != len(feature_columns):
        print(f"【警告】模型输出特征数 ({preds.shape[2]}) 与 CSV列数 ({len(feature_columns)}) 不匹配！")
        print("可能原因：")
        print("1. Run_EV_Multi.bat 中的 enc_in / c_out 设置错误。")
        print("2. 数据加载器截取了部分列。")
        print("程序将尝试继续，但列名可能对不上。")
    else:
        print("【校验】维度校验通过。")

    # 4. 筛选站点列
    # 找出所有名字里包含 'Station' 的列的索引
    station_indices = [i for i, col in enumerate(feature_columns) if 'Station' in col]
    station_names = [feature_columns[i] for i in station_indices]
    
    if not station_names:
        print("【警告】未在列名中检测到 'Station' 关键字。")
        return

    print(f"【筛选】检测到 {len(station_names)} 个站点，准备导出数据...")

    # 5. 导出逻辑
    # 提取测试集中【最后一个样本】的完整预测序列
    # 这是展示“未来预测能力”最常用的方式
    last_sample_idx = -1 
    
    export_dict = {}
    
    for i, col_idx in enumerate(station_indices):
        name = station_names[i]
        
        # 边界检查
        if col_idx >= preds.shape[2]:
            continue

        # 提取该站点的预测值和真实值 (96个时间步)
        p_data = preds[last_sample_idx, :, col_idx]
        t_data = trues[last_sample_idx, :, col_idx]
        
        export_dict[f"{name}_Pred"] = p_data
        export_dict[f"{name}_True"] = t_data

    # 6. 保存为 CSV
    output_filename = 'Final_Station_Forecasts.csv'
    
    try:
        df_export = pd.DataFrame(export_dict)
        df_export.to_csv(output_filename, index_label='Time_Step')
        
        print("="*60)
        print(f"【完成】导出成功！")
        print(f"文件位置: {os.path.abspath(output_filename)}")
        print(f"包含内容: {len(station_names)} 个站点的预测对比数据 (最后96步)")
        print("="*60)
    except Exception as e:
        print(f"【错误】保存CSV失败: {e}")

if __name__ == "__main__":
    export_station_data()