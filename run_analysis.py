# run_analysis.py

"""
执行数据分析、建模与可视化的主脚本
"""
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.analyzer import HouseAnalyzer

def main():
    print("="*60)
    print("广州市二手房数据分析与建模项目 - 分析流程")
    print("="*60)

    # 定义文件路径
    full_data_path = 'data/processed/houses_processed_full.csv'
    model_data_path = 'data/processed/houses_for_modeling.csv'
    report_path = 'data/processed/reports'

    # 检查报告目录是否存在
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    # 初始化并运行分析器
    analyzer = HouseAnalyzer(full_data_path, model_data_path, report_path)
    analyzer.run_analysis_pipeline()

if __name__ == "__main__":
    main()
