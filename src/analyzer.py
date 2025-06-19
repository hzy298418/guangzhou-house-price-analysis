# src/analyzer.py

"""
广州市二手房数据分析与建模模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestRegressor
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置Matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class HouseAnalyzer:
    def __init__(self, full_data_path, model_data_path, report_path):
        """
        初始化分析器
        :param full_data_path: 完整预处理数据的路径
        :param model_data_path: 用于建模的数据的路径
        :param report_path: 保存图表的报告路径
        """
        self.full_data_path = full_data_path
        self.model_data_path = model_data_path
        self.report_path = report_path
        self.df_full = None
        self.df_model = None
        print(f"报告将保存至: {self.report_path}")

    def load_data(self):
        """加载数据"""
        print("--- 1. 加载数据 ---")
        try:
            self.df_full = pd.read_csv(self.full_data_path)
            self.df_model = pd.read_csv(self.model_data_path)
            print("数据加载成功！")
            return True
        except FileNotFoundError:
            print(f"错误：找不到数据文件。请先运行 run_preprocessing.py。")
            return False

    def plot_core_variable_distribution(self):
        """绘制核心变量分布图"""
        print("--- 2.1 核心变量分布分析 ---")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('核心变量分布直方图', fontsize=16, y=1.02)
        
        sns.histplot(self.df_full['total_price'], kde=True, ax=axes[0], bins=50)
        axes[0].set_title('总价 (total_price) 分布')
        axes[0].set_xlabel('总价 (万元)')
        
        sns.histplot(self.df_full['area'], kde=True, ax=axes[1], bins=50)
        axes[1].set_title('面积 (area) 分布')
        axes[1].set_xlabel('面积 (平方米)')
        
        sns.histplot(self.df_full['house_age'], kde=True, ax=axes[2], bins=30)
        axes[2].set_title('房龄 (house_age) 分布')
        axes[2].set_xlabel('房龄 (年)')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_path}/midterm_core_variable_distribution.png', dpi=300)
        plt.close()
        print("核心变量分布图已保存。")

    def plot_price_by_district(self):
        """按行政区分析房价"""
        print("--- 2.2 地理位置对房价的影响 ---")
        plt.figure(figsize=(14, 7))
        district_order = self.df_full.groupby('district')['total_price'].median().sort_values(ascending=False).index
        sns.boxplot(data=self.df_full, x='district', y='total_price', order=district_order)
        plt.title('不同行政区的二手房总价分布 (按中位数排序)', fontsize=16)
        plt.xlabel('行政区')
        plt.ylabel('总价 (万元)')
        plt.xticks(rotation=45)
        plt.savefig(f'{self.report_path}/midterm_price_by_district.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("行政区房价分布图已保存。")

    def plot_attributes_vs_price(self):
        """绘制房屋属性与价格的关系图"""
        print("--- 2.3 房屋属性对房价的影响 ---")
        # 面积 vs. 总价
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df_full, x='area', y='total_price', alpha=0.3, s=15)
        sns.regplot(data=self.df_full, x='area', y='total_price', scatter=False, color='red', line_kws={'linewidth': 2})
        plt.title('建筑面积与总价的关系', fontsize=16)
        plt.xlabel('建筑面积 (平方米)')
        plt.ylabel('总价 (万元)')
        plt.savefig(f'{self.report_path}/midterm_area_vs_price.png', dpi=300)
        plt.close()

        # 房龄 vs. 总价
        sns.jointplot(data=self.df_full, x='house_age', y='total_price', kind='hex', height=7, gridsize=40)
        plt.suptitle('房龄与总价的关系 (六边形分箱图)', y=1.02, fontsize=16)
        plt.xlabel('房龄 (年)')
        plt.ylabel('总价 (万元)')
        plt.savefig(f'{self.report_path}/midterm_age_vs_price.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 其他属性分析
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        decoration_order = ['毛坯', '简装', '精装', '豪装']
        sns.boxplot(data=self.df_full, x='decoration', y='total_price', order=decoration_order, ax=axes[0])
        axes[0].set_title('不同装修情况的房价分布')
        sns.boxplot(data=self.df_full, x='bedroom_num', y='total_price', ax=axes[1])
        axes[1].set_title('不同卧室数量的房价分布')
        plt.savefig(f'{self.report_path}/midterm_other_attributes.png', dpi=300)
        plt.close()
        print("房屋属性关系图已保存。")

    def plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        print("--- 2.4 多变量相关性分析 ---")
        corr_features = ['total_price', 'unit_price', 'area', 'house_age', 'total_rooms', 
                         'bedroom_num', 'decoration_encoded', 'floor_type_encoded', 'district_level_encoded']
        corr_matrix = self.df_full[corr_features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('核心数值型特征相关性热力图', fontsize=16)
        plt.savefig(f'{self.report_path}/midterm_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("相关性热力图已保存。")
        
    def create_geospatial_heatmap(self):
        """创建地理空间热力图"""
        print("--- 3. 地理空间可视化 ---")
        if 'latitude' not in self.df_full.columns or 'longitude' not in self.df_full.columns:
            print("数据中缺少经纬度信息，跳过地理热力图生成。")
            return

        gz_map = folium.Map(location=[23.1291, 113.2644], zoom_start=11)
        heat_data = self.df_full[['latitude', 'longitude', 'unit_price']].dropna().values.tolist()
        HeatMap(heat_data, radius=12, blur=15).add_to(gz_map)
        map_path = f'{self.report_path}/midterm_price_heatmap.html'
        gz_map.save(map_path)
        print(f"价格热力图已生成并保存至: {map_path}")

    def analyze_feature_importance(self):
        """训练模型并分析特征重要性"""
        print("--- 4. 初步建模与特征重要性分析 ---")
        X = self.df_model.drop('total_price', axis=1)
        # 确保 unit_price 也不在特征中，因为它与总价直接相关
        if 'unit_price' in X.columns:
            X = X.drop('unit_price', axis=1)
            
        y = self.df_model['total_price']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
        plt.title('影响二手房总价的Top 15特征重要性排序', fontsize=16)
        plt.xlabel('特征重要性')
        plt.ylabel('特征')
        plt.savefig(f'{self.report_path}/midterm_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("特征重要性分析图已保存。")

    def run_analysis_pipeline(self):
        """执行完整的分析流程"""
        if self.load_data():
            self.plot_core_variable_distribution()
            self.plot_price_by_district()
            self.plot_attributes_vs_price()
            self.plot_correlation_heatmap()
            self.create_geospatial_heatmap()
            self.analyze_feature_importance()
            print("\n所有分析和可视化任务已完成！")
