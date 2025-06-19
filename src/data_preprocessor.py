"""
广州市二手房数据预处理模块
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class HouseDataPreprocessor:
    def __init__(self, data_path='data/raw/houses_basic.csv'):
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.preprocessing_report = {
            'original_shape': None,
            'final_shape': None,
            'missing_values': {},
            'outliers_removed': {},
            'feature_engineering': []
        }
        
    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        self.preprocessing_report['original_shape'] = self.df.shape
        print(f"数据加载完成，原始数据形状: {self.df.shape}")
        return self.df
    
    def parse_house_info(self):
        """解析房源信息字段"""
        print("\n正在解析房源信息...")
        
        if 'house_info' not in self.df.columns:
            return
        
        # 从house_info中提取信息
        for idx, info in enumerate(self.df['house_info'].fillna('')):
            if pd.isna(info):
                continue
                
            # 提取户型（如：3室2厅）
            layout_match = re.search(r'(\d+)室(\d+)厅', str(info))
            if layout_match:
                self.df.loc[idx, 'bedroom_num'] = int(layout_match.group(1))
                self.df.loc[idx, 'livingroom_num'] = int(layout_match.group(2))
            
            # 提取面积
            area_match = re.search(r'(\d+\.?\d*)平米', str(info))
            if area_match:
                self.df.loc[idx, 'area'] = float(area_match.group(1))
            
            # 提取朝向
            orientations = ['南', '北', '东', '西', '南北', '东西', '东南', '西南', '东北', '西北']
            for orient in orientations:
                if orient in str(info):
                    self.df.loc[idx, 'orientation'] = orient
                    break
            
            # 提取装修情况
            decorations = ['毛坯', '简装', '精装', '豪装', '其他']
            for decor in decorations:
                if decor in str(info):
                    self.df.loc[idx, 'decoration'] = decor
                    break
            
            # 提取楼层信息
            floor_match = re.search(r'(低|中|高)楼层', str(info))
            if floor_match:
                self.df.loc[idx, 'floor_type'] = floor_match.group(1) + '楼层'
            
            # 提取总楼层
            total_floor_match = re.search(r'共(\d+)层', str(info))
            if total_floor_match:
                self.df.loc[idx, 'total_floor'] = int(total_floor_match.group(1))
            
            # 提取建筑年份
            year_match = re.search(r'(\d{4})年', str(info))
            if year_match:
                self.df.loc[idx, 'build_year'] = int(year_match.group(1))
        
        # 解析follow_info
        if 'follow_info' in self.df.columns:
            for idx, info in enumerate(self.df['follow_info'].fillna('')):
                # 提取关注人数
                follow_match = re.search(r'(\d+)人关注', str(info))
                if follow_match:
                    self.df.loc[idx, 'followers'] = int(follow_match.group(1))
                
                # 提取发布时间
                time_patterns = [
                    (r'(\d+)年', 365),
                    (r'(\d+)个月', 30),
                    (r'(\d+)天', 1)
                ]
                
                for pattern, days in time_patterns:
                    match = re.search(pattern, str(info))
                    if match:
                        self.df.loc[idx, 'days_on_market'] = int(match.group(1)) * days
                        break
        
        print("房源信息解析完成")
    
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n正在处理缺失值...")
        
        # 记录原始缺失值情况
        missing_before = self.df.isnull().sum()
        self.preprocessing_report['missing_values']['before'] = missing_before.to_dict()
        
        # 数值型变量填充策略
        numeric_fills = {
            'bedroom_num': self.df['bedroom_num'].mode()[0] if not self.df['bedroom_num'].mode().empty else 2,
            'livingroom_num': self.df['livingroom_num'].mode()[0] if not self.df['livingroom_num'].mode().empty else 1,
            'area': self.df['area'].median(),
            'total_floor': self.df['total_floor'].median(),
            'followers': 0,
            'days_on_market': self.df['days_on_market'].median() if 'days_on_market' in self.df.columns else 30
        }
        
        for col, fill_value in numeric_fills.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(fill_value)
        
        # 类别型变量填充策略
        categorical_fills = {
            'orientation': '南',
            'decoration': '其他',
            'floor_type': '中楼层',
            'district': '其他'
        }
        
        for col, fill_value in categorical_fills.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(fill_value)
        
        # 建筑年份特殊处理
        if 'build_year' in self.df.columns:
            # 根据小区平均建筑年份填充
            if 'community' in self.df.columns:
                community_year = self.df.groupby('community')['build_year'].transform('median')
                self.df['build_year'] = self.df['build_year'].fillna(community_year)
            # 剩余的用中位数填充
            self.df['build_year'] = self.df['build_year'].fillna(self.df['build_year'].median())
        
        # 记录处理后的缺失值情况
        missing_after = self.df.isnull().sum()
        self.preprocessing_report['missing_values']['after'] = missing_after.to_dict()
        
        print(f"缺失值处理完成")
        print(f"处理前缺失值总数: {missing_before.sum()}")
        print(f"处理后缺失值总数: {missing_after.sum()}")
    
    def detect_and_handle_outliers(self):
        """检测和处理异常值"""
        print("\n正在检测和处理异常值...")
        
        outliers_info = {}
        
        # 使用IQR方法检测数值型变量的异常值
        numeric_columns = ['total_price', 'unit_price', 'area', 'bedroom_num', 
                          'livingroom_num', 'total_floor', 'build_year']
        
        for col in numeric_columns:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 特殊处理某些字段
            if col == 'total_price':
                lower_bound = max(lower_bound, 50)  # 总价最低50万
                upper_bound = min(upper_bound, 3000)  # 总价最高3000万
            elif col == 'unit_price':
                lower_bound = max(lower_bound, 10000)  # 单价最低1万
                upper_bound = min(upper_bound, 150000)  # 单价最高15万
            elif col == 'area':
                lower_bound = max(lower_bound, 20)  # 面积最小20平
                upper_bound = min(upper_bound, 500)  # 面积最大500平
            elif col == 'bedroom_num':
                lower_bound = max(lower_bound, 0)
                upper_bound = min(upper_bound, 8)
            elif col == 'total_floor':
                lower_bound = max(lower_bound, 1)
                upper_bound = min(upper_bound, 60)
            elif col == 'build_year':
                lower_bound = max(lower_bound, 1980)
                upper_bound = min(upper_bound, 2024)
            
            # 记录异常值
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # 处理异常值（截断处理）
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)
        
        self.preprocessing_report['outliers_removed'] = outliers_info
        
        # 打印异常值处理情况
        for col, info in outliers_info.items():
            if info['count'] > 0:
                print(f"{col}: 发现 {info['count']} 个异常值 ({info['percentage']:.2f}%), "
                      f"范围: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    
    def feature_engineering(self):
        """特征工程"""
        print("\n正在进行特征工程...")
        
        features_created = []
        
        # 1. 房龄
        if 'build_year' in self.df.columns:
            current_year = datetime.now().year
            self.df['house_age'] = current_year - self.df['build_year']
            features_created.append('house_age')
        
        # 2. 房间总数
        if 'bedroom_num' in self.df.columns and 'livingroom_num' in self.df.columns:
            self.df['total_rooms'] = self.df['bedroom_num'] + self.df['livingroom_num']
            features_created.append('total_rooms')
        
        # 3. 平均房间面积
        if 'area' in self.df.columns and 'total_rooms' in self.df.columns:
            self.df['avg_room_area'] = self.df['area'] / (self.df['total_rooms'] + 1)  # +1避免除零
            features_created.append('avg_room_area')
        
        # 4. 楼层比例
        if 'total_floor' in self.df.columns and 'floor_type' in self.df.columns:
            floor_ratio_map = {'低楼层': 0.25, '中楼层': 0.5, '高楼层': 0.75}
            self.df['floor_ratio'] = self.df['floor_type'].map(floor_ratio_map)
            features_created.append('floor_ratio')
        
        # 5. 价格相关特征
        if 'total_price' in self.df.columns and 'bedroom_num' in self.df.columns:
            self.df['price_per_bedroom'] = self.df['total_price'] / (self.df['bedroom_num'] + 1)
            features_created.append('price_per_bedroom')
        
        # 6. 市场热度指标
        if 'followers' in self.df.columns and 'days_on_market' in self.df.columns:
            self.df['market_heat'] = self.df['followers'] / (self.df['days_on_market'] + 1)
            features_created.append('market_heat')
        
        # 7. 户型是否合理（卧室数和客厅数的比例）
        if 'bedroom_num' in self.df.columns and 'livingroom_num' in self.df.columns:
            self.df['layout_ratio'] = self.df['bedroom_num'] / (self.df['livingroom_num'] + 1)
            features_created.append('layout_ratio')
        
        # 8. 区域分类（将11个区分为核心区、次核心区、外围区）
        if 'district' in self.df.columns:
            core_districts = ['天河区', '越秀区', '海珠区']
            sub_core_districts = ['荔湾区', '白云区', '黄埔区', '番禺区']
            
            def classify_district(district):
                if district in core_districts:
                    return '核心区'
                elif district in sub_core_districts:
                    return '次核心区'
                else:
                    return '外围区'
            
            self.df['district_level'] = self.df['district'].apply(classify_district)
            features_created.append('district_level')
        
        self.preprocessing_report['feature_engineering'] = features_created
        print(f"特征工程完成，创建了 {len(features_created)} 个新特征")
    def encode_categorical_variables(self):
        """编码类别变量"""
        print("\n正在编码类别变量...")
        
        # 确定类别变量
        categorical_columns = ['district', 'orientation', 'decoration', 
                              'floor_type', 'district_level']
        
        # 保存编码映射
        self.encoding_mappings = {}
        
        for col in categorical_columns:
            if col not in self.df.columns:
                continue
            
            # 对于有序变量使用有序编码
            if col == 'decoration':
                decoration_map = {'毛坯': 1, '简装': 2, '精装': 3, '豪装': 4, '其他': 2}
                self.df[f'{col}_encoded'] = self.df[col].map(decoration_map)
                self.encoding_mappings[col] = decoration_map
            
            elif col == 'floor_type':
                floor_map = {'低楼层': 1, '中楼层': 2, '高楼层': 3}
                self.df[f'{col}_encoded'] = self.df[col].map(floor_map)
                self.encoding_mappings[col] = floor_map
            
            elif col == 'district_level':
                district_level_map = {'外围区': 1, '次核心区': 2, '核心区': 3}
                self.df[f'{col}_encoded'] = self.df[col].map(district_level_map)
                self.encoding_mappings[col] = district_level_map
            
            # 对于无序类别变量使用独热编码
            else:
                # 使用pandas的get_dummies
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.encoding_mappings[col] = list(dummies.columns)
        
        print("类别变量编码完成")
    
    def create_analysis_report(self):
        """创建数据分析报告"""
        print("\n正在生成数据分析报告...")
        
        # 创建报告文件夹
        import os
        os.makedirs('data/processed/reports', exist_ok=True)
        
        # 1. 数据基本情况
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('数据集基本情况分析', fontsize=16)
        
        # 1.1 各区房源数量分布
        if 'district' in self.df.columns:
            district_counts = self.df['district'].value_counts()
            axes[0, 0].bar(district_counts.index, district_counts.values)
            axes[0, 0].set_title('各区房源数量分布')
            axes[0, 0].set_xlabel('区域')
            axes[0, 0].set_ylabel('房源数量')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 1.2 价格分布
        if 'total_price' in self.df.columns:
            axes[0, 1].hist(self.df['total_price'], bins=50, edgecolor='black')
            axes[0, 1].set_title('房源总价分布')
            axes[0, 1].set_xlabel('总价（万元）')
            axes[0, 1].set_ylabel('频数')
        
        # 1.3 面积分布
        if 'area' in self.df.columns:
            axes[1, 0].hist(self.df['area'], bins=50, edgecolor='black')
            axes[1, 0].set_title('房源面积分布')
            axes[1, 0].set_xlabel('面积（平方米）')
            axes[1, 0].set_ylabel('频数')
        
        # 1.4 单价分布
        if 'unit_price' in self.df.columns:
            axes[1, 1].hist(self.df['unit_price'], bins=50, edgecolor='black')
            axes[1, 1].set_title('房源单价分布')
            axes[1, 1].set_xlabel('单价（元/平方米）')
            axes[1, 1].set_ylabel('频数')
        
        plt.tight_layout()
        plt.savefig('data/processed/reports/basic_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 缺失值处理报告
        if self.preprocessing_report['missing_values']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 处理前
            before_missing = self.preprocessing_report['missing_values'].get('before', {})
            if before_missing:
                missing_df_before = pd.DataFrame.from_dict(before_missing, orient='index', columns=['缺失数量'])
                missing_df_before = missing_df_before[missing_df_before['缺失数量'] > 0]
                if not missing_df_before.empty:
                    missing_df_before.plot(kind='bar', ax=ax1, legend=False)
                    ax1.set_title('处理前各字段缺失值数量')
                    ax1.set_xlabel('字段名')
                    ax1.set_ylabel('缺失数量')
                    ax1.tick_params(axis='x', rotation=45)
            
            # 处理后
            after_missing = self.preprocessing_report['missing_values'].get('after', {})
            if after_missing:
                missing_df_after = pd.DataFrame.from_dict(after_missing, orient='index', columns=['缺失数量'])
                missing_df_after = missing_df_after[missing_df_after['缺失数量'] > 0]
                if not missing_df_after.empty:
                    missing_df_after.plot(kind='bar', ax=ax2, legend=False)
                    ax2.set_title('处理后各字段缺失值数量')
                    ax2.set_xlabel('字段名')
                    ax2.set_ylabel('缺失数量')
                    ax2.tick_params(axis='x', rotation=45)
                else:
                    ax2.text(0.5, 0.5, '无缺失值', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('处理后各字段缺失值数量')
            
            plt.tight_layout()
            plt.savefig('data/processed/reports/missing_values_report.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 异常值处理报告
        if self.preprocessing_report['outliers_removed']:
            outliers_df = pd.DataFrame.from_dict(self.preprocessing_report['outliers_removed'], orient='index')
            if 'count' in outliers_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                outliers_df['count'].plot(kind='bar', ax=ax)
                ax.set_title('各字段异常值数量')
                ax.set_xlabel('字段名')
                ax.set_ylabel('异常值数量')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加百分比标注
                for i, (idx, row) in enumerate(outliers_df.iterrows()):
                    if 'percentage' in row:
                        ax.text(i, row['count'] + 0.5, f"{row['percentage']:.1f}%", 
                               ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('data/processed/reports/outliers_report.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print("数据分析报告生成完成")
    
    def save_processed_data(self):
        """保存处理后的数据"""
        print("\n正在保存处理后的数据...")
        
        # 保存完整数据
        self.df.to_csv('data/processed/houses_processed_full.csv', 
                      index=False, encoding='utf-8-sig')
        
        # 保存用于建模的数据（只包含数值型和编码后的变量）
        model_columns = []
        
        # 数值型变量
        numeric_columns = ['total_price', 'unit_price', 'area', 'bedroom_num', 
                          'livingroom_num', 'total_floor', 'build_year', 
                          'followers', 'days_on_market', 'house_age', 
                          'total_rooms', 'avg_room_area', 'floor_ratio',
                          'price_per_bedroom', 'market_heat', 'layout_ratio']
        
        for col in numeric_columns:
            if col in self.df.columns:
                model_columns.append(col)
        
        # 编码后的类别变量
        encoded_columns = [col for col in self.df.columns if '_encoded' in col or col.startswith(('district_', 'orientation_'))]
        model_columns.extend(encoded_columns)
        
        # 保存建模数据
        model_df = self.df[model_columns].copy()
        model_df.to_csv('data/processed/houses_for_modeling.csv', 
                       index=False, encoding='utf-8-sig')
        
        # 保存数据字典
        data_dict = {
            'numeric_features': [col for col in model_columns if col in numeric_columns],
            'categorical_features': encoded_columns,
            'target_variable': 'total_price',
            'encoding_mappings': self.encoding_mappings,
            'shape': model_df.shape,
            'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open('data/processed/data_dictionary.json', 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        self.preprocessing_report['final_shape'] = model_df.shape
        
        # 保存预处理报告
        with open('data/processed/preprocessing_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.preprocessing_report, f, ensure_ascii=False, indent=2)
        
        print(f"处理后数据已保存，最终数据形状: {model_df.shape}")
        print(f"原始数据: data/processed/houses_processed_full.csv")
        print(f"建模数据: data/processed/houses_for_modeling.csv")
        print(f"数据字典: data/processed/data_dictionary.json")
    
    def generate_summary_statistics(self):
        """生成汇总统计信息"""
        print("\n正在生成汇总统计...")
        
        summary_stats = {}
        
        # 基本统计
        summary_stats['total_records'] = len(self.df)
        summary_stats['total_features'] = len(self.df.columns)
        
        # 价格统计
        if 'total_price' in self.df.columns:
            summary_stats['price_stats'] = {
                'mean': self.df['total_price'].mean(),
                'median': self.df['total_price'].median(),
                'std': self.df['total_price'].std(),
                'min': self.df['total_price'].min(),
                'max': self.df['total_price'].max(),
                'q25': self.df['total_price'].quantile(0.25),
                'q75': self.df['total_price'].quantile(0.75)
            }
        
        # 面积统计
        if 'area' in self.df.columns:
            summary_stats['area_stats'] = {
                'mean': self.df['area'].mean(),
                'median': self.df['area'].median(),
                'std': self.df['area'].std(),
                'min': self.df['area'].min(),
                'max': self.df['area'].max()
            }
        
        # 区域分布
        if 'district' in self.df.columns:
            summary_stats['district_distribution'] = self.df['district'].value_counts().to_dict()
        
        # 户型分布
        if 'bedroom_num' in self.df.columns:
            summary_stats['bedroom_distribution'] = self.df['bedroom_num'].value_counts().to_dict()
        
        # 保存统计信息
        import json
        with open('data/processed/summary_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)
        
        print("汇总统计信息已生成")
        
        return summary_stats
    
    def run_full_preprocessing(self):
        """执行完整的预处理流程"""
        print("="*50)
        print("开始数据预处理流程")
        print("="*50)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 解析房源信息
        self.parse_house_info()
        
        # 3. 处理缺失值
        self.handle_missing_values()
        
        # 4. 检测和处理异常值
        self.detect_and_handle_outliers()
        
        # 5. 特征工程
        self.feature_engineering()
        
        # 6. 编码类别变量
        self.encode_categorical_variables()
        
        # 7. 生成分析报告
        self.create_analysis_report()
        
        # 8. 生成汇总统计
        self.generate_summary_statistics()
        
        # 9. 保存处理后的数据
        self.save_processed_data()
        
        print("\n="*50)
        print("数据预处理完成！")
        print("="*50)
        
        return self.df

def main():
    """主函数"""
    # 创建必要的文件夹
    import os
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/processed/reports', exist_ok=True)
    
    # 初始化预处理器
    preprocessor = HouseDataPreprocessor()
    
    # 运行完整预处理流程
    processed_df = preprocessor.run_full_preprocessing()
    
    # 打印最终结果摘要
    print("\n### 预处理结果摘要 ###")
    print(f"原始数据形状: {preprocessor.preprocessing_report['original_shape']}")
    print(f"最终数据形状: {preprocessor.preprocessing_report['final_shape']}")
    print(f"创建的新特征: {', '.join(preprocessor.preprocessing_report['feature_engineering'])}")
    
    return processed_df

if __name__ == "__main__":
    main()