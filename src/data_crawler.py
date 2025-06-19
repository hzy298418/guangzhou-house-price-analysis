"""
广州市二手房数据爬取模块
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from fake_useragent import UserAgent
import json
import os
from tqdm import tqdm
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_crawler.log'),
        logging.StreamHandler()
    ]
)

class GuangzhouHouseCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.base_url = "https://gz.lianjia.com/ershoufang/"
        self.districts = [
            'tianhe', 'yuexiu', 'haizhu', 'liwan', 'baiyun', 
            'huangpu', 'panyu', 'huadu', 'nansha', 'zengcheng', 'conghua'
        ]
        self.district_names = {
            'tianhe': '天河区', 'yuexiu': '越秀区', 'haizhu': '海珠区',
            'liwan': '荔湾区', 'baiyun': '白云区', 'huangpu': '黄埔区',
            'panyu': '番禺区', 'huadu': '花都区', 'nansha': '南沙区',
            'zengcheng': '增城区', 'conghua': '从化区'
        }
        
    def get_headers(self):
        """获取随机请求头"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def parse_house_detail(self, house_url):
        """解析房源详情页"""
        try:
            time.sleep(random.uniform(1, 3))
            response = self.session.get(house_url, headers=self.get_headers(), timeout=10)
            if response.status_code != 200:
                logging.warning(f"访问失败: {house_url}")
                return None
                
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 提取详细信息
            detail = {}
            
            # 基本信息
            detail['url'] = house_url
            detail['house_id'] = house_url.split('/')[-1].replace('.html', '')
            
            # 价格信息
            price_elem = soup.find('span', class_='total')
            if price_elem:
                detail['total_price'] = float(price_elem.text)
            
            unit_price_elem = soup.find('span', class_='unitPriceValue')
            if unit_price_elem:
                detail['unit_price'] = float(unit_price_elem.text.replace('元/平米', ''))
            
            # 房屋基本信息
            base_info = soup.find('div', class_='base')
            if base_info:
                lis = base_info.find_all('li')
                for li in lis:
                    label = li.find('span').text if li.find('span') else ''
                    value = li.text.replace(label, '').strip()
                    
                    if '房屋户型' in label:
                        detail['layout'] = value
                    elif '建筑面积' in label:
                        detail['area'] = float(value.replace('㎡', ''))
                    elif '房屋朝向' in label:
                        detail['orientation'] = value
                    elif '装修情况' in label:
                        detail['decoration'] = value
                    elif '所在楼层' in label:
                        detail['floor'] = value
                    elif '建筑类型' in label:
                        detail['building_type'] = value
                    elif '梯户比例' in label:
                        detail['elevator_ratio'] = value
                    elif '配备电梯' in label:
                        detail['has_elevator'] = value
            
            # 交易信息
            transaction_info = soup.find('div', class_='transaction')
            if transaction_info:
                lis = transaction_info.find_all('li')
                for li in lis:
                    label = li.find('span').text if li.find('span') else ''
                    value = li.text.replace(label, '').strip()
                    
                    if '挂牌时间' in label:
                        detail['listing_time'] = value
                    elif '交易权属' in label:
                        detail['ownership'] = value
                    elif '房屋用途' in label:
                        detail['usage'] = value
                    elif '房屋年限' in label:
                        detail['property_years'] = value
            
            # 小区信息
            community_elem = soup.find('div', class_='communityName')
            if community_elem:
                detail['community'] = community_elem.find('a').text
            
            # 位置信息
            area_elem = soup.find('div', class_='areaName')
            if area_elem:
                area_info = area_elem.find('span', class_='info')
                if area_info:
                    links = area_info.find_all('a')
                    if len(links) >= 2:
                        detail['district'] = links[0].text
                        detail['area'] = links[1].text
            
            # 获取经纬度（如果有地图信息）
            script_tags = soup.find_all('script')
            for script in script_tags:
                if 'resblockPosition' in script.text:
                    try:
                        import re
                        lng_match = re.search(r"resblockPosition:'([\d.]+),([\d.]+)'", script.text)
                        if lng_match:
                            detail['longitude'] = float(lng_match.group(1))
                            detail['latitude'] = float(lng_match.group(2))
                    except:
                        pass
            
            return detail
            
        except Exception as e:
            logging.error(f"解析房源详情出错: {house_url}, 错误: {str(e)}")
            return None
    
    def get_district_houses(self, district, max_pages=20):
        """获取某个区的房源列表"""
        houses = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}{district}/pg{page}/"
            logging.info(f"正在爬取: {self.district_names.get(district, district)} 第{page}页")
            
            try:
                time.sleep(random.uniform(2, 4))
                response = self.session.get(url, headers=self.get_headers(), timeout=10)
                
                if response.status_code != 200:
                    logging.warning(f"访问失败: {url}")
                    continue
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                # 查找房源列表
                house_list = soup.find('ul', class_='sellListContent')
                if not house_list:
                    logging.warning(f"未找到房源列表: {url}")
                    break
                
                items = house_list.find_all('li', class_='clear')
                if not items:
                    logging.info(f"{district} 第{page}页无更多房源")
                    break
                
                for item in items:
                    try:
                        # 获取房源链接
                        link_elem = item.find('div', class_='title').find('a')
                        if link_elem:
                            house_url = link_elem.get('href')
                            
                            # 获取基本信息
                            house_info = {
                                'url': house_url,
                                'title': link_elem.text.strip(),
                                'district_code': district,
                                'district': self.district_names.get(district, district)
                            }
                            
                            # 获取价格
                            price_elem = item.find('div', class_='totalPrice')
                            if price_elem:
                                house_info['total_price'] = float(price_elem.find('span').text)
                            
                            unit_price_elem = item.find('div', class_='unitPrice')
                            if unit_price_elem:
                                unit_price_text = unit_price_elem.get('data-price', '')
                                if unit_price_text:
                                    house_info['unit_price'] = float(unit_price_text)
                            
                            # 获取房源基本信息
                            house_info_elem = item.find('div', class_='houseInfo')
                            if house_info_elem:
                                info_text = house_info_elem.text
                                house_info['house_info'] = info_text
                            
                            # 获取关注信息
                            follow_elem = item.find('div', class_='followInfo')
                            if follow_elem:
                                house_info['follow_info'] = follow_elem.text
                            
                            houses.append(house_info)
                            
                    except Exception as e:
                        logging.error(f"解析房源信息出错: {str(e)}")
                        continue
                
            except Exception as e:
                logging.error(f"爬取页面出错: {url}, 错误: {str(e)}")
                continue
        
        return houses
    
    def crawl_all_districts(self, max_pages_per_district=10, save_interval=100):
        """爬取所有区的房源数据"""
        all_houses = []
        
        for district in self.districts:
            logging.info(f"开始爬取 {self.district_names.get(district, district)} 的房源")
            district_houses = self.get_district_houses(district, max_pages_per_district)
            all_houses.extend(district_houses)
            
            # 定期保存
            if len(all_houses) % save_interval == 0:
                self.save_checkpoint(all_houses)
            
            logging.info(f"{self.district_names.get(district, district)} 爬取完成，"
                        f"获取 {len(district_houses)} 条房源")
            
            # 区域间休息
            time.sleep(random.uniform(5, 10))
        
        return all_houses
    
    def save_checkpoint(self, houses, filename=None):
        """保存检查点"""
        if filename is None:
            filename = f"data/cache/houses_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(houses, f, ensure_ascii=False, indent=2)
        logging.info(f"检查点已保存: {filename}")
    
    def crawl_details(self, houses, batch_size=50):
        """批量爬取房源详情"""
        detailed_houses = []
        
        for i in tqdm(range(0, len(houses), batch_size), desc="爬取房源详情"):
            batch = houses[i:i+batch_size]
            
            for house in batch:
                if 'url' in house:
                    detail = self.parse_house_detail(house['url'])
                    if detail:
                        # 合并基本信息和详情
                        house.update(detail)
                        detailed_houses.append(house)
            
            # 批次间休息
            time.sleep(random.uniform(3, 5))
            
            # 保存进度
            if len(detailed_houses) % 100 == 0:
                self.save_checkpoint(detailed_houses, 
                    f"data/cache/detailed_houses_{len(detailed_houses)}.json")
        
        return detailed_houses

def main():
    """主函数"""
    crawler = GuangzhouHouseCrawler()
    
    # 1. 爬取基本信息
    logging.info("开始爬取房源基本信息...")
    houses = crawler.crawl_all_districts(max_pages_per_district=5)  # 先爬取少量测试
    
    # 保存基本信息
    df_basic = pd.DataFrame(houses)
    df_basic.to_csv('data/raw/houses_basic.csv', index=False, encoding='utf-8-sig')
    logging.info(f"基本信息已保存，共 {len(houses)} 条")
    
    # 2. 爬取详细信息（可选）
    # logging.info("开始爬取房源详情...")
    # detailed_houses = crawler.crawl_details(houses[:100])  # 先爬取100条测试
    # 
    # df_detailed = pd.DataFrame(detailed_houses)
    # df_detailed.to_csv('data/raw/houses_detailed.csv', index=False, encoding='utf-8-sig')
    # logging.info(f"详细信息已保存，共 {len(detailed_houses)} 条")

if __name__ == "__main__":
    main()
