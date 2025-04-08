import holidays
import pandas as pd
from datetime import datetime

# 创建中国的节假日对象
cn_holidays = holidays.China()

# 读取 Excel 数据
data = pd.read_excel("2020哈尔滨轨道交通站点客流数据-0706.xlsx", sheet_name=7)
data = data.to_numpy()


# 检查某个日期是否为节假日
def check_holiday(date_str):
    date_str = date_str.strip("[]'")  # 去掉方括号和引号
    date_str = date_str.split("T")[0]  # 只保留日期部分
    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if date in cn_holidays and (date.weekday() in [5, 6]):
        return 3
    elif date in cn_holidays:
        return 2
    elif date.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
        return 1
    else:
        return 0


# 创建一个空列表用于存储结果
holiday_list = []

# 遍历数据中的日期并检查假期
for i in range(len(data)):
    date_str = str(data[i][0])  # 假设日期在第一列
    holiday_status = check_holiday(date_str)  # 检查假期
    holiday_list.append(holiday_status)  # 将结果添加到列表中

# 将结果写入 DataFrame
result_df = pd.DataFrame({
    'Date': [str(data[i][0]) for i in range(len(data))],  # 日期列
    'Holiday Status': holiday_list  # 假期状态列
})

# 写入 Excel 文件
result_df.to_excel("holiday_check_results.xlsx", index=False)

print("结果已成功写入 holiday_check_results.xlsx")