

# 训练时使用的数据提供器
class DataProvder:
    def __init__(self):
        pass

    # 总的数据量
    def getDataNum(self):
        return 0

    # 根据id获取某一组数据
    # 返回的是输入和期望输出
    def getDataByIndex(self, index) -> (str, str):
        pass