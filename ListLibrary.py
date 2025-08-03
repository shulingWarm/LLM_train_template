
# 用于在list1里面插入第2个list
def insert_list(list1, list2, target_value: int, hit_offset=0):
    # 遍历查找第一个目标值的位置
    for i in range(len(list1)):
        if list1[i] == target_value:
            # 看一下后续的字符还有多少
            rest_num = len(list1) - 1
            if (rest_num < hit_offset):
                hit_offset = rest_num
            # 找到后，在目标值位置后插入 list2
            return list1[:i+1+hit_offset] + list2 + list1[i+1+hit_offset:]
    raise RuntimeError(f'Cannot find target value: {target_value}')
    # 未找到目标值，返回原列表
    return list1