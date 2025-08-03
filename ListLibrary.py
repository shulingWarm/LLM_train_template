
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

# 寻找某个数值在list里面首次出现的位置
def find_in_list(in_list, target_value):
    try:
        return in_list.index(target_value)
    except ValueError:
        return -1

# 在list里面随便添加几个token
# 将list2里面的内容均匀的插入在list1里面，形成一个新的list
# 例如list1: [2,3,4,5,6,7,8]
# list2: [21,22,23]
# begin_offset是1,对应的是list1里面的"3"
# 这种情况下，将21,22,23均匀插入到list1从1号开始的位置
# 因为1号后面有5个数字，所以插入步长就是: 5/3
# 然后按照这个步长插入
def separate_insert_token(list1, list2, begin_offset):
    if len(list1) <= begin_offset:
        raise RuntimeError(f'list size: {len(list1)} < begin_offset: {begin_offset}')
    
    # 计算剩余长度和插入步长
    rest_length = len(list1) - begin_offset
    # 处理list2为空的情况
    if len(list2) == 0:
        return list1
    
    insert_step = rest_length / len(list2)
    
    # 初始化结果列表和指针
    ret_list = list1[:begin_offset+1]  # 包含0~begin_offset
    id1 = begin_offset + 1  # list1剩余部分的起始索引
    id2 = 0  # list2的起始索引
    next_insert = len(ret_list)  # 初始插入位置 (begin_offset+1)

    # 双指针遍历，直到list1和list2都处理完毕
    while id1 < len(list1) or id2 < len(list2):
        # 如果当前是插入点 并且还有list2元素要插入
        if id2 < len(list2) and len(ret_list) >= next_insert:
            ret_list.append(list2[id2])
            id2 += 1
            next_insert += insert_step
        # 否则插入list1当前元素
        else:
            ret_list.append(list1[id1])
            id1 += 1
            
    return ret_list