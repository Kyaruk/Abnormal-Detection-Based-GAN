# apriori算法
from numba import cuda, jit
import re


# 加载数据
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 创建集合C1
# @jit(target_backend='cuda', nopython=False)
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


# @jit(target_backend='cuda', nopython=False)
def scanD(D, Ck, minSupport):
    ssCnt = {}
    # print("numItems",numItems)
    # print(list(D))
    for tid in D:
        # print(tid)
        for can in Ck:
            # print("====================")
            # print(can)
            if can.issubset(tid):
                if can not in ssCnt:
                    # if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    # print("numItems",numItems)
    # print(list(D))
    # print(ssCnt)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# 创建Ck
# @jit(target_backend='cuda', nopython=False)
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    # print("============aprioriGen begin==============",Lk,k,lenLk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            # print("==========aprioriGen================",i,j,L1,L2)
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


# 实现apriori算法
# @jit(target_backend='cuda', nopython=False)
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    # L1用于存储项集大小为1的元素，它是一个集合，集合中的每个元素是一个大小为1的频繁项集
    L = [L1]
    k = 2
    # print("L1")
    # print(L1)
    # print("L")
    # print(L)
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        # Lk用于存储项集大小为k的元素，它是一个几乎，集合中的每个元素是一个大小为k的频繁项集
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# 关联规则生成函数.L表示求出的频繁项集合，supportData表示集合对应的支持度
# @jit(target_backend='cuda')
def generateRules(L, supportData, minConf=0.7):
    # print("======================begin generateRules====================")
    # print("L",L)
    # print("supportData",supportData)
    # print("supportData",supportData)
    bigRuleList = []
    # i从1开始，只获取有两个或者更多元素的集合
    for i in range(1, len(L)): \
            # freqSet表示一个频繁项集合
        for freqSet in L[i]:
            # print("freqSet")
            # print(freqSet)
            # item表示频繁项集合中的元素，H1是将一个频繁项集合从frozenset类型变成list类型，并且frozenset中每个元素类型从整型变成frozenset类型
            H1 = [frozenset([item]) for item in freqSet]
            # print("========H1",H1)
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 计算频繁项集freqSet中各个元素出现的条件概率.每次只考虑频繁项集中，（n-1个元素）-》（1个元素）的关联规则。不需要考虑小于n-1个元素到1个元素的关联规则，因为这个关联规则在本频繁项集的子集合中会出现（频繁项集的子集也是频繁项集）
# @jit(target_backend='cuda')
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    # print("==============begin calcConf===============")
    # print("===========freqSet",freqSet)
    # print("===========H",H)
    for conseq in H:
        # print("conseq",conseq)
        # print("freqSet-conseq",freqSet-conseq)
        # print("supportData[freqSet-conseq]",supportData[freqSet-conseq])
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # print(freqSet - conseq, '--->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# @jit(target_backend='cuda')
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


def my_print(data_name, data):
    print(data_name + ":")
    print(data)
    print("=====================")


def my_apriori(minSupport, temp_data):
    # my_print("dataSet", temp_data)
    c1 = createC1(temp_data)
    # print(c1)
    d = list(map(set, temp_data))
    # print(len(list(d)))
    l1, supp_data_0 = scanD(d, c1, minSupport)
    # my_print("L1", l1)
    # my_print("supp_data_0", supp_data_0)

    l, supp_data = apriori(temp_data, minSupport)
    # my_print("L", l)
    # my_print("supp_data", supp_data)
    my_rules = []
    my_rules = generateRules(l, supp_data, minConf=minSupport)
    my_rules.sort(key=lambda temp_rule: (-1 * temp_rule[2], temp_rule[0]))
    # print("rules" + ":")
    # for my_rule in my_rules:
    #     print(my_rule)
    # print("=====================")
    return my_rules, l


def get_multlayer_attr(attr_list, max_length, temp_col):
    result = []
    list = frozenset2list(attr_list)
    # TCP链接的基本特征
    # TCP链接的信息特征
    # 基于时间的网络流量统计特征
    # 基于主机的网络流量统计特征
    mult_attr_lists = [
        ["duration.*", "protocol_type.*", "service.*", "flag.*", "src_bytes.*", "dst_bytes.*", "land.*",
         "wrong_fragment.*", "urgent.*"],
        ["hot.*", "num_failed_logins.*", "logged_in.*", "num_compromised.*", "root_shell.*", "su_attempted.*",
         "num_root.*", "num_file_creations.*", "num_shells.*", "num_access_files.*", "num_outbound_cmds.*",
         "is_hot_login.*", "is_guest_login.*", ],
        ["count.*", "srv_count.*", "serror_rate.*", "srv_serror_rate.*", "rerror_rate.*", "srv_rerror_rate.*",
         "same_srv_rate.*", "diff_srv_rate.*", "srv_diff_host_rate.*"],
        ["dst_host_count.*", "dst_host_srv_count.*", "dst_host_same_srv_rate.*", "dst_host_diff_srv_rate.*",
         "dst_host_same_src_port_rate.*", "dst_host_srv_diff_host_rate.*", "dst_host_serror_rate.*",
         "dst_host_srv_serror_rate.*", "dst_host_rerror_rate.*", "dst_host_srv_rerror_rate.*"],
    ]
    for mult_attr_list in mult_attr_lists:
        temp_result = get_attr(list=mult_attr_list, temp_col=temp_col)
        result.append(temp_result)
    return result


def frozenset2list(list_frozen):
    result = []
    for temp in list_frozen:
        temp_result = []
        if len(temp) != 0:
            for temp_item in temp:
                temp_result.append(list(temp_item))
            result.append(temp_result)
    return result


def get_attr(list, temp_col):
    result = []
    for temp_tuple in temp_col:
        for pattern in list:
            if re.match(pattern, temp_tuple[0]):
                result.append(temp_tuple[1])
                break
    return result


if __name__ == "__main__":
    print("debug")
