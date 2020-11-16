# utils.py
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

def extract_tls_info(s):
    tls_key_list = ['C', 'ST', 'L', 'O', 'OU', 'CN', 'emailAddress', 'unknown', 'serialNumber']
    s = s.split(',')
    s = [x.split('/') for x in s]
    s = sum(s, [])
    res = {}
    for x in s:
        if '=' not in x:
            continue
        x = x.split('=')
        key = x[0].strip(' ')
        value = x[1].strip(' ')
        if key in tls_key_list:
            res[key] = value
    
    return res

def process_oneHot_by_cnt(dataset, key, threshold=0, vcnt=None):
    col = dataset[key]
    if vcnt is None:
        vcnt = col.value_counts()
        if threshold>0:
            if isinstance(threshold, int):
                vcnt = vcnt[vcnt>=threshold]
            else:
                if isinstance(threshold, float):
                    threshold = len(dataset)*threshold
                    vcnt = vcnt[vcnt>=threshold]
                else:
                    UserWarning("In function process_oneHot_by_cnt `threshold` should be of int or float type. ")
                    return dataset, None
    dtype = np.uint8
    for vkey in vcnt.index:
        vkey_col = np.array(col==vkey, dtype=dtype)
        dataset[key+str(vkey)] = vkey_col
    return dataset, vcnt


def process_srcAddress(dataset):
    '''process_srcAddress(统计每条数据的 srcAddress 在数据集中的出现次数并除以?，然后添加到数据信息中)

    Args:
        dataset     原始数据集
    Returns:
        dataset     添加 srcAddress_count 列后的数据集
    '''
    addr_list = list(dataset['srcAddress'].values)
    count_list = [0]*len(addr_list)
    for i in range(len(addr_list)):
        count_list[i] = addr_list.count(addr_list[i])
    count_list = [x for x in count_list]
    dataset['srcAddress_count'] = count_list
    return dataset

def process_destAddress(dataset):
    '''process_destAddress(统计每条数据的 destAddress 在数据集中的出现次数并除以？，然后添加到数据信息中)

    Args:
        dataset     原始数据集
    Returns:
        dataset     添加 destAddress_count 列后的数据集
    '''
    addr_list = list(dataset['destAddress'].values)
    count_list = [0]*len(addr_list)
    for i in range(len(addr_list)):
        count_list[i] = addr_list.count(addr_list[i])
    dataset['destAddress_count'] = count_list

    return dataset

def process_port(dataset, col_name):
    '''process_port(将端口号除以？)
    Args:
        dataset     原始数据集
    Returns:
        dataset     更新端口值后的数据集
    '''

    # MAX = dataset[col_name].values.max()
    # for idx in range(dataset.shape[0]):
    #     dataset.loc[idx, col_name] = dataset.loc[idx, col_name] * 1.0 / (MAX * 2)

    # port_list = list(dataset[col_name].values)
    # MAX = max(port_list)
    # port_list = [x*1.0/(MAX*2) for x in port_list]
    # dataset[col_name] = port_list

    return dataset

def process_tlsVersion(dataset):

    tls_list = list(dataset['tlsVersion'].values)
    tls_dic = {'TLS 1.1': 0.0, 'TLS 1.3': 0.014012143858010275, 'TLSv1': 29.69283276450512, 'UNDETERMINED': 0.13636363636363638, 'TLS 1.2': 0.6491481374530754, 'other': 0.0}
    tls_list = [tls_dic.get(x, tls_dic['other']) for x in tls_list]
    
    dataset['tlsVersion'] = tls_list

    return dataset

def process_tlsSni(dataset):

    tlsSni_list = list(dataset['tlsSni'].values)
    prefix = [''] * len(tlsSni_list)
    postfix = [''] * len(tlsSni_list)
    for idx in range(len(tlsSni_list)):
        s = tlsSni_list[idx]
        if s == "":
            continue
        s = s.strip('www.')
        point_idx = s.find('.')
        if point_idx < 0:
            prefix[idx] = s
        else:
            prefix[idx] = s[:point_idx]
            postfix[idx] = s[point_idx+1:]
    
    onehotencoders = []
    # prefix
    onehotencoder = OneHotEncoder(categories='auto', sparse=False, dtype=np.int8, handle_unknown='ignore')
    prefix_onehot = onehotencoder.fit_transform(np.array(prefix).reshape(-1,1))
    dataset = pd.concat([dataset, pd.DataFrame(prefix_onehot)], axis=1)
    onehotencoders.append({'tlsSni_prefix':onehotencoder})
    # postfix
    onehotencoder = OneHotEncoder(categories='auto', sparse=False, dtype=np.int8, handle_unknown='ignore')
    postfix_onehot = onehotencoder.fit_transform(np.array(postfix).reshape(-1,1))
    dataset = pd.concat([dataset, pd.DataFrame(postfix_onehot)], axis=1)
    onehotencoders.append({'tlsSni_postfix':onehotencoder})
    # remove tlsSni
    # dataset = dataset.drop(['tlsSni'], axis=1)
    return dataset, onehotencoders

def process_tlsIssuerDn(dataset):

    tls_key_list = ['C', 'ST', 'L', 'O', 'OU', 'CN', 'emailAddress', 'unknown', 'serialNumber']
    tlsSubject_list = list(dataset['tlsSubject'].values)
    tlsIssuerDn_list = list(dataset['tlsIssuerDn'].values)
    similarity = [0]*len(tlsIssuerDn_list)

    for idx in range(len(tlsIssuerDn_list)):
        subj = tlsSubject_list[idx]
        issue = tlsIssuerDn_list[idx]
        if subj == issue:
            similarity[idx] = 1.0
            continue
        subj = extract_tls_info(subj)
        issue = extract_tls_info(issue)
        MAX = max([len(subj), len(issue)])
        same = 0
        for key in tls_key_list:
            if subj.get(key, None) and subj.get(key, None) == issue.get(key, None):
                same += 1
        similarity[idx] = same*1.0 / MAX
    dataset['tlsSimilarity'] = similarity

    return dataset

def process_tlsSni_type(dataset):

    tlsSni_list = list(dataset['tlsSni'].values)

    postfix_type = [0] * len(tlsSni_list)
    postfix_len = [0] * len(tlsSni_list)
    prefix_type = [0] * len(tlsSni_list)
    point_count = [0] * len(tlsSni_list)
    middle_len = [0] * len(tlsSni_list)
    total_len = [0] * len(tlsSni_list)

    postfix = [""] * len(tlsSni_list)
    for idx in range(len(tlsSni_list)):
        s = tlsSni_list[idx]
        if '.' in s:
            res = s.split('.')
            postfix[idx] = res[-1]
            postfix_len[idx] = len(res[-1])
            prefix_type[idx] = ('www' in res[0])*1
            point_count[idx] = len(res)
            if 'www' in res[0]:
                middle_len[idx] = len(res[1])
            else:
                middle_len[idx] = len(res[0])
        total_len[idx] = len(s)    

    # dic = {'': 4.1722663408674405, 'me': 5.454545454545454, 'local': 0.0, 'link': 40.0, 'website': 38, 'tv': 0.0, 'net': 0.31277150304083406, 'ms': 0.0, '2': 0.0, 'gdn': 44, 'xyz': 18.461538461538463, 'cc': 0.0, 'ga': 14, 'co': 0.0, 'sb': 33, 'cn': 0.0, 'org': 0.0, 'so': 0.0, '174': 0.0, 'ru': 1696.6666666666667, 'io': 0.18433179723502305, 'com': 0.3168887288440763, 'top': 899.9999999999999, 'im': 0.0, '108': 0.0, 'digit': 0.025, 'other': 0.5360824742268041}

    # postfix_type = [dic['digit'] if x.isdigit() else dic.get(x, dic['other']) for x in postfix]
        
    dataset['tlsSni_postfix_type'] = postfix
    dataset['tlsSni_postfix_len'] = postfix_len
    dataset['tlsSni_prefix_type'] = prefix_type
    dataset['tlsSni_point_count'] = point_count
    dataset['tlsSni_middle_len'] = middle_len
    dataset['tlsSni_total_len'] = total_len

    return dataset

def process_tlsSubject_len(dataset):

    tlsSubject_list = list(dataset['tlsSubject'].values)

    tls_key_list = ['C', 'ST', 'L', 'O', 'OU', 'CN', 'emailAddress', 'unknown', 'serialNumber']
    tls_info_len = [[0]*len(tlsSubject_list) for _ in range(len(tls_key_list)+3)]
    for idx in range(len(tlsSubject_list)):
        tmp = extract_tls_info(tlsSubject_list[idx])
        for key_idx in range(len(tls_key_list)):
            key = tls_key_list[key_idx]
            tls_info_len[key_idx][idx] = len(tmp.get(key, ''))
            if tls_info_len[key_idx][idx] != 0:
                tls_info_len[-2][idx] += 1
        if tmp == {}:
            tls_info_len[-1][idx] = 1
        tls_info_len[-3][idx] = len(tlsSubject_list[idx])

    for key_idx in range(len(tls_key_list)):
        key = tls_key_list[key_idx]
        dataset[key+"_len"] = tls_info_len[key_idx]

    dataset['tlsSubject_total_len'] = tls_info_len[-3]
    dataset['tlsSubject_type_count'] = tls_info_len[-2]
    dataset['tlsSubject_empty'] = tls_info_len[-1] 

    return dataset    


def process_tlsSubject_other(dataset):

    tlsSubject_list = list(dataset['tlsSubject'].values)

    tls_XX = [0]*len(tlsSubject_list)
    tls_star = [0]*len(tlsSubject_list)
    tls_default = [0]*len(tlsSubject_list)
    tls_some_state = [0]*len(tlsSubject_list)

    for idx in range(len(tlsSubject_list)):
        tmp = extract_tls_info(tlsSubject_list[idx])
        for key, value in tmp.items():
            if value=='XX':
                tls_XX[idx] = 1
            elif value=='*':
                tls_star[idx] = 1
            elif 'default' in value.lower():
                tls_default[idx] = 1
            elif 'Some-State' in value:
                tls_some_state[idx] = 1
 
    dataset['tls_XX'] = tls_XX
    dataset['tls_star'] = tls_star
    dataset['tls_default'] = tls_default
    dataset['tls_some_state'] = tls_some_state

    return dataset


def process_bytes(dataset):

    bytesout_list = list(dataset['bytesOut'].values)
    bytesin_list = list(dataset['bytesIn'].values)

    pktin_list = list(dataset['pktsIn'].values)
    pktout_list = list(dataset['pktsOut'].values)

    bytesin_rate = [0] * len(bytesout_list)
    bytesout_rate = [0] * len(bytesout_list)

    for idx in range(len(bytesout_list)):
        if pktout_list[idx] > 0:
            bytesout_rate[idx] = bytesout_list[idx] / pktout_list[idx]
        else:
            bytesout_rate[idx] = 1000000

        if pktin_list[idx] > 0:
            bytesin_rate[idx] = bytesin_list[idx] / pktin_list[idx]
        else:
            bytesin_rate[idx] = 1000000

    dataset['tls_bytesin_rate'] = bytesin_rate
    dataset['tls_bytesout_rate'] = bytesout_rate

    return dataset


def process_tlsSubject_type(dataset):

    tlsSubject_list = list(dataset['tlsSubject'].values)
    
    ''' C_type, CN_type '''
    # C_dic = {'': 0.3878787878787879, 'XX': 750, '--': 0.0, 'DE': 1.1764705882352942, 'DK': 0.0, 'JP': 0.0, 'CNstore': 0.0, 'CN': 0.017035775127768313, 'US': 0.6221461187214612, 'AU': 1046.0, 'MY': 0.0, 'GB': 540.0, 'other': 18.125000000000007}
    # CN_dic = {'': 1.851012390450287, 'CMCC': 0.0, '1': 0.0, 'svn': 0.0, 'me': 20.909090909090907, 'org': 0.1234567901234568, 'localdomain': 0.0, 'link': 8, '*': 82, 'top': 31, 'net': 2.0, 'ms': 0.0, 'DBAPP': 0.0, 'info': 20, 'local': 0.0, 'XX': 17, 'sb': 33, 'sslvpn': 0.0, 'cn': 0.006082725060827251, 'io': 0.10695187165775401, '0': 0.0, 'com': 0.29000969932104753, 'im': 0.0, 'other': 8.768115942028999}
    C_list = []
    CN_list = []

    ST_list = []
    L_list = []
    O_list = []
    emailAddress_list = []
    serialNumber_list = []

    # tls_key_list = ['C', 'ST', 'L', 'O', 'OU', 'CN', 'emailAddress', 'unknown', 'serialNumber']
    # tls_key_list = ['ST', 'L', 'O', 'emailAddress', 'serialNumber']
    for idx in range(len(tlsSubject_list)):
        tmp = extract_tls_info(tlsSubject_list[idx])
        C_list.append(tmp.get('C', ''))
        CN_list.append(tmp.get('CN', '').split('.')[-1])
        ST_list.append(tmp.get('ST', ''))
        L_list.append(tmp.get('L', ''))
        O_list.append(tmp.get('O', ''))
        emailAddress_list.append(tmp.get('emailAddress', ''))
        serialNumber_list.append(tmp.get('serialNumber', ''))

    # C_type = [C_dic.get(x, C_dic['other']) for x in C_list]
    # CN_type = [CN_dic.get(x, CN_dic['other']) for x in CN_list]
    
    dataset['tls_C_type'] = C_list
    dataset['tls_CN_type'] = CN_list
    dataset['ST'] = ST_list
    dataset['L'] = L_list
    dataset['O'] = O_list
    dataset['emailAddress'] = emailAddress_list
    dataset['serialNumber'] = serialNumber_list

    return dataset


def drop_repeat_rows(dataset):

    # return dataset.drop_duplicates([x for x in dataset.columns if x!='eventId'], keep='first')
    return dataset


def process_port_adjacency(dataset):

    idx = range(dataset.shape[0])
    dataset['idx'] = idx

    ips_label = dataset[['srcAddress', 'destAddress', 'srcPort', 'idx']]
    ips_label = ips_label.sort_values(by = ['srcAddress', 'destAddress', 'srcPort'])
    ips_label = list(ips_label.values)

    port_adjacency = [0] * dataset.shape[0]
    for idx in range(dataset.shape[0]):
        cur = list(ips_label[idx])
        if idx == dataset.shape[0] - 1:
            next = ['', '', 0]
        else:
            next = list(ips_label[idx+1])
        if idx == 0:
            before = ['', '', 0]
        else:
            before = list(ips_label[idx-1])
        min_ = 1000000
        if cur[0] == before[0] and cur[1] == before[1]:
            min_ = cur[2] - before[2]
        if cur[0] == next[0] and cur[1] == next[1]:
            tmp = next[2] - cur[2]
            if tmp < min_:
                min_ = tmp
        if min_ != 1000000:
            port_adjacency[cur[-1]] = min_
        else:
            port_adjacency[cur[-1]] = 350

    dataset['srcPort_adjacency'] = port_adjacency
    dataset = dataset.drop(['idx'], axis=1)

    return dataset


def ExtractTlsSubject(data, onehotencoders=None, handle_key='tlsSubject'):
    tls_key_list = ['C', 'ST', 'L', 'O', 'OU', 'CN']
    #tls_key_hash_len = [65536, 65536, 65536, 65536, 65536, 65536]
    for ncol in tls_key_list:
        data[ncol] = ''
    for idx, row in enumerate(data.iterrows()):
        tlsSubject_str = row[1][handle_key]
        if tlsSubject_str=='':
            continue
        tlsSubject_list = tlsSubject_str.split(',')
        tlsSubject_list = [item.split('/') for item in tlsSubject_list]
        tlsSubject_list = sum(tlsSubject_list, [])
        i=0
        while i < len(tlsSubject_list):
            if ('=' not in tlsSubject_list[i]):
                if i==0:
                    del tlsSubject_list[0]
                tlsSubject_list[i-1] += tlsSubject_list[i]
                del tlsSubject_list[i]
            else:
                tlsSubject_list[i] = tlsSubject_list[i].strip(' ')
                i += 1
        tlsSubject_list = [item.split('=') for item in tlsSubject_list]
        
        try:
            for key, value in tlsSubject_list:
                if key in tls_key_list:
                    data.loc[idx, key] = value
        except:
            pass

    if not (onehotencoders is None):
        for key in tls_key_list:
            x = onehotencoders[key].transform(data[key].to_numpy().reshape(-1,1))
            data = pd.concat([data, pd.DataFrame(x)], axis=1)
    else:
        onehotencoders = {}
        for key in tls_key_list:
            onehotencoder = OneHotEncoder(categories='auto', sparse=False, dtype=np.int8, handle_unknown='ignore')
            x = onehotencoder.fit_transform(data[key].to_numpy().reshape(-1,1))
            data = pd.concat([data, pd.DataFrame(x)], axis=1)
            onehotencoders[key] = onehotencoder
    return data, onehotencoders


def ELFHash(str):
    hvalue = 0
    for i in str:
        hvalue = (hvalue<<4) + ord(i)
        if (hvalue & 0xF0000) != 0:
            x = hvalue & 0xF0000
            hvalue ^= (x >> 10)
            hvalue &= ~x
            hvalue &= 0xFFFF
    return hvalue



def ProcessData(dataset, encoders=None, keyList=None):
    keyList = {
        # 'srcPort': 16,
        'destPort': 1/1100,
        'tlsVersion': 0,
        'tlsSni_postfix_type': 5/13200,
        'tls_C_type': 5/13200,
        'tls_CN_type': 5/13200, 
        'srcAddress': 1.0,
        'destAddress': 1.0,
        'ST':5/13200,
        'L':5/13200,
        'O':5/13200,
        'emailAddress': 5/13200,
        'serialNumber': 5/13200
    } if keyList==None else keyList
    onhot_list = ['destPort', 'tlsVersion', 'tlsSni_postfix_type', 'tls_C_type', 'tls_CN_type'] + ['ST', 'L', 'O']
    # , 'emailAddress', 'serialNumber'
    if encoders==None:
        dataset = process_tlsSni_type(dataset)
        dataset = process_tlsSubject_type(dataset)
        encoders = []
        for key in onhot_list:  
            dataset, a_encoder = process_oneHot_by_cnt(dataset, key, keyList[key])
            encoders.append(a_encoder)
        dataset = process_port_adjacency(dataset)
        dataset = process_tlsIssuerDn(dataset)
        dataset = process_tlsSubject_len(dataset)
        dataset = process_bytes(dataset)

    else:
        dataset = process_tlsSni_type(dataset)
        dataset = process_tlsSubject_type(dataset)
        for idx, key in enumerate(onhot_list): 
            dataset, _ = process_oneHot_by_cnt(dataset, key, keyList[key], vcnt=encoders[idx])
        dataset = process_port_adjacency(dataset)
        dataset = process_tlsIssuerDn(dataset)
        dataset = process_tlsSubject_len(dataset)
        dataset = process_bytes(dataset)

    return dataset, encoders
