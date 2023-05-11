import h5py


# 记录txt数据
def list2txt(cache_data, txt_name='cache_data.txt'):
    with open(txt_name, 'w', encoding='utf-8') as cache_data_txt:
        cache_data_str = str(cache_data)
        cache_data_txt.write(cache_data_str)
    print(f'data write to: {txt_name}')


# 读取txt数据
def txt2list(txt_name):
    with open(txt_name, 'r', encoding='utf-8') as cache_data_txt:
        cache_data = cache_data_txt.read()
        cache_data = eval(cache_data)
    return cache_data


# 读取h5文件
def save_h5(filename, group_list, data_mat_list):
    with h5py.File(filename, 'w') as hf:
        for i, group_name in enumerate(group_list):
            hf.create_dataset('/' + group_name, data=data_mat_list[i])
    print(f'h5 file has been saved to: {filename}')


# 读取h5文件
def read_h5(file_name):
    group_list = []
    data_mat_list = []
    with h5py.File(file_name, 'r') as hf:
        for key in hf.keys():
            group_list.append(hf[key].name[1:])  # 去除斜杠
            data_mat_list.append(hf[key][:])
    return group_list, data_mat_list
