import random

def stats(data_list,class2id):
    num_of_class = {k: 0  for k, v in class2id.items()}
    dict_num = {str(i):0 for i in range(1,9)}
    labels_nums = {}
    total_len_labels = 0
    for x in data_list:
        labels = x[1]
        total_len_labels += len(eval(labels))

        
        labels_nums[' '.join(sorted(eval(labels)))] = labels_nums.get(' '.join(sorted(eval(labels))), 0) + 1
        # print(labels)
        dict_num[str(len(eval(labels)))] += 1
        for label in eval(labels):
            num_of_class[label] += 1
    # print(num_of_class)
    # print(dict_num)
    print(f"average length of labels: {total_len_labels / len(data_list)}")
    labels_nums = dict(sorted(labels_nums.items(), key=lambda item: item[1], reverse=True))
    # for key, value in labels_nums.items():
    #     print(f"{key}: {value}")
    return num_of_class,labels_nums

def filter_list(data_list,labels_nums_dict):
    filter_data_list = []
    for data in data_list:

        if data[1] == "['Love']":

            probability = random.uniform(0, 1)
            if probability <= 0.5:
                filter_data_list.append(data)
        
        elif labels_nums_dict[' '.join(sorted(eval(data[1])))] >= 50:
            filter_data_list.append(data)
    
    return filter_data_list


# df['Text'] = df['Text'].str.replace(' ', '')

# new_df = df[['Text','Lables']].copy()


# # 构建class_to_id字典
# class2id = get_class_to_id(df)
# # 构建id_to_class字典
# id2class = {v: k for k, v in class2id.items()}



# train_batch_size = 128
# valid_batch_size = 128


# data_list = new_df.values.tolist()
# # print(data_list)

# labels_nums_dict = stats(data_list,class2id)



# stats(filter_data_list,class2id)
# 设置随机数种子以确保可复现性
# random.seed(20)

# # 随机化数据集的索引
# indices = list(range(len(data_list)))
# # indices = indices[0:10]
# random.shuffle(indices)

# # 计算训练集和测试集的划分点
# train_size = 0.8  # 80% 的数据作为训练集
# split_point = int(len(indices) * train_size)

# # 划分数据集
# train_indices = indices[:split_point]
# test_indices = indices[split_point:]

# # 构建训练集和测试集
# train_dataset = [data_list[i] for i in train_indices]
# test_dataset = [data_list[i] for i in test_indices]

# print("FULL Dataset: {}".format(len(data_list)))
# print("TRAIN Dataset: {}".format(len(train_dataset)))
# print("TEST Dataset: {}".format(len(test_dataset)))

# tongji(train_dataset,class2id)
# tongji(test_dataset,class2id)


# print(config['bert_vocab_path'])
# tokenizer = BertTokenizer.from_pretrained("/home/idal-01/neu_cx/wcx__/Bert-Multi-Label-Text-Classification-master/pybert/pretrain/bert/bert-base-chinese")
# train_dataset = CustomDataset(train_dataset, tokenizer, 256, class2id, id2class,train = True,augment=True) 
# test_dataset = CustomDataset(test_dataset, tokenizer, 256, class2id, id2class,train = True)

# print(test_dataset[0])
# print(test_dataset[0])
# print(test_dataset[0])
# print(test_dataset[0])
