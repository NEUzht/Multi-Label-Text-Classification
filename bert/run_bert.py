import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from transformers import BertTokenizer, BertModel, BertConfig
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.model.models import *
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport,F1Score,Precision,Recall
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler

from utils.dataset import *
from models.model import *
from utils.loss import *
from utils.scheduler import *
from utils.draw import *
from utils.stats import *
import pandas as pd
import pickle
warnings.filterwarnings("ignore")


def run_train(args):
    df = pd.read_csv("Train.csv",encoding='utf-8')

    # df['Text'] = df['Text'].str.replace(' ', '')

    new_df = df[['Text','Labels']].copy()


    # 构建class_to_id字典
    class2id = get_class_to_id(df)
    # 构建id_to_class字典
    id2class = {v: k for k, v in class2id.items()}
    
    with open(config['result'] / 'class2id.pkl', 'wb') as f:
        pickle.dump(class2id, f)
    with open(config['result'] / 'id2class.pkl', 'wb') as f:
        pickle.dump(id2class, f)
    print(class2id,"\n",id2class)
    

    train_batch_size = 32
    valid_batch_size = 32


    data_list = new_df.values.tolist()

    labels_nums_dict = stats(data_list, class2id)

    
    # data_list = filter_list(data_list, labels_nums_dict)
    # 设置随机数种子以确保可复现性
    stats(data_list,class2id)
    random.seed(10)

    # 随机化数据集的索引
    indices = list(range(len(data_list)))
    # indices = indices[0:1000]
    random.shuffle(indices)

    # 计算训练集和测试集的划分点
    train_size = 0.8  # 80% 的数据作为训练集
    split_point = int(len(indices) * train_size)

    # 划分数据集
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # 构建训练集和测试集
    train_dataset = [data_list[i] for i in train_indices]
    test_dataset = [data_list[i] for i in test_indices]

    print("FULL Dataset: {}".format(len(data_list)))
    print("TRAIN Dataset: {}".format(len(train_dataset)))
    print("TEST Dataset: {}".format(len(test_dataset)))


    # tokenizer = BertTokenizer.from_pretrained('/home/idal-01/neu_cx/wcx_/bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(config['bert_vocab_path'])
    train_dataset = CustomDataset(train_dataset, tokenizer, args.train_max_seq_len, class2id, id2class,train = True,augment=True) 
    test_dataset = CustomDataset(test_dataset, tokenizer, args.train_max_seq_len, class2id, id2class,train = True)

    # print(train_dataset[0])

    train_params = {'batch_size': train_batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': valid_batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }

    train_dataloader = DataLoader(train_dataset, **train_params)
    valid_dataloader = DataLoader(test_dataset, **test_params)
    
    # ------- model
    logger.info("initializing model")
    # if args.resume_path:
    #     args.resume_path = Path(args.resume_path)
    #     model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(id2class))
    # else:
    #     model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(id2class))
        
    #     model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(id2class))
    base_model = transformers.BertModel.from_pretrained(config['bert_model_dir'])
    input_size = 768
    num_classes = 8

    if args.method_name == 'fnn':
        model = Transformer(base_model, num_classes, input_size)
    elif args.method_name == 'gru':
        model = Gru_Model(base_model, num_classes, input_size)
    elif args.method_name == 'lstm':
        model = Lstm_Model(base_model, num_classes, input_size)
    elif args.method_name == 'bilstm':
        model = BiLstm_Model(base_model, num_classes, input_size)
    elif args.method_name == 'rnn':
        model = Rnn_Model(base_model, num_classes, input_size)
    elif args.method_name == 'textcnn':
        model = TextCNN_Model(base_model, num_classes)
    elif args.method_name == 'attention':
        model = Transformer_Attention(base_model, num_classes)
    elif args.method_name == 'lstm+textcnn':
        model = Transformer_CNN_RNN(base_model, num_classes)
    elif args.method_name == 'lstm_textcnn_attention':
        model = Transformer_CNN_RNN_Attention(base_model, num_classes)
    else:
        raise ValueError('unknown method')

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    # print(model)
    param_optimizer = list(model.named_parameters())
    # print(param_optimizer)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # print(optimizer_grouped_parameters)
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)


    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args= args,model=model,logger=logger,criterion=BCEWithLogLoss(),optimizer=optimizer,
                      scheduler=scheduler,early_stopping=None,training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics=[AUC(average='micro', task_type='binary'),
                                     MultiLabelReport(id2label=id2class),
                                     F1Score(average="samples",search_thresh=True),
                                     AccuracyThresh(thresh=0.28)
                                    #  Precision(thresh=0.5,search_thresh=True),
                                    #  Recall(thresh=0.38,search_thresh=False)
                                     ])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)

def run_test(args):
    from pybert.test.predictor import Predictor


    df = pd.read_csv("test.csv",encoding='utf-8')

    
    with open(config['result'] / 'class2id.pkl', 'rb') as f:
        class2id = pickle.load(f)
    with open(config['result'] / 'id2class.pkl', 'rb') as f:
        id2class = pickle.load(f)


    tokenizer = BertTokenizer.from_pretrained(config['bert_vocab_path'])
    test_dataset = df['Text'].copy().values.tolist()
    
    test_dataset = CustomDataset(test_dataset, tokenizer, args.train_max_seq_len, class2id, id2class,train = False)

    # print(training_set[0])


    test_params = {'batch_size': 32,
                    'shuffle': False,
                    'num_workers': 4
                    }

    test_dataloader = DataLoader(test_dataset, **test_params)
    base_model = transformers.BertModel.from_pretrained(config['bert_model_dir'])
    model = Transformer_Attention(base_model, len(id2class))
    model_dict = torch.load("model.pth")
    # print(model_dict)
    model.load_state_dict(model_dict)
    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu)
    result = predictor.predict(data=test_dataloader)
    
    result = np.array(result) >= 0.2
    # print(result)
    labels = []
    for i in range(result.shape[0]):
        label = []
        for j in range(result.shape[1]):
            if result[i][j]:
                label.append(id2class[j])
        labels.append(label)
    index_ = []
    label =  []
    for index, data in enumerate(labels):
        index_.append(index + 1)
        label.append(str(data))
    # 将列表转为字典，然后创建 DataFrame
    data_dict = {"ID": index_, "Labels": labels}
    df = pd.DataFrame(data_dict)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(config["result"] / "predict.csv",encoding='utf-8', index=False)
    print("predict finish!")
def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='kaggle', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--method_name", default='attention', type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    # parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    init_logger(log_file=config['log_dir'] / f'{args.arch}-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)
    

    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)



if __name__ == '__main__':
    main()
