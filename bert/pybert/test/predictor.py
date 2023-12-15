#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar
from tqdm import tqdm
class Predictor(object):
    def __init__(self,model,logger,n_gpu):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)

    def predict(self,data):
        all_logits = []
        predict_bar = tqdm(data,desc= "caojijiayou",total = len(data))
        result = []
        for step, batch in enumerate(predict_bar, 0):
            self.model.eval()
            with torch.no_grad():
                ids = batch['ids'].to(self.device, dtype = torch.long)
                mask = batch['mask'].to(self.device, dtype = torch.long)
                token_type_ids = batch['token_type_ids'].to(self.device, dtype = torch.long)

                logits = self.model(ids, mask, token_type_ids)
                logits = logits.sigmoid()
                # print(logits.shape)
                all_logits.append(logits)
                
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        all_logits = torch.cat(all_logits, dim=0).cpu().numpy().tolist()
        return all_logits






