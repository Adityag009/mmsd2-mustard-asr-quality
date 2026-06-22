from torch.utils.data import Dataset
import logging
import os
from PIL import Image
import json
import pickle

logger = logging.getLogger(__name__)

WORKING_PATH = r"X:\ASR\MMSD2.0-main\data"

class MyDataset(Dataset):
    def __init__(self, mode, text_name=None, limit=None, *args, **kwargs):
        
        self.text_name = "mustard_text" 
        
        audio_path = os.path.join(WORKING_PATH, "extracted_features", "audio_291.pkl")
        with open(audio_path, 'rb') as f:
            self.audio_features = pickle.load(f)
            
        self.data = self.load_data(mode, limit)
        self.image_ids = list(self.data.keys())
        
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH, "dataset_image", "extracted_frames", str(id)+".jpg")
    
    def load_data(self, mode, limit):
        cnt = 0
        data_set = dict()
        
        json_path = os.path.join(WORKING_PATH, self.text_name, f"{mode}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f1:
                datas = json.load(f1)
                
            for data in datas:
                if limit is not None and cnt >= limit:
                    break

                image = str(data['image_id'])
                sentence = data['text']
                label = data['label']
 
                image_file_path = os.path.join(WORKING_PATH, "dataset_image", "extracted_frames", f"{image}.jpg")
                
                if os.path.isfile(image_file_path) and image in self.audio_features:
                    data_set[image] = {"text": sentence, 'label': label}
                    cnt += 1
                    
        return data_set

    def image_loader(self, id):
        return Image.open(self.data[id]["image_path"])
        
    def text_loader(self, id):
        return self.data[id]["text"]

    def __getitem__(self, index):
        id = self.image_ids[index]
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        audio_feature = self.audio_features[str(id)]
        
        return text, image_feature, audio_feature, label, id

    def __len__(self):
        return len(self.image_ids)
        
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        audio_list = []
        label_list = []
        id_list = []
        
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            audio_list.append(instance[2]) 
            label_list.append(instance[3]) 
            id_list.append(instance[4])    
            
        return text_list, image_list, audio_list, label_list, id_list
