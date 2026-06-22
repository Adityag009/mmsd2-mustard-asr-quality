import os
import sys
from model_audio import MV_CLIP         
from train_audio_500 import train, evaluate_acc_f1            
from data_set_audio import MyDataset     
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import wandb
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    
    #  ORIGINAL ARGUMENTS 
    parser.add_argument('--device', default='0', type=str, help='device number')
    parser.add_argument('--model', default='MV_CLIP', type=str, help='the model name', choices=['MV_CLIP'])
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    
    
    parser.add_argument('--weight_text', default=0.33, type=float, help='Alpha: text modality weight')
    parser.add_argument('--weight_image', default=0.33, type=float, help='Beta: image/video modality weight')
    parser.add_argument('--weight_audio', default=0.34, type=float, help='Gamma: audio modality weight')
    parser.add_argument('--audio_size', default=768, type=int, help='Size of audio features in .pkl file')
    
     # PAPER's CONFIG ARGUMENTS 
    parser.add_argument('--epochs', default=500, type=int, help='Paper num epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Paper batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='Paper learning rate')
    parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')
    parser.add_argument('--shared', default=1024, type=int, help='Dummy variable for CLIP')
    parser.add_argument('--projection', default=256, type=int, help='Dummy variable for CLIP')
    parser.add_argument('--gpu', default='0', type=str, help='GPU slot number')
    parser.add_argument('--context', default='True', type=str, help='Context flag')
    parser.add_argument('--speaker', default='False', type=str, help='Speaker flag')
    parser.add_argument('--classification_report', default='True', type=str, help='Report flag')
    
    #  PREVIOUS LEARNING ARGUMENTS 
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam', help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = set_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- HARDWARE CHECK: Currently running on {device} ---")

    seed_everything(args.seed)

    wandb.init(
        project="MMSD2.0",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.watch_called = False  

    train_data = MyDataset(mode='train', text_name=args.text_name, limit=None)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)

    if args.model == 'MV_CLIP':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = MV_CLIP(args)
    else:
        raise RuntimeError('Error model name!')

    model.to(device)
    
    checkpoint_path = '../output_dir/MV_CLIP/model_500.pt' 
    
    print(f"\n Loading pre-trained brain from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.eval() 

    wandb.watch(model, log="all")

    print("\n RUNNING TEST SET WITH NEW WEIGHTS ")
    print(f"Current Weights -> Text: {args.weight_text} | Image: {args.weight_image} | Audio: {args.weight_audio}")
    
    with torch.no_grad(): 
        test_acc, test_f1, test_precision, test_recall = evaluate_acc_f1(
            args, model, device, test_data, processor, macro=True, mode='test'
        )
    
    print(f"\n Final Test Accuracy: {test_acc*100:.2f}% | F1 Score: {test_f1*100:.2f}%")

if __name__ == '__main__':
    main()
