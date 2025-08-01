import os
import argparse
import torch
from model import Model
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help="'train' or 'test'")
    parser.add_argument('--model_save_path', type=str, default='models',
                        help='Base directory for saving the models')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device id, e.g., cuda:0 or cpu')
    parser.add_argument('--out_csv', type=str, default='testing_variances.csv',
                        help='specify uncertainty csv name')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='specify learning rate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional checkpoint to load in test mode')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='specify data directory')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model_save_path = os.path.join('model', args.model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

    log_dir = os.path.join('logs', args.model_save_path)
    os.makedirs(log_dir, exist_ok=True)

    if args.mode == 'test' and args.checkpoint is None:
        args.checkpoint = os.path.join(model_save_path, 'model.pth')

    model = Model(mode=args.mode).to(device)
    
    data_dir = args.data_dir
    out_csv = args.out_csv
    lr = args.lr
    
    solver = Solver(model=model,out_csv=out_csv,data_dir=data_dir,
                    model_save_path=model_save_path,lr=lr,
                    log_dir=log_dir,
                    
                    device=device)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test(checkpoint=args.checkpoint,output_csv=out_csv)
    else:
        print("Unrecognized mode.")

if __name__ == '__main__':
    main()
