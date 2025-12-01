import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from dataloader import HSTDataset
from framework import HSTPOI
from tools import get_mapper, get_config, update_config, custom_collate, train_epoch, run_test

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=256)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument('--dataset', type=str, default='GW')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--dim', type=int, default=8, help='must be a multiple of 4')
parser.add_argument('--graph', type=int, default=1, help='Use Hypergraph or not')
parser.add_argument('--at', type=str, default='attn', help='arrival time module type')
parser.add_argument('--encoder', type=str, default='LWtrans', help='encoder type')
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--epoch', type=int, default=70, help='epoch num')
args = parser.parse_args()

# 环境
gpu_list = args.gpu
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
dataset_path = f'./data/{args.dataset}'
timestamp = time.strftime("%Y%m%d%H%M", time.localtime())

# 保存
save_path = args.dataset
save_dir = f"./configuration/{save_path}"
config_path = f"{save_dir}/settings.yml"
device = torch.device("cuda")

test_only = args.test

if __name__ == '__main__':
    get_mapper(dataset_path=dataset_path)

    # 更新
    update_config(config_path, key_list=['Model', 'use_graph_net'], value=args.graph)
    update_config(config_path, key_list=['Model', 'seed'], value=args.seed)
    update_config(config_path, key_list=['Model', 'at_type'], value=args.at)
    update_config(config_path, key_list=['Embedding', 'base_dim'], value=args.dim)
    update_config(config_path, key_list=['Encoder', 'encoder_type'], value=args.encoder)
    update_config(config_path, key_list=['Model', 'batch_size'], value=args.batch)
    update_config(config_path, key_list=['Model', 'epoch'], value=args.epoch)
    config = get_config(config_path, easy=True)

    # 初始化
    dataset = HSTDataset(config=config, dataset_path=dataset_path, device=device, load_mode='train')
    batch_size = config.Model.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                            collate_fn=lambda batch: custom_collate(batch, device, config))
    model = HSTPOI(config)
    model.to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total training samples: {len(dataloader) * batch_size} | Total trainable parameters: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Adam_optimizer.initial_lr,
                                         weight_decay=config.Adam_optimizer.weight_decay)

    print(f"Dataset: {args.dataset} | Device: {device} | Model: {config.Encoder.encoder_type}")
    print(f"AT type: {config.Model.at_type} | Graph type: {config.Model.use_graph_net} | dim: {config.Embedding.base_dim}")

    if test_only:
        save_dir = f'../configuration/{save_path}'
        run_test(dataset_path=dataset_path, model_path=save_dir, model=model, device=device, epoch=69, test_only=test_only)
        exit()

    best_val_loss = float("inf")
    start_time = time.time()
    num_epochs = config.Model.epoch
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader) * 1,
        num_training_steps=len(dataloader) * num_epochs,
    )

    # 报告
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "report.txt"), "w") as report_file:
        print('Train batches:', len(dataloader))
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            average_loss = train_epoch(model, dataloader, optimizer, ce_loss_fn, warmup_scheduler)

            epoch_str = f"================= Epoch [{epoch + 1}/{num_epochs}] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =================\n"

            if average_loss <= best_val_loss:
                epoch_str += f"Best Loss: {best_val_loss:.6f} ---> {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"
                best_val_loss = average_loss
            else:
                epoch_str += f"Best Loss: {best_val_loss:.6f} | Epoch Loss: {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"

            report_file.write(epoch_str + '\n\n')
            report_file.flush()
            print(epoch_str)
            if (epoch+1) % config.Model.test_epoch == 0:
                torch.cuda.synchronize()  # 保证 GPU 计算完成
                t_infer_start = time.time()

                run_test(dataset_path=dataset_path, model_path=save_dir, model=model, device=device, epoch=epoch, test_only=test_only)
                torch.cuda.synchronize()
                t_infer_end = time.time()
                infer_time = t_infer_end - t_infer_start
                print(f"Inference time at epoch {epoch + 1}: {infer_time:.2f} s")

                with open(os.path.join(save_dir, "report.txt"), "a") as f:
                    f.write(f"Inference time at epoch {epoch + 1}: {infer_time:.2f} s\n")
    end_time = time.time()
    total_time = end_time - start_time

    with open(os.path.join(save_dir, "report.txt"), "a") as report_file:
        report_file.write(f"Total Running Time: {total_time:.2f} seconds\n")
    print(f"\nModel done.\n")
