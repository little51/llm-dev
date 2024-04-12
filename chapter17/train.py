import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_model(args):
    model_config = transformers.modeling_gpt2.GPT2Config.\
        from_json_file(
            args.model_config)
    print('模型配置参数:\n' + model_config.to_json_string())
    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(
            config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.\
            from_pretrained(
                args.pretrained_model)
    model.train()
    model.to(device)
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('参数个数: {}'.format(num_parameters))
    return model, model_config


def init_variable(args, model_config):
    n_ctx = model_config.n_ctx
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    multi_gpu = False
    full_len = 0
    overall_step = 0
    running_loss = 0
    return n_ctx, tb_writer, multi_gpu, full_len, \
        overall_step, running_loss


def calc_total_steps(args, full_len):
    for i in tqdm(range(args.num_pieces)):
        with open(args.tokenized_data_path +
                  'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for
                             item in f.read().strip().split()])
    total_steps = int(full_len / args.stride * args.epochs /
                      args.batch_size / args.gradient_accumulation)
    print('总步数 = {}'.format(total_steps))
    return total_steps


def fp16_support(args, model, total_steps):
    optimizer = transformers.AdamW(
        model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_steps,
        t_total=total_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install apex from https://github.com/nvidia/apex")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
    return optimizer, scheduler, model


def train(args):
    # 初始化模型参数
    model, model_config = init_model(args)
    # 初始化变量
    n_ctx, tb_writer, multi_gpu, full_len, overall_step, running_loss = \
        init_variable(args, model_config)
    assert args.log_step % args.gradient_accumulation == 0
    # 计算总步数
    total_steps = calc_total_steps(args, full_len)
    # fp16支持
    optimizer, scheduler, model = fp16_support(args, model, total_steps)
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[
                             int(i) for i in device.split(',')])
        multi_gpu = True
    print('开始训练')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for epoch in range(args.epochs):
        print('轮次 {}'.format(epoch + 1))
        now = datetime.now()
        print('时间: {}'.format(now))
        x = np.linspace(0, args.num_pieces - 1,
                        args.num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(args.tokenized_data_path +
                      'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += args.stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // args.batch_size):
                # 准备数据
                batch = samples[step *
                                args.batch_size: (step + 1)
                                * args.batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(
                    batch_inputs).long().to(device)
                # 前向推理，计算损失函数
                outputs = model.forward(
                    input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]
                # 获取损失函数
                if multi_gpu:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                # 反向传播，计算当前梯度
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer),
                            max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                # 根据梯度更新网络参数
                if (overall_step + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % args.log_step == 0:
                    tb_writer.add_scalar(
                        'loss', loss.item() *
                        args.gradient_accumulation, overall_step)
                    print('时间: {}:{}. 步数 {} of \
                        分片 {} of 轮次 {}, 损失 {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * args.gradient_accumulation /
                        (args.log_step / args.gradient_accumulation)))
                    running_loss = 0
                overall_step += 1
            piece_num += 1
        # 保存检查点
        print('保存模型，轮次 {}'.format(epoch + 1))
        if not os.path.exists(args.output_dir +
                              'model_epoch{}'.format(epoch + 1)):
            os.mkdir(args.output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module \
            if hasattr(model, 'module') else model
        model_to_save.save_pretrained(
            args.output_dir + 'model_epoch{}'.format(epoch + 1))
        print('轮次 {} 完成'.format(epoch + 1))
        then = datetime.now()
        print('时间: {}'.format(then))
        print('本轮用时: {}'.format(then - now))
    # 保存最终模型
    if not os.path.exists(args.output_dir + 'final_model'):
        os.mkdir(args.output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir + 'final_model')
    print('训练完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0',
                        type=str, required=False, help='推理卡列表')
    parser.add_argument('--model_config',
                        default='config/model_config_small.json',
                        type=str, required=False,
                        help='模型参数配置文件')
    parser.add_argument('--tokenized_data_path', default='tokenized/',
                        type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='训练轮次')
    parser.add_argument('--batch_size', default=8, type=int,
                        required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float,
                        required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000,
                        type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int,
                        required=False,
                        help='汇报loss步数,gradient accumulation整数倍')
    parser.add_argument('--stride', default=768, type=int,
                        required=False,
                        help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1,
                        type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true',
                        help='fp16混合精度')
    parser.add_argument('--fp16_opt_level', default='O1',
                        type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)
    parser.add_argument('--num_pieces', default=100,
                        type=int, required=False,
                        help='tokenized分片数')
    parser.add_argument('--output_dir', default='model/',
                        type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='',
                        type=str, required=False,
                        help='训练模型检查点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/',
                        type=str, required=False,
                        help='Tensorboard路径')

    args = parser.parse_args()
    train(args)
