import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import datetime
import random
import numpy as np
import os

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

# Windows 显存管理优化
os.environ['CURL_CA_BUNDLE'] = ''
# 消除 AllocatorConfig 警告，使用新版写法
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def vali(model, vali_loader, criterion, args, accelerator):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # --- Late Fusion: Remove covariate dimensions from batch_y before loss calculation ---
            if getattr(args, 'cov_dim', 0) > 0 and args.features == 'M':
                batch_y = batch_y[:, :, args.cov_dim:]
            # -----------------------------------------------------------------------------------

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)
            total_loss.append(loss)
            
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def test(model, test_loader, args, accelerator):
    preds = []
    trues = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # Encoder - Decoder 推理
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            preds.append(outputs)
            trues.append(batch_y)

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    
    target_dim = preds.shape[-1]
    
    print(f"【测试结果】预测矩阵形状: {preds.shape} (预期目标特征数应为 {args.c_out})")
    print(f"【测试结果】真实矩阵形状: {trues.shape}")

    # =========================================================================
    # 第一步：计算标准化 (Standardized) 指标 (学术论文对比标准)
    # =========================================================================
    preds_std = preds[:, :, :target_dim]
    trues_std = trues[:, :, :target_dim]
    std_mae, std_mse, std_rmse, std_mape, std_mspe, std_r2 = metric(preds_std, trues_std)

    # =========================================================================
    # 第二步：逆归一化 (Inverse Transform)
    # =========================================================================
    print("【系统】正在生成真实刻度结果 (Inverse Transform)...")
    
    if test_loader.dataset.scale:
        # 1. 独立处理预测值 preds
        pad_dim_p = args.enc_in - preds.shape[-1]
        if pad_dim_p > 0:
            pad_zeros_p = np.zeros((preds.shape[0], preds.shape[1], pad_dim_p))
            preds_padded = np.concatenate([preds, pad_zeros_p], axis=-1)
        else:
            preds_padded = preds
            
        # 2. 独立处理真实值 trues
        pad_dim_t = args.enc_in - trues.shape[-1]
        if pad_dim_t > 0:
            pad_zeros_t = np.zeros((trues.shape[0], trues.shape[1], pad_dim_t))
            trues_padded = np.concatenate([trues, pad_zeros_t], axis=-1)
        else:
            trues_padded = trues
            
        # 3. 将 3D 转为 sklearn 需要的 2D 数组
        preds_padded_2d = preds_padded.reshape(-1, preds_padded.shape[-1])
        trues_padded_2d = trues_padded.reshape(-1, trues_padded.shape[-1])
        
        # 4. 执行逆归一化
        preds_real_2d = test_loader.dataset.inverse_transform(preds_padded_2d)
        trues_real_2d = test_loader.dataset.inverse_transform(trues_padded_2d)
        
        # 5. 从 2D 还原回 3D 形状
        preds_real = preds_real_2d.reshape(preds_padded.shape)
        trues_real = trues_real_2d.reshape(trues_padded.shape)
    else:
        preds_real = preds
        trues_real = trues

    # 对齐切片区：统一切除多余的协变量维度
    preds_real = preds_real[:, :, :target_dim]
    trues_real = trues_real[:, :, :target_dim]

    # =========================================================================
    # 第三步：计算真实刻度 (Real Scale) 指标
    # =========================================================================
    real_mae, real_mse, real_rmse, real_mape, real_mspe, real_r2 = metric(preds_real, trues_real)
    
    print("------------------------------------")
    print(f"【系统】学术论文标准 (标准化)   -> MSE:{std_mse:.4f}, MAE:{std_mae:.4f}")
    print(f"【系统】业务评估指标 (真实刻度) -> MSE:{real_mse:.4f}, MAE:{real_mae:.4f}, R2:{real_r2:.4f}")
    print("------------------------------------")
    
    # 保存真实预测结果供后续画图使用
    folder_path = './test_results/' + args.model_id + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path + 'pred_original.npy', preds_real)
    np.save(folder_path + 'true_original.npy', trues_real)

    # 最终返回值：返回标准的 MAE/MSE 供原代码其它部分可能的使用，同时附带真实的 R2
    return std_mae, std_mse, std_rmse, std_mape, std_mspe, real_r2

def main():
    parser = argparse.ArgumentParser(description='Time-LLM for EV Load Forecasting')

    # Basic Config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast', help='task name')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='TimeLLM', help='model name')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # Data Loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset loader type')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--percent', type=int, default=100, help='percent of training data') # 修复: 补回缺失的参数

    # Forecasting Task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # Model Define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--cov_dim', type=int, default=7, help='dimension of covariates in data, placed at the front of columns')
    #parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True) # 修复: 补回缺失的参数
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='used in prompt_bank')
    #parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model selection')
    parser.add_argument('--llm_model', type=str, default='Qwen/Qwen2.5-32B-Instruct-AWQ', help='LLM model path or name')
    #parser.add_argument('--llm_dim', type=int, default=768, help='LLM model dimension')
    parser.add_argument('--llm_dim', type=int, default=5120, help='LLM model dimension (Qwen2.5-32B-Instruct-AWQ is 5120)')
    #parser.add_argument('--llm_layers', type=int, default=6, help='number of LLM layers')
    parser.add_argument('--llm_layers', type=int, default=14, help='LLM model layers')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--llm_skip', type=int, default=1, help='skip LLM layers')
    
    # Custom / Windows Specific
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive pause after each epoch')
    parser.add_argument('--causal_prompt_path', type=str, default=None, help='Path to causal prompt text file')

    # 兼容 DeepSpeed 自动传入的参数
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DeepSpeed')
    parser.add_argument('--deepspeed', type=str, default='', help='DeepSpeed config file path')

    args = parser.parse_args()
    
    # Load Causal Prompt if provided
    args.causal_prompt = ""
    if args.causal_prompt_path and os.path.exists(args.causal_prompt_path):
        try:
            with open(args.causal_prompt_path, 'r', encoding='utf-8') as f:
                args.causal_prompt = f.read().strip()
            print(f"【系统】已加载因果Prompt: {len(args.causal_prompt)} 字符")
        except Exception as e:
            print(f"【⚠️警告】无法读取因果Prompt文件: {e}")

    # NEW: Load General Prompt (Content) for Time-LLM
    args.content = "Time Series Forecasting" # Default
    if args.model == 'TimeLLM':
        args.prompt_domain = 1 # Force enable prompt domain
        prompt_path = os.path.join('./dataset/prompt_bank', args.data + '.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                args.content = f.read().strip()
            print(f"【系统】已加载通用Prompt (Domain): {len(args.content)} 字符")
        else:
            print(f"【⚠️警告】未找到通用Prompt文件: {prompt_path}, 使用默认描述。")
            
    # 强制修正 LLM 维度
    if args.llm_model == 'GPT2':
        args.llm_dim = 768
    elif args.llm_model == 'LLAMA':
        # Llama-7B hidden size is 4096
        args.llm_dim = 4096
    elif args.llm_model == 'BERT':
        # BERT-base hidden size is 768
        args.llm_dim = 768
    elif '27b' in args.llm_model.lower() or '32b' in args.llm_model.lower():
        args.llm_dim = 5120

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    for ii in range(args.itr):
        # Setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.distil, args.des, ii)

        if args.model == 'TimeLLM':
            # Append cluster/prompt info to setting so checkpoints don't collision if model_id is not unique
            if args.causal_prompt_path:
                 cluster_suffix = os.path.basename(args.causal_prompt_path).replace('.txt', '')
                 setting += f"_{cluster_suffix}"

        if args.is_training:
            print(f"【系统】正在加载训练数据 (Data Provider)...")
        train_data, train_loader = data_provider(args, flag='train')
        print(f"【系统】训练数据加载完成。")
        
        if args.is_training:
             print(f"【系统】正在加载验证数据...")
        vali_data, vali_loader = data_provider(args, flag='val')
        print(f"【系统】正在加载测试数据...")
        test_data, test_loader = data_provider(args, flag='test')
        print(f"【系统】所有数据加载完毕。")

        if args.model == 'TimeLLM':
            print(f"【系统】正在初始化 Time-LLM 模型 (Backbone: {args.llm_model})...")
            if args.llm_model == 'GPT2' and os.path.exists('./gpt2'):
                print(f"【系统】检测到本地 GPT2 模型文件夹，将优先从 ./gpt2 加载")
            model = TimeLLM.Model(args)
            print(f"【系统】模型初始化完成。")
        else:
            model = eval(args.model).Model(args).float()

        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)


        model, model_optim, train_loader, vali_loader, test_loader, scheduler = accelerator.prepare(
            model, model_optim, train_loader, vali_loader, test_loader, scheduler)

        criterion = nn.MSELoss()

        # Resume logic
        start_epoch = 0
        if args.resume:
            state_path = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(state_path):
                print(f"【系统】正在从 {state_path} 恢复训练状态...")
                checkpoint = torch.load(state_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"【系统】恢复成功，从第 {start_epoch + 1} 轮开始继续训练。")
            else:
                print("【系统】未找到 Checkpoint，将开始新的训练。")

        if args.is_training:
            for epoch in range(start_epoch, args.train_epochs):
                iter_count = 0
                train_loss = []

                model.train()
                epoch_time = time.time()
                
                # Progress bar
                progress_bar = tqdm(train_loader, desc=f"[{args.model_id}] Epoch {epoch + 1}/{args.train_epochs}", disable=not accelerator.is_local_main_process)
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(progress_bar):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(accelerator.device)
                    batch_y = batch_y.float().to(accelerator.device)
                    batch_x_mark = batch_x_mark.float().to(accelerator.device)
                    batch_y_mark = batch_y_mark.float().to(accelerator.device)

                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            if args.output_attention:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                            f_dim = -1 if args.features == 'MS' else 0
                            outputs = outputs[:, -args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                            
                            # --- Late Fusion: Remove covariate dimensions from batch_y ---
                            if getattr(args, 'cov_dim', 0) > 0 and args.features == 'M':
                                batch_y = batch_y[:, :, args.cov_dim:]
                            # -----------------------------------------------------------
                            
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        
                        # --- Late Fusion: Remove covariate dimensions from batch_y ---
                        if getattr(args, 'cov_dim', 0) > 0 and args.features == 'M':
                            batch_y = batch_y[:, :, args.cov_dim:]
                        # -----------------------------------------------------------
                        
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    accelerator.backward(loss)
                    model_optim.step()

                    if args.lradj == 'TST':
                        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                        scheduler.step()

                    progress_bar.set_postfix(loss=loss.item())

                # End of epoch ops
                train_loss = np.average(train_loss)
                vali_loss = vali(model, vali_loader, criterion, args, accelerator)
                test_loss = vali(model, test_loader, criterion, args, accelerator)

                print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss))
                
                early_stopping(vali_loss, model, path)
                
                # Save Checkpoint
                state_path = os.path.join(path, 'checkpoint.pth')
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                }
                torch.save(state, state_path)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if args.lradj != 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args)
                
                # Clear CUDA Cache
                torch.cuda.empty_cache()

                # Interactive Pause Logic
                if args.interactive:
                    print("\n" + "="*60)
                    print(f"【暂停】Epoch {epoch + 1}/{args.train_epochs} 已完成。")
                    print(f"当前时间: {datetime.datetime.now().strftime('%H:%M:%S')}")
                    print(f"Train Loss: {train_loss:.5f} | Vali Loss: {vali_loss:.5f}")
                    print(">>> 请确认上述指标正常。")
                    print(">>> 按 [回车键] 继续下一轮训练 (或按 Ctrl+C 终止)...")
                    print("="*60)
                    try:
                        user_input = input("按 [回车键] 继续下一轮，或输入 'test' 并回车以运行测试并查看结果: ")
                        if user_input.strip().lower() == 'test':
                             try:
                                 print("\n【系统】正在运行测试集评估 (使用当前Epoch模型)...")
                                 test(model, test_loader, args, accelerator)
                                 print("【系统】测试完成，结果已保存。")
                                 print("【提示】您现在可以检查 pred.npy 或运行 export_results.py 查看详细指标。")
                             except Exception as e:
                                 print(f"\n【错误】测试过程中发生异常: {str(e)}")
                                 import traceback
                                 traceback.print_exc()
                             
                             input("按 [回车键] 继续训练...")
                    except KeyboardInterrupt:
                        print("\n【系统】用户中断训练。")
                        exit()
                    except:
                        pass

            # Training finished, load best model for test
            best_model_path = path + '/' + 'checkpoint.pth'
            model.load_state_dict(torch.load(best_model_path)['model_state_dict'])

            print("------------------------------------")
            test_results = test(model, test_loader, args, accelerator)
            print("------------------------------------")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()