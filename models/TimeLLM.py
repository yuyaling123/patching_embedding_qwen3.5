from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        # Note: nf will be dynamically sized based on whether we use covariates
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # --- Dual-Patch Fusion: Covariate configurations ---
        self.cov_dim = getattr(configs, 'cov_dim', 0)
        self.main_dim = configs.enc_in - self.cov_dim
        # ---------------------------------------------------

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('./gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    './gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    './gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    './gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    './gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        # ===== Qwen 分支 =====
        elif 'qwen' in configs.llm_model.lower():
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
            import torch
            import gc
            
            print(f"Loading Qwen model from {configs.llm_model}...")
            self.llm_config = AutoConfig.from_pretrained(
                configs.llm_model, 
                trust_remote_code=True
            )
            
            # 【终极防爆显存：强制层数上限】
            # 14层在 16GB V100 上依然会占满 14.7GB 导致 OOM。强行限制最高 8 层！
            if configs.llm_layers > 8:
                print(f"【⚠️安全警告】配置的层数 {configs.llm_layers} 会导致 16GB 显存 OOM。已自动为您安全截断至 8 层！")
                configs.llm_layers = 8
            
            # 【截断核心 1】：强行修改 Config，只保留用户指定的安全层数
            self.llm_config.num_hidden_layers = configs.llm_layers
            
            # 清除之前的碎片以防万一
            gc.collect()
            torch.cuda.empty_cache()
            
            # 纯净的 4-bit 量化，专供 V100
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 【截断核心 2】：必须把修改后的 config 传进去！
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                configs.llm_model,
                config=self.llm_config,             # <--- 必须加上这一行才能真正截断
                trust_remote_code=True,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                device_map={"": 0},                 # <--- 【硬核修复】：禁止 auto 瞎分配！强行锁死在 GPU 0 上
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                configs.llm_model,
                trust_remote_code=True
            )
            
            # 对齐 Padding Token 以免报错
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm_model.config.pad_token_id = self.tokenizer.pad_token_id
                
            # 获取 Qwen 的 hidden_size 供后续特征投影使用
            self.llm_dim = self.llm_config.hidden_size 
        # ====================================
        
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        
        # New: Causal Prompt Support
        self.causal_prompt = getattr(configs, 'causal_prompt', '')

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding_main = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        self.patch_embedding_cov = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout) if self.cov_dim > 0 else None

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        
        # --- Dual-Patch Fusion: Covariate fusion layer ---
        self.cov_fusion = nn.Sequential(
            nn.Linear(2 * configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model)
        )
        
        # --- Attention Pooling for Covariates ---
        self.cov_attn_pool = nn.Sequential(
            nn.Linear(configs.d_model, 1)
        ) if self.cov_dim > 0 else None
            
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(self.main_dim, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_main = Normalize(self.main_dim, affine=False)
        self.normalize_cov = Normalize(self.cov_dim, affine=False) if self.cov_dim > 0 else None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, N_total = x_enc.shape
        
        # --- Dual-Patch Fusion: Split covariates from main sequence ---
        if self.cov_dim > 0:
            x_cov = x_enc[:, :, :self.cov_dim].clone()
            x_main = x_enc[:, :, self.cov_dim:].clone()
        else:
            x_cov = None
            x_main = x_enc.clone()
        # --------------------------------------------------------------
        
        # Normalize main and covariate sequences separately
        x_main = self.normalize_main(x_main, 'norm')
        if self.cov_dim > 0:
            x_cov = self.normalize_cov(x_cov, 'norm')

        N = x_main.shape[2]
        x_enc_flat = x_main.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # [OPTIMIZATION] Removed verbose stats (min/max/med/lags/trend) to save tokens & compute.
        prompt = []
        for b in range(x_enc_flat.shape[0]):
            # Simplified Prompt
            prompt_content = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
            )
            
            if self.causal_prompt:
                prompt_content += f" Causal Relations: {self.causal_prompt}"
            
            # Additional context if needed, but keeping it minimal:
            # prompt_content += "Input statistics: omitted for efficiency."
            
            prompt_content += "<|end_prompt|>"
            
            prompt.append(prompt_content)
            


        x_enc_flat = x_enc_flat.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=950).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_main.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_main = x_main.permute(0, 2, 1).contiguous()
        main_tokens, n_main = self.patch_embedding_main(x_main)
        
        if self.cov_dim > 0:
            x_cov = x_cov.permute(0, 2, 1).contiguous()
            cov_tokens, n_cov = self.patch_embedding_cov(x_cov)
            
            D = main_tokens.shape[-1]
            main_tokens_reshaped = main_tokens.reshape(B, self.main_dim, self.patch_nums, D)
            cov_tokens_reshaped = cov_tokens.reshape(B, self.cov_dim, self.patch_nums, D)
            
            attn_score = self.cov_attn_pool(cov_tokens_reshaped)         # [B, N_cov, P, 1]
            attn_weight = torch.softmax(attn_score, dim=1)
            cov_context = (cov_tokens_reshaped * attn_weight).sum(dim=1) # [B, P, D]
            
            cov_context_expand = cov_context.unsqueeze(1).expand(-1, self.main_dim, -1, -1)
            
            fused_tokens = torch.cat([main_tokens_reshaped, cov_context_expand], dim=-1)
            fused_tokens = self.cov_fusion(fused_tokens)
            
            fused_tokens = fused_tokens.reshape(B * self.main_dim, self.patch_nums, D)
            n_vars = self.main_dim
        else:
            fused_tokens = main_tokens
            n_vars = n_main

        enc_out = self.reprogramming_layer(fused_tokens, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # Safe Chunking Logic (Preserved from User Edit)
        dec_out_list = []
        chunk_size = 4 
        for i in range(0, llama_enc_out.shape[0], chunk_size):
            chunk_input = llama_enc_out[i : i + chunk_size]
            chunk_out = self.llm_model(inputs_embeds=chunk_input).last_hidden_state
            dec_out_list.append(chunk_out)
        
        dec_out = torch.cat(dec_out_list, dim=0)
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        dec_out = dec_out[:, :, :, -self.patch_nums:]
        
        dec_out = self.output_projection(dec_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_main(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
