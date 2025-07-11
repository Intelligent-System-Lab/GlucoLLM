"""
Gluco-LLM: Glucose Time Series Forecasting with Large Language Models

This file is part of a project based on Time-LLM (https://github.com/KimMeen/Time-LLM)
Original code licensed under Apache License 2.0.
Modifications and glucose-specific implementations copyright (c) 2025 Gluco-LLM Research Team

Time-LLM: Time Series Forecasting by Reprogramming Large Language Models
Paper: https://arxiv.org/abs/2310.01728
Repository: https://github.com/KimMeen/Time-LLM
"""

from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        # === Add: Store age and gender from configs ===
        self.age = getattr(configs, 'age', None)
        self.gender = getattr(configs, 'gender', None)
        # === End add ===
        # Add time_features_count from configs
        self.time_features_count = configs.time_features_count  # Get the number of time features from configs

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    '/mmfs1/projects/j.li/llama_7b_lqrft/',
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
                    '/mmfs1/projects/j.li/llama_7b_lqrft/',
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
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    '/mmfs1/projects/j.li/gpt2_lqrft',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    '/mmfs1/projects/j.li/gpt2_lqrft',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
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
        elif configs.llm_model == 'DEEPSEEK':
            model_name = 'deepseek-ai/deepseek-coder-1.3b-base'
            print(f"Using device: {self.device}")
            
            self.deepseek_config = AutoConfig.from_pretrained(model_name)
            self.deepseek_config.num_hidden_layers = configs.llm_layers
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            
            self.d_llm = self.deepseek_config.hidden_size
            
            try:
                print("Attempting to load Deepseek model...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.deepseek_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to(self.device)
                print(f"Deepseek model loaded successfully! Model dimension: {self.d_llm}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

            try:
                print("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False
                )
                print("Tokenizer loaded successfully!")
            except Exception as e:
                print(f"Error loading tokenizer: {str(e)}")
                raise
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

        self.description = configs.content

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.feature_names = configs.feature_names

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc, dim=1).values
            lags = self.calcute_lags(x_enc)
            trends = x_enc.diff(dim=1).sum(dim=1)
            
            current_features = x_mark_enc[b, -1, :]
            for i, name in enumerate(self.feature_names):
                feature_idx = self.time_features_count + i
                value = current_features[feature_idx].item()
            
            additional_info = self._format_additional_features(current_features)
            
            # === Age and gender info ===
            age_info = f"Age: {self.age if self.age is not None else 'Unknown'}"
            gender_info = f"Gender: {self.gender if self.gender is not None else 'Unknown'}"

            prompt_ = (
                f"<|start_prompt|>\n"
                f"Dataset description: {self.description}"
                f"\nPatient information: {age_info}, {gender_info}\n"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information.\n"
                f"Input statistics:\n"
                f"- min value: {min_values[b].tolist()[0]:.3f}\n"
                f"- max value: {max_values[b].tolist()[0]:.3f}\n"
                f"- median value: {medians[b].tolist()[0]:.3f}\n"
                f"- trend: {'upward' if trends[b] > 0 else 'downward'}\n"
                f"- top 5 lags: {str(lags[b].tolist())}\n"
                f"Current status: {additional_info}\n"
                f"<|end_prompt|>"
            )
            # === Add: Print prompt ===
            print(prompt_)
            # === End add ===
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # Handle different model outputs: DeepSeek uses .logits, others use .last_hidden_state
        outputs = self.llm_model(inputs_embeds=llama_enc_out)
        if hasattr(self, 'deepseek_config'):  # DeepSeek model
            dec_out = outputs.logits
        else:  # LLAMA, GPT2, BERT models
            dec_out = outputs.last_hidden_state
            
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def _format_additional_features(self, features_data):
        """Format additional features into prompt text following specific rules"""
        info_parts = []
        
        # Get feature values from features_data using index mapping
        feature_values = {}
        for i, name in enumerate(self.feature_names):
            feature_idx = self.time_features_count + i
            value = features_data[feature_idx].item() if torch.is_tensor(features_data[feature_idx]) else features_data[feature_idx]
            feature_values[name] = value
        
        # Format basal insulin information
        if feature_values['basal'] is not None and feature_values['basal'] > 0:
            info_parts.append(
                f"\n- Current basal insulin injection amount: {feature_values['basal']:.2f}. "
            )
        else:
            info_parts.append("\n- No basal insulin injection now. ")
            if feature_values['TimePassedLastBasal'] != -1:
                info_parts.append(
                    f"\n- Last basal injection: {feature_values['TimePassedLastBasal']:.0f} minutes ago. "
                )
        
        # Format bolus insulin information
        if feature_values['bolus'] is not None and feature_values['bolus'] > 0:
            info_parts.append(
                    f"\n- Current bolus insulin injection amount: {feature_values['bolus']:.2f}. "
            )
        else:
            info_parts.append("\n- No bolus insulin injection now. ")
            if feature_values['TimePassedLastBolus'] != -1:
                info_parts.append(
                        f"\n- Last bolus injection: {feature_values['TimePassedLastBolus']:.0f} minutes ago. "
                )
        
        # Format meal information based on MealCarbs
        if feature_values['MealCarbs'] is not None and feature_values['MealCarbs'] > 0:
            info_parts.append(
                f"\n- Current meal carbohydrates of {feature_values['MealCarbs']:.2f}. "
            )
        else:
            info_parts.append("\n- No meal taken now. ")
            if feature_values['TimePassedLastMeal'] != -1:
                info_parts.append(
                        f"\n- Last meal taken: {feature_values['TimePassedLastMeal']:.0f} minutes ago. "
                )
        
        return "".join(info_parts)


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
