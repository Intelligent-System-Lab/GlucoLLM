# Gluco-LLM: Glucose Time Series Forecasting with Large Language Models

A specialized implementation for glucose time series forecasting based on the Time-LLM framework, designed for diabetes management applications.


## Overview

This project extends the [Time-LLM](https://github.com/KimMeen/Time-LLM) framework to specifically handle glucose time series forecasting. It leverages large language models (GPT2, LLAMA, Deepseek) to predict future glucose levels based on historical measurements and additional features like insulin injections and meal information.


## Features

- **Multi-Model Support**: Compatible with GPT2, LLAMA, and Deepseek models
- **Glucose-Specific Features**: Handles historical measurements and additional features like insulin injections and meal information
- **Time Series Forecasting**: Predicts glucose levels for diabetes management
- **Flexible Architecture**: Supports both training and inference modes
- **Comprehensive Evaluation**: Includes MAE, RMSE, and other metrics


## Installation

```bash
pip install -r requirements.txt
```


## Usage

### Training Mode Example
```bash
  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 588_train.csv \
  --test_data_path 588_test.csv \
  --model_id 588 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len $s_len \
  --label_len $l_len \
  --pred_len $p_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
```

### Testing/Inference Mode Example
```bash
  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_glucose.py \
  --task_name short_term_forecast \
  --is_training 0 \
  --separate_test yes \
  --root_path ./dataset/glucose/ \
  --data_path 588_train.csv \
  --test_data_path 588_test.csv \
  --model_id 588 \
  --model $model_name \
  --data Glucose \
  --features S \
  --seq_len $s_len \
  --label_len $l_len \
  --pred_len $p_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
```


## Data Format

The project expects glucose data in CSV format with the following columns:
- `ts`: Timestamp
- `glucose`: Target glucose values
- Additional features: insulin injection data, meal information, etc.


## Model Architecture

The model combines:
1. **Time Series Encoding**: Converts glucose time series into patch embeddings
2. **LLM Integration**: Uses large language models for sequence processing
3. **Prompt Engineering**: Incorporates domain-specific prompts for glucose forecasting
4. **Feature Fusion**: Combines time series data with additional medical features


## Results

The model outputs predictions in the `./results/` directory with detailed metrics including:
- Average prediction error
- Last prediction error
- MAE and RMSE metrics


## Configuration

Key parameters:
- `--seq_len`: Input sequence length
- `--pred_len`: Prediction horizon
- `--llm_model`: Choice of LLM (GPT2, LLAMA, BERT)
- `--llm_layers`: Number of LLM layers to use


## Citation

If you use this code in your research, please cite both this project and the original Time-LLM paper:

```bibtex
@misc{li2024glucollm,
  title={Gluco-LLM: LLM-Powered Personalized Glucose Prediction in Type 1 Diabetes},
  author={qingrui.li, Kapileshwor Ray Amat, and Juan Li},
  year={2025},
  url={https://github.com/your-username/gluco-LLM}
}

@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Acknowledgement

This project is developed based on [Time-LLM](https://github.com/KimMeen/Time-LLM), which is licensed under the Apache License, Version 2.0.

**Original Time-LLM Paper:**
Jin, Ming, et al. "Time-LLM: Time series forecasting by reprogramming large language models." International Conference on Learning Representations (ICLR), 2024. [arxiv.org/abs/2310.01728](https://arxiv.org/abs/2310.01728)

**Original Repository:** https://github.com/KimMeen/Time-LLM

This project retains substantial portions of the TimeLLM codebase for time series forecasting with large language models. All modifications and new contributions specific to glucose forecasting are copyright (c) 2025 Gluco-LLM Research Team.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Contact

For questions and issues, please open an issue on GitHub. 
