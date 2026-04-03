@echo off
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%PYTHONPATH%;.

:: ================= CONFIGURATION =================
:: Input Variables: 88 Stations + 7 Weather/Price = 95
set INPUT_VARS=86
:: =================================================

echo ========================================================
echo Starting Time-LLM for EV Load Forecasting (Qwen3.5-27B Version)
echo Workspace: WORKSPACE/timellm_improving/patching_embedding
echo Model Dir: ../local_models/Qwen3.5-27B
echo --------------------------------------------------------

D:\Anaconda3\envs\timellm\python.exe run_main.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/EV_Data/ ^
  --data_path EV_Load_Cleaned.csv ^
  --model_id EV_Qwen_Test ^
  --model TimeLLM ^
  --data EV_Data ^
  --features M ^
  --seq_len 96 ^
  --label_len 24 ^
  --pred_len 24 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in %INPUT_VARS% ^
  --dec_in %INPUT_VARS% ^
  --c_out %INPUT_VARS% ^
  --target Station_10 ^
  --batch_size 1 ^
  --learning_rate 0.0001 ^
  --llm_model ../local_models/Qwen3.5-27B ^
  --llm_dim 5120 ^
  --llm_layers 64

pause