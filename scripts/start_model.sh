#!/bin/bash
model-path=$1
# # 启动第一组服务
python3 -m fastchat.serve.controller --port 11001 &
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11001 --model-path $1 --worker-address http://localhost:11002 --port 11002 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11001 --host localhost --port 11003 & 

python3 -m fastchat.serve.controller --port 11004 &
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11004 --model-path $1 --worker-address http://localhost:11005 --port 11005 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11004 --port 11006 &

# 启动第二组服务
python3 -m fastchat.serve.controller --port 11007 &
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11007 --model-path $1 --worker-address http://localhost:11008 --port 11008 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11007 --port 11009

python3 -m fastchat.serve.controller --port 11011 &
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11011 --model-path $1 --worker-address http://localhost:11012 --port 11012 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11011 --port 11013 &

python3 -m fastchat.serve.controller --port 11014 &
CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11014 --model-path $1 --worker-address http://localhost:11015 --port 11015 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11014 --port 11016 &

python3 -m fastchat.serve.controller --port 11017 &
CUDA_VISIBLE_DEVICES=5 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11017 --model-path $1 --worker-address http://localhost:11018 --port 11018 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11017 --port 11019 &

python3 -m fastchat.serve.controller --port 11021 &
CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11021 --model-path $1 --worker-address http://localhost:11022 --port 11022 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11017 --port 11023 &

python3 -m fastchat.serve.controller --port 11024 &
CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11024 --model-path $1 --worker-address http://localhost:11025 --port 11025 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11024 --port 11026 &