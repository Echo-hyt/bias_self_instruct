#!/bin/bash

# # 启动第一组服务
python3 -m fastchat.serve.controller --port 11001 &
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11001 --model-path /data/workspace/huangyuting/google/gemma-7b/ --worker-address http://localhost:11002 --port 11002 &
python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11001 --host localhost --port 11003 & 

# python3 -m fastchat.serve.controller --port 11004 &
# CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11004 --model-path /data/workspace/huangyuting/google/gemma-7b/ --worker-address http://localhost:11005 --port 11005 &
# python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11004 --port 11006 &

# # 启动第二组服务
# python3 -m fastchat.serve.controller --port 11007 &
# CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --controller-address http://localhost:11007 --model-path /data/workspace/huangyuting/google/gemma-7b/ --worker-address http://localhost:11008 --port 11008 &
# python3 -m fastchat.serve.openai_api_server --controller-address http://localhost:11007 --port 11009
