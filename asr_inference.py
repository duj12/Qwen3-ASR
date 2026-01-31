#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Qwen3-ASR批量转录脚本
从wav.scp文件读取音频路径，输出对应的转录文本
"""

import torch
import argparse
from qwen_asr import Qwen3ASRModel
import os


def read_wav_scp(wav_scp_path):
    """
    读取wav.scp文件，返回音频ID到路径的映射
    wav.scp格式: uttid1 /path/to/uttid1.wav
                 uttid2 /path/to/uttid2.wav
    """
    wav_dict = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                uttid = parts[0]
                audio_path = ' '.join(parts[1:])
                wav_dict[uttid] = audio_path
    return wav_dict


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Batch Transcription")
    parser.add_argument("--wav_scp", type=str, required=True, help="Path to wav.scp file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output text file")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B", 
                       help="Path to the Qwen3-ASR model, default is Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for inference, default is 8")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type for model, default is bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0", 
                       help="Device to run the model on, default is cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--language", type=str, default=None, 
                       help="Language for ASR, default is None (will auto-detect)")
    parser.add_argument("--use_vllm", action="store_true", 
                       help="Use vLLM for inference acceleration")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                       help="GPU memory utilization for vLLM, default is 0.7")
    parser.add_argument("--forced_aligner", type=str, default=None,
                       help="Path to forced aligner model")
    parser.add_argument("--return_time_stamps", action="store_true",
                       help="Return time stamps in transcription")
    
    args = parser.parse_args()
    
    # 获取本地模型路径
    model_path = get_local_model_path(args.model_path)
    forced_aligner_path = get_local_model_path(args.forced_aligner) if args.forced_aligner else None
    
    # 映射字符串到torch数据类型
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # 根据是否使用vLLM选择不同的模型加载方式
    if args.use_vllm:
        print(f"Loading model with vLLM from: {model_path}")
        model_kwargs = {
            "model": model_path,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_inference_batch_size": args.batch_size if args.batch_size > 0 else -1,
            "max_new_tokens": args.max_new_tokens,
        }
        
        if forced_aligner_path:
            model_kwargs["forced_aligner"] = forced_aligner_path
            model_kwargs["forced_aligner_kwargs"] = {
                'dtype': dtype,
                'device_map': args.device,
            }
        
        model = Qwen3ASRModel.LLM(**model_kwargs)
    else:
        print(f"Loading model from: {model_path}")
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=args.device,
            max_inference_batch_size=args.batch_size if args.batch_size > 0 else -1,
            max_new_tokens=args.max_new_tokens,
            forced_aligner=forced_aligner_path,
        )
    
    # 读取wav.scp文件
    print(f"Reading wav.scp from: {args.wav_scp}")
    wav_dict = read_wav_scp(args.wav_scp)
    
    # 准备音频路径列表和ID列表
    uttids = list(wav_dict.keys())
    audio_paths = list(wav_dict.values())
    
    print(f"Total {len(uttids)} utterances to process")
    
    # 根据语言参数决定是否为每个音频指定语言
    if args.language:
        # 如果指定了统一语言，则所有音频都使用该语言
        languages = [args.language] * len(audio_paths)
    else:
        # 如果未指定语言，则让模型自动检测
        languages = [None] * len(audio_paths)
    
    # 批量转录
    results = model.transcribe(
        audio=audio_paths,
        language=languages,
        return_time_stamps=args.return_time_stamps
    )
    
    # 写入输出文件
    print(f"Writing results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, (uttid, result) in enumerate(zip(uttids, results)):
            text = result.text if hasattr(result, 'text') else str(result)
            f.write(f"{uttid} {text}\n")
            print(f"Processed {i+1}/{len(uttids)}: {uttid}")
    
    print("Done!")


def get_local_model_path(model_path):
    """
    获取本地模型路径
    如果模型路径是相对路径（如 Qwen/Qwen3-ASR-1.7B），则尝试转换为本地绝对路径
    """
    if not model_path:
        return None
        
    # 如果已经是绝对路径且存在，则直接返回
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    
    # 如果是相对路径，尝试在当前脚本所在目录下查找
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(script_dir, model_path)
    
    if os.path.exists(local_model_path):
        print(f"Found local model at: {local_model_path}")
        return local_model_path
    else:
        print(f"Local model not found at: {local_model_path}")
        print(f"Using model identifier: {model_path}")
        return model_path

if __name__ == "__main__":
    main()