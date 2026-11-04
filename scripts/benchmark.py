import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.models.resnet import ResNet9

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark custom ResNet-9 performance")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for benchmarking")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--bench-runs", type=int, default=50, help="Number of timed iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    return parser.parse_args()

def count_parameters(model):
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if getattr(p,"requires_grad", True))
    return total_params, trainable_params

def benchmark_latency(model, x, target, device, warmup=10, runs=50):
    is_cuda=device.type=="cuda"
    criterion=nn.CrossEntropyLoss()
    
    def run_step():
        preds = model(x)
        loss = criterion(preds, target)
        if hasattr(model, "backward"):
            # Analytical backprop hook
            grad = torch.softmax(preds, dim=-1)
            grad[torch.arange(target.size(0)), target] -= 1.0
            grad /= target.size(0)
            model.backward(grad)
        elif loss.grad_fn is not None:
            loss.backward()
        return loss
    
    for _ in range(warmup):
        preds=model(x)
        loss=criterion(preds, target)
        loss.backward()
        if hasattr(model, "zero_grad"):
            model.zero_grad()
            
        if is_cuda:
            torch.cuda.synchronize()
            
        fwd_times=[]
        model.eval()
        with torch.no_grad():
            for _ in range(runs):
                if is_cuda:
                    start_event=torch.cuda.Event(enable_timing=True)
                    end_event=torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _=model(x)
                    end_event.record()
                    torch.cuda.synchronize()
                    fwd_times.append(start_event.elapsed_time(end_event))
                else:
                    t0=time.perf_counter()
                    _=model(x)
                    t1=time.perf_counter()
                    fwd_times.append((t1-t0)*1000.0)
                    
        
        fwd_bwd_times=[]
        model.train()
        for _ in range(runs):
            if hasattr(model, "zero_grad"):
                model.zero_grad()
                
            if is_cuda:
                start_event=torch.cuda.Event(enable_timing=True)
                end_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
                preds=model(x)
                loss=criterion(preds, target)
                end_event.record()
                torch.cuda.synchronize()
                fwd_bwd_times.append(start_event.elapsed_time(end_event))
                
            else:
                t0=time.perf_counter()
                preds=model(x)
                loss=criterion(preds, target)
                loss.backward()
                t1=time.perf_counter()
                fwd_bwd_times.append((t1-t0)*1000.0)
            
        mean_fwd = sum(fwd_times) / len(fwd_times)
        mean_fwd_bwd = sum(fwd_bwd_times) / len(fwd_bwd_times)
        return mean_fwd, mean_fwd_bwd
        
def main():
    args=parse_args()
    device=torch.device(args.device)
    print(f"============================================================")
    print(f"[*] ResNet-9 Performance Benchmark Harness")
    print(f"[*] Target Device: {device} | Batch Size: {args.batch_size}")
    print(f"============================================================")
    
    model=ResNet9(num_classes=10).to(device)
    total_p, train_p=count_parameters(model)
    print(f"[+] Total Parameters:     {total_p:,}")
    print(f"[+] Trainable Parameters: {train_p:,}")
    
    x=torch.randn(args.batch_size,3,32,32, device=device)
    target=torch.randint(0,10,(args.batch_size,), device=device)
    
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
        
    print(f"\n[*] Running warmup ({args.warmup_runs} iters) and benchmark ({args.bench_runs} iters)...")
    mean_fwd_ms, mean_fwd_bwd_ms=benchmark_latency(
        model,x,target,device,warmup=args.warmup_runs, runs=args.bench_runs
        
    )
    
    fwd_fps = (args.batch_size / (mean_fwd_ms / 1000.0))
    train_fps = (args.batch_size / (mean_fwd_bwd_ms / 1000.0))

    print(f"\n[+] Results:")
    print(f"    - Forward Latency:         {mean_fwd_ms:.2f} ms / batch")
    print(f"    - Forward Throughput:      {fwd_fps:.1f} images / sec")
    print(f"    - Forward + Backprop Time: {mean_fwd_bwd_ms:.2f} ms / batch")
    print(f"    - Training Throughput:     {train_fps:.1f} images / sec")
          
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"    - Peak VRAM Allocated:     {peak_vram_mb:.2f} MB")

    print(f"============================================================")
    print(f"[SUCCESS] Benchmark completed successfully.")


if __name__ == "__main__":
    main()