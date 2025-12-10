import argparse
import mmcv
import torch
import warnings
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test
from mmdet.datasets import replace_ImageToTensor
import projects.mmdet3d_plugin 

def main():
    # 1. 准备配置
    config_file = 'projects/configs/bevformer/bevformer_tiny.py'
    checkpoint_file = 'ckpts/bevformer_tiny_epoch_24.pth'
    out_file = 'work_dirs/results.pkl'
    
    cfg = Config.fromfile(config_file)
    
    # --- 关键修复：清洗配置 ---
    # 强制修正数据配置
    cfg.data.test = cfg.data.val
    cfg.data.test.ann_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
    
    # 这里的 samples_per_gpu 是给 DataLoader 用的，不能传给 Dataset
    # 我们先把它从 dataset 配置里删掉（如果存在的话），防止报错
    if 'samples_per_gpu' in cfg.data.test:
        cfg.data.test.pop('samples_per_gpu')
    if 'workers_per_gpu' in cfg.data.test:
        cfg.data.test.pop('workers_per_gpu')

    # 2. 构建数据集
    print('\n[Step 1] Loading Dataset...')
    try:
        dataset = build_dataset(cfg.data.test)
    except KeyError as e:
        print(f"❌ Error: {e}")
        return
    except TypeError as e:
        print(f"❌ TypeError Detail: {e}")
        print("Tip: There are still invalid keys in cfg.data.test")
        return

    print(f'✅ Dataset loaded! Length: {len(dataset)}')
    
    if len(dataset) == 0:
        print('❌ Error: Dataset length is 0. Check your pkl path!')
        return

    # 构建 DataLoader (在这里使用 samples_per_gpu)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1, # 强制为 1
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 3. 构建模型
    print('[Step 2] Building Model...')
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 4. 加载权重
    print(f'[Step 3] Loading Checkpoint from {checkpoint_file}...')
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    
    # 5. 模型包装
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # 6. 开始推理
    print('[Step 4] Start Inference (Running on 81 samples)...')
    outputs = single_gpu_test(model, data_loader, show=False)
    
    # 7. 保存结果
    print(f'[Step 5] Saving results to {out_file}...')
    mmcv.dump(outputs, out_file)

    # 8. 计算指标
    print('\n[Step 6] Evaluating Metrics...')
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # 清理评估参数
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs['metric'] = 'bbox'
    
    print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()
