import mmcv
import torch
import os
import shutil
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test
import projects.mmdet3d_plugin 

def main():
    # === 配置区域 ===
    config_file = 'projects/configs/bevformer/bevformer_tiny.py'
    checkpoint_file = 'ckpts/bevformer_tiny_epoch_24.pth'
    vis_dir = 'work_dirs/vis_tiny_final'
    
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    
    # =================
    
    cfg = Config.fromfile(config_file)
    
    # 1. 基础配置修正
    cfg.data.test = cfg.data.val
    cfg.data.test.ann_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if 'samples_per_gpu' in cfg.data.test: cfg.data.test.pop('samples_per_gpu')

    # ==========================================================
    # 2. 【关键修复】: 注入点云加载流程
    # 目的：为了欺骗 dataset.show() 函数，防止报 'NoneType' 错误
    # ==========================================================
    
    # 定义加载点云的操作
    load_points_config = dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')
    )
    
    # 我们需要把这个操作塞到数据处理流程(pipeline)的最前面
    # 检查 pipeline 是直接的列表还是嵌套在 MultiScaleFlipAug 里
    if isinstance(cfg.data.test.pipeline, list):
        # 简单列表情况
        print('[Fix] Injecting LoadPointsFromFile into pipeline...')
        cfg.data.test.pipeline.insert(0, load_points_config)
    else:
        # 复杂嵌套情况 (通常 BEVFormer 是这种情况)
        print('[Fix] Injecting LoadPointsFromFile into MultiScaleFlipAug...')
        # 找到 transforms 列表并插入
        if 'transforms' in cfg.data.test.pipeline:
             cfg.data.test.pipeline['transforms'].insert(0, load_points_config)

    # ==========================================================

    # 3. 构建数据集 (这时候数据集里已经有点云加载功能了)
    print('\n[Step 1] Loading Dataset...')
    dataset = build_dataset(cfg.data.test)
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 4. 构建模型
    print('[Step 2] Building Model...')
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print(f'[Step 3] Loading Checkpoint...')
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # 5. 推理
    print(f'[Step 4] Running Inference...')
    outputs = single_gpu_test(model, data_loader, show=False)
    
    # 6. 画图
    print(f'\n[Step 5] Rendering Images (Now with Point Cloud support)...')
    print(f'   -> Target Directory: {vis_dir}')
    print('   -> This will take 2-3 minutes. Please wait...')
    
    # 这里的 show=True/False 在某些版本有歧义，我们用 out_dir 确保保存
    # 注意：生成的图片里可能也会包含一张点云的鸟瞰图，这是附赠的
    dataset.show(outputs, vis_dir, show=False)
    
    print(f'\n✅ Visualization Done! Images saved in {vis_dir}')

if __name__ == '__main__':
    main()
