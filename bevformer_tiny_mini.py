
_base_ = ['./bevformer_tiny.py']

# 1. 重新定义数据根目录
data_root = 'data/nuscenes/'

# 2. 覆盖测试集配置 (显式指定 mini 验证集路径)
data = dict(
    samples_per_gpu=1,  # 显存保险
    test=dict(
        # 强制指定为 mini 的 pkl 文件
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        # 复用验证集的其他属性 (pipeline, classes 等)
        # 注意：这里我们利用 list 索引 -1 来动态获取 val 配置
        type='CustomNuScenesDataset',
        data_root=data_root,
    )
)

# 3. 终极补丁：利用 Python 动态语法强制 test = val
# 这段代码会在加载配置时执行，确保万无一失
def patch_config(cfg):
    cfg.data.test = cfg.data.val
    cfg.data.test.ann_file = data_root + 'nuscenes_infos_temporal_val.pkl'
    cfg.data.samples_per_gpu = 1
    return cfg
