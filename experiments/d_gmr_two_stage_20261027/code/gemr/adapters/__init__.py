def build_adapter(name, device):
    # Native source packages collide (models/utils/training). One backbone per process.
    if name == 'moment_detr':
        from .moment_detr_adapter import MomentDETRAdapter as Adapter
    elif name == 'qd_detr':
        from .qd_detr_adapter import QDDETRAdapter as Adapter
    elif name == 'cg_detr':
        from .cg_detr_adapter import CGDETRAdapter as Adapter
    elif name == 'eatr':
        from .eatr_adapter import EaTRAdapter as Adapter
    elif name == 'flashvtg':
        from .flashvtg_adapter import FlashVTGAdapter as Adapter
    else:
        raise ValueError(name)
    return Adapter(name, device)
