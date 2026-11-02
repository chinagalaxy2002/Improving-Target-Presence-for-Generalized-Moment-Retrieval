"""Retain each route's native architecture, targets, losses and prediction exporter."""
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from study_paths import CORE, PROJECT


class Adapter:
    def __init__(self, backbone, variant):
        self.name = backbone
        self.variant = variant
        self.bs = 8 if backbone == "moment_detr" else 32
        if backbone == "moment_detr":
            sys.path.insert(0, str(PROJECT / "gmr_multiroute_kit"))
            import run_train
            self.native = run_train
            self.opt, self.base, self.criterion = run_train.setup(variant)
            self.opt.num_workers = 2
            self.opt.eval_bsz = 8
            self.train_data, self.val_data = self.dataset("train"), self.dataset("val")
        else:
            spec = importlib.util.spec_from_file_location("native_multiroute_engine", CORE / "run.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.experiment = module.Experiment(backbone, variant).build()
            self.experiment.microbatch = self.bs
            self.opt = self.experiment.options
            self.base = self.experiment.model
            self.criterion = self.experiment.criterion
            self.train_data = self.experiment.train_data
            self.val_data = self.experiment.val_data
        self.opt.num_workers = 2

    def dataset(self, split):
        if self.name == "moment_detr":
            path = PROJECT / "soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard" / f"{split}.jsonl"
            return self.native.StartEndDataset(**self.native.build_dataset_config(
                self.opt, str(path), load_labels=True, keep_empty_gt=True))
        if self.name == "eatr":
            return self.experiment.module.SoccerDataset(self.opt, split)
        return self.experiment.module.Soccer(split)

    def loader(self, data, training=False, generator=None):
        if self.name != "moment_detr":
            loader = self.experiment.loader(data, training)
            if generator is not None:
                loader.generator = generator
                loader.sampler.generator = generator
            return loader
        return DataLoader(data, batch_size=self.bs if training else self.opt.eval_bsz,
            shuffle=training, num_workers=2, collate_fn=self.native.start_end_collate,
            generator=generator, pin_memory=True, persistent_workers=True)

    def unpack(self, batch):
        if self.name == "moment_detr":
            inputs, targets = self.native.prepare_batch_inputs(batch[1], self.opt.device)
            return batch[0], inputs, targets, {}
        return self.experiment.unpack(batch)

    def predict(self, model, data):
        model.eval()
        if self.name == "moment_detr":
            from evaluate import compute_mr_results
            with torch.no_grad():
                predictions, _ = compute_mr_results(None, model, self.loader(data), self.opt, criterion=None)
            return predictions
        self.experiment.model = model
        return self.experiment.predict(data)

    def native_loss(self, output, targets, epoch):
        native = self.criterion.native
        if hasattr(native, "epoch"):
            native.epoch = epoch - 1
        losses = self.criterion(output, targets)
        return sum(value * self.criterion.weight_dict[key] for key, value in losses.items()
                   if key in self.criterion.weight_dict)
