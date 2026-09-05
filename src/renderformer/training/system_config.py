import dataclasses

from renderformer.data.loaders import TriangleRenderH5DatasetConfig
from renderformer.models.config import RenderTransformerConfig
from renderformer.training.trainer_config import TrainerConfig


@dataclasses.dataclass
class SystemConfig:
    model_config: RenderTransformerConfig = RenderTransformerConfig()
    dataset_config: TriangleRenderH5DatasetConfig = TriangleRenderH5DatasetConfig(h5_folder_path='TBD')
    trainer_config: TrainerConfig = TrainerConfig()

    def to_dict(self):
        return dataclasses.asdict(self)
