# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

from kraken.lib.xml import XMLPage
from kraken.train import BLLASegmentationDataModule
from lightning.pytorch.utilities.parsing import save_hyperparameters
from torch.utils.data import Subset, random_split

logger: logging.Logger = logging.getLogger(__name__)


def data_module_init_patch(self, data_config) -> None:
    """
    A patched version of BLLASegmentationDataModule.__init__ that supports an
    optional `image_extension` attribute on the data config. When set, the
    image filename derived from the PAGE-XML `imageFilename` attribute is
    replaced by the XML file's base name plus the given extension.
    """
    super(BLLASegmentationDataModule, self).__init__()
    # fix lightning auto discovery (no `__class__` cell in this frame)
    save_hyperparameters(
        self,
        given_hparams={'data_config': data_config},
    )

    all_files = [getattr(data_config, x) for x in ['training_data', 'evaluation_data', 'test_data']]
    image_extension = getattr(data_config, 'image_extension', None)

    def _apply_image_extension(page: XMLPage) -> None:
        if image_extension is not None:
            page.imagename = page.filename.parent / f'{page.filename.name.split(".")[0]}{image_extension}'

    if data_config.format_type in ['xml', 'page', 'alto']:

        def _parse_xml_set(ds_type, dataset):
            if not dataset:
                return None
            logger.info(f'Parsing {len(dataset) if dataset else 0} XML files for {ds_type} data')
            data = []
            for pos, file in enumerate(dataset):
                try:
                    page = XMLPage(file, filetype=data_config.format_type)
                    _apply_image_extension(page)
                    data.append({'doc': page.to_container()})
                except Exception as e:
                    logger.warning(f'Failed to parse {file}: {e}')
            return data

        training_data = _parse_xml_set('training', all_files[0])
        evaluation_data = _parse_xml_set('evaluation', all_files[1])
        self.test_data = _parse_xml_set('test', all_files[2])
    elif data_config.format_type is None:
        training_data = data_config.training_data
        logger.info(f'Using {len(training_data) if training_data else 0} Segmentation objects for training data')
        evaluation_data = data_config.evaluation_data
        logger.info(f'Using {len(evaluation_data) if evaluation_data else 0} Segmentation objects for evaluation data')
        self.test_data = data_config.test_data
        logger.info(f'Using {len(self.test_data) if self.test_data else 0} Segmentation objects for test data')
    else:
        raise ValueError(f'format_type {data_config.format_type} not in [alto, page, xml, None].')

    if training_data and evaluation_data:
        train_set = self._build_dataset(training_data,
                                        augmentation=data_config.augment,
                                        im_transforms=None,
                                        class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                                       'baselines': self.hparams.data_config.line_class_mapping,
                                                       'regions': self.hparams.data_config.region_class_mapping})
        self.train_set = Subset(train_set, range(len(train_set)))
        val_set = self._build_dataset(evaluation_data,
                                      im_transforms=None,
                                      class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                                     'baselines': self.hparams.data_config.line_class_mapping,
                                                     'regions': self.hparams.data_config.region_class_mapping})

        self.val_set = Subset(val_set, range(len(val_set)))
    elif training_data:
        train_set = self._build_dataset(training_data,
                                        augmentation=data_config.augment,
                                        im_transforms=None,
                                        class_mapping={'aux': {'_start_separator': 0, '_end_separator': 1},
                                                       'baselines': self.hparams.data_config.line_class_mapping,
                                                       'regions': self.hparams.data_config.region_class_mapping})

        train_len = int(len(train_set) * data_config.partition)
        val_len = len(train_set) - train_len
        logger.info(f'No explicit validation data provided. Splitting off '
                    f'{val_len} (of {len(train_set)}) samples to validation '
                    'set.')
        self.train_set, self.val_set = random_split(train_set, (train_len, val_len))
    elif self.test_data:
        pass
    else:
        raise ValueError('Invalid specification of training/evaluation/test data.')


class OctopyTrainer:
    @staticmethod
    def register() -> None:
        BLLASegmentationDataModule.__init__ = data_module_init_patch  # ty:ignore[invalid-assignment]
        logger.info(f'Plugin: {OctopyTrainer.__name__} registered')
