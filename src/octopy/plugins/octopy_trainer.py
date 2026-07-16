# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
import inspect
import logging
from os import PathLike
from pathlib import Path
import re
from typing import Callable, Literal, Sequence
import warnings

from kraken.containers import Segmentation, Region, BaselineLine, BBoxLine
from kraken.lib import default_specs, vgsl
from kraken.lib.dataset import BaselineSet, ImageInputTransforms
from kraken.lib.models import validate_hyper_parameters
from kraken.lib.train import SegmentationModel
from kraken.lib.xml import XMLPage
import lightning as L
from lightning.pytorch.utilities.parsing import save_hyperparameters
from lxml import etree  # ty:ignore[unresolved-import]
import torch
from torch.utils.data import Subset, random_split


logger: logging.Logger = logging.getLogger(__name__)


page_regions = {
    'TextRegion': 'text',
    'ImageRegion': 'image',
    'LineDrawingRegion': 'line drawing',
    'GraphicRegion': 'graphic',
    'TableRegion': 'table',
    'ChartRegion': 'chart',
    'MapRegion': 'map',
    'SeparatorRegion': 'separator',
    'MathsRegion': 'maths',
    'ChemRegion': 'chem',
    'MusicRegion': 'music',
    'AdvertRegion': 'advert',
    'NoiseRegion': 'noise',
    'UnknownRegion': 'unknown',
    'CustomRegion': 'custom'
}


def parse_page_patch(self):
    with open(self.filename, 'rb') as fp:
        base_directory = self.filename.parent

        try:
            doc = etree.parse(fp)
        except etree.XMLSyntaxError as e:
            raise ValueError(f'Parsing {self.filename} failed: {e}')

    if (image := doc.find('.//{*}Page')) is None or image.get('imageFilename') is None:
        raise ValueError(f'No valid image filename found in PageXML file {self.filename}')
    page_default_direction = {
        'left-to-right': 'L',
        'right-to-left': 'R',
        'top-to-bottom': 'L',
        'bottom-to-top': 'R'
    }.get(image.get('readingDirection'), None)

    page_default_lang = self._parse_page_langs(image)

    if self.image_extension is None:
        if image is None or image.get('imageFilename') is None:
            raise ValueError(f'No valid image filename found in PageXML file {self.filename}')
        self.imagename = base_directory / image.get('imageFilename')
    else:
        self.imagename = base_directory / f'{self.filename.name.split(".")[0]}{self.image_extension}'
        
    self.image_size = int(image.get('imageWidth')), int(image.get('imageHeight'))

    # parse region type and coords
    region_data = defaultdict(list)
    tr_region_order = []

    self._tag_set = set(('default',))
    tmp_transkribus_line_order = defaultdict(list)

    for region in image.iterfind('./{*}*'):
        if not any([True if region.tag.endswith(k) else False for k in page_regions.keys()]):
            continue
        region_id = region.get('id')
        coords = region.find('./{*}Coords')
        try:
            coords = self._parse_page_coords(coords.get('points'))
        except Exception:
            logger.info(f'Region {region_id} without coordinates')
            coords = None
        tags = {}
        rtype = region.get('type')
        # parse transkribus-style custom field if possible
        region_default_lang = self._parse_page_langs(region, page_default_lang)
        if (custom_str := region.get('custom')) is not None:
            cs = self._parse_page_custom(custom_str)
            if not rtype and 'structure' in cs and 'type' in cs['structure'][0]:
                rtype = cs['structure'][0]['type']
            # transkribus-style reading order
            if (reg_ro := cs.get('readingOrder')) is not None and (reg_ro_idx := reg_ro[0].get('index')) is not None:
                tr_region_order.append((region_id, int(reg_ro_idx)))
            tags.update(cs)

        if region_default_lang is None:
            region_default_lang = page_default_lang

        # fall back to default region type if nothing is given
        if not rtype:
            rtype = page_regions[region.tag.split('}')[-1]]

        tags['type'] = [{'type': rtype}]
        region_data[rtype].append(Region(id=region_id, boundary=coords, tags=tags, language=region_default_lang))  # ty:ignore[invalid-argument-type]

        region_default_direction = {
            'left-to-right': 'L',
            'right-to-left': 'R',
            'top-to-bottom': 'L',
            'bottom-to-top': 'R'
        }.get(region.get('readingDirection'))

        # register implicit reading order
        self._orders['region_implicit']['order'].append(region_id)

        # parse line information
        for line in region.iterfind('./{*}TextLine'):
            line_id = line.get('id')
            base = line.find('./{*}Baseline')
            baseline = None
            try:
                baseline = self._parse_page_coords(base.get('points'))
            except Exception:
                logger.info(f'TextLine {line_id} without baseline')
                if self.type == 'baselines':
                    continue

            pol = line.find('./{*}Coords')
            boundary = None
            try:
                boundary = self._parse_page_coords(pol.get('points'))
            except Exception:
                logger.info(f'TextLine {line_id} without polygon')
                if self.type == 'bbox':
                    continue

            text = ''
            manual_transcription = line.find('./{*}TextEquiv')
            if manual_transcription is not None:
                transcription = manual_transcription
            else:
                transcription = line
            for el in transcription.findall('.//{*}Unicode'):
                if el.text:
                    text += el.text
            # retrieve line tags if custom string is set and contains
            tags = {}
            custom_str = line.get('custom')
            if custom_str:
                cs = self._parse_page_custom(custom_str)
                if (structure := cs.get('structure')) is not None and (ltype := structure[0].get('type')):
                    tags['type'] = [{'type': ltype}]
                if (line_ro := cs.get('readingOrder')) is not None and (line_ro_idx := line_ro[0].get('index')) is not None:
                    # look up region index from parent
                    reg_cus = self._parse_page_custom(line.getparent().get('custom'))
                    if 'readingOrder' not in reg_cus or 'index' not in reg_cus['readingOrder']:
                        logger.info('Incomplete `custom` attribute reading order found.')
                    else:
                        tmp_transkribus_line_order[int(reg_cus['readingOrder'][0]['index'])].append((int(line_ro_idx), line_id))
                tags.update(cs)

            # get base text direction
            line_dir = {
                'left-to-right': 'L',
                'right-to-left': 'R',
                'top-to-bottom': 'L',
                'bottom-to-top': 'R'
            }.get(line.get('readingDirection'), None)
            if region_default_direction and line_dir is None:
                line_dir = region_default_direction
            elif page_default_direction and line_dir is None:
                line_dir = page_default_direction

            line_langs = self._parse_page_langs(line, region_default_lang)
            line_split = None
            if (split := tags.get('split', None)) is not None and len(split) == 1:
                line_split = split[0]['type']
                tags.pop('split')

            if self.type == 'baselines':
                line_obj = BaselineLine(
                    id=line_id,
                    baseline=baseline,  # ty:ignore[invalid-argument-type]
                    boundary=boundary,  # ty:ignore[invalid-argument-type]
                    text=text,
                    tags=tags,  # ty:ignore[invalid-argument-type]
                    language=line_langs,
                    split=line_split,
                    base_dir=line_dir,  # ty:ignore[invalid-argument-type]
                    regions=[region_id]
                )
            elif self.type == 'bbox':
                flat_box = [point for pol in boundary for point in pol]  # ty:ignore[not-iterable]
                xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
                ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
                line_obj = BBoxLine(
                    id=line_id,
                    bbox=(xmin, ymin, xmax, ymax),
                    text=text,
                    tags=tags,  # ty:ignore[invalid-argument-type]
                    language=line_langs,
                    split=line_split,
                    base_dir=line_dir,  # ty:ignore[invalid-argument-type]
                    regions=[region_id]
                )

            self._lines[line_id] = line_obj
            # register implicit reading order
            self._orders['line_implicit']['order'].append(line_id)

    # add transkribus-style region order
    self._orders['region_transkribus'] = {
        'order': [x[0] for x in sorted(tr_region_order, key=lambda k: k[1])],
        'is_total': True if len(set(map(lambda x: x[0], tr_region_order))) == len(tr_region_order) else False,
        'description': 'Explicit region order from `custom` attribute'
    }

    self._regions = region_data

    if tmp_transkribus_line_order:
        # sort by regions
        tmp_reg_order = sorted(((k, v) for k, v in tmp_transkribus_line_order.items()), key=lambda k: k[0])
        # flatten
        tr_line_order = []
        for _, lines in tmp_reg_order:
            tr_line_order.extend([x[1] for x in sorted(lines, key=lambda k: k[0])])
        self._orders['line_transkribus'] = {
            'order': tr_line_order,
            'is_total': True,
            'description': 'Explicit line order from `custom` attribute'
        }

    # parse explicit reading orders if they exist
    ro_el = doc.find('.//{*}ReadingOrder')
    if ro_el is not None:
        reading_orders = ro_el.getchildren()
        # UnorderedGroup at top-level => treated as multiple reading orders
        if len(reading_orders) == 1 and reading_orders[0].tag.endswith('UnorderedGroup'):
            reading_orders = reading_orders.getchildren()

        def _parse_group(el):

            _ro = []
            if el.tag.endswith('UnorderedGroup'):
                _ro = [_parse_group(x) for x in el.iterchildren()]
                is_total = False  # NOQA
            elif el.tag.endswith('OrderedGroup'):
                _ro.extend(_parse_group(x) for x in el.iterchildren())
            else:
                return el.get('regionRef')
            return _ro

        for ro in reading_orders:
            is_total = True
            self._orders[ro.get('id')] = {
                'order': _parse_group(ro),
                'is_total': is_total,
                'description': ro.get('caption') if ro.get('caption') else ''
            }

    if len(self._tag_set) > 1:
        self.has_tags = True
    else:
        self.has_tags = False

    self.filetype = 'page'


def xmlpage_init_patch(
    self,
    filename: str | PathLike,
    filetype: Literal['xml', 'alto', 'page'] = 'xml',
    linetype: Literal['baselines', 'bbox'] = 'baselines',
    image_extension: str | None = None
) -> None:
    object.__init__(self)
    self.filename = Path(filename)
    self.filetype = filetype
    self.type = linetype
    self.image_extension = image_extension

    self._regions = {}
    self._lines = {}
    self._orders = {
        'line_implicit': {
            'order': [], 
            'is_total': True, 
            'description': 'Implicit line order derived from element sequence'
        },
        'region_implicit': {
            'order': [], 
            'is_total': True, 
            'description': 'Implicit region order derived from element sequence'
        }
    }

    if filetype == 'xml':
        self._parse_xml()
    elif filetype == 'alto':
        self._parse_alto()
    elif filetype == 'page':
        self._parse_page()

def segmentation_module_init_patch(
    self,
    hyper_params: dict = None,  # ty:ignore[invalid-parameter-default]
    load_hyper_parameters: bool = False,
    progress_callback: Callable[[str, int], Callable[[None], None]] = lambda string, length: lambda: None,  # ty:ignore[invalid-parameter-default]
    message: Callable[[str], None] = lambda *args, **kwargs: None,
    output: str = 'model',
    spec: str = default_specs.SEGMENTATION_SPEC,
    model: PathLike | str | None = None,
    training_data: Sequence[PathLike | str] | Sequence[Segmentation] = None,  # ty:ignore[invalid-parameter-default]
    evaluation_data: Sequence[PathLike | str] | Sequence[Segmentation] | None = None,
    partition: float | None = 0.9,
    num_workers: int = 1,
    force_binarization: bool = False,
    format_type: Literal['path', 'alto', 'page', 'xml'] | None = 'path',
    suppress_regions: bool = False,
    suppress_baselines: bool = False,
    valid_regions: Sequence[str] | None = None,
    valid_baselines: Sequence[str] | None = None,
    merge_regions: dict[str, str] | None = None,
    merge_baselines: dict[str, str] | None = None,
    merge_all_baselines: str | None = None,
    merge_all_regions: str | None = None,
    bounding_regions: Sequence[str] | None = None,
    resize: Literal['fail', 'both', 'new', 'add', 'union'] = 'fail',
    topline: bool | None = False,
    image_extension: str | None = None,
) -> None:
    """
    A LightningModule encapsulating the training setup for a page
    segmentation model.

    Setup parameters (load, training_data, evaluation_data, ....) are
    named, model hyperparameters (everything in
    `kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS`) are in in the
    `hyper_params` argument.

    Args:
        hyper_params (dict): Hyperparameter dictionary containing all fields
                                from
                                kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS
        **kwargs: Setup parameters, i.e. CLI parameters of the segtrain() command.
    """

    L.LightningModule.__init__(self)

    self.best_epoch = -1
    self.best_metric = 0.0
    self.best_model = None

    self.model = model
    self.num_workers = num_workers

    if resize == "add":
        resize = "union"
        warnings.warn("'add' value for resize has been deprecated. Use 'union' instead.", DeprecationWarning)
    elif resize == "both":
        resize = "new"
        warnings.warn("'both' value for resize has been deprecated. Use 'new' instead.", DeprecationWarning)
    self.resize = resize

    self.output = output
    self.bounding_regions = bounding_regions
    self.topline = topline

    hyper_params_ = default_specs.SEGMENTATION_HYPER_PARAMS.copy()

    if model:
        logger.info(f'Loading existing model from {model}')
        self.nn = vgsl.TorchVGSLModel.load_model(model)

        if self.nn.model_type not in [None, 'segmentation']:
            raise ValueError(f'Model {model} is of type {self.nn.model_type} while `segmentation` is expected.')

        if load_hyper_parameters:
            hp = self.nn.hyper_params
        else:
            hp = {}
        hyper_params_.update(hp)
        batch, channels, height, width = self.nn.input
    else:
        self.nn = None

        spec = spec.strip()
        if spec[0] != '[' or spec[-1] != ']':
            raise ValueError(f'VGSL spec "{spec}" not bracketed')
        self.spec = spec
        blocks = spec[1:-1].split(' ')
        m = re.match(r'(\d+),(\d+),(\d+),(\d+)', blocks[0])
        if not m:
            raise ValueError(f'Invalid input spec {blocks[0]}')
        batch, height, width, channels = [int(x) for x in m.groups()]

    if hyper_params:
        hyper_params_.update(hyper_params)

    validate_hyper_parameters(hyper_params_)
    self.hyper_params = hyper_params_
    #self.save_hyperparameters()
    

    save_hyperparameters(
        self,
        given_hparams={  # fix lightning auto discovery
            'hyper_params': hyper_params,
            'load_hyper_parameters': load_hyper_parameters,
            'output': output,
            'spec': spec,
            'model': model,
            'partition': partition,
            'num_workers': num_workers,
            'force_binarization': force_binarization,
            'format_type': format_type,
            'suppress_regions': suppress_regions,
            'suppress_baselines': suppress_baselines,
            'valid_regions': valid_regions,
            'valid_baselines': valid_baselines,
            'merge_regions': merge_regions,
            'merge_baselines': merge_baselines,
            'merge_all_baselines': merge_all_baselines,
            'merge_all_regions': merge_all_regions,
            'bounding_regions': bounding_regions,
            'resize': resize,
            'topline': topline,
        },
        frame=inspect.currentframe(),
    )

    if format_type in ['xml', 'page', 'alto']:
        logger.info(f'Parsing {len(training_data)} XML files for training data')
        _training_data = []
        for file in training_data:
            try:
                _training_data.append(
                    XMLPage(file, format_type, image_extension=image_extension).to_container()  # ty:ignore[invalid-argument-type, unknown-argument]
                ) 
            except Exception as e:
                logger.warning(f'Failed to parse {file}: {e}')
        training_data = _training_data
        if evaluation_data:
            _evaluation_data = []
            logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
            for file in evaluation_data:
                try:
                    _evaluation_data.append(
                        XMLPage(file, format_type, image_extension=image_extension).to_container()  # ty:ignore[invalid-argument-type, unknown-argument]
                    )  
                except Exception as e:
                    logger.warning(f'Failed to parse {file}: {e}')
            evaluation_data = _evaluation_data
    elif not format_type:
        pass
    else:
        raise ValueError(f'format_type {format_type} not in [alto, page, xml, None].')

    if not training_data:
        raise ValueError('No training data provided. Please add some.')

    transforms = ImageInputTransforms(
        batch,
        height,
        width,
        channels,
        self.hyper_params['padding'],  # ty:ignore[invalid-argument-type]
        valid_norm=False,
        force_binarization=force_binarization
    )

    self.example_input_array = torch.Tensor(
        batch,
        channels,
        height if height else 400,
        width if width else 300
    )

    # set multiprocessing tensor sharing strategy
    if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
        logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')

    if not valid_regions:
        valid_regions = None
    if not valid_baselines:
        valid_baselines = None

    if suppress_regions:
        valid_regions = []
        merge_regions = None
    if suppress_baselines:
        valid_baselines = []
        merge_baselines = None

    train_set = BaselineSet(
        line_width=self.hyper_params['line_width'],  # ty:ignore[invalid-argument-type]
        im_transforms=transforms,
        augmentation=self.hyper_params['augment'],  # ty:ignore[invalid-argument-type]
        valid_baselines=valid_baselines,  # ty:ignore[invalid-argument-type]
        merge_baselines=merge_baselines,  # ty:ignore[invalid-argument-type]
        valid_regions=valid_regions,  # ty:ignore[invalid-argument-type]
        merge_regions=merge_regions,  # ty:ignore[invalid-argument-type]
        merge_all_baselines=merge_all_baselines,
        merge_all_regions=merge_all_regions
    )

    for page in training_data:
        train_set.add(page)  # ty:ignore[invalid-argument-type]

    if evaluation_data:
        val_set = BaselineSet(
            line_width=self.hyper_params['line_width'],  # ty:ignore[invalid-argument-type]
            im_transforms=transforms,
            augmentation=False,
            valid_baselines=valid_baselines,  # ty:ignore[invalid-argument-type]
            merge_baselines=merge_baselines,  # ty:ignore[invalid-argument-type]
            valid_regions=valid_regions,  # ty:ignore[invalid-argument-type]
            merge_regions=merge_regions,  # ty:ignore[invalid-argument-type]
            merge_all_baselines=merge_all_baselines,
            merge_all_regions=merge_all_regions
        )

        for page in evaluation_data:
            val_set.add(page)  # ty:ignore[invalid-argument-type]

        train_set = Subset(train_set, range(len(train_set)))
        val_set = Subset(val_set, range(len(val_set)))
    else:
        train_len = int(len(train_set)*partition)  # ty:ignore[unsupported-operator]
        val_len = len(train_set) - train_len
        logger.info(f'No explicit validation data provided. Splitting off '
                    f'{val_len} (of {len(train_set)}) samples to validation '
                    'set.')
        train_set, val_set = random_split(train_set, (train_len, val_len))

    if len(train_set) == 0:
        raise ValueError('No valid training data provided. Please add some.')

    if len(val_set) == 0:
        raise ValueError('No valid validation data provided. Please add some.')

    # overwrite class mapping in validation set
    val_set.dataset.num_classes = train_set.dataset.num_classes  # ty:ignore[unresolved-attribute]
    val_set.dataset.class_mapping = train_set.dataset.class_mapping  # ty:ignore[unresolved-attribute]

    self.train_set = train_set
    self.val_set = val_set


class OctopyTrainer:
    @staticmethod
    def register() -> None:
        XMLPage._parse_page = parse_page_patch
        XMLPage.__init__ = xmlpage_init_patch  # ty:ignore[invalid-assignment]
        SegmentationModel.__init__ = segmentation_module_init_patch  # ty:ignore[invalid-assignment]
        logger.info(f'Plugin: {OctopyTrainer.__name__} registered')
