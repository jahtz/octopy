# Copyright 2024 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This project includes code from the kraken project,
# available at https://github.com/mittagessen/kraken and licensed under
# Apache 2.0 license https://github.com/mittagessen/kraken/blob/main/LICENSE.

from pathlib import Path
from typing import Literal, Optional, Union

import click
from PIL import Image
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.containers import Segmentation
from kraken.lib.exceptions import KrakenInvalidModelException

from pypxml import PageXML, XMLType
from modules.util import parse_path_list, parse_path, parse_suffix, expand_path_list


# CLI input to kraken text direction (--text-direction )
TEXT_DIRECTION = {
    'l2r': 'horizontal-lr',
    'r2l': 'horizontal-rl',
    't2b': 'vertical-lr',
    'b2t': 'vertical-rl',
}


# Mapping from models regions to PageXML regions
REGION_MAPPING = {
    'text': (XMLType.TextRegion, 'paragraph'),
    'paragraph':(XMLType.TextRegion, 'paragraph'),
    'image': (XMLType.ImageRegion, None),
    'header': (XMLType.TextRegion, 'header'),
    'signature-mark': (XMLType.TextRegion, 'signature-mark'),
    'catch-word': (XMLType.TextRegion, 'catch-word'),
    'drop-capital': (XMLType.TextRegion, 'drop-capital'),
    'separator': (XMLType.SeparatorRegion, None),
    'page-number': (XMLType.TextRegion, 'page-number'),
    'footnote': (XMLType.TextRegion, 'footnote'),
    'marginalia': (XMLType.TextRegion, 'marginalia'),
    'table': (XMLType.TableRegion, None),
    'other': (XMLType.TextRegion, 'other'),

    'Title': (XMLType.TextRegion, 'caption'),
    'Illustration': (XMLType.ImageRegion, None),
    'Commentary': (XMLType.TextRegion, 'footnote'),

    'unknown': (XMLType.UnknownRegion, None)
}


def coords_to_pagexml(coords: list[Union[list[int], tuple[int, int]]]) -> str:
    """
    Build a PageXML coordinate string from a list of sublists/tuples containing x,y coordinates

    :param coords: List of x,y coordinates, where each coordinate is a tuple or list.
    :return: PageXML coords string.
    """
    return ' '.join([f'{point[0]},{point[1]}' for point in coords])


def kraken_to_pxml(res: Segmentation, image_width: int, image_height: int,
                   creator: str, skip_unknown: bool = True) -> PageXML:
    """
    Converts a Kraken Segmentation object to a PageXML object.

    :param res: Kraken result Segmentation object.
    :param image_width: Width of image in pixels.
    :param image_height: Height of image in pixels.
    :param creator: Creator tag content for PageXML metadata.
    :param skip_unknown: Skip unknown regions, else use UnknownRegion. Defaults to True.
    :return: PageXML object containing all information from Kraken Segmentation object.
    """
    pxml_obj = PageXML.new(creator=creator)
    name_parts = Path(res.imagename).name.split('.')
    page_obj = pxml_obj.create_page(imageFilename=f'{name_parts[0]}.{name_parts[-1]}',
                                    imageWidth=image_width,
                                    imageHeight=image_height)
    region_counter = 1
    line_counter = 1
    for region_type, regions in res.regions.items():
        if region_type not in REGION_MAPPING:
            click.echo('Unknown region type: ' + region_type)
            if skip_unknown:
                continue
            region_type = 'unknown'  # use UnknownRegion as default
        for region in regions:
            region_obj = page_obj.create_element(REGION_MAPPING[region_type][0],
                                                 id=f'r_{region_counter:03d}',
                                                 type=REGION_MAPPING[region_type][1])
            region_obj.create_element(XMLType.Coords, points=coords_to_pagexml(region.boundary))
            region_counter += 1
            for line in res.lines:
                if region.id in line.regions:
                    line_obj = region_obj.create_element(XMLType.TextLine, id=f'l_{line_counter:03d}')
                    line_counter += 1
                    if line.boundary:
                        line_obj.create_element(XMLType.Coords, points=coords_to_pagexml(line.boundary))
                    if line.baseline:
                        line_obj.create_element(XMLType.Baseline, points=coords_to_pagexml(line.baseline))
    return pxml_obj


def segment(images: list[Path],
            glob: str = '*',
            models: Optional[list[Path]] = None,
            output: Optional[Path] = None,
            text_direction: Literal['l2r', 'r2l', 't2b', 'b2t'] = 'l2r',
            suffix: str = '.xml',
            creator: str = 'octopy',
            device: str = 'cpu',
            default_polygon: Optional[tuple[int, int, int, int]] = None,
            sort_lines: bool = False,
            drop_empty: bool = False) -> None:
    """
    Segment images using Kraken and save the results as XML files (PageXML format).


    :param images: List of image files to be segmented. Supports multiple file paths or directories
        (when used with the -g option).
    :param glob: Specify a glob pattern to match image files when processing directories in IMAGES.
    :param models: List of custom segmentation models. If not provided, the default Kraken model is used.
    :param output: Directory to save the output PageXML files. Defaults to the parent directory of each input image.
    :param text_direction: Set the text direction for segmentation.
    :param suffix: Append a suffix to the output PageXML file names. For example, using `.seg.xml` results in
        filenames like `imagename.seg.xml`
    :param creator: Specify the creator of the PageXML file. This can be useful for tracking the origin of
        segmented files.
    :param device: Specify the device for running the model (e.g., `cpu`, `cuda:0`). Refer to PyTorch documentation
        for supported devices.
    :param default_polygon: If the polygonizer fails to create a polygon around a baseline, use this option to create
        a default polygon instead of discarding the baseline. The offsets are defined as left, top, right, and bottom.
    :param sort_lines: Sort text lines in each TextRegion based on their centroids according to the specified text
        direction. (Feature not yet implemented)
    :param drop_empty: Automatically drop empty TextRegions from the output. (Feature not yet implemented)
    """
    # load images
    images = expand_path_list(images, glob)
    if len(images) < 1:
        click.echo('No files found!', err=True)
        return
    click.echo(f'{len(images)} image(s) found.')

    # create output directory:
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    # load models
    torch_models = []
    for model in models:
        try:
            nn = TorchVGSLModel.load_model(model)
        except Exception as e:
            click.echo(f'Error loading model {model}: {e}', err=True)
            return
        if nn.model_type != 'segmentation':
            raise KrakenInvalidModelException(f'Invalid model type {nn.model_type} for {model}')
        if 'class_mapping' not in nn.user_metadata:
            raise KrakenInvalidModelException(f'Segmentation model {model} does not contain valid class mapping')
        torch_models.append(nn)

    # default polygon
    default_polygon = None if default_polygon is None or len(default_polygon) != 4 else default_polygon
    
    # process files
    with click.progressbar(images, label='Segmenting images', show_pos=True, show_eta=True, show_percent=True,
                           item_show_func=lambda f: f.name if f is not None else '') as image_iterator:
        for image in image_iterator:
            im = Image.open(image)
            res: Segmentation = blla.segment(im=im,
                                             text_direction=TEXT_DIRECTION[text_direction],
                                             model=torch_models if torch_models else None,
                                             device=device,
                                             default_polygon=default_polygon)
            filename = image.name.split('.')[0] + suffix
            output_file = output.joinpath(filename) if output is not None else image.parent.joinpath(filename)
            xml = kraken_to_pxml(res, image_width=im.size[0], image_height=im.size[1], creator=creator)
            xml.to_xml(output_file)


@click.command('segment', help="""
    Segment images using Kraken and save the results as XML files (PageXML format). 

    This tool processes one or more images and segments them using a trained Kraken model. 
    The segmented results are saved as XML files, corresponding to each input image. 
    
    IMAGES: List of image files to be segmented. Supports multiple file paths, wildcards, 
    or directories (when used with the -g option).
    """)
@click.help_option('--help')
@click.argument('images',
                type=click.Path(exists=True, file_okay=True, dir_okay=False),
                callback=parse_path_list,
                nargs=-1)
@click.option('-g', '--glob',
              help='Specify a glob pattern to match image files when processing directories in IMAGES.',
              type=click.STRING,
              default='*',
              show_default=True)
@click.option('-m', '--model', 'models',
              help='Path to a custom segmentation model. If not provided, the default Kraken model is used. '
                   'Multiple models can be specified.',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              multiple=True,
              callback=parse_path_list)
@click.option('-o', '--output',
              help='Directory to save the output PageXML files. Defaults to the parent directory of each input image.',
              type=click.Path(exists=False, file_okay=False, dir_okay=True),
              callback=parse_path)
@click.option('--text-direction', 'text_direction',
              help='Set the text direction for segmentation.',
              type=click.Choice(['l2r', 'r2l', 't2b', 'b2t']),
              default='l2r',
              show_default=True)
@click.option('-s', '--suffix', 'suffix',
              help='Append a suffix to the output PageXML file names. '
                   'For example, using `.seg.xml` results in filenames like `imagename.seg.xml`',
              type=click.STRING,
              callback=parse_suffix,
              default='.xml',
              show_default=True)
@click.option('-c', '--creator',
              help='Specify the creator of the PageXML file. '
                   'This can be useful for tracking the origin of segmented files.',
              type=click.STRING,
              default='octopy')
@click.option('-d', '--device',
              help='Specify the device for running the model (e.g., `cpu`, `cuda:0`). '
                   'Refer to PyTorch documentation for supported devices.',
              type=click.STRING,
              default='cpu',
              show_default=True)
@click.option('--default-polygon', 'default_polygon',
              help='If the polygonizer fails to create a polygon around a baseline, '
                   'use this option to create a default polygon instead of discarding the baseline. '
                   'The offsets are defined as left, top, right, and bottom.',
              type=click.Tuple([int, int, int, int]),
              nargs=4)
@click.option('--sort-lines', 'sort_lines',
              help='Sort text lines in each TextRegion based on their centroids according to the '
                   'specified text direction. (Feature not yet implemented)',
              is_flag=True)
@click.option('--drop-empty', 'drop_empty',
              help='Automatically drop empty TextRegions from the output. (Feature not yet implemented)',
              is_flag=True)
def segment_cli(**kwargs):
    segment(**kwargs)
