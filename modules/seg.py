from pathlib import Path

import click
from PIL import Image
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.segmentation import calculate_polygonal_environment

from pagexml import PageXML, ElementType
from pagexml.geometry import Polygon


"""
Module: Segmentation
Segmentes a set of images using Kraken and outputs the results as PageXML files.
"""


__all__ = ['segment']


REGION_TYPES = [
    'paragraph', 
    'heading', 
    'caption', 
    'header', 
    'footer', 
    'page-number', 
    'drop-capital', 
    'credit',    
    'floating', 
    'signature-mark', 
    'catch-word', 
    'marginalia', 
    'footnote', 
    'footnote-continued', 
    'endnote',
    'TOC-entry', 
    'list-label', 
    'other'
]


def segment(
    files: list[Path],
    model: Path,
    output: Path | None = None,
    output_suffix: str = '.xml',
    device: str = 'cpu',
    creator: str = 'octopy',
    recalculate: int | None = None,
):
    """
    Segments a set of images using a Kraken model
    
    :param files: list of image files to segment.
    :param model: path to Kraken model file.
    :param output: output directory to save the segmented files.
    :param output_suffix: suffix to append to the output file name. e.g. '.seg.xml' results in 'imagename.seg.xml'.
    :param device: device to run the model on. (see Kraken guide)
    :param creator: creator of the PageXML file.
    :param recalculate: recalculate line polygons with this factor. Increases compute time significantly.
    """

    def recalculate_masks(im: Image, res: dict, v_scale: int = 0, h_scale: int = 0):
        """
        Recalculate masks with scale factors.
        Increased mask quality but significantly higher compute time.

        :param im: Image object.
        :param res: Kraken segmentation result dictionary.
        :param v_scale: vertical scale factor.
        :param h_scale: horizontal scale factor.
        :return: overrides result masks
        """
        baselines = list([line['baseline'] for line in res['lines']])
        calculated_masks = calculate_polygonal_environment(im, baselines, scale=(v_scale, h_scale))
        for i, line in enumerate(res['lines']):
            if (mask := calculated_masks[i]) is not None:
                line['boundary'] = mask  # override mask with newly calculated mask

    def kraken_to_list(res: dict) -> list:
        """
        Parses Kraken results.

        :param res: kraken result dictionary.
        :return: list of regions containing dictionaries with type, coords and lines attributes.
        """
        regions: list = []
        for region_type, region_data in res['regions'].items():
            for coords in region_data:
                if (region_type := region_type.lower()) not in REGION_TYPES:
                    region_type = 'other'
                regions.append({
                    'type': region_type,
                    'coords': Polygon.from_kraken_coords(coords),
                    'lines': []
                })
        for line in res['lines']:
            coords_bl = Polygon.from_kraken_coords(line['baseline'])
            coords_mask = Polygon.from_kraken_coords(line['boundary'])
            # find corresponding region
            for region in regions:
                if region['coords'].contains(coords_mask.center()):
                    region['lines'].append({
                        'bl': coords_bl,
                        'mask': coords_mask
                    })
                    break  # line only belongs to one region
        return regions


    if len(files) == 0:
        click.echo('No files found!', err=True)
        return
    click.echo(f'{len(files)} file(s) found.')

    # create output directory:
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    # load model
    try:
        torch_model = TorchVGSLModel.load_model(model)
    except Exception as e:
        click.echo(f'Error loading model: {e}', err=True)
        return
    
    with click.progressbar(files, label='Segmenting files', show_pos=True, show_eta=True, show_percent=True,
                           item_show_func=lambda f: f.name if f is not None else '') as images:
        for image in images:
            img = Image.open(image)
            np = image.name.split('.')  # filename parts

            res = blla.segment(img, model=torch_model, device=device)

            if recalculate is not None:
                res = recalculate_masks(image, res, v_scale=recalculate)

            res = kraken_to_list(res)

            # create PageXML object
            pxml = PageXML.new(creator)
            page = pxml.create_page(imageFilename=f'{np[0]}.{np[-1]}', imageWidth=str(img.size[0]), imageHeight=str(img.size[1]))

            lid = 0
            for rid, rdata in enumerate(res):
                region = page.create_element(ElementType.TextRegion, type=rdata['type'], id=f'r_{rid:03d}')
                region.create_element(ElementType.Coords, points=rdata['coords'].to_page_coords())
                for ldata in rdata['lines']:
                    line = region.create_element(ElementType.TextLine, id=f'l_{lid:03d}')
                    if (mask := ldata['mask']) is not None:
                        line.create_element(ElementType.Coords, points=mask.to_page_coords())
                    if (bl := ldata['bl']) is not None:
                        line.create_element(ElementType.Baseline, points=bl.to_page_coords())
                    lid += 1
            pxml.to_xml((image.parent if output is None else output).joinpath(f'{np[0]}{output_suffix}'))
            
