from pathlib import Path

import click
from PIL import Image
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.segmentation import calculate_polygonal_environment

from pagexml import Polygon, PageXML, ElementType


"""
Module: Segmentation
Segments a set of images using Kraken and outputs the results as PageXML files.
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
    drop_empty_regions: bool = False,
    default_polygon: int | None = None
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
    :param drop_empty_regions: Drops empty regions
    :param default_polygon: fallback to bbox with fixed height if polygonizer fails.
    """

    def recalculate_masks(im: Image, kraken_res: dict, v_scale: int = 0, h_scale: int = 0,
                          default: int | None = None) -> dict:
        """
        Recalculate masks with scale factors.
        Increased mask quality but significantly higher compute time.

        :param im: Image object.
        :param kraken_res: Kraken segmentation result dictionary.
        :param v_scale: vertical scale factor.
        :param h_scale: horizontal scale factor.
        :param default: fallback to bbox with fixed height if polygonizer fails.
        :return: overrides result masks
        """
        baselines = list([x['baseline'] for x in kraken_res['lines']])
        calculated_masks = calculate_polygonal_environment(im, baselines, scale=(v_scale, h_scale), fallback=default)
        for i, l in enumerate(res['lines']):
            if (m := calculated_masks[i]) is not None:
                l['boundary'] = m  # override mask with newly calculated mask
        return res

    def kraken_to_list(kraken_res: dict, drop_regions: bool) -> list:
        """
        Parses Kraken results.

        :param kraken_res: kraken result dictionary.
        :param drop_regions: Drops empty regions
        :return: list of regions containing dictionaries with type, coords and lines attributes.
        """
        regions: list = []
        for region_type, region_data in kraken_res['regions'].items():
            for coords in region_data:
                if (region_type := region_type.lower()) not in REGION_TYPES:
                    region_type = 'other'
                regions.append({
                    'type': region_type,
                    'coords': Polygon.from_kraken_coords(coords),
                    'lines': []
                })
        for l in kraken_res['lines']:
            coords_bl = Polygon.from_kraken_coords(l['baseline'])
            coords_mask = Polygon.from_kraken_coords(l['boundary'])
            # find corresponding region
            for r in regions:
                if r['coords'].contains(coords_mask.center()):
                    r['lines'].append({
                        'bl': coords_bl,
                        'mask': coords_mask
                    })
                    break  # line only belongs to one region
        if drop_regions:
            for r in regions:
                if len(r['lines']) == 0:
                    regions.remove(r)
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

            res = blla.segment(img, model=torch_model, device=device, fallback=default_polygon)

            if recalculate is not None:
                res = recalculate_masks(img, res, v_scale=recalculate)

            res = kraken_to_list(res, drop_empty_regions)

            # create PageXML object
            pxml = PageXML.new(creator)
            page = pxml.create_page(imageFilename=f'{np[0]}.{np[-1]}', imageWidth=str(img.size[0]),
                                    imageHeight=str(img.size[1]))

            lid = 0
            for rid, rdata in enumerate(res):
                region = page.create_element(ElementType.TextRegion, type=rdata['type'], id=f'r_{rid:03d}')
                region.create_element(ElementType.Coords, points=rdata['coords'].to_page_coords())
                for l_data in rdata['lines']:
                    line = region.create_element(ElementType.TextLine, id=f'l_{lid:03d}')
                    if (mask := l_data['mask']) is not None:
                        line.create_element(ElementType.Coords, points=mask.to_page_coords())
                    if (bl := l_data['bl']) is not None:
                        line.create_element(ElementType.Baseline, points=bl.to_page_coords())
                    lid += 1
            pxml.to_xml((image.parent if output is None else output).joinpath(f'{np[0]}{output_suffix}'))
