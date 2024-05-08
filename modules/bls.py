from pathlib import Path

import click
from PIL import Image
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.segmentation import calculate_polygonal_environment

from modules.preproc import normalize as im_nrm
from modules.preproc import binarize as im_bin
from modules.helper import normalize_suffix, path_parser
from pagexml import Polygon, PageXML, ElementType


__all__ = ['bls_workflow']

REGION_TYPES = ['paragraph', 'heading', 'caption', 'header', 'footer', 'page-number', 'drop-capital', 'credit',
                'floating', 'signature-mark', 'catch-word', 'marginalia', 'footnote', 'footnote-continued', 'endnote',
                'TOC-entry', 'list-label', 'other']


def _kraken_parser(res: dict) -> list:
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


def _recalc_masks(im: Image, res: dict, v_scale: int = 0, h_scale: int = 0):
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


def bls_workflow(
        _input: tuple,
        output: Path | None = None,
        model: Path | None = None,
        binarize: bool = False,
        normalize: bool = False,
        segment: bool = False,
        binsuffix: str = '.bin.png',
        nrmsuffix: str = '.nrm.png',
        segsuffix: str = '.xml',
        device: str = 'cpu',
        creator: str = 'ZPD Wuerzburg',
        scale: int | None = None,
        threshold: int = 50
):
    """
    Implements baseline segmentation module workflow.
    Takes input images and binarize (for segmentation), normalize (for recognition) and segment them.
    Outputs PageXML files.

    :param _input: tuple of glob expression for input files.
    :param output: change output directory.
    :param model: Kraken model used for segmentation. Only necessary if segment is set to True.
    :param binarize: create binarized image files.
    :param normalize: create normalized image files.
    :param segment: segment images using Kraken.
    :param binsuffix: suffix of binarized images.
    :param nrmsuffix: suffix of normalized images.
    :param segsuffix: suffix of PageXML files.
    :param device: device for computation.
    :param creator: creator tag in PageXML metadata.
    :param scale: If set, recalculate line polygons with this factor. Increases compute time significantly.
    :param threshold: threshold for binarization.
    :return: None
    """

    # load files
    fl = path_parser(_input)
    if len(fl) == 0:
        click.echo('No files found!', err=True)
        return
    click.echo(f'Files found: {len(fl)}')

    # set and create output directory
    output.mkdir(parents=True, exist_ok=True)

    # load Torch model
    if model is None and segment:
        click.echo('No model set for segmentation!', err=True)
        return
    m = None
    if segment:
        m = TorchVGSLModel.load_model(model)
        click.echo('Model loaded')

    with click.progressbar(fl, label='Processing files', show_pos=True, show_eta=True, show_percent=True,
                           item_show_func=lambda x: f'Current file: {x.name}' if x is not None else '') as images:
        for image in images:
            im = Image.open(image)  # load image
            np = image.name.split('.')  # filename parts

            # normalize
            if normalize:
                im_nrm(im, output.joinpath(f'{np[0]}{normalize_suffix(nrmsuffix)}'))

            # binarize
            im = im_bin(im, output.joinpath(f'{np[0]}{normalize_suffix(binsuffix)}') if binarize else None, threshold)

            # segment
            if segment:
                res = blla.segment(im, model=m, device=device)
                if scale:
                    _recalc_masks(im, res, v_scale=scale)

                res = _kraken_parser(res)  # parse kraken output to usable data

                # build PageXML
                pxml = PageXML.new(creator)
                page = pxml.create_page(imageFilename=f'{np[0]}.{np[-1]}', imageWidth=str(im.size[0]), imageHeight=str(im.size[1]))

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
                pxml.to_xml(output.joinpath(f'{np[0]}{normalize_suffix(segsuffix)}'))
