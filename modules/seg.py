from pathlib import Path
from typing import Optional

import click
from PIL import Image
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.segmentation import calculate_polygonal_environment

from pagexml import PageXML, Element, ElementType
from pagexml.geometry import Polygon
from modules.preproc import binarize
from modules.postproc import (pxml_remove_empty_regions, 
                              pxml_merge_overlapping_regions, 
                              pxml_sort_lines, pxml_sort_regions)


"""
Module: Segmentation
Segmentes a set of images using Kraken and outputs the results as PageXML files.
"""


__all__ = ['segment']


FALLBACK_LINE_POLYGON_BASELINE = (0, 10, 0, 5)
FALLBACK_LINE_POLYGON_TOPLINE = (0, 5, 0, 10)
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
    segmentation_model: Path,
    xml_output: Optional[Path] = None,
    xml_suffix: str = '.xml',
    xml_creator: str = 'octopy',
    device: str = 'cpu',
    recalculate: Optional[int] = None,
    fallback_line_polygon: bool = False,
    drop_empty_regions: bool = False,
    merge_overlapping_regions: bool = False,
    sort_regions: bool = False,
    sort_lines: bool = False,
):
    """
    Segment a set of images using Kraken

    :param files: list of input image files.
    :param segmentation_model: path to kraken pytorch model file.
    :param xml_output: output directory for PageXML files. Writes file to same directory as input file if not set.
    :param xml_suffix: suffix for PageXML files.
    :param xml_creator: sets creator in meta data.
    :param device: device to run network on (see kraken guide).
    :param recalculate: if set, recalculate line polygons with this factor. Increases compute time significantly.
    :param fallback_line_polygon: Sets line polygon to fixed size around baseline if kraken polygonizer fails. Drops line if set to None.
    :param drop_empty_regions: Drops regions without lines.
    :param merge_overlapping_regions: merge overlapping text regions.
    :param sort_regions: sort regions by their center x coordinates (from left to right).
    :param sort_lines: sort lines for each region by their center y coordinates (from top to bottom).
    """

    def recalculate_masks(img: Image, res: dict, topline: bool, 
                          fallback: Optional[tuple[int, int, int, int]] = None,  
                          v_scale: int = 0, h_scale: int = 0):
        """
        Recalculate masks with scale factors.
        Increased mask quality but significantly higher compute time.
        This method needs to be rewritten on Kraken >= 5.0.0.

        :param img: Image object.
        :param res: Kraken segmentation result dictionary.
        :param model: torch model object.
        :param fallback: fallback tuple or None
        :param v_scale: vertical scale factor.
        :param h_scale: horizontal scale factor.
        :return: overrides result masks
        """
        baselines = list([line['baseline'] for line in res['lines']])
        calculated_masks = calculate_polygonal_environment(
            img, 
            baselines, 
            scale=(v_scale, h_scale), 
            fallback_line_polygon=fallback, 
            topline=topline
        )
        for i, line in enumerate(res['lines']):
            if (mask := calculated_masks[i]) is not None:
                line['boundary'] = mask  # override mask with newly calculated mask

    def kraken_to_regions(kraken_res: dict) -> list[Element]:
        """
        Reads kraken output and parses it to a list of PageXMl Elements.
        This method needs to be rewritten on Kraken >= 5.0.0.

        :param kraken_res: Result from kraken segmentation.
        """
        # parse regions
        regions: list[Element] = []
        rid = 0
        for region_type, region_data in kraken_res['regions'].items():
            for coords in region_data:
                if (region_type := region_type.lower()) not in REGION_TYPES:
                    region_type = 'other'
                r = Element.new(ElementType.TextRegion, type=region_type, id=f'r_{rid:03d}')
                r.create_element(ElementType.Coords, points=Polygon.from_kraken_coords(coords).to_page_coords())
                regions.append(r)
                rid += 1

        # parse lines
        lines: list[Element] = []
        lid = 0
        for line in res['lines']:
            l = Element.new(ElementType.TextLine, id=f'l_{lid:03d}')
            l.create_element(ElementType.Coords, points=Polygon.from_kraken_coords(line['boundary']).to_page_coords())
            l.create_element(ElementType.Baseline, points=Polygon.from_kraken_coords(line['baseline']).to_page_coords())
            lines.append(l)
            lid += 1
        
        # merge lines into matching regions
        for line in lines:
            for region in regions:
                if Polygon.from_page_coords(region.get_coords_element()['points']).contains(Polygon.from_page_coords(line.get_coords_element()['points']).center()):
                    region.add_element(line)
                    break

        return regions


    if len(files) < 1:
        click.echo('No files found!', err=True)
        return
    click.echo(f'{len(files)} file(s) found.')

    # create output directory:
    if xml_output is not None:
        xml_output.mkdir(parents=True, exist_ok=True)

    # load model
    try:
        torch_model = TorchVGSLModel.load_model(segmentation_model)
        topline = torch_model.user_metadata['topline'] if 'topline' in torch_model.user_metadata else False
        if fallback_line_polygon is None:
            fb = None
        else:
            fb = FALLBACK_LINE_POLYGON_TOPLINE if topline else FALLBACK_LINE_POLYGON_BASELINE
    except Exception as e:
        click.echo(f'Error loading model: {e}', err=True)
        return
    
    # process files
    with click.progressbar(files, label='Segmenting files', show_pos=True, show_eta=True, show_percent=True,
                           item_show_func=lambda f: f.name if f is not None else '') as images:
        for image in images:
            im = Image.open(image)

            if not blla.is_bitonal(im):
                im = binarize(image=im)

            res = blla.segment(im=im, model=torch_model, device=device, fallback_line_polygon=fb)
            if recalculate is not None:
                recalculate_masks(im, res, topline=topline, v_scale=recalculate, fallback=fb)

            # postprocess output
            regions = kraken_to_regions(res)
            if drop_empty_regions:
                pxml_remove_empty_regions(regions, inplace=True)
            if merge_overlapping_regions:
                regions = pxml_merge_overlapping_regions(regions)
            if sort_regions:
                pxml_sort_regions(regions, inplace=True)
            if sort_lines:
                for region in regions:
                    pxml_sort_lines(region.elements, inplace=True)
            
            # create PageXML object
            pxml = PageXML.new(xml_creator)
            page = pxml.create_page(imageFilename=f'{image.name.split(".")[0]}{image.suffix}', imageWidth=str(im.size[0]), imageHeight=str(im.size[1]))
            for region in regions:
                page.add_element(region)
            pxml.to_xml((image.parent if xml_output is None else xml_output).joinpath(f'{image.name.split(".")[0]}{xml_suffix}'))
