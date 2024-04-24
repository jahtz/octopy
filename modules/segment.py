from pathlib import Path

from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib.segmentation import calculate_polygonal_environment
from PIL import Image

from pagexml import Polygon, PageXML, ElementType


# Methods that need to be changed on Kraken 5.x update:
# parse_result, recalculate_masks

def parse_result(res: dict) -> list:
    """ Parses kraken output """
    regions: list = []

    # parse region data
    for region_type, region_data in res['regions'].items():
        for r in region_data:
            regions.append({
                'rtype': region_type,
                'coords': Polygon.from_kraken_coords(r),
                'lines': []
            })

    # parse line data and assign to regions
    for line in res['lines']:
        bl_coords = Polygon.from_kraken_coords(line['baseline'])
        mask_coords = Polygon.from_kraken_coords(line['boundary'])
        for region in regions:
            if region['coords'].contains(mask_coords.center()):
                region['lines'].append({
                    'baseline': bl_coords,
                    'mask': mask_coords
                })
                break  # line can only belong to one region
    return regions


def recalculate_masks(im, res, scale: int):
    """ Recalculate masks from baselines with given vertical scale factor """
    baselines = list([line['baseline'] for line in res['lines']])
    calculated_masks = calculate_polygonal_environment(im, baselines, scale=(scale, 0))
    for i, line in enumerate(res['lines']):
        if (mask := calculated_masks[i]) is not None:
            line['boundary'] = mask  # override mask with newly calculated mask


def build_pagexml(regions: list, out_file: Path, creator: str, attributes: dict):
    """ Creates PageXML file based on Kraken segmentation """
    pxml = PageXML.new(creator)  # create pagexml object
    page = pxml.create_page(**attributes)  # create page element
    line_id_counter = 0

    for i, region_data in enumerate(regions):
        region = page.create_element(ElementType.TextRegion, type=region_data['rtype'], id=f'r_{i+1:03d}')
        region.create_element(ElementType.Coords, points=region_data['coords'].to_page_coords())

        for line_data in region_data['lines']:
            line = region.create_element(ElementType.TextLine, id=f'l_{line_id_counter+1:03d}')
            if (mask := line_data['mask']) is not None:
                line.create_element(ElementType.Coords, points=mask.to_page_coords())
            if (baseline := line_data['baseline']) is not None:
                line.create_element(ElementType.Baseline, points=baseline.to_page_coords())
            line_id_counter += 1

    pxml.to_xml(out_file)  # save file


def image_segment(
        im: Image,
        im_name: str,
        out_file: Path,
        creator: str,
        model: TorchVGSLModel,
        device: str = 'cpu',
        scale: int | None = None
):
    """
    Segment an input image and outputs PageXML file.

    :param im: Image object.
    :param im_name: filename of image
    :param out_file: path to output xml file.
    :param creator: creator tag content of xml file.
    :param model: torch VGSLModel.
    :param device: set device to run neural network on.
    :param scale: vertical scale factor for mask recalculation.
    :return: None
    """

    # calculate segmentation
    result = blla.segment(im, model=model, device=device)

    # recalculate masks, overrides masks in result
    if scale is not None:
        recalculate_masks(im, result, scale)

    # write pagexml
    filename = im_name.split('.')
    build_pagexml(parse_result(result), out_file, creator, {
        'imageFilename': f'{filename[0]}.{filename[-1]}',
        'imageWidth': str(im.size[0]),
        'imageHeight': str(im.size[1])
    })
