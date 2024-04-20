from pathlib import Path

from kraken import blla
from kraken.lib import vgsl
from kraken.lib.segmentation import calculate_polygonal_environment
from PIL import Image

from pagexml import Polygon, PageXML, ElementType


# Methods that need to be changed on Kraken 5.x update:
# parse_result, recalculate_masks

def parse_result(res: dict) -> list:
    """ Parses kraken output """
    data = []
    # read region data
    for region_type, regions in res['regions'].items():
        for coords in regions:
            data.append({
                'rtype': region_type,
                'coords': Polygon.from_kraken_coords(coords),
                'lines': []
            })

    # read line data
    for line in res['lines']:
        baseline = Polygon.from_kraken_coords(line['baseline'])
        mask = Polygon.from_kraken_coords(line['boundary'])
        for region in data:
            if region['coords'].contains(mask.center()):
                region['lines'].append({
                    'baseline': baseline,
                    'mask': mask
                })
                break
    return data


def recalculate_masks(im, res, scale: int):
    """ Recalculate masks from baselines with given vertical scale factor """
    baselines = list([line['baseline'] for line in res['lines']])
    masks = calculate_polygonal_environment(im, baselines, scale=(scale, 0))
    for i, line in enumerate(res['lines']):
        line['boundary'] = masks[i]


def build_pagexml(regions: list, out_file: Path, creator: str, attributes: dict):
    """ Creates PageXML file based on Kraken segmentation """
    pxml = PageXML.new(creator)
    p = pxml.create_page(**attributes)
    lc = 0
    for i, region in enumerate(regions):
        r = p.create_element(ElementType.TextRegion, type=region['rtype'], id=f'r_{i+1:03d}')
        r.create_element(ElementType.Coords, points=region['coords'].to_page_coords())
        for line in region['lines']:
            l = r.create_element(ElementType.TextLine, id=f'l_{lc+1:03d}')
            l.create_element(ElementType.Coords, points=line['mask'].to_page_coords())
            l.create_element(ElementType.Baseline, points=line['baseline'].to_page_coords())
            lc += 1
    pxml.to_xml(out_file)


def image_segment(im: Image, im_name: str, out_file: Path, creator: str, model: Path, device: str = 'cpu', scale: int | None = None):
    """
    Segment an input image and outputs PageXML file.

    :param im: Image object.
    :param im_name: filename of image
    :param out_file: path to output xml file.
    :param creator: creator tag content of xml file.
    :param model: path to torch model.
    :param device: set device to run neural network on.
    :param scale: vertical scale factor for mask recalculation.
    :return: None
    """

    # calculated and checked in main script:
    # if not blla.is_bitonal(im):
    #    click.echo(f'Image {fp.name} is not binarized! Skipping...')
    #    return

    # load model
    m = vgsl.TorchVGSLModel.load_model(model)

    # calculate segmentation
    result = blla.segment(im, model=m, device=device)

    # recalculate masks
    if scale is not None:
        recalculate_masks(im, result, scale)

    # write pagexml
    filename = im_name.split('.')
    build_pagexml(parse_result(result), out_file, creator, {
        'imageFilename': f'{filename[0]}.{filename[-1]}',
        'imageWidth': str(im.size[0]), 'imageHeight': str(im.size[1])
    })
