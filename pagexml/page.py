from typing_extensions import Self

from lxml import etree

from .types import ElementType
from .element import Element


class Page:
    def __init__(self, attributes: dict):
        self._attributes: dict = attributes
        self._ro: list[str] = []  # reading order by region id's
        self._elements: list[Element] = []

    def __len__(self) -> int:
        """ Return number of elements """
        return len(self._elements)

    def __iter__(self) -> Self:
        """ Iterate through the list of elements """
        self.__n = 0
        return self

    def __next__(self) -> Element:
        """ Iterate through the list of elements """
        if self.__n < len(self._elements):
            element = self._elements[self.__n]
            self.__n += 1
            return element
        else:
            raise StopIteration

    def __getitem__(self, key: str | int) -> str | Element | None:
        """ Get attribute or element with brackets operator """
        if isinstance(key, int) and key < len(self._elements):
            return self._elements[key]
        elif isinstance(key, str) and key in self._attributes:
            return self._attributes[key]
        return None

    def __setitem__(self, key: str | int, value: str | Element):
        """ Set attribute or element with brackets operator """
        if isinstance(key, int) and isinstance(value, Element) and key < len(self._elements):
            self._elements[key] = value
        elif isinstance(key, str) and isinstance(value, str):
            self._attributes[str(key)] = str(value)

    def __contains__(self, key: str | Element) -> bool:
        """ Check if attribute or element exists """
        if isinstance(key, str):
            return key in self._attributes
        elif isinstance(key, Element):
            return key in self._elements
        return False

    @classmethod
    def new(cls, **attributes: dict):
        """ Create a new Page object from scratch """
        return cls(attributes)

    @classmethod
    def from_etree(cls, tree: etree.Element) -> Self:
        """ Create a new Page object from a xml etree element """
        page = cls(dict(tree.items()))

        # reading order
        if (ro := tree.find('./{*}ReadingOrder')) is not None:
            if (ro_elements := tree.findall('../{*}RegionRefIndexed')) is not None:
                page._ro = list([i.get('regionRef') for i in sorted(list(ro_elements), key=lambda i: i.get('index'))])
            tree.remove(ro)

        # elements
        for element in tree:
            page.add_element(Element.from_etree(element), reading_order=False)
        return page

    def to_etree(self) -> etree.Element:
        """ Convert the Page object to a xml etree element """
        # create page element
        page = etree.Element('Page', **self._attributes)

        # create reading order element
        if len(self._ro) > 0:
            reading_order = etree.SubElement(page, 'ReadingOrder')
            order_group = etree.SubElement(reading_order, 'OrderedGroup', id='g0')  # does id matter?
            for i, rid in enumerate(self._ro):
                etree.SubElement(order_group, 'RegionRefIndexed', index=str(i), regionRef=rid)

        # add elements
        for element in self._elements:
            page.append(element.to_etree())

        return page

    @property
    def attributes(self) -> dict:
        """ Page attributes """
        return self._attributes

    @property
    def elements(self) -> list[Element]:
        """ List of elements """
        return self.elements

    @property
    def reading_order(self) -> list[str]:
        """ List of region id's in reading order """
        return self._ro

    @reading_order.setter
    def reading_order(self, new_reading_order: list[str]):
        """ Set the reading order """
        self._ro = new_reading_order

    def set_attribute(self, key: str, value: str):
        """ Set an attribute """
        self._attributes[key] = value

    def remove_attribute(self, key: str):
        """ Remove an attribute """
        self._attributes.pop(key, None)

    def add_element(self, element: Element, index: int | None = None, reading_order: bool = True):
        """ Add an element to the elements list. """
        if index is None:
            self._elements.append(element)
            if element.is_region and reading_order:
                self._ro.append(element.attributes['id'])
        else:
            self._elements.insert(index, element)
            if element.is_region and reading_order:
                self._ro.insert(index, element.attributes['id'])

    def create_element(self, etype: ElementType, index: int = None, **attributes: dict) -> Element:
        """ Create a new element and add it to the elements list """
        element = Element.new(etype, **attributes)
        self.add_element(element, index)
        return element

    def remove_element(self, element: Element):
        """ Remove an element from the elements list """
        self._elements.remove(element)

    def remove_element_by_index(self, index: int) -> Element:
        """ Remove an element from the elements list by its index """
        return self._elements.pop(index)

    def get_regions(self) -> list[Element]:
        """ Get all regions """
        return list([e for e in self._elements if e.is_region()])
