from typing_extensions import Self

from lxml import etree

from .types import ElementType


class Element:
    def __init__(self, etype: ElementType, attributes: dict):
        self._etype: ElementType = etype
        self._attributes: dict = attributes
        self._elements: list[Element] = []
        self._text: str | None = None

    def __len__(self) -> int:
        """ Return number of elements """
        return len(self._elements)

    def __iter__(self) -> Self:
        """ Iterate through the list of elements """
        self.__n = 0
        return self

    def __next__(self) -> Self:
        """ Iterate through the list of elements """
        if self.__n < len(self._elements):
            element = self._elements[self.__n]
            self.__n += 1
            return element
        else:
            raise StopIteration

    def __getitem__(self, key: str | int) -> str | Self | None:
        """ Get attribute or element with brackets operator """
        if isinstance(key, int) and key < len(self._elements):
            return self._elements[key]
        elif isinstance(key, str) and key in self._attributes:
            return self._attributes[key]
        return None

    def __setitem__(self, key: str | int, value: str | Self):
        """ Set attribute or element with brackets operator """
        if isinstance(key, int) and isinstance(value, Element) and key < len(self._elements):
            self._elements[key] = value
        elif isinstance(key, str) and isinstance(value, str):
            self._attributes[str(key)] = str(value)

    def __contains__(self, key: str | Self) -> bool:
        """ Check if attribute or element exists """
        if isinstance(key, str):
            return key in self._attributes
        elif isinstance(key, Element):
            return key in self._elements
        return False

    @classmethod
    def new(cls, etype: ElementType, **attributes: dict):
        """ Create a new Element object from scratch """
        return cls(etype, attributes)

    @classmethod
    def from_etree(cls, tree: etree.Element) -> Self:
        """ Create a new Element object from a xml etree element """
        element = cls(ElementType(tree.tag.split('}')[1]), dict(tree.items()))
        element.text = tree.text
        for child in tree:
            element.add_element(Element.from_etree(child))
        return element

    def to_etree(self) -> etree.Element:
        """ Convert the Element object to a xml etree element """
        # create element
        element = etree.Element(self._etype.value, **self._attributes)
        if self._text is not None:
            element.text = self._text

        # add elements
        for child in self._elements:
            element.append(child.to_etree())

        return element

    @property
    def etype(self) -> ElementType:
        """ Element type """
        return self._etype

    @property
    def attributes(self) -> dict:
        """ Element attributes """
        return self._attributes

    @property
    def text(self) -> str | None:
        """ Element text """
        return self._text

    @text.setter
    def text(self, value: str | None):
        """ Set the element text """
        self._text = value

    @property
    def elements(self) -> list[Self]:
        """ List of elements """
        return self._elements

    def is_region(self) -> bool:
        """ Check if the element is a region """
        return 'Region' in self._etype.value

    def contains_text(self) -> bool:
        """ Check if the element contains text """
        return self._text is not None

    def set_attribute(self, key: str, value: str):
        """ Set an attribute """
        self._attributes[key] = value

    def remove_attribute(self, key: str):
        """ Remove an attribute """
        self._attributes.pop(key, None)

    def add_element(self, element: Self, index: int | None = None):
        """ Add an element to the elements list. """
        if index is None:
            self._elements.append(element)
        else:
            self._elements.insert(index, element)

    def create_element(self, etype: ElementType, index: int = None, **attributes: dict) -> Self:
        """ Create a new element and add it to the elements list """
        element = Element.new(etype, **attributes)
        self.add_element(element, index)
        return element

    def remove_element_by_index(self, index: int) -> Self:
        """ Remove an element from the elements list by its index """
        return self._elements.pop(index)

    def get_coords_element(self) -> Self | None:
        """ Returns the first Coords element. None if nothing found """
        for element in self._elements:
            if element.etype == ElementType.Coords:
                return element
        return None

    def get_baseline_element(self):
        """ Returns the first Baseline element. None if nothing found """
        for element in self._elements:
            if element.etype == ElementType.Baseline:
                return element
        return None
