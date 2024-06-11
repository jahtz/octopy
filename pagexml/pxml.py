from pathlib import Path
from typing_extensions import Self
from datetime import datetime

from lxml import etree

from .page import Page
from . import schema


class PageXML:
    def __init__(self, creator: str, created: str, last_change: str):
        self._creator: str = creator
        self._created: str = created
        self._last_change: str = last_change
        self._pages: list[Page] = []

    def __len__(self) -> int:
        """ Return number of pages """
        return len(self._pages)

    def __iter__(self) -> Self:
        """ Iterate through the list of pages """
        self.__n = 0
        return self

    def __next__(self) -> Page:
        """ Iterate through the list of pages """
        if self.__n < len(self._pages):
            page = self._pages[self.__n]
            self.__n += 1
            return page
        else:
            raise StopIteration

    def __getitem__(self, key: int) -> Page | None:
        """ Get page with brackets operator """
        if key < len(self._pages):
            return self._pages[key]
        return None

    def __setitem__(self, key: int, value: Page):
        """ Set page with brackets operator """
        if key < len(self._pages):
            self._pages[key] = value

    def __contains__(self, key: Page) -> bool:
        """ Check if page exists """
        if isinstance(key, Page):
            return key in self._pages
        return False

    @classmethod
    def new(cls, creator: str = 'ZPD Wuerzburg'):
        """ Create a new PageXML object from scratch """
        return cls(creator, datetime.now().isoformat(), datetime.now().isoformat())

    @classmethod
    def from_etree(cls, tree: etree.Element) -> Self:
        """ Create a new PageXML object from a xml etree element """
        # PageXML element with metadata
        if (md_tree := tree.find('./{*}Metadata')) is not None:
            if (creator := md_tree.find('./{*}Creator')) is not None:
                creator = creator.text
            if (created := md_tree.find('./{*}Created')) is not None:
                created = created.text
            if (last_change := md_tree.find('./{*}LastChange')) is not None:
                last_change = last_change.text
            pxml = cls(creator, created, last_change)
        else:
            pxml = cls.new()

        # page elements
        if (pages := tree.findall('./{*}Page')) is not None:
            for page_tree in pages:
                pxml.add_page(Page.from_etree(page_tree))

        return pxml

    @classmethod
    def from_xml(cls, fp: str | Path) -> Self:
        """ Create a new PageXML object from a xml string """
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(fp, parser).getroot()
        return cls.from_etree(tree)

    def to_etree(self):
        """ Convert the PageXML object to a xml etree element """
        self.last_change_now()

        # create root element
        xsi_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance", 'schemaLocation')
        nsmap = {None: schema.xmlns, 'xsi': schema.xmlns_xsi}
        root = etree.Element('PcGts', {xsi_qname: schema.xsi_schemaLocation}, nsmap=nsmap)

        # create metadata element
        metadata = etree.SubElement(root, 'Metadata')
        etree.SubElement(metadata, 'Creator').text = self._creator
        etree.SubElement(metadata, 'Created').text = self._created
        etree.SubElement(metadata, 'LastChange').text = self._last_change

        # create page elements
        for page in self._pages:
            root.append(page.to_etree())

        return root

    def to_xml(self, fp: str | Path):
        """ Write the PageXML object to a file """
        with open(fp, 'wb') as f:
            f.write(etree.tostring(self.to_etree(), pretty_print=True, encoding='utf-8', xml_declaration=True))

    @property
    def creator(self) -> str:
        """ Creator of the PageXML file """
        return self._creator

    @creator.setter
    def creator(self, creator: str):
        """ Set the creator of the PageXML file """
        self._creator = creator

    @property
    def created(self) -> str:
        """ Date and time of the creation of the PageXML file (ISO format)"""
        return self._created

    @created.setter
    def created(self, created: str):
        """ Set the date and time of the creation of the PageXML file (ISO format)"""
        self._created = created

    @property
    def last_change(self) -> str:
        """ Date and time of the last change of the PageXML file (ISO format)"""
        return self._last_change

    @last_change.setter
    def last_change(self, last_change: str):
        """ Set the date and time of the last change of the PageXML file (ISO format)"""
        self._last_change = last_change

    def last_change_now(self):
        """ Update the last_change attribute to the current time """
        self._last_change = datetime.now().isoformat()

    @property
    def pages(self) -> list[Page]:
        """ List of pages """
        return self._pages

    def add_page(self, page: Page, index: int = None):
        """ Add a page to the pages list """
        if index is None:
            self._pages.append(page)
        else:
            self._pages.insert(index, page)

    def create_page(self, index: int = None, **attributes) -> Page:
        """ Create a new page and add it to the pages list """
        page = Page.new(**attributes)
        self.add_page(page, index)
        return page

    def remove_page(self, page: Page):
        """ Remove a page from the pages list """
        self._pages.remove(page)

    def remove_page_by_index(self, index: int) -> Page:
        """ Remove a page at a specific index """
        return self._pages.pop(index)

    def clear_pages(self):
        """ Remove all pages """
        self._pages.clear()
