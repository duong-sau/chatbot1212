import itertools
from bs4 import BeautifulSoup
# define
from Static.Define import Tag


def groupSoupByTag(soup: BeautifulSoup, tag_type: Tag, group_type: Tag) -> None:
    head_list = soup.find_all(tag_type)
    for e in head_list:
        els = [i for i in itertools.takewhile(lambda x: x.name not in [e.name, 'script'], e.next_siblings)]
        section = soup.new_tag(name=group_type)
        section['id'] = head_list.index(e)
        e.wrap(section)
        for new_ in els:
            section.append(new_)
