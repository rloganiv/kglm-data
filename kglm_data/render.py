"""
Functions for rendering date and quantity literals.
"""
from collections import namedtuple
import re

from pint import UnitRegistry
from pint.errors import UndefinedUnitError
ureg = UnitRegistry()


# See https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Dates_and_numbers
Format = namedtuple('Format', ['format_string', 'include_era',
                               'remove_leading_zeros'])


YEAR_FORMATS = [
    Format('%Y', True, True)
]


MONTH_FORMATS = [
    Format('%B %Y', True, True),
    Format('%b %Y', True, True),
    *YEAR_FORMATS
]

DAY_FORMATS = [
    Format('%d %B %Y', True, True),
    Format('%d %b %Y', True, True),
    Format('%B %d, %Y', True, True),
    Format('%b %d, %Y', True, True),
    Format('%d %B', False, True),
    Format('%d %b', False, True),
    Format('%B %d', False, True),
    Format('%b %d', False, True),
    Format('%Y-%m-%d', False, False),
    *MONTH_FORMATS
]

RE_LEADING_ZEROS = re.compile(r'((?<=\s)0+|^0+)')
RE_ISO_8601 = re.compile(r'(?P<year>[+-][0-9]+)-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})(?=T)')
RE_WIKIDATA_ENTITY = re.compile(r'http://www.wikidata.org/entity/Q[0-9]+')

class Date(object):

    LONG_MONTHS = [None, 'januari', 'februari', 'mars', 'april', 'maj', 'juni', 'juli',
                   'augusti', 'september', 'oktober', 'november', 'december']

    SHORT_MONTHS = [None, 'jan', 'feb', 'mar', 'apr', 'maj', 'jun', 'jul', 'aug',
                    'sep', 'oct', 'nov', 'dec']

    def __init__(self, year, month=None, day=None):
        self._year = year
        self._month = month
        self._day = day

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    @property
    def day(self):
        return self._day

    def strftime(self, format_string):
        out = format_string
        if self._day:
            out = out.replace('%d', '%02d' % self._day)
        if self._month:
            out = out.replace('%b', Date.SHORT_MONTHS[self._month])
            out = out.replace('%m', '%02d' % self._month)
            out = out.replace('%B', Date.LONG_MONTHS[self._month])
        out = out.replace('%Y', '%d' % abs(self._year))
        return out


def parse_iso8601(iso_string: str):
    match = RE_ISO_8601.match(iso_string)
    if match:
        year = int(match.group('year'))
        month = int(match.group('month'))
        day = int(match.group('day'))
        return Date(year, month, day)


def custom_strftime(formats, date):
    out = []
    for format in formats:
        date_string = date.strftime(format.format_string)
        if format.remove_leading_zeros:
            date_string = RE_LEADING_ZEROS.sub('', date_string)
        out.append(date_string)
        if format.include_era:
            is_bc = date.year < 0
            era_strings = ['BC', 'BCE'] if is_bc else ['AD', 'CE']
            for era_string in era_strings:
                out.append(' '.join((era_string, date_string)))
                out.append(' '.join((date_string, era_string)))
    return out


def render_time(value):
    posix_string = value['time']
    precision = int(value['precision'])
    date = parse_iso8601(posix_string)
    if precision == 11:  # Day level precision
        return custom_strftime(DAY_FORMATS, date)
    if precision == 10:  # Month level prevision
        return custom_strftime(MONTH_FORMATS, date)
    elif precision < 10:  # Year level precision or less
        return custom_strftime(YEAR_FORMATS, date)


def render_quantity(value, alias_db):
    amount = float(value['amount'])
    unit = value['unit']
    if unit == '1':
        return [str(amount)]
    else:
        Q_ = ureg.Quantity
        try:
            quantity = Q_(amount, unit)
        except UndefinedUnitError:
            units = []
            if RE_WIKIDATA_ENTITY.match(unit):
                units = alias_db.get(unit.split("/")[-1], [])
            return [str(amount)] + [str(amount) + " " + u for u in units]
        f_string = '{0:{1:}f} {2:{3:}}'
        precisions = ['', '1.0', '0.1', '0.2']
        unit_formats = ['', '~']
        out = [quantity.format_babel(locale='sv_SE')]
        for precision in precisions:
            for unit_format in unit_formats:
                out.append(f_string.format(quantity.magnitude,
                                           precision,
                                           quantity.units,  # TODO: Translate units
                                           unit_format))
        return out


def process_literal(value, alias_db):
    literal = aliases = None
    if value['type'] == 'time':
        literal = 'T::%i::%s' % (value['value']['precision'],
                                 value['value']['time'])
        aliases = render_time(value['value'])
    elif value['type'] == 'quantity':
        literal = 'V::%0.4f::%s' % (float(value['value']['amount']),
                                    value['value']['unit'])
        aliases = render_quantity(value['value'], alias_db)
    # elif value['type'] == 'string':
    #     literal = value['value']
    #     aliases =  [value['value']]
    # elif value['type'] == 'monolingualtext':
    #     literal = value['value']['text']
    #     aliases = [value['value']['text']]
    return literal, aliases

