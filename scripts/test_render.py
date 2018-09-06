import render


test_date = {
    'type': 'time',
    'value': {
        'time': '-0020-01-01T00:00:00Z',
        'timezone': 0,
        'after': 0,
        'before': 0,
        'precision': 11,
        'calendarmodel': 'http://www.wikidata.org/entity/Q1985786',
    }
}

test_quantity = {
    'type': 'quantity',
    'value': {
        'amount': '+170',
        'unit': 'feet'
    }
}

print(test_date)
print(render.process_literal(test_date))
test_date['value']['precision'] = 9
print(render.process_literal(test_date))

print(test_quantity)
print(render.process_literal(test_quantity))
