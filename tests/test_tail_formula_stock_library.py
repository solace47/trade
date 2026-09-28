from trade_research.tail_formula_stock_library import namespace, previous, rename


def test_renaming_is_token_based_and_invertible():
    mapping = {'HDV01': 'U01HDV01', 'W01': 'X01'}
    text = 'HDV01+HDV010+W01+VW01+REF(HDV01,B0)'
    renamed = rename(text, mapping)
    assert renamed == 'U01HDV01+HDV010+X01+VW01+REF(U01HDV01,B0)'
    assert rename(renamed, {v:k for k,v in mapping.items()}) == text


def test_colliding_block_features_and_helpers_stay_distinct():
    blocks = [('first', {'W01': 'HDV01/V01'}, previous.HEADER+'HDV01:=REF(V,B0);\n'),
              ('second', {'W01': 'HDV01+V01'}, previous.HEADER+'HDV01:=SUM(V,4);\n')]
    expressions, header, mappings = namespace(blocks)
    assert expressions['X01'] == 'U01HDV01/V01'
    assert expressions['X02'] == 'U02HDV01+V01'
    assert 'U01HDV01:=REF(V,B0);' in header
    assert 'U02HDV01:=SUM(V,4);' in header
    assert expressions['V01'] == previous.EXPRESSIONS['V01']
    assert mappings[0]['features'] != mappings[1]['features']
