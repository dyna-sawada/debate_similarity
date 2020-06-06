## get some info (test)

import pandas as pd
import re


def convert_to_node_info_html(original_text, xml):
    span_text_list = []
    for value in root.iter('annotation'):
        span_id = value.find('id').text
        start = int(value.find('start').text)
        end = int(value.find('end').text)
        span_text = original_text[start:end]
        span_text_list.append(span_text)
    return span_text_list

def convert_to_edge_info_html(original_text, xml):
    edge_info_html = ''
    root = ET.fromstring(xml)
    for value in root.iter('link'):
        from_id = value.find('from').text
        to_id = value.find('to').text
        label = value.find('.//label')
        relation_type = (label.text if label is not None else '')
    return from_id, to_id, relation_type


df = pd.read_csv('./original_data.tsv', delimiter='\t')
print(df)
original_text = (df[df['name'] == debate_id])[:1]['text'].values[0]
print(original_text)

"""
xml = zip_data.read(filename)
node_info_html = convert_to_node_info_html(original_text, xml)
edge_info_html = convert_to_edge_info_html(original_text, xml)
"""
