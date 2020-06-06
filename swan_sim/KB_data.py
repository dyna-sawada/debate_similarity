## KB_data

from collections import Counter
import pandas as pd
import re
import xml.etree.ElementTree as ET
import zipfile



class KnowledgeBaseData:
    def __init__(self, xml, original_text):
        self.xml = xml
        self.root = ET.fromstring(self.xml)
        self.original_text = original_text


    def get_span_text_info(self):
        """
        ## get segmented texts
        << in : xml_file, original_text >>
        xml_file : annotation_result, export file from Swan
        original_text : import file used annotation on Swan

        << out : span_text_dict, LO_span_texts_list >>
        span_text_dict : {span_id : text}
        LO_span_texts_list : [ [LO1_text1, LO1_text2, ...], [LO2_text1, ...], ... ]
                             for culculating word_freqs
        """
        span_text_dict = {}
        LO_span_texts_list = []
        for value in self.root.iter('annotation'):
            span_id = int(value.find('id').text)
            start = int(value.find('start').text)
            end = int(value.find('end').text)
            span_text = self.original_text[start:end]
            ## clean
            span_text = re.sub(r'[,.!?:;"]', '', span_text).lower()

            span_text_dict[span_id] = span_text
            ## ready for word_freqs
            span = value.find('spanType')
            span_type = (span.text if span is not None else '')

            if span_type == 'LO':
                LO_span_texts_list += span_text.split(" ")

        return span_text_dict, LO_span_texts_list


    def get_FB_info(self):
        """
        # make FB dictionaly
        << in : xml, span_text_dict >>
        << out : FB_dict >>
        FB_dict : {to_id : FB_text}
        """
        span_text_dict, _ = self.get_span_text_info()
        FB_dict = {}
        for value in self.root.iter('link'):
            from_id = int(value.find('from').text)
            to_id = int(value.find('to').text)
            label = value.find('.//label')
            relation_type = (label.text if label is not None else '')
            fb_labels = ["GP", "BP", "GP-edge", "BP-edge"]
            if relation_type in fb_labels:
                FB_dict[to_id] = span_text_dict[from_id]
                continue
            else:
                continue

        return FB_dict


    def KB_data(self):
        """
        # make KB_data
        << in : xml, span_text_dict, FB_dict >>
        << out : KB_data >>
        KB_data : [ [from_text, to_text, relation, FB_texts] ]
        """
        KB_data = []
        span_text_dict, _ = self.get_span_text_info()
        FB_dict = self.get_FB_info()
        for value in self.root.iter('link'):
            FB = []
            from_id = int(value.find('from').text)
            to_id = int(value.find('to').text)
            label = value.find('.//label')
            relation_type = (label.text if label is not None else '')

            fb_labels = ["GP", "BP", "GP-edge", "BP-edge"]
            if relation_type in fb_labels:
                continue
            else:
                if from_id in FB_dict:
                    if to_id in FB_dict:
                        FB.append(FB_dict[from_id])
                        FB.append(FB_dict[to_id])
                    else:
                        FB.append(FB_dict[from_id])
                elif to_id in FB_dict:
                    FB.append(FB_dict[to_id])
                else:
                    FB.append("None")

            KB_data.append([span_text_dict[from_id], span_text_dict[to_id], relation_type, ",".join(FB)])

        return KB_data
