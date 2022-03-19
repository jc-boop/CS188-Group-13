import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
import requests
import nltk
import spacy
# for spacy and web dictionary install reqs
# pip install -U spacy
# python -m spacy download en_core_web_sm

from tqdm import tqdm
from .utils import DataProcessor
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)

nlp = spacy.load("en_core_web_sm")

class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        ##################################################
        # TODO: Use json python package to load the data
        # properly.
        # We recommend separately storing the two
        # complementary statements into two individual
        # `examples` using the provided class
        # `Coms2SenseSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # Make sure to add to the examples strictly
        # following the `_1` and `_2` order, that is,
        # `sent_1`'s info should go in first and then
        # followed by `sent_2`'s, otherwise your test
        # results will be messed up!
        # For the guid, simply use the row number (0-
        # indexed) for each data instance, i.e. the index
        # in a for loop. Use the same guid for statements
        # coming from the same complementary pair.
        # Make sure to handle if data do not have
        # labels field.
        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))
        
        examples = []

        for i in range(len(data)):
            datum = data[i]
            guid = i
            sentence_1 = datum["sent_1"]
            sentence_2 = datum["sent_2"]

            if "label_1" in datum:
                label_1 = self.label2int[datum["label_1"]]
            else:
                label_1 = None

            if "label_2" in datum:
                label_2 = self.label2int[datum["label_2"]]
            else:
                label_2 = None

            domain = datum["domain"]
            scenario = datum["scenario"]
            numeracy = eval(datum["numeracy"])

            # ------------------------ Additional Knowledge Augmentation --------------
            # ready to find nouns,verbs,Name entities for each sentence
            nlp_list_1 = nlp(sentence_1)
            nlp_list_2 = nlp(sentence_2)

            new_sent_1 = ""
            new_sent_2 = ""

            # named entities list
            list_ent1 = []
            list_ent2 = []

            # nouns list
            list_noun1 = []
            list_noun2 = []
            
            # check the named entities
            for ent1 in nlp_list_1.ents:
                list_ent1.append(ent1.text)
            
            for ent2 in nlp_list_2.ents:
                list_ent2.append(ent2.text)

            # augment additional knowledge if true label ie: 1
            if (label_1 == 1):
                for token1 in nlp_list_1:
                    # potential tag list approach
                    if (token1.pos_ == 'ADJ' or token1.pos_ == 'ADP' or \
                        token1.pos_ == 'PART' or token1.pos_ == 'NOUN' or \
                        token1.pos_ == 'NUM' or  token1.pos_ == 'AUX' or \
                        token1.pos_ == 'ADV' or token1.pos_ == 'CONJ' or  \
                        token1.pos_ == 'CCONJ' or \
                        token1.pos_ == 'VERB' or token1.text in list_ent1):

                        # add unique nouns
                        if (token1.pos_ == 'NOUN'):

                            if token1.text.lower() not in list_noun1:
                                list_noun1.append(token1.text.lower())


                # embedding the additional knowledge 
                info1 = get_additional_knowledge(list_noun1)

                if (info1 != None):
                    sentence_1 = sentence_1[0:-1] + info1


            if (label_2 == 1):
                for token2 in nlp_list_2:
                    # store the part('s, not), noun and verb
                    if (token2.pos_ == 'ADJ' or token2.pos_ == 'ADP' or \
                        token2.pos_ == 'PART' or token2.pos_ == 'NOUN' or \
                        token2.pos_ == 'NUM' or token2.pos_ == 'AUX' or  \
                        token2.pos_ == 'ADV' or token2.pos_ == 'CONJ' or  \
                        token2.pos_ == 'CCONJ' or \
                        token2.pos_ == 'VERB' or token2.text in list_ent1):

                        if (token2.pos_ == 'NOUN'):

                            if token2.text.lower() not in list_noun2:
                                list_noun2.append(token2.text.lower())

                # embedding the additional knowledge
                info2 = get_additional_knowledge(list_noun2)

                if (info2 != None):
                    sentence_2 = sentence_2[0:-1] + info2

            # ----------------End of Additional Knowledge Augmentation --------------
            
            example_1 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=sentence_1,
                label=label_1,
                domain=domain,
                scenario=scenario,
                numeracy=numeracy
            )
            example_2 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=sentence_2,
                label=label_2,
                domain=domain,
                scenario=scenario,
                numeracy=numeracy
            )
            examples.append(example_1)
            examples.append(example_2)

            # loop through the tokens inside the list and find

            # embedding ConceptNet knowedge
            
        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


def get_additional_knowledge(word_list):

    # augment additional knowledge from conceptNet

    addi_info1 = ", where "

    len_list = len(word_list)

    if (len_list > 0):

        if (len_list > 2):
            len_list = 2

        for i in range(0,len_list):

            # check whether request succeed or not
            try:
                cn = requests.get('http://api.conceptnet.io/c/en/' + word_list[i], timeout=5)
            except requests.exceptions.Timeout as e:  # This is the correct syntax
                return None
            
            cn = cn.json()

            len_edges = len(cn['edges'])

            for index in range(0, len_edges):

                string1 = cn['edges'][index]['surfaceText'] 

                # check for null values and language mode
                if string1 != None and cn['edges'][index]["start"]["language"] == "en" and \
                    cn['edges'][index]["end"]["language"] == "en":

                    string1= str(string1)

                    string1 = string1.replace("[", "")
                    string1 = string1.replace("*", "")
                    string1 = string1.replace(".", "")
                    string1 = string1.replace("]", "").lower()

                    
                    addi_info1 = addi_info1 + string1 + ", "
                    
                    # only add one edge from each noun
                    break

        addi_info1 = addi_info1[0:-2] + ".\n"

        return addi_info1

    else: 
        return None
if __name__ == "__main__":

    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()



