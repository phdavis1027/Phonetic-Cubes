from re import I
from typing import NamedTuple
import networkx as nx
from collections import namedtuple
import pyarrow.feather as feather
import pandas as pd
from os import path, write
import copy
from enum import Enum
from itertools import chain, combinations
import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns

reducer = umap.UMAP()
inventories = feather.read_feather("../public/contrastive-symmetry/inventories.feather")

class Features(Enum):
    ANTERIOR = 'anterior',
    LABIAL = 'labial',
    BACK = 'back',
    HIGH = 'high',
    ATR = 'atr',
    LATERAL = 'lateral',
    DISTRIBUTED = 'distributed',
    LONG = 'long',
    SPREAD = 'spread',
    LOW = 'low',
    SYLLABIC = 'syllabic',
    CONTINUANT = 'continuant',
    NASAL = 'nasal',
    SONORANT = 'sonorant',
    CORONAL = 'coronal',
    TENSE = 'tense',
    CONSTR = 'constr',
    VOCALIC = 'vocalic',
    STRIDENT = 'stident',
    EXTRA = 'extra',
    CONSONANTAL = 'consonantal',
    VOICE = 'voice',
    ROUND = 'round'


def powerset(set: list):
    return chain.from_iterable(combinations(set,r) for r in range(len(set) + 1))

features_powerset = powerset([f for f in Features])
def extract_segments(inv: str, data: pd.DataFrame = inventories, inv_type: str ='Whole') -> tuple:
    segments = []
    inv_records = data.loc[data['language'] == inv]
    inv_records = inv_records.loc[inv_records['segment_type'] == inv_type]
    print(inv_records)
    for row in inv_records.itertuples():
        segment = []
        if row.voice == '+':
            segment.append(Features.VOICE)
        if row.anterior == '+':
            segment.append(Features.ANTERIOR)
        if row.labial == '+':
            segment.append(Features.LABIAL)
        if row.back == '+':
            segment.append(Features.BACK)
        if row.high == '+':
            segment.append(Features.HIGH)
        if row.ATR == '+':
            segment.append(Features.ATR)
        if row.lateral == '+':
            segment.append(Features.LATERAL)
        if row.distributed == '+':
            segment.append(Features.DISTRIBUTED)
        if row.LONG == '+':
            segment.append(Features.LONG)
        if row.spread == '+':
            segment.append(Features.SPREAD)
        if row.low == '+':
            segment.append(Features.LOW)
        if row.syllabic == '+':
            segment.append(Features.SYLLABIC)
        if row.continuant == '+':
            segment.append(Features.CONTINUANT)
        if row.nasal == '+':
            segment.append(Features.NASAL)
        if row.sonorant == '+':
            segment.append(Features.SONORANT)
        if row.coronal == '+':
            segment.append(Features.CORONAL)
        if row.tense == '+':
            segment.append(Features.TENSE)
        if row.constr == '+':
            segment.append(Features.CONSTR)
        if row.vocalic == '+':
            segment.append(Features.VOCALIC)
        if row.strident == '+':
            segment.append(Features.STRIDENT)
        if row.EXTRA == '+':
            segment.append(Features.EXTRA)
        if row.consonantal == '+':
            segment.append(Features.CONSONANTAL)
        if row.round == '+':
            segment.append(Features.ROUND)
        segments.append(segment)

    return segments

# @segments : an iterable of feature bundles, which themselves are 
# are iterables
#
# returns : the smallest seat of features required to distinguish
# that set of features from each other
# 
# this implementation uses the top-down approach from
# mackie and mielke 2011

def distinguishes(segments, features):
    indices = []
    for segment in segments:
        seg_index = {f:0 for f in features} # has to be a dictionary so order doesn't get in the way
        for k in seg_index.keys():
            if k in segment: 
                seg_index[k] = 1 # assign an index to every segment based on the segments
        if seg_index in indices: # if we've seen that index before, it can't be distinguished
            return False
        else:
            indices.append(seg_index)
    return True

def feature_economist(segments: list) -> NamedTuple:
    result = namedtuple("result", "features num_features")
    total_set = {feature for feature in Features}
    n = len(total_set)
    last_valid = set()
    last_length = n
    for subset in features_powerset:
        current_length = 23 - len(subset)
        clone = copy.copy(total_set) 
        for member in subset:
            clone.remove(member)        
        if not distinguishes(segments, clone):
            return result(last_valid, n)
        else:
            if current_length < last_length:
                n = current_length
                last_length = current_length
            last_valid = clone


def feature_economist_bottom_up(segments: list):
    for subset in features_powerset:
        if distinguishes(segments, subset):
            return subset



class Inventory:
    def __init__(self, segment_store: set = set()):
        self.segment_store = segment_store
        self.feature_data = feature_economist_bottom_up(segment_store) 
        self.current_index = 0
        self.feature_map = dict()
        self.index_map = dict()
        for feature in self.feature_data:
            self.feature_map[feature] = self.current_index
            self.index_map[self.current_index] = feature
            self.current_index += 1

    def gen_cube(self) -> nx.Graph:
        G = nx.hypercube_graph(self.current_index)

        coord_set = set()
        for segment in self.segment_store:
            coord_set.add(self.gen_coord(segment))

        attrs = {node : (1 if node in coord_set else 0) for node in G.nodes}
        nx.set_node_attributes(G, attrs, "active")
        write_graph_file('user', G)

    def gen_coord(self, segment) -> tuple:
        coord = [0 for i in range(len(self.feature_map))]
        for feature in segment:
            if feature in self.feature_data:
                coord[self.feature_map[feature]] = 1
        return tuple(coord)


def write_graph_file(user: str, graph: nx.Graph) -> None:
    with open(path.abspath("../public/json/cube.json"),'w') as file:
        file.write(nx.jit_data(graph))

hat = Inventory(extract_segments('eng', inv_type='Stop/affricate'))
hat.gen_cube()


''' 
    def __init__(self, num_features: int, feature_map=dict(), segment_store=set(), segments_first=False):
        if not segments_first:
            if feature_map:
                assert(len(feature_map) == num_features)
            else:
                self.current_index = 0    
            self.num_features = num_features
            self.feature_map = feature_map 
            self.segment_store = segment_store
            self.segments_first = segments_first

    '''

'''
    def add_feature(self, feature):       #essentially all this does is bind a member of the features enum
        assert feature in Features        #to a particular row of the index to any given node
        self.feature_map[self.current_index] = feature                    
        assert self.current_index < len(self.feature_map)
        self.current_index += 1
    '''
    
    # In this program, a segment is essentially just an index 
    # to a node on a hypercube
    # Here's the trick: depending on the entire geometry of its surrounding inventory
    # the same segment can have different specifications in different languages
    # So, to include a segment, you ONLY specify the features
    # given value '+' (or '1' as an index) and everything else is given value zero


'''
    def add_segment(self, segment_vals):
        assert type(segment_vals) == type(tuple()) 
        for val in segment_vals:
            assert val in self.feature_map.values()
        

        segment_spec = [0 for i in range(self.num_features)]
        for i, feature in self.feature_map.items():
            if feature in segment_vals:
                segment_spec[i] = 1
        self.segment_store.add(str(segment_spec)) 
        return segment_spec

'''
