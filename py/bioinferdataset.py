import sys
import numpy as np
import dgl

sys.path.append("../lib/BioInfer_software_1.0.1_Python3/")
sys.path.append("../py/")
from BIParser import BIParser
from BasicClasses import RelNode

import os
import json

from config import ENTITY_PREFIX, PREDICATE_PREFIX
from collections import Counter, OrderedDict

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.nn import functional as functional

from config import (
    ENTITY_PREFIX,
    PREDICATE_PREFIX,
    EPOCHS,
    WORD_EMBEDDING_DIM,
    VECTOR_DIM,
    HIDDEN_DIM,
    RELATION_EMBEDDING_DIM,
    BATCH_SIZE,
    MAX_LAYERS,
    MAX_ENTITY_TOKENS,
)


class BioInferDataset(Dataset):
    def __init__(
        self, xml_file, entity_prefix=ENTITY_PREFIX, predicate_prefix=PREDICATE_PREFIX
    ):
        self.entity_prefix = entity_prefix
        self.predicate_prefix = predicate_prefix
        self.parser = BIParser()
        with open(xml_file, "r") as f:
            self.parser.parse(f)

        self.vocab_dict = self.create_vocab_dictionary(self.parser)
        entities = self.get_entities(self.parser)
        predicates = self.get_predicates(self.parser)
        elements = entities + predicates
        self.element_to_idx = {elements[i]: i for i in range(len(elements))}
        self.schema = self.get_schema(self.parser, self.element_to_idx)
        self.inverse_schema = self.invert_schema(self.schema)

    def __len__(self):
        return len(self.parser.bioinfer.sentences.sentences)

    def __getitem__(self, idx):
        sentence = self.parser.bioinfer.sentences.sentences[idx]
        entities, entity_locs = self.get_entities_from_sentence(sentence)
        entity_names, entity_locs, entity_spans = self.entities_to_tensors(
            entities, entity_locs
        )
        graphs, nkis, node_idx_to_element_idxs = self.get_relation_graphs_from_sentence(
            sentence, entity_locs
        )

        sample = {
            "text": sentence.getText(),
            "tokens": self.sent_to_idxs(sentence.getText(), self.vocab_dict),
            "element_names": entity_names,
            "element_locs": entity_locs,
            "entity_spans": entity_spans,
            "relation_graphs": graphs,
            "node_idx_to_element_idxs": node_idx_to_element_idxs,
        }
        return sample

    def create_vocab_dictionary(self, parser):
        vocab = set()

        for s in parser.bioinfer.sentences.sentences:
            for token in s.tokens:
                vocab.add(token.getText())

        vocab_size = len(vocab)
        vocab_index_list = [index for index in range(1, vocab_size)]

        vocab_dict = dict(zip(vocab, vocab_index_list))
        vocab_dict["UNK"] = 0
        return vocab_dict

    def get_entities(self, parser):
        entities = set()
        for s in parser.bioinfer.sentences.sentences:
            for e in s.entities:
                entity_type = e.type.name
                if "RELATIONSHIP" not in entity_type:
                    entities.add(f"{self.entity_prefix}{entity_type}")

        return list(entities)

    def get_predicates(self, parser):
        predicates = [
            f"{self.predicate_prefix}{p}"
            for p in parser.bioinfer.ontologies["Relationship"].predicates.keys()
        ]

        return list(set(predicates))

    def get_entities_from_sentence(self, sentence):
        entity_locs = {}
        entities = []
        i = 0
        for e in sentence.entities:
            entity_type = e.type.name
            if "RELATIONSHIP" not in entity_type:
                entity = (
                    f"{ENTITY_PREFIX}{entity_type}",
                    tuple([st.token.sequence for st in e.subTokens]),
                )
                entities.append(entity)
                entity_locs[e.id] = i
                i += 1
        return entities, entity_locs

    def get_relation_graphs_from_sentence(self, sentence, element_locs):
        graphs = []
        nkis = []
        node_idx_to_element_idxs = []
        for f in sentence.formulas:
            g, nki = self.pairs_to_graph(self.construct_graph_pairs(f.rootNode))
            nkis.append(nki)
            g.ndata["element_indices"] = torch.tensor(
                [
                    element_locs[nki[n.item()]]
                    if nki[n.item()] in element_locs.keys()
                    else -2
                    for n in g.nodes()
                ]
            )
            graphs.append(g)
            node_idx_to_element_idxs.append(
                {
                    k: v
                    for k, v in zip(
                        g.nodes().tolist(), g.ndata["element_indices"].tolist()
                    )
                }
            )
        return graphs, nkis, node_idx_to_element_idxs

    def entities_to_tensors(self, entities, entity_locs):
        if len(entities) > 0:
            entity_names = torch.tensor(
                [torch.tensor(self.element_to_idx[e[0]]) for e in entities]
            )
            entity_names = entity_names.reshape(-1, 1)
            entity_spans = torch.stack(
                [
                    functional.pad(
                        torch.tensor(e[1]),
                        pad=(0, MAX_ENTITY_TOKENS - len(e[1])),
                        mode="constant",
                        value=-1,
                    )
                    for e in entities
                ]
            )
            return entity_names, entity_locs, entity_spans
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

    def get_relnode_argument_types(self, relnode) -> tuple:
        arguments = set()
        for a in relnode.arguments:
            if a.isEntity():
                arguments.add(f"{self.entity_prefix}{a.entity.type.name}")
            elif a.isPredicate():
                arguments.add(f"{self.predicate_prefix}{a.predicate.name}")
            else:
                raise ValueError

        return tuple(sorted(list(arguments)))

    def sent_to_idxs(self, sentence, vocab_dict):
        token_list = sentence.split()

        index_list = []
        for token in token_list:
            if token in vocab_dict:
                index_list.append(vocab_dict[token])
            else:
                index_list.append(vocab_dict["UNK"])
        return torch.LongTensor(index_list)

    def get_schema(self, parser, element_to_idx):
        schema = {}
        for s in parser.bioinfer.sentences.sentences:
            for f in s.formulas:
                if f.rootNode.isPredicate() and not f.rootNode.isEntity():
                    predicate_name = f.rootNode.predicate.name
                    key = f"{self.predicate_prefix}{predicate_name}"
                    num_key = self.element_to_idx[key]
                    if num_key not in schema.keys():
                        schema[num_key] = Counter()
                    arguments = self.get_relnode_argument_types(f.rootNode)
                    try:
                        arguments = tuple(
                            sorted([self.element_to_idx[arg] for arg in arguments])
                        )
                    except:
                        print(f.rootNode)
                    schema[num_key][arguments] += 1
                else:
                    raise ValueError("formula rootNode should not be Entity")
        return schema

    def invert_schema(self, schema):
        inverted_schema = {}

        for rel, argsets in schema.items():
            for argset in argsets:
                if argset not in inverted_schema.keys():
                    inverted_schema[argset] = Counter()
                inverted_schema[argset][rel] += 1

        return inverted_schema

    def construct_graph_pairs(self, node):
        pairs = []
        filler_prefix = "x.{}"
        filler_idx = 1
        if node.isPredicate():
            node_type = f"{self.predicate_prefix}{node.predicate.name}"
        elif node.isEntity():
            node_type = f"{self.entity_prefix}{node.entity.type.name}"
        else:
            raise ValueError("node is neither predicate nor entity")

        if not node.entity:
            node_entity_id = filler_prefix.format(filler_idx)
            filler_idx += 1
        else:
            node_entity_id = node.entity.id

        for arg in node.arguments:
            pairs.append(
                [node_entity_id, arg.entity.id, self.element_to_idx[node_type]]
            )
            pairs += self.construct_graph_pairs(arg)

        return pairs

    def pairs_to_graph(self, pairs):
        if len(pairs) > 0:
            g = np.array(pairs)
            node_keys = {v: i for i, v in enumerate(np.unique(g[:, :2].flatten()))}
            u = g[:, 0]
            v = g[:, 1]
            u = np.vectorize(node_keys.get)(u)
            v = np.vectorize(node_keys.get)(v)
            g[:, 0] = u
            g[:, 1] = v
            element_names = [-1 for _ in np.unique(g[:, :2])]
            for row in g:
                element_names[int(row[0])] = int(row[2])
            g = dgl.graph((u, v))
            node_keys_inverse = {v: k for k, v in node_keys.items()}
            g.ndata["element_names"] = torch.tensor(element_names)
            return g, node_keys_inverse
        else:
            return EMPTY_GRAPH, {}