import sys

sys.path.append("../lib/BioInfer_software_1.0.1_Python3/")
sys.path.append("../py/")
from BIParser import BIParser
from BasicClasses import RelNode

import os
import json

from config import ENTITY_PREFIX, PREDICATE_PREFIX
from collections import Counter, OrderedDict


class BioInferTaskConfiguration:
    def __init__(self):
        self.schema = None
        self.element_to_idx = None
        self.entity_prefix = None
        self.predicate_prefix = None

    def _create_schema(self, parser: BIParser) -> dict:
        schema = {}
        for s in parser.bioinfer.sentences.sentences:
            for f in s.formulas:
                if f.rootNode.isPredicate() and not f.rootNode.isEntity():
                    predicate_name = f.rootNode.predicate.name
                    key = f"{self.predicate_prefix}{predicate_name}"
                    if key not in schema.keys():
                        schema[key] = Counter()
                    arguments = self._get_relnode_argument_types(f.rootNode)
                    schema[key][arguments] += 1
                else:
                    raise ValueError("formula rootNode should not be Entity")
        return schema

    def _invert_schema(self, schema):
        inverted_schema = {}

        for rel, argsets in schema.items():
            for argset in argsets:
                if argset not in inverted_schema.keys():
                    inverted_schema[argset] = Counter()
                inverted_schema[argset][rel] += 1

        return inverted_schema

    def _combine_schema_keys(self, schema):
        schema_for_write = {}
        for predicate, argset in schema.items():
            schema_for_write[predicate] = {" ".join(k): v for k, v in argset.items()}

        return schema_for_write

    def _split_schema_keys(self, schema):
        split_schema = {}
        for predicate, argset in schema.items():
            split_schema[predicate] = {
                tuple(k.split(" ")): v for k, v in argset.items()
            }

        return split_schema

    def _get_relnode_argument_types(self, relnode: RelNode) -> tuple:
        arguments = set()
        for a in relnode.arguments:
            if a.isEntity():
                arguments.add(f"{self.entity_prefix}{a.entity.type.name}")
            elif a.isPredicate():
                arguments.add(f"{self.predicate_prefix}{a.predicate.name}")
            else:
                raise ValueError

        return tuple(sorted(list(arguments)))

    def _get_entities(self, parser, entity_prefix: str) -> dict:
        entities = set()
        for s in parser.bioinfer.sentences.sentences:
            for e in s.entities:
                entity_type = e.type.name
                if 'RELATIONSHIP' not in entity_type:
                    entities.add(f"{ENTITY_PREFIX}{entity_type}")
        entities = list(entities)

        return entities

    def _get_predicates(self, schema) -> dict:
        return list(schema.keys())

    def to_json(self, path) -> None:
        schema_to_write = {}
        schema_to_write["schema"] = self._combine_schema_keys(self.schema)
        schema_to_write["element_to_idx"] = self.element_to_idx

        with open(path, "w") as f:
            json.dump(schema_to_write, f)

    def from_json(
        self,
        path,
        entity_prefix: str = ENTITY_PREFIX,
        predicate_prefix: str = PREDICATE_PREFIX,
    ) -> None:
        with open(path, "r") as f:
            json_loaded = json.load(f)

        self.entity_prefix = entity_prefix
        self.predicate_prefix = predicate_prefix
        self.schema = self._split_schema_keys(json_loaded["schema"])
        self.inverted_schema = self._invert_schema(self.schema)
        self.element_to_idx = json_loaded["element_to_idx"]

        return self

    def from_parser(
        self,
        parser: BIParser,
        entity_prefix: str = ENTITY_PREFIX,
        predicate_prefix: str = PREDICATE_PREFIX,
    ) -> None:
        self.entity_prefix = entity_prefix
        self.predicate_prefix = predicate_prefix
        self.schema = self._create_schema(parser)
        self.inverted_schema = self._invert_schema(self.schema)
        entities = self._get_entities(parser,ENTITY_PREFIX)
        predicates = self._get_predicates(self.schema)
        elements = entities + predicates
        self.element_to_idx = {elements[i]: i for i in range(len(elements))}

        return self

    def __repr__(self):
        return str(
            {
                "schema": self.schema,
                "element_to_idx": self.element_to_idx,
                "entity_prefix": self.entity_prefix,
                "predicate_prefix": self.predicate_prefix,
            }
        )
