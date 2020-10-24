# BioInfer supporting software tools
# Copyright (C) 2006 University of Turku
#
# This is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this software in the file COPYING. If not, see
# http://www.gnu.org/licenses/lgpl.html

from BasicClasses import (Sentences, Sentence,
                          Token, SubToken, Entity,
                          Linkage, Link, NestedSubtoken,
                          Formulas,Formula,RelNode,
                          EntityNode,
                          EntityNesting, Linkages,BioInfer)
from OntologyClasses import (Ontology,RelType,Predicate,EntityType)

import xml.sax
import xml.sax.saxutils
import xml.sax.handler
import sys
from optparse import OptionParser,OptionGroup

class BIParser (xml.sax.handler.ContentHandler,
                 xml.sax.handler.DTDHandler,
                 xml.sax.handler.EntityResolver,
                 xml.sax.handler.ErrorHandler,object):
    """
    The parser used to process the BioInfer XML file. The parsing process is implemented so
    that most of the logic of building the corpus objects lies on the C{__init__} methods
    of the classes. The parser has very little logic of its own.

    The parsing works as follows:
      - Each XML tag is assigned a corresponding class, whose instance is created upon encountering the tag.
      
      - The XML parser maintains a stack of instances which correspond to the current hierarchy during the XML parsing.
        Every time a tag is closed, the appropriate instance is removed from the stack. Every time a tag is opened,
        the appropriate instance is pushed on the stack.
        
      - When a tag is opened, the appropriate class is looked up and an instance created. Certain parameters are
        given to the instance when creating as B{named arguments}

          - C{parser} The instance of the parser.

          - C{oStack} The current stack of instances in the parsing.

          - C{attrs} A string->string mapping of the XML attributes for the tag.

    It is the responsibility of the __init__ method of an instance to integrate the instance into the data representation
    of the corpus. In particular C{oStack[-1]} is useful, because it is the instance above the current one hierarchically.
    For example, when a C{Sentence} instance is created, there will be a C{Sentences} instance in C{oStack[-1]}. The C{Sentence}
    instance would then call C{oStack[-1].addSentence(self)} to integrate itself to the data representation.

    A typical way to extend the capabilities of the basic classes is to subclass and extend some of them. For example, if one
    would like to extend the C{Sentence} class, one would create a C{MySentence} class inherited from C{Sentence}. It is then
    necessary to create instances of C{MySentence} when parsing the XML file. This is achieved by passing a named parameter
    of the form C{sentenceCls=MySentence} in the C{__init__} method of the parser: C{myParser=BIParser(sentenceCls=MySentence)}.
    Any named argument whose name ends with I{Cls} is recognized as an assignment of a class to XML tag. The default assignment
    is specified in the class variable C{defaultClasses}.
    """

    defaultClasses={"sentencesCls":Sentences,
                    "sentenceCls":Sentence,
                    "tokenCls":Token,
                    "subtokenCls":SubToken,
                    "entityCls":Entity,
                    "linkageCls":Linkage,
                    "linkCls":Link,
                    "nestedsubtokenCls":NestedSubtoken,
                    "formulasCls":Formulas,
                    "formulaCls":Formula,
                    "relnodeCls":RelNode,
                    "entitynodeCls":EntityNode,
                    "entitynestingCls":EntityNesting,
                    "linkagesCls":Linkages,
                    "bioinferCls":BioInfer,
                    "ontologyCls":Ontology,
                    "reltypeCls":RelType,
                    "predicateCls":Predicate,
                    "entitytypeCls":EntityType,
                    }

    def __init__(self,**args):
        self.classCfg=dict(BIParser.defaultClasses)
        classArgs=((k,v) for (k,v) in args.items() if k.endswith("Cls"))
        self.classCfg.update(classArgs)
        self.objectStack=[]
        self.nameStack=[]
        self.bioinfer=None

    def startElement(self,name,attrs):
        if name.lower()=="bioinfer": #Outer tag handled separately
            if not self.bioinfer:
                self.bioinfer=self.classCfg.get("bioinferCls",None)(parser=self,oStack=self.objectStack,attrs=attrs)
            elementObject=self.bioinfer
        else:
            elementClass=self.classCfg.get(name.lower()+"Cls",None)
            elementObject=elementClass(parser=self,oStack=self.objectStack,attrs=attrs)
        self.nameStack.append(name)
        self.objectStack.append(elementObject)

    def endElement(self,name):
        del self.nameStack[-1]
        del self.objectStack[-1]

    def parse(self,lines):
        parser=xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(self)
        parser.parse(lines)
