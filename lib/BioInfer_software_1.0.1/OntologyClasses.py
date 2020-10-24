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

from BasicClasses import BIObject, BIXMLWriteable

class Ontology (BIObject, BIXMLWriteable):
    """
    The representation of an ontology. Instances of C{Ontology} hold
    the root C{OntologyNode} of the ontology tree. For simplicity, we
    call ontology items I{predicates}.
    """
    
    XMLTag="ontology"

    persistentAttrs=["type"]

    def __init__(self,oStack,attrs,**args):
        self.type=attrs["type"]
        oStack[-1].addOntology(self)
        oStack[-1].currentOntology=self
        self.predicates={}
        self.instances={}

    def addSpecialization(self,spec):
        self.rootNode=spec

    def registerPredicate(self,pred):
        self.predicates[pred.name]=pred

    def findPredicate(self,name):
        return self.predicates[name]

    def writeXMLNestedItems(self,out,indent):
        self.rootNode.writeXML(out,indent)

class OntologyNode (BIObject, BIXMLWriteable):

    def __init__(self,oStack,parser,attrs,**args):
        self.specs=[]
        self.predicates=[]
        self.ontology=parser.bioinfer.currentOntology

    def addSpecialization(self,spec):
        self.specs.append(spec)

    def addPredicate(self,pred):
        self.predicates.append(pred)

    def writeXMLNestedItems(self,out,indent):
        for s in self.predicates:
            s.writeXML(out,indent+2)
        for s in self.specs:
            s.writeXML(out,indent+2)
    
class RelType (OntologyNode):

    XMLTag="reltype"

    persistentAttrs=["name","comment"]

    def __init__(self,oStack,attrs,**args):
        OntologyNode.__init__(self,oStack=oStack,attrs=attrs,**args)
        self.name=attrs["name"]
        if "comment" in attrs.keys():
            self.comment=attrs["comment"]
        oStack[-1].addSpecialization(self)

class Predicate (BIObject, BIXMLWriteable):

    XMLTag="predicate"

    persistentAttrs=["name"]

    def __init__(self,parser,oStack,attrs,**args):
        self.name=attrs["name"]
        if "comment" in attrs.keys():
            self.comment=attrs["comment"]
        oStack[-1].addPredicate(self)
        parser.bioinfer.currentOntology.registerPredicate(self)

class EntityType (OntologyNode):

    XMLTag="entitytype"

    persistentAttrs=["name"]

    def __init__(self,oStack,attrs,parser,**args):
        OntologyNode.__init__(self,oStack=oStack,attrs=attrs,parser=parser,**args)
        self.name=attrs["name"]
        if "comment" in attrs.keys():
            self.comment=attrs["comment"]
        oStack[-1].addSpecialization(self)
        parser.bioinfer.currentOntology.registerPredicate(self)
    
