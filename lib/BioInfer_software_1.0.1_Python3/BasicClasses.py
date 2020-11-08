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

import xml.sax.saxutils
import itertools

currentSentence=None
DEFAULT_INDENT=2

class BIObject (object):

    XMLTag=None

    def __init__(self,attrs):
        pass


class BIXMLWriteable (object):
    """Definition of methods and variables for XML output. Any object that is included in the XML output needs to inherit
    from this class. The XML output mechanism works as follows.

      - Each object defines a class variable C{XMLTag} which is a string containing the tag to be used for the object in the XML output. This tag is used when writing the XML, not when reading the XML.
    
      - Each object defines iterable (typically list or tuple)
      C{persistentAttrs} which lists (as strings) the names of
      instance variables whose values should be included as attributes
      in the XML output. For instance, if an instance contains the
      variable C{id} and the string id is in its C{persistentAttrs},
      the attribute I{id=} will be output in the XML representation of
      this object. The builtin C{str} is used to convert values to
      strings.
    
      - Sometimes the C{persistentAttrs} mechanism is not
      sufficient. That is, for example, the case when a reference to
      an other object is held, and its id should be used as an
      attribute. In that case, an object may define a method
      C{computeXMLArgs()} which returns an iterable of two-tuples,
      where each tuple is (attributeName,attributeValue) - both
      strings - that specifies the attribute names and values to be
      included into the XML output. It is possible to combine
      C{persistentAttrs} with C{computeXMLArgs()}.

      - If the object defines C{writeXMLNestedItems} method, it is called
      so that the object may output any nested objects. It should do so
      by calling their C{writeXML} method. See the definition of C{writeXMLNestedItems}
      in any of the objects to see an example.

      @cvar persistentAttrs: The list of attributes to be included in
      the XML output. By default an empty list, that is, no attributes
      are included by default.
    """

    persistentAttrs=[]

    def writeXMLOpen(self,out,indent=0,closing=False):
        """
        Writes opening tag into the XML output. Depending on the parameter C{closing}, the opening will be
        a simple opening tag <tag> or a complete tag <tag/>. 
        
        @param out: Output stream to which the XML output is printed.
        @param indent: Indentation level of the output.
        @type indent: Integer
        @param closing: If true, the complete tag <tag/> is produced. If false, only the opening tag <tag> is produced.
        @type closing: bool
        """
        if closing:
            closeChar="/"
        else:
            closeChar=""
        print(" "*indent+"<%s>"%(" ".join((self.XMLTag,self.XMLAttrs())).strip()+closeChar), file=out)

    def XMLAttrs(self):
        """
        Computes the string representation of the object's attributes, which are included into the XML tag. Invokes
        the mechanism for attribute printing (described above) to obtain all printable attributes of the object

        @return: A string containing the XML-encoded, textual representation of the object's attributes
        """
        
        try:
            args2=self.computeXMLArgs() #Is the computeXMLArgs method defined by this object?
        except AttributeError:
            args2=() #no
        args1=((a,str(self.__dict__[a])) for a in self.persistentAttrs if a in self.__dict__) #Gather the values indicated in persistentAttrs
        xmlStrings=(str(a)+"="+xml.sax.saxutils.quoteattr(s) for a,s in itertools.chain(args1,args2))
        return " ".join(xmlStrings)

    def writeXML(self,out,indent,**kwArgs):
        """
        Outputs the complete XML representation of the given object.

           1. The XML tag, including the attributes is printed.

           2. If defined, the nested object are recursively printed via a call to C{writeXMLNestedItems}.

           3. The closing tag is printed. If there are no nested objects, the complete <tag/> was used already in step 1.

        @param out: The output stream to which the XML representation should be printed.
        @param indent: The indentation level of the XML block.
        @type indent: Integer
        @param kwArgs: Any keyword arguments will be passed to writeXMLNestedItems
        """
        
        try: #check if the nestedItemsPresent method is defined
            hasNestedItems=self.nestedItemsPresent()
        except:
            hasNestedItems=True
        try:
            foo=self.writeXMLNestedItems
            hasWriteXMLNestedItemsMethod=True
        except:
            hasWriteXMLNestedItemsMethod=False
        if hasNestedItems and hasWriteXMLNestedItemsMethod:
            self.writeXMLOpen(out,indent,False) #Write a <...> tag
            self.writeXMLNestedItems(out,indent+DEFAULT_INDENT,**kwArgs) #Write nested items
            self.writeXMLClose(out,indent) #Write a </...> tag
        else:
            self.writeXMLOpen(out,indent,True) #Write a <.../> tag, there are no nested items

    def writeXMLClose(self,out,indent=0):
        """
        Prints the closing tag of the object: </tag>.
        @param out: The output stream to which the output is printed.
        @param indent: The indentation level of the XML block.
        @type indent: Integer
        """
        print(" "*indent+"</%s>"%self.XMLTag, file=out)


class BioInfer (BIObject, BIXMLWriteable):
    """
    The top-level class in the hierarchy of objects which represents the corpus. It holds the L{Sentences} object as well
    as the L{ontologies<OntologyClasses.Ontology>}. A single instance of this class is created and maintained by the
    XML parser. It corresponds to the top-level item in the XML file.

    @ivar sentences: An instance of L{Sentences}, holding the sentences of the corpus.
    @type sentences: L{Sentences}
    @ivar ontologies: A dictionary which stores the ontologies. The key correponds to the ontology type.
    @type ontologies: Dictionary
    """

    XMLTag="bioinfer"

    def __init__(self,parser,attrs,**args):
        BIObject.__init__(self,attrs)
        self.sentences=None
        self.ontologies={}

    def addOntology(self,ontology):
        """
        Insert an ontology into the C{ontologies} dictionary. The key is C{ontology.type}. If an ontology already exists
        in the dictionary under the given key, it gets replaced by the new ontology.

        @param ontology: The L{OntologyClasses.Ontology} to be stored.
        """
        self.ontologies[ontology.type]=ontology

    def writeXMLNestedItems(self,out,indent,includeOntologies=False):
        if includeOntologies:
            for o in list(self.ontologies.values()):
                o.writeXML(out,indent)
        self.sentences.writeXML(out,indent)

    def regenId(self):
        """
        Regenerate the C{id} instance variables. Only to be used when
        the corpus data was changed (re-tokenized, etc...).  This is
        not to be used unless you know exactly what you are doing.
        """
        self.sentences.regenId()

    def isValid(self):
        """
        Validity check for the parsed corpus. This returns C{False} is any of the required corpus components is missing.
        """
        if not self.sentences:
            return False
        for i in ('Relationship','Entity'):
            if i not in self.ontologies:
                return False
        return True

class Sentences (BIObject,BIXMLWriteable):
    """
    C{Sentences} holds the sentences of the corpus. One instance corresponds
    to one XML tag I{sentences}.

    @ivar sentences: List of L{Sentence} objects.
    @type sentences: list
    """

    XMLTag="sentences"

    def __init__(self,oStack,attrs,parser,**args):
        """
        Initializes a C{Sentences} object. Called by the XML parser
        which also provides the arguments.

        @param oStack: List of objects representing the current nesting of tags in the XML document. The C{oStack} here will be an empty list, because I{sentences} is the top-level tag.
        @param attrs:  The dictionary of attributes of the I{sentences} tag, as provided by the XML parser
        """

        oStack[-1].sentences=self
        BIObject.__init__(self,attrs)
        self.sentences=[]
        """A list of `Sentence` objects."""

    def addSentence(self,sentence):
        """
        Appends C{sentence} to the end of the list of sentences.

        @param sentence: The sentence to be appended.
        @type sentence: Instance of L{Sentence<BasicClasses.Sentence>}, or derived.
        """
        self.sentences.append(sentence)

    def writeXMLNestedItems(self,out,indent):
        for s in self.sentences:
            s.writeXML(out,indent)

    def regenId(self):
        """
        Regenerate the C{id} instance variables. Only to be used when
        the corpus data was changed (re-tokenized, etc...).  This is
        not to be used unless you know exactly what you are doing.
        """
        for s in self.sentences:
            s.regenId()


class Sentence (BIObject,BIXMLWriteable):
    """
    C{Sentence} holds all annotation for a single sentence: entities, formulas, and linkages.

    @ivar id: The unique id of the sentence. It is also used as part of the ids of all objects that belong to this sentence.
    @type id: String
    @ivar origText: The original, untokenized text of the sentence
    @type origText: String
    @ivar tokens: List of L{Tokens<BasicClasses.Token>}. The tokens are/must be ordered in the same order as they appear in the XML file (and that order preserves the sentence order).
    @type tokens: list
    @ivar formulas: List of L{Formulas<BasicClasses.Formula>}. They are ordered in the same order as they appear in the XML file.
    @type formulas: list
    @ivar linkages: Dictionary of L{Linkages<BasicClasses.Linkage>}, where the linkage type serves as key.
    @type linkages: dictionary
    @ivar entitiesById: Dictionary of L{Entities<BasicClasses.Entity>}, where the entity id serves as key.
    @type entitiesById: dictionary
    @cvar persistentAttrs: The attributes written in the XML file for each sentence.
    @type persistentAttrs: List of strings
    @ivar sentences: The C{sentences} instance to which this sentence belongs.
    @type sentences: L{sentences}
    """

    XMLTag="sentence"
    persistentAttrs=["id","origText"]

    def __init__(self,oStack,attrs,**args):
        """
        Initializes a C{Sentence} object. Called by the XML parser
        which also provides the arguments. Also calls
        L{addSentence(self)<BasicClasses.Sentences.addSentence>} on
        C{oStack[-1]} to register self with the C{Sentences}.

        @param oStack: List of objects representing the current
        nesting of tags in the XML document. In particular,
        C{oStack[-1]} is expected to be a L{Sentences} instance (or
        derived), or any class that defines
        L{addSentence<BasicClasses.Sentences.addSentence>}.

        @param attrs: The dictionary of attributes of the I{sentence}
        tag, as provided by the XML parser.
        """
        global currentSentence
        BIObject.__init__(self,attrs)
        currentSentence=self
        self.tokens=[]
        self.id=attrs["id"] #compulsory attr, throws exception if missing
        self.origText=attrs["origText"]
        self.entitiesById={}
        self.entities=[]
        self.formulas=[]
        self.linkages={}
        self.tokenSequence=-1
        self.subTokenSequence=-1
        oStack[-1].addSentence(self)

    def getTokenSequence(self):
        """
        Returns the sequence number of the next token to be included into the sequence. A special method only used during XML parsing.
        """
        self.tokenSequence+=1
        return self.tokenSequence

    def getSubTokenSequence(self):
        """
        Returns the sequence number of the next subtoken to be included into the sequence. A special method only used during XML parsing.
        """
        self.subTokenSequence+=1
        return self.subTokenSequence

    def getText(self):
        """
        A space-delimited string of all tokens in the sentence. 
        @return: The text of the sentence (with spaces at token boundaries)
        @rtype: string
        """
        return " ".join(st.getText() for st in self.tokens)

    def regenId(self):
        """
        Regenerate the C{id} instance variables. Only to be used when
        the corpus data was changed (re-tokenized, etc...).  This is
        not to be used unless you know exactly what you are doing.
        """
        for seq,token in enumerate(self.tokens):
            token.id="t.%s.%d"%(self.id,seq)
            token.regenId()
        for seq,entity in enumerate(self.entities): #There is no natural order for entities, but the self.entities takes care the order is preserved as in the XML file
            entity.id="e.%s.%d"%(self.id,seq)

    def addToken(self,token):
        """
        Appends a L{token<Token>} at the end of the list of tokens C{tokens}.

        @param token: The token to be appended.
        """
        self.tokens.append(token)
        token.sentence=self
        token.sequence=len(self.tokens)-1


    def addEntity(self,entity):
        """
        Includes the entity into the dictionary C{entities}, key being C{entity.id}.

        @param entity: The entity to be included.
        """
        entity.sentence=self
        self.entitiesById[entity.id]=entity
        self.entities.append(entity)

    def writeXMLNestedItems(self,out,indent):
        for t in self.tokens:
            t.writeXML(out,indent)
        for e in self.entities:
            e.writeXML(out,indent)
        for e in self.entities:
            e.writeXMLNestedEntities(out,indent)
        print(" "*indent+"<linkages>", file=out)
        for (t,l) in list(self.linkages.items()):
            l.writeXML(out,indent+DEFAULT_INDENT)
        print(" "*indent+"</linkages>", file=out)
        print(" "*indent+"<formulas>", file=out)
        for f in self.formulas:
            f.writeXML(out,indent+DEFAULT_INDENT)
        print(" "*indent+"</formulas>", file=out)
        

class Token (BIObject,BIXMLWriteable):
    """
    The representation of a single token as collection of subtokens.

    @ivar id: A unique id assigned to the token in the corpus.
    @type id: string
    @ivar subTokens: A list of L{subtokens<SubToken>}. The subtokens are ordered in the same order as they appear in the XML file, which is the same order as they appear in the sentence. This order must be preserved.
    @type subTokens: list
    @ivar charOffset: An offset (zero-based) to the string L{self.sentence.origText<Sentence.origText>}, that is, the original untokenized text of the sentence. To be used when translating the corpus to some of the character-offset based formats.
    @ivar sentence: The sentence to which this token belongs.
    @type sentence: L{sentence}
    """
    
    XMLTag="token"
    persistentAttrs=["id","charOffset"]

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        self.subTokens=[]
        self.id=attrs["id"]
        self.charOffset=attrs["charOffset"]
        self.sequence=currentSentence.getTokenSequence()
        oStack[-1].addToken(self)

    def regenId(self):
        """
        Regenerate the C{id} instance variables. Only to be used when
        the corpus data was changed (re-tokenized, etc...).  This is
        not to be used unless you know exactly what you are doing.
        """
        for seq,subToken in enumerate(self.subTokens):
            subToken.id="st.%s.%d"%(self.id.split(".",1)[1],seq)

    def getText(self,sep=""):
        """
        The text of the token, assembled from the subtoken texts.
        @param sep: The separator to be used when joining the subtoken texts.
        """
        return sep.join(st.text for st in self.subTokens)

    def addSubToken(self,subToken):
        """
        Add a new L{subToken<SubToken>} to the list of subtokens C{self.subTokens}. The subtokens must be added in the correct order.

        @param subToken: The subtoken to be added.
        @type subToken: L{SubToken}
        """
        subToken.token=self
        self.subTokens.append(subToken)

    def writeXMLNestedItems(self,out,indent):
        for st in self.subTokens:
            st.writeXML(out,indent)

class SubToken (BIObject,BIXMLWriteable):
    """
    The representation of a subToken.

    @ivar id: The unique identifier of this subtoken in the corpus.
    @type id: string
    @ivar text: The text of the subtoken.
    @type text: string
    @ivar sequence: The sequential number of the subtoken in the B{sentence} (that is, not the Token). First subtoken in the sentence is 0, second is 1, etc.
    @type sequence: integer
    @ivar token: The L{token<Token>} to which this subtoken belongs.
    @type token: L{Token}
    """

    XMLTag="subtoken"
    persistentAttrs=["id","text"]

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        self.token=oStack[-1]
        self.text=attrs["text"]
        self.id=attrs["id"]
        self.sequence=currentSentence.getSubTokenSequence()
        oStack[-1].addSubToken(self)
        #entities

    def getCharOffset(self):
        """
        Returns the zero-based character offset of this subtoken in the
        string representing the original untokenized text of the sentence.
        """
        # calculate as on the character offset of the token plus the
        # lengths of preceding subtokens
        offset = int(self.token.charOffset)
        for st in [st for st in self.token.subTokens if st.sequence<self.sequence]:
            offset += len(st.text)
        return offset

class Link (BIObject,BIXMLWriteable):
    """
    The representation of a link in a linkage.

    @ivar token1: The first L{token<Token>} of the link.
    @type token1: L{Token}
    @ivar token2: The second L{token<Token>} of the link.
    @type token2: L{Token}
    @ivar category: The list of categories of the link as strings.
    @type category: list
    @ivar type: The link type of this link
    @type type: string
    """

    XMLTag="link"
    persistentAttrs=["type"]

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        token1IDComponents=attrs["token1"].split(".")
        token2IDComponents=attrs["token2"].split(".")
        if len(token1IDComponents)==3:
            del token1IDComponents[0] #Remove the "t" from the begining of token ids
        if len(token2IDComponents)==3:
            del token2IDComponents[0] #Remove the "t" from the begining of token ids
        s1,t1=(int(x) for x in token1IDComponents)
        s2,t2=(int(x) for x in token2IDComponents)
        self.token1=oStack[-1].sentence.tokens[t1]
        self.token2=oStack[-1].sentence.tokens[t2]
        self.category=attrs["category"].split(",")
        self.type=attrs["type"]
        oStack[-1].addLink(self)

    def __getMacro(self):
        if "macro" in self.category:
            return True
        return False

    macro=property(__getMacro)

    def computeXMLArgs(self):
        return ("token1",self.token1.id),("token2",self.token2.id),("category",",".join(self.category))


class Entity (BIObject,BIXMLWriteable):
    """
    The representation of Entity as a list of subtokens.

    @ivar sentence: The L{sentence<Sentence>} this entity belongs to.
    @type sentence: L{Sentence}
    @ivar id: The unique id of the entity in the corpus.
    @type id: string
    @ivar subTokens: A list of L{subtokens<SubToken>} which constitute this entity.
    @type subTokens: list
    @ivar nestedEntities: The entities which are directly nested in the current entity.
    @type nestedEntities: list
    @ivar formulaNodesUsingMe: The list of L{formula nodes<FormulaNode>} which refer to this entity, using it as their text binding: both Relationships (L{RelNode}) and Entities (L{EntityNode}). While the
    code is written in a general way, in the actual corpus, the same entity is never used both as a relationship and entity text binding.
    @type formulaNodesUsingMe: list
    @ivar type: The L{entity type<OntologyClasses.EntityType>} of this entity. The type is a node in an ontology.
    @type type: L{EntityType<OntologyClasses.EntityType>}
    """
    
    XMLTag="entity"
    persistentAttrs=["id","annotation","other"]

    def __init__(self,oStack,parser,attrs,**args):
        BIObject.__init__(self,attrs)
        self.sentence=oStack[-1]
        self.id=attrs["id"]
        self.subTokens=[]
        self.nestedEntities=[]
        self.formulaNodesUsingMe=[]
        oStack[-1].addEntity(self)
        typeName=attrs["type"]
        try:
            self.type=parser.bioinfer.ontologies["Entity"].findPredicate(typeName)
        except KeyError:
            print("Unknown",typeName)
        try:
            self.annotation=attrs["annotation"]
        except KeyError:
            self.annotation=None
        try:
            self.other=attrs["other"]
        except KeyError:
            self.other=None

    def getText(self,getType=False):
        """
        Returns the text of the entity.
        """
        text, prevSt = "", None
        for st in self.subTokens:
            # only add separating space if the SubTokens are not consequtive
            # in the original text of the sentence.
            if prevSt is not None and prevSt.getCharOffset()+len(prevSt.text) != st.getCharOffset():
                text += " "
            text += st.text
            prevSt = st
            
        if getType:
            text += " ("+self.type.name+")"

        return text

    def getCharOffsets(self):
        """
        Returns the character offsets to the original sentence text of the
        SubTokens of which this entity consists.
        @return: List of (from,to) tuples, where from and to are integer
        offsets.
        """

        if len(self.subTokens) == 0:
            return []

        # start with a list of one (from,to) offset pair for each
        # subtoken, and then merge ranges where the subtokens are
        # either consequtive or separated by a single space in the
        # original text.

        offsets =[(st.getCharOffset(), st.getCharOffset()+len(st.text)-1) for st in self.subTokens]

        text = self.subTokens[0].token.sentence.origText
        merged, start, end = [], offsets[0][0], offsets[0][1]
        for o in offsets[1:]:
            if (end+1 == o[0]) or (end+2 == o[0] and text[end+1] == ' '):
                end = o[1]
            else:
                merged.append((start, end))
                start, end = o[0], o[1]
        merged.append((start, end))

        return merged

    def addSubToken(self,subToken):
        """
        Add a subToken to the entity.

        @param subToken: The subToken to be added.
        @type subToken: L{SubToken}
        """
        self.subTokens.append(subToken)

    def addNestedEntity(self,entity):
        """
        Add an entity to the list of entities nested within the current entity

        @param entity: The entity to be added
        @type entity: L{Entity}
        """
        self.nestedEntities.append(entity)

    def writeXMLNestedItems(self,out,indent):
        for s in self.subTokens:
            print(" "*indent+"<nestedsubtoken id=%s/>"%xml.sax.saxutils.quoteattr(s.id), file=out)

    def writeXMLNestedEntities(self,out,indent):
        for e in self.nestedEntities:
            print(" "*indent+"<entitynesting outerid=%s innerid=%s/>"%(xml.sax.saxutils.quoteattr(self.id),xml.sax.saxutils.quoteattr(e.id)), file=out)

    def registerFormulaNode(self,node):
        """
        Append a formula node to the list of formula nodes that use this entity as a text binding
        @param node: The node to be added to the list.
        @type node: L{FormulaNode}
        """
        self.formulaNodesUsingMe.append(node)

    def isFormulaRelationship(self): #Am I being used as a text binding of a predicate?
        """
        Test whether the entity is being used as a text binding of a relationship formula node (L<RelNode>). In the corpus, never
        C{isFormulaRelationship} and C{isFormulaEntity} return C{True} at the same time.
        """
        return bool([n for n in self.formulaNodesUsingMe if n.isPredicate()])

    def isFormulaEntity(self): #Am I being used as a text binding of formula entity?
        """
        Test whether the entity is being used as a text binding of an entity formula node (L<EntityNode>). In the corpus, never
        C{isFormulaRelationship} and C{isFormulaEntity} return C{True} at the same time.
        """
        return bool([n for n in self.formulaNodesUsingMe if n.isEntity()])

    def computeXMLArgs(self):
        return (("type",self.type.name),)
        
class NestedSubtoken (BIObject):
    """
    This class is only used during the XML parsing and it corresponds
    to the I{nestedsubtoken} tag and mediates the inclusion
    of subtokens into entities. Instances of this class
    are not preserved outside of the parsing process.
    """

    def __init__(self,oStack,attrs,**args):
        global currentSentence
        BIObject.__init__(self,attrs)
        idComponents=attrs["id"].split(".")
        if len(idComponents)==4: #New ID numbering
            del idComponents[0] #Remove the "st" in subtokenIDs
        s,t,st=(int(i) for i in idComponents)
        oStack[-1].addSubToken(currentSentence.tokens[t].subTokens[st])

class Formulas (BIObject):
    """
    This class is only used during the XML parsing and it corresponds
    to the I{formulas} tag and mediates the inclusion
    of formulas into sentences. Instances of this class
    are not preserved outside of the parsing process.
    """

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        self.formulas=oStack[-1].formulas
        self.sentence=oStack[-1]

    def addFormula(self,formula):
        """
        This method adds a formula to the current sentence. It is
        called by C{Formula.__init__} during XML parsing.
        @param formula: The formula to be added
        @type formula: L{Formula}
        """
        self.formulas.append(formula)
        formula.sentence=self.sentence

class Formula (BIObject,BIXMLWriteable):
    """
    The representation of a single relationship formula. The formulas
    are built as trees, where each node in the tree is a L{FormulaNode}, which
    can be either a relationship (L{RelNode}), or an entity L{EntityNode}.
    The C{Formula} instance holds the root node of the formula tree.

    @ivar rootNode: The root node of the formula.
    @type rootNode: L{RelNode}
    """

    XMLTag="formula"

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        oStack[-1].addFormula(self)

    def addArgument(self,root):
        """
        Define the root node of the formula.
        @param root: The root node.
        @type root: L{RelNode}
        """
        self.rootNode=root

    def writeXMLNestedItems(self,out,indent):
        self.rootNode.writeXML(out,indent)

class FormulaNode (BIObject,BIXMLWriteable):
    """
    The base class from which formula nodes are inherited.
    It implements the methods common to formula nodes.

    @ivar arguments: The arguments (children) of this formula node. This list remains empty for leaves in the
    formula tree, that is, the L{EntityNodes<EntityNode>}.
    @type arguments: list
    @ivar entity: The L{Entity} used as text binding for this formula node. In the rare cases when the formula node
    does not have a text binding, the C{entity} is set to C{None}.
    @type entity: L{Entity}
    """
    
    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        self.arguments=[]
        self.myArgumentPosition=oStack[-1].addArgument(self)
        self.entity=None
        tbEntityId=None
        try:
            tbEntityId=attrs["textbindingentity"]
        except KeyError:
            pass
        try:
            tbEntityId=attrs["entity"]
        except KeyError:
            pass
        if tbEntityId:
            self.setEntity(tbEntityId)
            self.entity.registerFormulaNode(self) #Inform the entity it's being used in a formula

    def addArgument(self,argument):
        """
        Add argument to the list of arguments of this formula node.
        @param argument: The argument to be added
        @type argument: FormulaNode
        @return: The position of the argument in this node (zero-based integer). The return value is used by the argument to remember its position among the arguments of the parent formula node.
        @rtype: integer
        """
        argument.parent=self
        self.arguments.append(argument)
        return len(self.arguments)-1 #Replies back the argument position - which may be in turn reported to Entity objects

    def setEntity(self,entityId):
        """
        Resolve entityId into an L{Entity} and set the entity as the formula node text binding entity.
        @param entityId: Id of the entity which is used as the text binding entity of this formula node.
        @type entityId: string
        """
        self.entity=currentSentence.entitiesById[entityId]

    def writeXMLNestedItems(self,out,indent):
        for a in self.arguments:
            a.writeXML(out,indent)

    def computeXMLArgs(self):
        if self.entity:
            return (("entity",self.entity.id),)
        else:
            return ()

    def nestedItemsPresent(self):
        """
        True if this formula node has arguments, that is, if it is not a leaf in the formula tree.
        """
        return bool(self.arguments)

    def isPredicate(self):
        """
        Return C{True} if the formula node represents a predicate (i.e. relationship). In the C{FormulaNode} class, this
        method returns C{False}. It overriden in the relationship node classes inherited from C{FormulaNode} to return C{True}.
        """
        return False

    def isEntity(self):
        """
        Return C{True} if the formula node represents a predicate (i.e. relationship). In the C{FormulaNode} class, this
        method returns C{False}. It overriden in the C{EntityNode} class inherited from C{FormulaNode} to return C{True}.
        """
        return False

    def getEntities(self,recursive=False):
        """
        Return a list of all entities used as a text binding in this node and, if C{recursive==True}, in the
        subtree rooted by this node.
        """
        result = []
        if self.entity:
            result.append(self.entity)
        if recursive:
            result.extend(j for i in self.arguments for j in i.getEntities(True))
        return result
        
class RelNode (FormulaNode):
    """
    A formula node which represents a relationship among its arguments. The arguments,
    formula nodes themselves, are in the instance variable C{arguments}. The text binding
    entity of the predicate, if any, is in the instance variable C{entity}.

    @ivar predicate: A L{Predicate<OntologyClasses.Predicate>} instance representing the predicate of the relationship.
    @type predicate: L{OntologyClasses.Predicate}
    """

    XMLTag="relnode"

    def __init__(self,oStack,parser,attrs,**args):
        FormulaNode.__init__(self,oStack,attrs)
        predicateName=attrs["predicate"]
        try:
            self.predicate=parser.bioinfer.ontologies["Relationship"].findPredicate(predicateName)
        except:
            print("Unknown predicate",predicateName)


    def computeXMLArgs(self):
        upstream=FormulaNode.computeXMLArgs(self)
        return upstream+(("predicate",self.predicate.name),)

    def isPredicate(self):
        """
        True.
        """
        return True

    def getText(self, tBinding=True, descend=False):
        result = self.predicate.name
        if tBinding and self.entity:
            result = self.entity.getText()+": "+result
        if descend:
            result += "("+", ".join([i.getText(tBinding,descend) for i in self.arguments])+")"
        return result


class EntityNode (FormulaNode):
    """
    A formula node which represents an entity argument. The entity referenced by this
    node is in the instance variable C{entity}.
    """

    XMLTag="entitynode"
    
    def __init__(self,oStack,attrs,**args):
        FormulaNode.__init__(self,oStack,attrs)

    def isEntity(self):
        """
        True.
        """
        return True

    def getText(self, *args, **kwArgs):
        if self.entity:
            return self.entity.getText()
        else:
            return ""

class Linkage (BIObject,BIXMLWriteable):
    """
    A representation of a single linkage as a collection of links between tokens. Each sentence
    may have several linkages, which are distinguished by linkage type. The type as annotated
    by the corpus annotators has type \"raw\"

    @ivar links: A list of links in no particular order. The order of the links in the XML file is used.
    @type links: list
    @ivar sentence: The sentence to which this linkage belongs
    @type sentence: L{Sentence}
    @ivar type: The type of the linkage as a string.
    @type type: string
    @ivar reliability: The reliability of the automatic linkage typing for this sentence. Can be either I{high} or I{medium}. High reliability is assigned to link types obtained from non-panic parses and is close to 100% reliable.
    @type reliability: string
    """

    XMLTag="linkage"
    persistentAttrs=["type"]

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        self.type=attrs["type"]
        try:
            self.reliability=attrs["reliability"]
        except KeyError:
            self.reliability=None
        self.links=[]
        self.sentence=oStack[-2]
        oStack[-1].addLinkage(self)

    def addLink(self,link):
        """
        Add a new link to the linkage.

        @param link: The link to be added
        @type link: L{Link}
        """
        link.linkage=self
        self.links.append(link)

    def writeXMLNestedItems(self,out,indent):
        for l in self.links:
            l.writeXML(out,indent)
            

class EntityNesting (BIObject):
    """
    This class is only used during the XML parsing and it corresponds
    to the I{entitynesting} tag and mediates the inclusion
    of entities into other entities to represent entity nesting. Instances of this class
    are not preserved outside of the parsing process.
    """

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        oStack[-1].entitiesById[attrs["outerid"]].addNestedEntity(oStack[-1].entitiesById[attrs["innerid"]])


class Linkages (BIObject,BIXMLWriteable):
    """
    This class is only used during the XML parsing and it corresponds
    to the I{linkages} tag and mediates the inclusion
    of linkages into the sentence instances. Instances of this class
    are not preserved outside of the parsing process.
    """

    def __init__(self,oStack,attrs,**args):
        BIObject.__init__(self,attrs)
        self.linkages=oStack[-1].linkages
        self.sentence=oStack[-1]

    def addLinkage(self,linkage):
        """
        Passes a linkage into the currently open sentence (during XML parsing).

        @param linkage: The linkage to be appended to the sentence.
        @type linkage: L{Linkage}
        """
        self.linkages[linkage.type]=linkage
        linkage.sentence=self.sentence
