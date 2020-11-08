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

import sys
from BIParser import BIParser
from optparse import OptionParser,OptionGroup

def printText(sentence):
    """
    Prints the original, untokenized text of the given sentence.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    print(sentence.origText, end=' ')

def printTokenizedText(sentence):
    """
    Prints the tokenized text of the given sentence, with tokens separated
    by whitespace.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    for t in sentence.tokens:
        print(t.getText(), end=' ')

def printTokenOffsets(sentence):
    """
    Prints the tokens of the given sentence in the format
    \"token(TOKEN_IDX, [FROM_OFFSET-TO_OFFSET], 'TOKEN_TEXT')
    where FROM_OFFSET and TO_OFFSET are character offsets into the
    original sentence text.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    for i, t in enumerate(sentence.tokens):
        text     = t.getText()
        from_off = int(t.charOffset)
        to_off   = from_off + len(text) - 1
        print("token(%d, [%d-%d], '%s')" % (i, from_off, to_off, text), end=' ')

def printLinkageDependencies(sentence, linkage):
    """
    Prints the dependencies of the given linkage for the given
    sentence in \"FROM-TO[TYPE]\" format, where FROM and TO are token
    offsets and TYPE is the type of the dependency. If the type is
    None, the bracketed part is not output. Additionally, NP
    macro-dependencies are typed \"macro\". Output dependencies are
    separated by space.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    @param linkage: a string identifying the type of linkage to output.
    @type linkage: String
    """

    if linkage not in sentence.linkages:
        print("missing '%s' linkage for sentece %s" % (linkage, sentence.id), file=sys.stderr)
        sys.exit(1)
    links = sentence.linkages[linkage].links

    for l in links:
        if l.type is not None:
            type_string = "[%s]" % l.type
        else:
            type_string = ""

        if l.macro:
            type_string = "[macro]"

        print("%s-%s%s" % (l.token1.sequence, l.token2.sequence, type_string), end=' ')

def printBasicDependencies(sentence):
    """
    Prints the dependencies of the \"basic\" (unexpanded) linkage using
    L{printLinkageDependencies}.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    printLinkageDependencies(sentence, "unexpanded")

def printParallelDependencies(sentence):
    """
    Prints the dependencies of the typed linkage with NP
    macro-dependencies expanded in parallel using
    L{printLinkageDependencies}.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    printLinkageDependencies(sentence, "parallel")

def printSerialDependencies(sentence):
    """
    Prints the dependencies of the typed linkage with NP
    macro-dependencies expanded serially using
    L{printLinkageDependencies}.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    printLinkageDependencies(sentence, "serial")

def printEntities(sentence):
    """
    Prints the entities of the given sentence in the format
    \"TYPE(ID, [CHAR_OFFSETS], 'TEXT')\", where TYPE
    is the type of the entity (found in the entity type ontology),
    ID is the unique identifier of the entity,
    CHAR_OFFSETS is a comma-separated list of \"FROM_OFF-TO_OFF\"
    character offsets specifying where in the untokenized sentence
    text the entity is found, and TEXT is the text of the token.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """

    # omit relationship entities
    entities = [e for e in list(sentence.entitiesById.values()) if not e.isFormulaRelationship()]

    for e in entities:
        offset_text = ",".join(["%d-%d" % a for a in e.getCharOffsets()])
        print("%s(%s, [%s], '%s')" % (e.type.name, e.id, offset_text, e.getText()), end=' ')

def relationshipToString(r):
    """
    Returns a string representing the given relationship in
    the format required by L{printRelationships}.
    @param r: the relationship to convert to a string.
    @type r: L{BasicClasses.Formula}
    """

    # get bound text character offsets, if any
    if r.entity is None:
        st_offsets = []
    else:
        st_offsets = r.entity.getCharOffsets()

    s = "%s([%s], " % (r.predicate.name, ",".join(["%d-%d" % a for a in st_offsets]))

    # recursively build the string for the included entites and relationships
    for a in r.arguments:
        if a.isEntity():
            s += a.entity.id
        else:
            s += relationshipToString(a)

        if a != r.arguments[-1]:
            s += ", "

    s += ")"
    return s

def printRelationships(sentence):
    """
    Prints the relationships of the given sentence in the format
    \"TYPE([CHAR_OFFSETS], ARGS)\" where TYPE is the
    type of the relationship (found in the relationship type ontology),
    CHAR_OFFSETS is a comma-separated list of \"FROM_OFF-TO_OFF\"
    character offsets specifying the text binding of the relationship
    in the untokenized sentence text, and ARGS is a comma-separated list
    of the arguments of the relationship, where entities are represented
    by their unique identifiers and relationship arguments by their
    strings as defined (recursively) above.
    @param sentence: The L{sentence<BasicClasses.Sentence>} from which to
    extract the information to output.
    @type sentence: L{BasicClasses.Sentence}
    """
    for r in [f.rootNode for f in sentence.formulas]:
        print(relationshipToString(r), end=' ')

"""
Defines the actions for the extractor along
with the command-line options for choosing each action.
Each of the list element if a 5-tuple with the following structure:
(NAME, FUNC, SHORTARG, LONGARG, HELP), where
- NAME: an identifier for the action, also used to identify the outpyt type
- FUNC: the function to call to implement the action
- SHORTARG: a short format argument (e.g. \"-t\") for invoking the action
- LONGARG: a long format argument (e.g. \"--text\") for invoking the action
- HELP: a help string defining the action.
"""
actions = [
    ("TXT", printText, "-t", "--text", "Extract the original, untokenized text of the sentence."),
    ("TOK", printTokenizedText, "-k", "--toktext", "Extract the tokenized text of the sentence, with tokens separated by space. Dependency annotations refer to these tokens sequentially by index, the first token indexed by 0."),
    ("OFF", printTokenOffsets, "-o", "--offset", "Extract the tokenized text of the sentence if offset format: \"token(INDEX, [FROM-TO], 'TEXT')\", where INDEX is the token index, FROM and TO character offsets in the untokenized text, and TEXT the text of the token."),
    ("DEP", printBasicDependencies, "-d", "--dependency", "Extract the \"basic\", unexpanded dependencies in FROM-TO[TYPE] format, where FROM and TO are token indices and TYPE the type of the link. NP macro-dependencies are typed \"macro\"."),
    ("PAR", printParallelDependencies, "-p", "--parallel", "Extract dependencies in FROM-TO[TYPE] format, where NP macro-dependencies have been expanded in \"parallel\", so that each of the words covered by the NP macro-dependency is connected to the head of the NP."),
    ("SER", printSerialDependencies, "-s", "--serial", "Extract dependencies in FROM-TO[TYPE] format, where NP macro-dependencies have been expanded \"serially\", so that each of the words covered by the NP macro-dependency is connected to the next word in sequence."),
    ("ENT", printEntities, "-e", "--entity", "Extract the entities annotated in the sentences. The output format is TYPE(ID, [CHAR_OFFSETS], 'TEXT'), where TYPE is the type of the entity, ID is the unique identifier of the entity, CHAR_OFFSETS is a comma-separated list of FROM-TO character offsets specifying where in the untokenized sentence text the entity is found, and TEXT is the text of the entity."),
    ("REL", printRelationships, "-r", "--relationship", "Extract the annotated relationships in the format TYPE([CHAR_OFFSETS], ARGS), where TYPE is the type of the relationship, CHAR_OFFSETS is a comma-separated list of FROM-TO character offsets specifying the text binding of the relationship in the untokenized sentence text, and ARGS is a comma-separated list of the arguments of the relationship. ARGS can contain either entity identifiers or other relationships.")
    ]

if __name__=="__main__":
    # set up the option parser and parse command-line options
    usage="\n\n%prog -h or --help\n%prog [OPTIONS]\n\nA program for extracting parts of the BioInfer corpus annotation in simplified\nformat from the corpus XML file."
    optionParser=OptionParser(usage)

    group1=OptionGroup(optionParser,"*** Standard usage options ***")
    group1.add_option("-b","--bioInferFile",action="store",dest="bioinferXmlFile",metavar="FILENAME",default=None,help="The XML file holding the BioInfer corpus. This parameter is compulsory.")
    optionParser.add_option_group(group1)

    group2=OptionGroup(optionParser,"*** Extraction options ***")
    for a in actions:
        group2.add_option(a[2],a[3],dest=a[0],help=a[4], default=0, action="store_true")
    optionParser.add_option_group(group2)

    options,args=optionParser.parse_args()


    # check that the input XML file is set
    if not options.bioinferXmlFile:
        print("You must specify the --bioInferFile (-b) option.", file=sys.stderr)
        optionParser.print_help()
        sys.exit(-1)


    # require that at least one of the extraction actions is set.
    if len([a[0] for a in actions if options.__dict__[a[0]]]) == 0:
        print("Please specify at least one extraction option.", file=sys.stderr)
        optionParser.print_help()
        sys.exit(1)

    try:
        bioinferFile=open(options.bioinferXmlFile,"rt")
    except IOError as e:
        print("Failed to open '%s': %s" % (e.filename, e.strerror))
        sys.exit(1)


    # parse the XML file
    parser=BIParser()
    parser.parse(bioinferFile)
    bioinferFile.close()


    # for each sentence and all action for which the option is set
    # (a[0] is true), print the sentence and action identifiers,
    # invoke the action function (action[1]), and finally print
    # a newline.    
    for s in parser.bioinfer.sentences.sentences:
        for action in [a for a in actions if options.__dict__[a[0]]]:
            print("%s:%s:" % (s.id, action[0]), end=' ')
            action[1](s)
            print()
