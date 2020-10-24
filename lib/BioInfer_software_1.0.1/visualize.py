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
tkinterMessage="""ERROR: Failed to import Tkinter

This program uses the standard Python library TkInter to implement the
graphical user interface features. Importing this library failed,
suggesting that this library is not installed on your system or is not
in PATH. In order to run this program, you will need to install
TkInter (Tk support for Python).

TkInter is included by default on Windows. If you see this message on
Windows, you may need to check your installation or install a new
version of Python.

The following webpage may give you some pointers on how to install
TkInter on non-Windows operating systems:

http://tkinter.unpythonic.net/wiki/How_20to_20install_20Tkinter
"""

try:
    from Tkinter import *
except ImportError:
    print tkinterMessage
    sys.exit(-1)
import tkFileDialog as fd
import tkMessageBox as mb
import tkSimpleDialog as sd
from tkFont import Font
from BIParser import BIParser

class TextAsListbox(Frame):
    """
    Text widget modified to resemble Listbox. We need the tagging ability of text widget.
    """

    def __init__(self,master,visualiser,title,**args):
        """
        Initialises the object.

        @param master: The parent widget.
        @param visualiser: The parent Visualiser object.
        @param args: Optional widget configuration.
        """
        Frame.__init__(self,master,**args)
        self.visualiser = visualiser
        self.rowconfigure(1,weight=1)
        self.columnconfigure(0,weight=1)

        Label(self, text=title).grid(sticky=W)
        self.list = Text(self,wrap=NONE,state="disabled",height=1,width=1)
        self.list.grid(sticky=N+E+W+S)

        self.list.scroll_y = Scrollbar(self, orient=VERTICAL,
                               command=self.list.yview, width=12, bd=1)
        self.list.scroll_y.grid(row=1, column=1, sticky=N+S)
        self.list.scroll_x = Scrollbar(self, orient=HORIZONTAL,
                               command=self.list.xview, width=12, bd=1)
        self.list.scroll_x.grid(row=2, column=0, sticky=E+W)
        self.list['xscrollcommand'] = self.list.scroll_x.set
        self.list['yscrollcommand'] = self.list.scroll_y.set

        self.list.bind("<Button-1>",self.click)
        self.list.bind("<Enter>", self.focus)
        self.list.bind("<Down>", self.down)
        self.list.bind("<Up>", self.up)
        self.list.bind("<Right>", self.right)
        self.list.bind("<Left>", self.left)

    def click(self,event):
        self.select(str(int(self.list.index(CURRENT).split(".")[0])))
        
    def select(self,line):
        line = str(line)
        start = line+".0 linestart"
        end = line+".0 lineend"
        self.list.mark_set(INSERT,start)
        self.list.mark_set(CURRENT,start)
        self.list.tag_remove(SEL,1.0,END)
        self.list.tag_add(SEL,start,end)

    def right(self,event):
        return "break"

    def left(self,event):
        return "break"

    def down(self,event):
        line = int(self.list.index(INSERT).split(".")[0])+1
        lastline = int(self.list.index(END).split(".")[0])-1
        if line<lastline:
            self.select(str(line))
        return "break"

    def up(self,event):
        self.select(str(int(self.list.index(INSERT).split(".")[0])-1))
        return "break"

    def focus(self,event):
        self.list.focus_set()

    def clear(self):
        """
        Clears the list.
        """
        self.list.config(state="normal")
        self.list.delete("1.0",END)
        for i in self.list.tag_names():
            self.list.tag_delete(i)
        self.list.config(state="disabled")

    def getIndex(self):
        """
        Gives the index for the sentence under the cursor.
        """
        indexStr = self.list.index(INSERT)
        return int(indexStr.split(".")[0])-1


class Pulldown(Menubutton):
    """
    A simple class for creating a pulldown menu.
    """
    def __init__(self, master, preText, alts, function, **args):
        """
        Initialises an object.

        @param master: The parent widget.
        @param preText: The predefined text.
        @param alts: The list of alternatives.
        @param function: The function to be called as function(item) when item is is chosen.
        @param args: Optional widget configuration.
        """
        Menubutton.__init__(self, master, args)
        self.preText = preText
        self.function = function
        self['text'] = self.preText
        self.setTypes(alts)

    def getType(self,offset=0):
        if not self.types:
            return None
        
        if self.preText:
            curType = self['text'].split(": ")[1]
        else:
            curType = self['text']
        idx = self.types.index(curType)+offset
        idx = idx%len(self.types)
        return self.types[idx]

    def setTypes(self,alts):
        self.types = []
        menu = self['menu'] = Menu(self)
        for item in alts:
            self.types.append(item)
            menu.add_command(label=item,
                             command=lambda item=item: self.switch(item))
        if len(alts)>0:
            self.switch(alts[0])
        

    def switch(self, item):
        if item not in self.types:
            return
        
        if self.preText:
            self['text'] = self.preText + ": " + item
        else:
            self['text'] = item
        self.function(item)


class Statusbar(Frame):
    """
    A frame for displaying a predefined text preceding a changeable value.
    """

    def __init__(self,master,text="",**args):
        """
        Initialises a Statusbar object.

        @param master: The parent widget.
        @param text: The predefined text.
        @param args: Optional widget configuration.
        """
        Frame.__init__(self,master,**args)
        self.preText = text
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)
        self.text = Label(self, text=self.preText, anchor=W, **args)
        self.text.grid(sticky=W)

    def draw(self,text):
        """
        Sets the displayed value of the statusbar.

        @param text: The new value.
        """
        self.text.config(text=self.preText+text)
        self.text.update_idletasks()

    def clear(self):
        """
        Clears the statusbar. Note that the predefined text is still displayed.
        """
        self.text.config(text=self.preText)
        self.text.update_idletasks()


class Toolbar(Frame):
    """
    A frame for displaying buttons.
    """

    def __init__(self,master,**args):
        """
        Initialises a Toolbar object.

        @param master: The parent widget.
        @param args: Optional widget configuration.
        """
        Frame.__init__(self,master,**args)
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)
        self.buttons = []

    def addButton(self, text, function):
        """
        Adds a button to the toolbar.

        @param text: The text to be displayed in the button
        @param function: The callback function for the button
        """
        newButton = Button(self, text=text, command=function)
        newButton.pack(side=LEFT)
        self.buttons.append(newButton)


class ConstrainedCanvas(Canvas):
    """
    A variant of Canvas that constrains scrolling moveto requests
    through the yview and xview to the region (0.0,1.0). This avoids
    some undesireable behaviour when the shown canvas region is larger
    than the scrollregion.
    """
    def yview(self, *args):
        newargs = [a for a in args]
        if len(args) == 2 and args[0] == "moveto":
            if float(args[1]) < 0.0:
                newargs[1] = 0.0
            elif float(args[1]) > 1.0:
                newargs[1] = 1.0
        Canvas.yview(self, *newargs)
    
    def xview(self, *args):
        newargs = [a for a in args]
        if len(args) == 2 and args[0] == "moveto":
            if float(args[1]) < 0.0:
                newargs[1] = 0.0
            elif float(args[1]) > 1.0:
                newargs[1] = 1.0
        Canvas.xview(self, *newargs)


class DepView(Frame):
    """
    A frame for displaying dependencies.
    """

    def __init__(self,master,visualiser,**args):
        """
        Initializes the DepView.
        """
        Frame.__init__(self,master,**args)
        self.visualiser = visualiser
        self.rowconfigure(1,weight=1)
        self.columnconfigure(0,weight=1)

        self.canvas = None

        # configuration of the canvas view
        self.token_font_height = -12 # Tk: negative size guarantees pixels
        self.token_font = Font(font=("Helvetica", self.token_font_height))
        self.token_padding = 2
        self.token_text_margin = 10 + self.token_padding # extra space around the text

        self.rect_x_space     = 4
        self.arc_base_height  = 8

        self.link_type_font_height = -10
        self.link_type_font = Font(font=("Courier", self.link_type_font_height))

        self.entity_rect_height  = 5
        self.entity_rect_y_space = 2

        self.word_rect_start_y = 30
        self.horizontal_margin = 10
        self.top_margin        = 5
        self.bottom_margin     = 15

        self.default_link_type = ""
        self.entity_reserve_height = 100
        self.entity_rect_x = {}
        self.entity_rect_width = {}
        self.entity_rect_color = "#006ab3"
        self.relationship_rect_color = "#94ba5f"
        
        # entity and item references, filled in during drawing
        self.entity_by_id = {}
        self.drawn_entity_items = {}

        self.highlighted = None

        # layout of the view
        Label(self, text="Dependency").grid(sticky=W)
        self.canvas = ConstrainedCanvas(self, bg="#ffffff",
                   width=1, height=1,
                   scrollregion=(0,0,0,0),
                   borderwidth=1, relief=RIDGE)
        self.canvas.grid(sticky=N+E+W+S)

        # make scrollbars for the canvas
        self.canvas.scroll_y = Scrollbar(self, orient=VERTICAL,
                               command=self.canvas.yview, width=12, bd=1)
        self.canvas.scroll_y.grid(row=1, column=1, sticky=N+S)
        self.canvas.scroll_x = Scrollbar(self, orient=HORIZONTAL,
                               command=self.canvas.xview, width=12, bd=1)
        self.canvas.scroll_x.grid(row=2, column=0, sticky=E+W)
        self.canvas['xscrollcommand'] = self.canvas.scroll_x.set
        self.canvas['yscrollcommand'] = self.canvas.scroll_y.set

        self.canvas.bind("<Enter>", self.focus)

    def focus(self,event):
        """
        Called when this widget gets focus. Passes focus down to the
        canvas.
        """
        self.canvas.focus_set()

    def clear(self):
        """
        Clears the canvas.
        """

        self.default_link_type = ""
        self.canvas.delete(ALL)

    def drawTokens(self, sentence):
        """
        Draws the tokens of the given sentence, with rectangles
        surrounding the text of the tokens.
        @param sentence: the sentence from which the drawn tokens are
        taken.
        @type sentence: L{BasicClasses.Sentence}
        """

        token_color = "#ddddee"

        xpos = 0
        ypos = 0
        for t in sentence.tokens:
            xt = xpos
            yt = ypos
            for s in t.subTokens:
                text = self.canvas.create_text(xt, yt, text=s.text,
                                               anchor=NW, font=self.token_font,
                                               tags=('subtoken', s.id, t.id))
                xt = self.canvas.bbox(text)[2]-1

            bbox = self.canvas.bbox(t.id)
            coords = (bbox[0]-self.token_padding,
                      bbox[1]-self.token_padding,
                      bbox[2]+self.token_padding,
                      bbox[3]+self.token_padding)
            temp = self.canvas.create_rectangle(coords,fill=token_color,
                                                tags=('token', t.id))
            self.canvas.lower(temp)
            xpos = bbox[2]+self.token_text_margin

    def drawConnectingLine(self, start_rect, end_rect, tags):
        """
        Draws a line connecting the two given rectangles with the given tags.
        @param start_rect: The starting rectangle.
        @type start_rect: 4-element list of the rectangle coordinates.
        @param end_rect: The ending rectangle.
        @type end_rect: 4-element list of the rectangle coordinates.
        @param tags: The tags to assign to the created line.
        @type tags: Tuple
        """
        x1 = start_rect[2]
        y1 = (start_rect[1]+start_rect[3])/2
        x2 = end_rect[0]
        y2 = (end_rect[1]+end_rect[3])/2
        self.canvas.create_line(x1, y1, x2, y2,
                                stipple="gray50",
                                tags=tags,
                                state='hidden')

    def drawEntityRectangle(self, entity, start_st, end_st):
        """
        Draws a highlight rectangle under the given L{BasicClasses.Entity}
        spanning the given L{BasicClasses.SubToken} objects.
        @param entity: The Entity for which the rectangle will be drawn.
        @type entity: L{BasicClasses.Entity}
        @param start_st: The first SubToken to draw by the rectangle under.
        @type start_st: L{BasicClasses.SubToken}
        @param end_st: The last SubToken to draw the rectangle under.
        @type end_st: L{BasicClasses.SubToken}
        @return A 4-element list of the coordinates of the drawn rectangle.
        """
        c1 = self.canvas.bbox(start_st.id)
        c2 = self.canvas.bbox(end_st.id)
        x1 = c1[0]
        y1 = (c1[3]+c1[1]-self.entity_rect_height)/2
        x2 = c2[2]
        y2 = (c2[3]+c2[1]+self.entity_rect_height)/2
        coords = [x1,y1,x2,y2]

        color = self.entity_rect_color
        if entity.isFormulaRelationship():
            color = self.relationship_rect_color

        self.canvas.create_rectangle(coords,
                                     fill=color,
                                     tags=(entity.id,"BI_all",'entity'),
                                     state='hidden',
                                     width=0)
        return coords
        
    def drawEntities(self,sentence):
        """
        Draws highlight rectangles for all the entities in the given
        sentence. Entities consisting of nonconsequtive subtokens will
        be drawn with lines connecting the subtokens.
        @param sentence: the sentence from which the drawn entities are
        taken.
        @type sentence: L{BasicClasses.Sentence}
        """
        for e in sentence.entities:
            ordered_st = e.subTokens[:]
            ordered_st.sort(lambda a,b: a.sequence - b.sequence)

            # determine rectangles, combining adjacent subtokens.
            start_st, prev_st = None, None
            last_coords = None
            for st in ordered_st:
                if prev_st is not None and st.sequence == prev_st.sequence+1:
                    prev_st = st  # consequtive; combine
                else:
                    # non-consequtive; draw previous.
                    if start_st is not None:
                        coords = self.drawEntityRectangle(e, start_st, prev_st)

                        # draw connecting line to previous rectangle
                        if last_coords:
                            self.drawConnectingLine(last_coords, coords,
                                                    (e.id,"BI_all",'entity'))
                        last_coords = coords
                    start_st = st

                prev_st = st

            # draw rectangle for the last entity also.
            if start_st is not None and prev_st is not None:
                coords = self.drawEntityRectangle(e, start_st, prev_st)
                if last_coords:
                    self.drawConnectingLine(last_coords, coords,
                                            (e.id,"BI_all",'entity'))

    def drawLink(self, linkage_type, link):
        """
        Draws a single link. If the link has a type, the text of the type
        will be drawn on the link.
        @param linkage_type: the type of the linkage the link belongs to.
        The drawn link will be tagged with the linkage_type.
        @type linkage_type: String
        @param link: The link to draw.
        @type link: L{BasicClasses.Link}
        """
        start = link.token1.id
        end   = link.token2.id

        bbox_start = self.canvas.bbox(start)
        bbox_end = self.canvas.bbox(end)
        dist = abs(link.token2.sequence - link.token1.sequence)
                    
        start_x = (bbox_start[0]+bbox_start[2])/2
        start_y = bbox_start[1] - dist * self.arc_base_height
        end_x = (bbox_end[0]+bbox_end[2])/2
        end_y = bbox_end[1] + dist * self.arc_base_height

        # determine width, stipple and color by type
        width, color, ols = 1, "#000000", None
        if link.macro:
            width = 2
        elif "expanded" in link.category:
            color = '#0000A0'
            # ols = 'gray50'

        self.canvas.create_arc(start_x, start_y, end_x, end_y,
                               start=0, width=width, extent=180,
                               style=ARC, outline=color,
                               outlinestipple=ols,
                               tags=(linkage_type,'link'))

        # if a type is defined, draw it as text, with a blank
        # box below to clear the area
        if link.type != "None":
            xpos = (start_x+end_x)/2
            ypos = start_y
            text = self.canvas.create_text(xpos, ypos,
                                           text=link.type, anchor=CENTER,
                                           font=self.link_type_font,
                                           tags=(linkage_type, "linktype"),
                                           fill=color)

            bbox = self.canvas.bbox(text)
            rect = self.canvas.create_rectangle(bbox, fill='#ffffff',
                                                tags=(linkage_type,"linktype"),
                                                width=0)
                        
            self.canvas.lift(rect)
            self.canvas.lift(text)

    def drawLinks(self, sentence):
        """
        Draws all the links of the sentence for all linkages.
        """

        if len(sentence.linkages.keys())==0:
            #print >> sys.stderr, "No linkage found for sentence", sentence
            return
        else:
            self.default_link_type = sentence.linkages.keys()[0]
            for k,v in sentence.linkages.items():
                # draw links longest-first. This way if a link type texts
                # will not be "drawn over" by overlapping long links.
                def linklength_comp(a,b):
                    a_len = abs(a.token1.sequence - a.token2.sequence)
                    b_len = abs(b.token1.sequence - b.token2.sequence)
                    return b_len - a_len

                ordered_links = v.links[:]
                ordered_links.sort(linklength_comp)

                for link in ordered_links:
                    self.drawLink(k, link)

    def updateScrollregion(self):
        """
        Updates the scroll region of the canvas to accommodate
        everything that has been drawn on the canvas.
        """
        # currently only setting scrollregion to the bbox of ALL plus
        # some margin space. Could also consider hiding the scrollbars
        # when they're not necessary; this could be determined with
        # winfo_reqheight/winfo_height etc. (or use Autoscrollbar).

        bbox = self.canvas.bbox(ALL)
        if bbox is None:
            return
        
        self.canvas.config(scrollregion=(bbox[0]-self.horizontal_margin,
                                         bbox[1]-self.top_margin,
                                         bbox[2]+self.horizontal_margin,
                                         bbox[3]+self.bottom_margin))

    def draw(self, sentence):
        """
        Draws everything relating to the given sentence.
        """
        self.clear()

        self.drawTokens(sentence)
        self.drawLinks(sentence)
        self.drawEntities(sentence)

        self.updateScrollregion()

        self.visualiser.showLinks(self.default_link_type)

    def hideLinks(self):
        """
        Hides all links from view.
        """
        self.canvas.itemconfig('link',state='hidden')
        self.canvas.itemconfig('linktype',state='hidden')
        
    def showLinks(self,type):
        """
        Shows links belonging to the given linkage type.
        @param type: The linkage type to show.
        @type type: String
        """
        self.hideLinks()
        self.canvas.lift(type)
        self.canvas.itemconfig(type,state='normal')

    def highlight(self, tags):
        """
        Highlights the entities with the given tags.
        """
        for e in tags:
            self.canvas.itemconfig(e,state='normal')
            self.canvas.lift(e)

        bbox_token = self.canvas.bbox("token")
        for e in self.canvas.find_withtag('entity'):
            if self.canvas.itemcget(e,'state')=='normal':
                for i in self.canvas.gettags(e):
                    if i not in tags:
                        tags.append(i)

        if "BI_all" in tags:
            tags.remove("BI_all")
        if "entity" in tags:
            tags.remove("entity")

        # remove tags without a box.
        cleaned_tags = []
        for t in tags:
            if self.canvas.bbox(t) is not None:
                cleaned_tags.append(t)
        tags = cleaned_tags

        for t in tags:
            # empty coords may be returned for entities without a box
            if len(self.canvas.coords(t)) != 0:
                self.canvas.move(t,0,(bbox_token[3]-self.canvas.coords(t)[1]+
                                      self.entity_rect_y_space))

        # Sort the shortest boxes into the beginning.
        def bbox_comp(a,b):
            c1 = self.canvas.bbox(a)
            c2 = self.canvas.bbox(b)
            result = (c1[2]-c1[0]) - (c2[2]-c2[0])
            if result==0:
                return cmp(a,b)
            return int(result)

        tags.sort(bbox_comp)

        # Move the boxes down if they overlap.
        step = self.entity_rect_y_space + self.entity_rect_height
        for t in tags:
            while True:
                self.canvas.move(t,0,step)
                bbox = self.canvas.bbox(t)
                if bbox is not None:
                    c = self.canvas.find_overlapping(bbox[0],bbox[1],bbox[2],bbox[3])
                    d = self.canvas.find_withtag(t)
                    if c==d:
                        break

        # Update scrollregion: size may have changed.
        self.updateScrollregion()

    def unhighlight(self,tags):
        """
        Removes highlighting from entities with the given tags.
        """
        for i in tags:
            self.canvas.itemconfig(i, state='hidden')

class EntView(TextAsListbox):
    """
    A frame for displaying annotated named entities.
    """

    def __init__(self,master,visualiser,**args):
        """
        Initialises the object.

        @param master: The parent widget.
        @param visualiser: The parent Visualiser object.
        @param args: Optional widget configuration.
        """
        TextAsListbox.__init__(self,master,visualiser,"Entities",**args)

        self.list.bind("<Key-space>",self.newHighlight)
        self.list.bind("<Key-Return>",self.newHighlight)
        self.list.bind("<ButtonRelease-1>",self.newHighlight)
        self.list.bind("<Shift-Key-space>",self.addHighlight)
        self.list.bind("<Shift-Key-Return>",self.addHighlight)
        self.list.bind("<Shift-ButtonRelease-1>",self.addHighlight)

    def newHighlight(self, event):
        self.visualiser.unhighlight(["BI_all"])
        self.addHighlight(event)

    def addHighlight(self, event):
        tags = [i for i in event.widget.tag_names(INSERT)]
        if "BI_all" in tags:
            tags.remove("BI_all")
        if "sel" in tags:
            tags.remove("sel")
        self.visualiser.highlight(tags)

    def draw(self,sentence):
        """
        Lists the entities in the given sentence.

        @param sentence: The Sentence object to be shown.
        """
        self.clear()
        self.list.config(state="normal")
        for l,(k,v) in enumerate((k,v) for (k,v) in sentence.entitiesById.items() if not v.isFormulaRelationship()):
            text = v.getText(True).strip()
            self.list.insert(END,text+"\n")
            self.list.tag_add(k,str(l+1)+".0 linestart",str(l+1)+".0 lineend")
        self.list.tag_add("BI_all","1.0",END)
        self.list.config(state="disabled")

        self.font_height = -12  # Tk: negative size guarantees pixels
        self.font_family = "Helvetica"
        self.font = Font(font=(self.font_family, self.font_height))
        self.highlight_font = Font(font=(self.font_family, self.font_height, "bold"))
        self.highlight_color = '#000000'

        self.list.config(font=self.font)

    def highlight(self,tags):
        for i in [i for i in tags if i in self.list.tag_names()]:
            self.list.tag_raise(i)
            self.list.tag_config(i,foreground=self.highlight_color)
            self.list.tag_config(i,font=self.highlight_font)

    def unhighlight(self,tags):
        for i in [i for i in tags if i in self.list.tag_names()]:
            self.list.tag_raise(i)
            self.list.tag_config(i,foreground="#000000")
            self.list.tag_config(i,font=self.font)

class RelView(Frame):
    """
    A frame for displaying annotated relationships.
    """
    
    def __init__(self,master,visualiser,**args):
        """
        Initialises the object.

        @param master: The parent widget.
        @param visualiser: The parent Visualiser object.
        @param args: Optional widget configuration.
        """
        Frame.__init__(self,master,**args)
        self.visualiser = visualiser
        self.rowconfigure(1,weight=1)
        self.columnconfigure(0,weight=1)

        self.nodes = {}
        self.cursor = None

        Label(self, text="Relationships").grid(sticky=W)
        self.canvas = Canvas(self, height=1, width=1,
                             scrollregion=(0,0,0,0),
                             borderwidth=1, relief=RIDGE)
        self.canvas.grid(sticky='news')

        self.canvas.scroll_y = Scrollbar(self, orient=VERTICAL,
                               command=self.canvas.yview, width=12, bd=1)
        self.canvas.scroll_y.grid(row=1, column=1, sticky=N+S)
        self.canvas.scroll_x = Scrollbar(self, orient=HORIZONTAL,
                               command=self.canvas.xview, width=12, bd=1)
        self.canvas.scroll_x.grid(row=2, column=0, sticky=E+W)
        self.canvas['xscrollcommand'] = self.canvas.scroll_x.set
        self.canvas['yscrollcommand'] = self.canvas.scroll_y.set

        self.canvas.bind('<Down>', self.next)
        self.canvas.bind('<Up>', self.prev)
        self.canvas.bind('<Left>', self.ascend)
        self.canvas.bind('<Right>', self.descend)
        self.canvas.bind('<Home>', self.first)
        self.canvas.bind('<End>', self.last)
        self.canvas.bind('<MouseWheel>', self.roll)
        self.canvas.bind('<Button-4>', self.roll)
        self.canvas.bind('<Button-5>', self.roll)

        self.canvas.bind("<Enter>", self.focus)
        self.canvas.bind("<Button-1>",self.newClickHighlight)
        self.canvas.bind("<Shift-Button-1>",self.addClickHighlight)

        self.canvas.bind("<Key-space>",self.newHighlight)
        self.canvas.bind("<Key-Return>",self.newHighlight)
        self.canvas.bind("<Shift-Key-space>",self.addHighlight)
        self.canvas.bind("<Shift-Key-Return>",self.addHighlight)

        self.font_height = -12  # Tk: negative size guarantees pixels
        self.font_family = "Helvetica"

        self.font = Font(font=(self.font_family, self.font_height))
        self.highlight_font = Font(font=(self.font_family, self.font_height, "bold"))
        self.highlight_color = '#000000'
        
        self.row_height = abs(self.font_height) + 4
        self.indent = self.row_height + 10

    def roll(self,event):
        if event.delta:
            self.canvas.yview('scroll', event.delta, 'units')
        else:
            if event.num==5:
                self.canvas.yview('scroll', 1, 'units')
            else:
                self.canvas.yview('scroll', -1, 'units')

    def resetCursor(self):
        closest = self.canvas.find_closest(self.indent,self.row_height)
        bbox = self.canvas.bbox(closest)
        self.canvas.coords(self.cursor,bbox[0],bbox[1],bbox[2],bbox[3])

    def moveCursor(self,item):
        bbox = self.canvas.bbox(item)
        if bbox:
            self.canvas.coords(self.cursor,bbox[0],bbox[1],bbox[2],bbox[3])

    def resizeCursor(self):
        if self.cursor is None:
            return
        bbox = self.canvas.bbox(self.cursor)
        if bbox is None:
            return
        items = self.canvas.find_overlapping(bbox[0],bbox[1],bbox[2],bbox[3])
        selected = [i for i in items]
        selected.remove(self.cursor)
        bbox = self.canvas.bbox(selected[0])
        if bbox:
            self.canvas.coords(self.cursor,bbox[0],bbox[1],bbox[2],bbox[3])

    def focus(self,event=None):
        self.canvas.focus_set()

    def next(self,event=None):
        coords = self.canvas.coords(self.cursor)
        newY = (coords[1]+coords[3])//2+self.row_height
        i = self.canvas.find_overlapping(0,newY,self.canvas.bbox(ALL)[2],newY)
        bbox = self.canvas.bbox(i)
        if bbox: 
            self.canvas.coords(self.cursor,bbox[0],bbox[1],bbox[2],bbox[3])

    def prev(self,event=None):
        coords = self.canvas.coords(self.cursor)
        newY = (coords[1]+coords[3])//2-self.row_height
        i = self.canvas.find_overlapping(0,newY,self.canvas.bbox(ALL)[2],newY)
        bbox = self.canvas.bbox(i)
        if bbox:
            self.canvas.coords(self.cursor,bbox[0],bbox[1],bbox[2],bbox[3])

    def ascend(self,event=None):
        bbox = self.canvas.bbox(self.cursor)
        under_cur = self.canvas.find_enclosed(bbox[0],bbox[1],bbox[2],bbox[3])
        sel = [i for i in under_cur]
        sel.remove(self.cursor)
        if len(sel)==1 and 'expanded' in self.canvas.gettags(sel[0]):
            sel = sel[0]
            self.canvas.itemconfig(sel,
                                   text="+ "+self.nodes[sel][0].getText(descend=True))
            self.canvas.dtag(sel,'expanded')
            self.canvas.addtag_withtag('collapsed',sel)
            for e in self.nodes[sel][0].getEntities(True):
                self.canvas.addtag_withtag(e.id,sel)
            self.moveCursor(sel)

            selX = self.canvas.bbox(sel)[0]
            checkY = self.canvas.bbox(sel)[1]+self.row_height+5
            maxX,maxY = self.canvas.bbox(ALL)[2:]
            i = self.canvas.find_overlapping(0,checkY,maxX,checkY)
            while i and self.canvas.bbox(i)[0]>selX:
                self.nodes[sel][1].append(i)
                #print "save: "+str(i[0]) # debugging
                self.canvas.itemconfig(i,state="hidden")
                checkY += self.row_height
                i = self.canvas.find_overlapping(0,checkY,maxX,checkY)

            items = filter(lambda a: a>sel, [i for i in self.canvas.find_withtag(ALL)])
            #print "cursor: "+str(self.cursor) # debugging
            for i in items:
                self.canvas.move(i,0,-self.row_height*len(self.nodes[sel][1]))

        #else:
        #    self.prev()

    def descend(self,event=None):
        bbox = self.canvas.bbox(self.cursor)
        under_cur = self.canvas.find_enclosed(bbox[0],bbox[1],bbox[2],bbox[3])
        sel = [i for i in under_cur]
        sel.remove(self.cursor)
        if len(sel)==1 and 'collapsed' in self.canvas.gettags(sel[0]):
            sel = sel[0]
            self.canvas.itemconfig(sel,
                                   text="- "+self.nodes[sel][0].getText(descend=False))
            self.canvas.dtag(sel,'collapsed')
            for e in self.nodes[sel][0].getEntities(True):
                self.canvas.dtag(sel,e.id)
            self.canvas.addtag_withtag('expanded',sel)
            for e in self.nodes[sel][0].getEntities():
                self.canvas.addtag_withtag(e.id,sel)
            self.moveCursor(sel)

            checkY = self.canvas.bbox(sel)[1]+self.row_height+5
            maxX,maxY = self.canvas.bbox(ALL)[2:]
            items = filter(lambda a: a>sel, [i for i in self.canvas.find_withtag(ALL)])
            for i in items:
                self.canvas.move(i,0,self.row_height*len(self.nodes[sel][1]))

            while self.nodes[sel][1]:
                i = self.nodes[sel][1].pop()
                #print "restore: "+str(i[0]) # debugging
                self.canvas.itemconfig(i,state="normal")

        #else:
        #    self.next()

    def first(self,event=None):
        self.resetCursor()
        self.canvas.xview('moveto', 0.0)
        self.canvas.yview('moveto', 0.0)

    def last(self,event=None):
        newY = self.canvas.bbox(ALL)[3] - self.row_height//2
        i = self.canvas.find_overlapping(0,newY,self.canvas.bbox(ALL)[2],newY)
        bbox = self.canvas.bbox(i)
        if bbox: 
            self.canvas.coords(self.cursor,bbox[0],bbox[1],bbox[2],bbox[3])
        self.canvas.xview('moveto', 0.0)
        self.canvas.yview('moveto', 1.0)

    def clear(self):
        """
        Clears the list.
        """
        self.canvas.delete(ALL)
        self.nodes = {}

    def newClickHighlight(self,event):
        self.moveCursor(CURRENT)
        self.newHighlight(event)

    def addClickHighlight(self,event):
        self.moveCursor(CURRENT)
        self.addHighlight(event)

    def newHighlight(self, event):
        self.visualiser.unhighlight(["BI_all"])
        self.addHighlight(event)

    def addHighlight(self, event):
        if self.cursor is None:
            return
        bbox = self.canvas.bbox(self.cursor)
        all = self.canvas.find_enclosed(bbox[0],bbox[1],bbox[2],bbox[3])
        cur = [i for i in all]
        cur.remove(self.cursor)
        if len(cur)>0:
            tags = [i for j in cur for i in event.widget.gettags(j)]
            if "BI_all" in tags:
                tags.remove("BI_all")
            if "expanded" in tags:
                tags.remove("expanded")
            if "collapsed" in tags:
                tags.remove("collapsed")
            if "current" in tags:
                tags.remove("current")
            self.visualiser.highlight(tags)

    def draw(self,sentence):
        """
        Lists the relationships in the given sentence.

        @param sentence: The Sentence object to be shown.
        """
        def drawChildren(f,x,y):
            x += self.indent
            for i in f.arguments:
                y += self.row_height
                if i.arguments:
                    tags = ('BI_all','expanded')
                else:
                    tags = ('BI_all',)
                num = self.canvas.create_text(x, y, anchor='w',
                                text="- "+i.getText(descend=False),
                                tags=tags+tuple(e.id for e in i.getEntities()),
                                font=self.font)
                self.nodes[num] = (i,[])
                x,y = drawChildren(i,x,y)
            return (x-self.indent,y)

        self.clear()
        self.cursor = self.canvas.create_rectangle(0,0,0,0,state='hidden')

        x = 10
        y = - (self.row_height)//2 + 5
        for f in [f.rootNode for f in sentence.formulas]:
            y += self.row_height
            if f.arguments:
                tags = ('BI_all','expanded')
            else:
                tags = ('BI_all',)
            num = self.canvas.create_text(x, y, anchor='w',
                               text="- "+f.getText(descend=False),
                               tags=tags+tuple(e.id for e in f.getEntities()),
                               font=self.font)
            self.nodes[num] = (f,[])
            x,y = drawChildren(f,x,y)

        if self.nodes:
            bbox = self.canvas.bbox(ALL)
            self.canvas.configure(scrollregion=(0,0)+bbox[2:])
            self.canvas.itemconfig(self.cursor,state='normal')
            self.resetCursor()

    def highlight(self, tags):
        for i in [i for i in tags if len(self.canvas.find_withtag(i))>0]:
            self.canvas.itemconfig(i, fill=self.highlight_color)
            self.canvas.itemconfig(i, font=self.highlight_font)
        # redo cursor: highlight may change text size
        self.resizeCursor()

    def unhighlight(self,tags):
        for i in [i for i in tags if len(self.canvas.find_withtag(i))>0]:
            self.canvas.itemconfig(i, fill="#000000")
            self.canvas.itemconfig(i, font=self.font)
        # redo cursor: highlight may change text size
        self.resizeCursor()


class SentenceSelector(TextAsListbox):
    """
    A frame for selecting sentences.
    """

    def __init__(self,master,visualiser,**args):
        """
        Initialises the object.

        @param master: The parent widget.
        @param visualiser: The parent Visualiser object.
        @param args: Optional widget configuration.
        """
        TextAsListbox.__init__(self,master,visualiser,"Sentences",**args)
        self.list.bind("<ButtonRelease-1>",self.changeSentence)
        self.list.bind("<Key-space>",self.changeSentence)
        self.list.bind("<Key-Return>",self.changeSentence)

        self.font_height = -12   # Tk: negative size guarantees pixels
        self.font_family = "Helvetica"
        self.font = Font(font=(self.font_family, self.font_height))
        self.highlight_font = Font(font=(self.font_family, self.font_height, "bold"))
        self.highlight_color = '#000000'

        self.list.config(font=self.font)

    def draw(self,sentenceList):
        """
        Lists the sentences in the given list.

        @param sentenceList: The list to be shown.
        """
        self.clear()
        self.list.config(state="normal")
        for i in sentenceList:
            text = i.id+": "+i.getText()
            self.list.insert(END,text+"\n")
        self.list.config(state="disabled")

    def highlight(self,idx):
        idx = str(idx)
        start = idx+".0 linestart"
        end = idx+".0 lineend"
        self.list.tag_config("mytag",foreground=self.highlight_color)
        self.list.tag_config("mytag",font=self.highlight_font)
        self.list.tag_remove("mytag",1.0,END)
        self.list.tag_add("mytag",start,end)

    def changeSentence(self,event):
        """
        Callback function for changing the shown sentence.
        """
        self.visualiser.showSentenceByIndex(self.getIndex())
        line = str(self.getIndex()+1)
        self.highlight(line)


class Visualiser(Tk):
    """
    A Tk derivative for visualising BioInfer corpus.
    """

    def __init__(self):
        """
        Initialises the window and views.
        """
        Tk.__init__(self)
        self.parser = BIParser()

        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)

        # Views
        self.mainframe = Frame(self)

        # Annotation views
        self.depview = DepView(self.mainframe, self,
                               relief=RIDGE, borderwidth=1)
        self.entview = EntView(self.mainframe, self,
                               relief=RIDGE, borderwidth=1)
        self.relview = RelView(self.mainframe, self,
                               relief=RIDGE, borderwidth=1)
        self.selector = SentenceSelector(self.mainframe, self,
                                         relief=RIDGE, borderwidth=1)
        # "selectable" current sentence view. Disabled for now.
        #self.senview = Entry(self.mainframe, relief=RIDGE, borderwidth=1, state='readonly')

        # Bars
        self.barview = Frame(self.mainframe)
        self.toolbar = Toolbar(self.barview, relief=RIDGE, borderwidth=1)
        self.sentencebar = Statusbar(self.barview, "Sentence: ",
                                   relief=RIDGE, borderwidth=1, width=20)
        self.dependencybar = Pulldown(self.barview, "",
                                      [],
                                      self.depview.showLinks,
                                      relief=GROOVE, borderwidth=1, width=15)

        # Toolbar buttons
        self.toolbar.addButton("Open", self.openFile)
        self.toolbar.addButton("Help", self.displayAbout)
        self.toolbar.addButton("Quit", self.closeProgram)
        self.toolbar.addButton("Go to", self.search)

        # Layout of the window/views

        # Toolbar and statusbar are one-liners, share the top
        # and expand horizontally
        # Main view gets the rest of the space
        # Sentence selector at the very bottom
        self.mainframe.grid(sticky=N+E+W+S)
        self.mainframe.columnconfigure(0,weight=1)
        self.mainframe.columnconfigure(1,weight=1)
        self.mainframe.rowconfigure(1,weight=1)
        self.mainframe.rowconfigure(2,weight=1)
        self.mainframe.rowconfigure(3,weight=1)

        # Bars
        self.barview.grid(sticky=W, columnspan=2)
        
        self.toolbar.grid(sticky=W,row=0, column=0)
        self.sentencebar.grid(sticky=W, row=0, column=1)
        Label(self.barview, text="Linkage type:").grid(sticky=W,
                                                          row=0, column=2)
        self.dependencybar.grid(sticky=W, row=0, column=3)

        # Dependency view
        self.depview.grid(sticky=N+E+W+S, row=1, columnspan=2)
        self.depview.columnconfigure(0,weight=1)
        self.depview.rowconfigure(1,weight=1)

        # Relationship view
        self.relview.grid(sticky=N+E+W+S, row=2, column=0)
        self.relview.columnconfigure(0,weight=1)
        self.relview.rowconfigure(1,weight=1)

        # Entity view
        self.entview.grid(sticky=N+E+W+S, row=2, column=1)
        self.entview.columnconfigure(0,weight=1)
        self.entview.rowconfigure(1,weight=1)

        # Current sentence view
        # Disabled for now.
        #self.senview.grid(sticky=N+E+W+S, row=3, column=0, columnspan=2)
        #self.selector.rowconfigure(1,weight=1)

        # Sentence view
        self.selector.grid(sticky=N+E+W+S, row=3, column=0, columnspan=2)
        self.selector.columnconfigure(0,weight=1)
        self.selector.rowconfigure(1,weight=1)

        # Bind keyboard shortcuts
        self.bind_all("<Control-o>",self.openFile)
        self.bind_all("<Control-h>",self.displayAbout)
        self.bind_all("<Control-x>",self.closeProgram)
        self.bind_all("<Shift-Q>",self.changeDepType)
        self.bind_all("<Shift-W>",self.changeDepType)
        self.bind_all("<Control-s>",self.search)

    def showLinks(self,type):
        self.dependencybar.switch(type)

    def highlight(self,tags):
        self.entview.highlight(tags)
        self.depview.highlight(tags)
        self.relview.highlight(tags)
        
    def unhighlight(self,tags):
        self.entview.unhighlight(tags)
        self.depview.unhighlight(tags)
        self.relview.unhighlight(tags)

    def showCorpus(self):
        self.selector.draw(self.parser.bioinfer.sentences.sentences)
        self.showSentenceByIndex(0)

    def showSentenceByIndex(self,idx):
        """
        Shows a sentence.

        @param idx: The index of the sentence to be shown.
        """
        if self.parser.bioinfer is None:
            return

        self.sentencebar.draw(self.parser.bioinfer.sentences.sentences[idx].id)
        linktype = self.dependencybar.getType()

        # "selectable" sentence view is disabled.
        #self.senview.configure(state='normal')
        #self.senview.delete(0,END)
        #self.senview.insert(END,self.parser.bioinfer.sentences.sentences[idx].getText())
        #self.senview.configure(state='readonly')
        
        self.selector.highlight(idx+1)
        self.selector.select(idx+1)
        self.dependencybar.setTypes(self.parser.bioinfer.sentences.sentences[idx].linkages.keys())
        self.depview.draw(self.parser.bioinfer.sentences.sentences[idx])
        self.entview.draw(self.parser.bioinfer.sentences.sentences[idx])
        self.relview.draw(self.parser.bioinfer.sentences.sentences[idx])

        self.dependencybar.switch(linktype)

    def showSentenceById(self,uid):
        """
        Shows a sentence.

        @param uid: The id of the sentence to be shown.
        """
        idx = self.idToIdx(uid)
        if idx:
            self.showSentenceByIndex(idx)

    def idToIdx(self,uid):
        if self.parser.bioinfer is None:
            return None
        for i,j in enumerate(self.parser.bioinfer.sentences.sentences):
            if j.id==str(uid):
                return i
        return None

    def isValidId(self,uid):
        if self.idToIdx(uid) == None:
            return False
        return True
        
    def openFile(self,event=None):
        """
        Opens and parses a corpus. The file is selected with a dialog.
        """
        selected = fd.askopenfilename()

        if selected: # not None can fail, may return empty tuple
            newParser = BIParser()

            try:
                f=open(selected)
            except IOError, inst:
                mb.showinfo("IO error", inst.args[0])
                return

            try:
                newParser.parse(f)
            except:
                mb.showinfo("Parsing failed",
                            "The file might be corrupted.")
            else:
                if newParser.bioinfer.isValid():
                    self.parser = newParser
                    self.showCorpus()
                else:
                    mb.showinfo("Missing corpus component",
                                "The opened corpus does not contain all required components.")
            f.close()

    def closeProgram(self,event=None):
        """
        Quits.
        """
        self.quit()

    def changeDepType(self,event):
        if event.char=='Q':
            prevType = self.dependencybar.getType(-1)
            if prevType:
                self.dependencybar.switch(prevType)
        elif event.char=='W':
            nextType = self.dependencybar.getType(1)
            if nextType:
                self.dependencybar.switch(nextType)
    
    def search(self,event=None):
        selected = sd.askinteger("Search", "Sentence number:")
        if selected != None:
            if self.isValidId(selected):
                self.showSentenceById(selected)
            else:
                mb.showinfo("Not found", "Sentence number "+str(selected)+" not found")

    def displayAbout(self,event=None):
        """
        Displays an info box.
        """
        window = Toplevel()
        window.columnconfigure(0,weight=1)
        window.rowconfigure(0,weight=1)
        text = Text(window,width=80,bg='white')
        text.columnconfigure(0,weight=1)
        text.rowconfigure(0,weight=1)
        text.grid(sticky=N+E+W+S)
        text.insert(END,
"""
Bioinfer Visualizer help:

The Bioinfer Visualizer consists of four main views and a toolbar.
The three topmost views visualize the three annotation types
(dependency, entity, relationship) while the fourth view shows all
sentences.

The corpus file that is opened must contain the necessary ontologies
and the sentences along the annotation.


Toolbar:

The toolbar contains the following items:
* Buttons
   * Open : Open file
   * Help : Show this help
   * Quit : Close Program
   * Go to : Go to a sentence
* Sentence number
* Linkage type selector

The linkage type selector can be used to select the type of linkage
annotation shown for the sentence.



Dependency View       - The dependency view shows the given sentence, the
                        dependencies between the tokens, and the selected
                        entities/text bindings. Selected entities are
                        highlighted in blue, and the text bindings of
                        selected relationships in green. The type of the
                        shown linkage graph can be changed with the
                        selector in the toolbar or with shortcut keys.

Relationship View     - The relationship view shows the annotated relationships
                        in the given sentence.

Entity View           - The entity view shows the annotated entities in the
                        given sentence.

Sentence selector     - The sentence selector shows the sentences in the corpus
                        and allows selecting them for further inspection.



The available keyboard shortcuts:

Control-o : Open file
Control-h : Show this help
Control-x : Close Program
Shift-Q : Change to the previous linkage type
Shift-W : Change to the next linkage type
Control-s : Go to a sentence



How to use relationship view, entity view and sentence selector:
(You must have the mouse over the desired view in order to use the keyboard.)

Up/Down            Move to next/previous item.
arrow keys:

Spacebar/          Highlight/select an item.
Enter/
Left-Mouse:

Shift-Spacebar/    Add an item to the current selection.
Shift-Enter/
Shift-Left-Mouse:

(relationship view only)
Left arrow key     Collapse the current relationship.
Right arrow keys   Expanded the current relationship.

""")
        

if __name__=="__main__":
    # Initialise
    root = Visualiser()
    root.title("BioInfer Visualizer")
    root.mainloop()
