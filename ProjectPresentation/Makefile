DOCNAME=JLGouwsPresentation

all: report

.PHONY: clean

report:   
	lualatex --draftmode --interaction=batchmode $(DOCNAME).tex 
	biber $(DOCNAME).bcf --onlylog
	lualatex --draftmode --interaction=batchmode $(DOCNAME).tex 
	lualatex --interaction=batchmode $(DOCNAME).tex 

view: report
	xreader $(DOCNAME).pdf

quick:
	lualatex --interaction=batchmode $(DOCNAME).tex 

clean:
	rm *.blg *.bbl *.aux *.log
