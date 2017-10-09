#!/bin/bash

bibtex proposal
pdflatex proposal.tex
pdflatex proposal.tex
if [ "$(uname)" == "Darwin" ]; then
    open proposal.pdf
else
    xdg-open proposal.pdf
fi
