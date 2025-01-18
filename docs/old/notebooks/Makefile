IPYNB_FILES := $(wildcard *.ipynb)
HTML_FILES  := $(filter-out index.html, $(IPYNB_FILES:.ipynb=.html))

%.html: %.ipynb
	jupyter nbconvert --ExecutePreprocessor.timeout=None --output-dir="." --to html --execute $<

all: html

html: $(HTML_FILES)

$(IPYNB_FILES):
	jupyter nbconvert --ExecutePreprocessor.timeout=None --to notebook --inplace --execute $@

clean:
	@rm -rf $(HTML_FILES)

.PHONY: all html $(IPYNB_FILES) clean
