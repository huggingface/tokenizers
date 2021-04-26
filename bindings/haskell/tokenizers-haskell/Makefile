all: download test

clean-models:
	rm -f models/*

download: clean-models
	./get_bart.sh
	./get_roberta.sh
	./get_t5.sh

test:
	export LD_LIBRARY_PATH=`pwd`/lib; stack test

watch:
	rg --files -t hs | entr -s "export LD_LIBRARY_PATH=`pwd`/lib; stack test"

download-vocab:
	stack build haskell-test:exe:download-vocab
