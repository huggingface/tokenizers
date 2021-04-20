all: download test

clean:
	rm -f roberta-base-vocab.json*
	rm -f roberta-base-merges.txt*
	rm -f roberta-base-tokenizer.json*

download: clean
	./get_roberta.sh

test:
	export LD_LIBRARY_PATH=`pwd`/lib; stack test

watch:
	rg --files -t hs | entr -s "export LD_LIBRARY_PATH=`pwd`/lib; stack test"

download-vocab:
	stack build haskell-test:exe:download-vocab
