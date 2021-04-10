all: download run

clean:
	rm -f roberta-base-vocab.json*
	rm -f roberta-base-merges.txt*
	rm -f roberta-base-tokenizer.json*

download: clean
	./get_roberta.sh

run:
	export LD_LIBRARY_PATH=`pwd`/lib; stack run haskell-test

# watch:
#	ghcid -c "stack ghci haskell-test:haskell-test" -Tmain
watch:
	rg --files -t hs | entr -s "export LD_LIBRARY_PATH=`pwd`/lib;stack run haskell-test"

download-vocab:
	stack build haskell-test:exe:download-vocab
