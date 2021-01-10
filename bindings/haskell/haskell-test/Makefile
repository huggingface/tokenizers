all: download run

clean:
	rm -f roberta-base-vocab.json*
	rm -f roberta-base-merges.txt*
	rm -f roberta-base-tokenizer.json*

download: clean
	./get_roberta.sh

run:
	export LD_LIBRARY_PATH=`pwd`/lib; stack run haskell-test

