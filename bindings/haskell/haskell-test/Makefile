all: download run

clean:
	rm -f roberta-base-vocab.json*
	rm -f roberta-base-merges.txt*

download: clean
	./get_roberta.sh

run:
	export LD_LIBRARY_PATH=`pwd`/lib; stack run haskell-test

