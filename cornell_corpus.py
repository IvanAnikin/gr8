

from convokit import Corpus, download
corpus = Corpus(filename=download("movie-corpus"))

corpus.print_summary_stats()