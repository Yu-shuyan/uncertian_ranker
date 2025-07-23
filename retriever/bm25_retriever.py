from rank_bm25 import BM25Okapi
def bm25_retrieve(query, corpus, n, return_index=False):
    tokenized_corpus = []
    for doc in corpus:
        doc_tokens = doc.split()
        tokenized_corpus.append(doc_tokens)
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc = bm25.get_top_n(tokenized_query, corpus, n=n, return_index=return_index)
    return doc
