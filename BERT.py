from embedding4bert import Embedding4BERT
emb4bert = Embedding4BERT("bert-base-cased")
def Model_BERT(data):
    # pass
    BERT= []
    for i in range(len(data)):  # For all datasets  prep.shape[0]
         # bert-base-uncased
        tokens, embeddings = emb4bert.extract_word_embeddings(data)
        BERT.append(embeddings[0,:])
    return BERT
