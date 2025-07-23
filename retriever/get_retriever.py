from retriever.topk_retriever import TopkRetriever
#from retriever.dpp_retriever import DPPRetriever
def get_retriever(args):
    if args.retrieve_method == 'topk':
        return TopkRetriever
    elif args.retrieve_method == 'dpp':
        return DPPRetriever