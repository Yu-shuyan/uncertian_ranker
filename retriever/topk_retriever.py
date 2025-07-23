"""Topk Retriever"""



import torch

from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss





class TopkRetriever():
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.

    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """
    model = None

    def __init__(self,
                 train_data,
                 ice_num,
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.ice_num = ice_num
        self.train_data = train_data
        #gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)
        #gen_datalist = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        #self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        #co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        #self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()

    def create_index(self):
        # self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        # encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        # co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        # dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
        # res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        train_embed = self.forward(self.train_data, information="Embedding train set...")
        id_list = np.arange(len(train_embed))
        #print("len train_embed", len(train_embed))
        #len_embed = [len(embed) for embed in train_embed]
        #print("len_embed", len_embed)
        self.embed_list = np.stack([embed.squeeze() for embed in train_embed], axis=0)
        index.add_with_ids(self.embed_list, id_list)
        return index

    def knn_search(self, query,ice_num,return_index=False):

        test_embed = self.forward(query,information="Embedding test set...")
        #test_embed = np.stack([embed.squeeze() for embed in test_embed], axis=0)
        # print(test_embed.shape)
        rtr_sample_list = []
        rtr_idx_list = []
        for embed in test_embed:
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()
            rtr_sample_list.append([self.train_data[id] for id in near_ids])
            rtr_idx_list.append(near_ids)
        # rtr_idx_list = [[] for _ in range(len(res_list))]
        # logger.info("Retrieving data for test set...")
        # for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
        #     idx = entry['metadata']['id']
        #     embed = np.expand_dims(entry['embed'], axis=0)
        #     near_ids = self.index.search(embed, ice_num)[1][0].tolist()
        #     rtr_idx_list[idx] = near_ids
        if return_index:
            return rtr_idx_list, rtr_sample_list
        else:
            return rtr_sample_list

    def forward(self, raw_data, information=''):
        print(information)
        res = []
        #raw_data = self.tokenizer.encode_plus(raw_data, truncation=True, return_tensors='pt', verbose=False)['input_ids']
        raw_data = self.tokenizer(raw_data, return_tensors="pt", padding="max_length", max_length=70)['input_ids']
        #raw_data = self.tokenizer(raw_data, return_tensors="pt", truncation=True, verbose=False)['input_ids']


        for entry in raw_data:
            with torch.no_grad():
                raw_text = self.tokenizer.batch_decode(entry.unsqueeze(0), skip_special_tokens=True, verbose=False)
                res.append(self.model.encode(raw_text, show_progress_bar=False))


        #res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res

    def retrieve(self, query):
        return self.knn_search(query, self.ice_num)

    # def get_embedding(self, dataloader, process_bar=False, information=''):
    #     _dataloader = copy.deepcopy(dataloader)
    #     if process_bar:
    #         logger.info(information)
    #         _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
    #     noise = []
    #     out = []
    #     for idx, entry in enumerate(_dataloader):
    #         with torch.no_grad():
    #             original_label = self.original_label[idx]
    #             new_label = self.new_label[idx]
    #             if original_label == new_label:
    #                 noise.append(1)
    #             else:
    #                 noise.append(0)
    #             raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
    #             res = self.model.encode(raw_text, show_progress_bar=False)
    #             out.append(torch.Tensor(res))
    #     Out = torch.cat(out)
    #     noise = torch.Tensor(noise)
    #     print("noise", noise)
    #     torch.save(Out, './distribution/embedding_dup2.pth')
    #     torch.save(noise, './distribution/noise_dup2.pth')
    #     print("Successfully save out and noise.")
    #     return Out, noise
    #
    # def visualize(self, Out=None, noise=None):
    #     if Out is None:
    #         encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
    #         co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
    #         dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
    #         self.original_label = self.dataset_reader.dataset["train"]["label"]
    #         self.new_label = self.dataset_reader.dataset["train"]["new_label"]
    #         Out, noise = self.get_embedding(dataloader, process_bar=True, information="Embedding test set...")
    #     X = Out.detach().cpu().numpy()
    #     y = noise.detach().cpu().numpy()
    #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, perplexity=20)
    #     X_tsne1 = tsne.fit_transform(X)
    #     print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne1.shape[-1]))
    #     x_min, x_max = X_tsne1.min(0), X_tsne1.max(0)
    #     X_norm1 = (X_tsne1 - x_min) / (x_max - x_min)
    #     plt.rcParams.update({'font.size': 15})
    #     plt.scatter(X_norm1[:, 0], X_norm1[:, 1], s=2, c=y)
    #     plt.legend()
    #     plt.xticks([])
    #     plt.yticks([])
    #     figure_name = "visualize_dup2.png"
    #     plt.savefig("distribution/{}".format(figure_name))
    #     plt.close()
    #     print("save figure to {}".format(figure_name))
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
