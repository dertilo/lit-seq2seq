import os

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from chatbot_tutorial_code.chatbot_tutorial import build_pairs_voc, batch2TrainData, \
    load_and_trim, build_train_batches
from chatbot_tutorial_code.load_preprocess_data import write_formatted_data, \
    loadPrepareData



class Seq2SeqDataModule(LightningDataModule):

    def __init__(
        self,
        corpus_name="cornell movie-dialogs corpus",
        data_dir=os.environ.get("DATA_DIR", "data"),
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size=32,
        n_iteration = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.batch_size = batch_size
        self.data_dir = data_dir
        print(data_dir)
        self.corpus_name = corpus_name
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.datafile = None
        self.n_iteration = n_iteration

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        corpus = os.path.join(self.data_dir, self.corpus_name)
        self.datafile = os.path.join(corpus, "formatted_movie_lines.txt")
        if not os.path.isfile(self.datafile):
            write_formatted_data(corpus, self.datafile)


    def train_dataloader(self):
        pairs,voc = load_and_trim(self.corpus_name,self.datafile)
        training_batches = build_train_batches(self.batch_size, self.n_iteration, pairs, voc)


        loader = DataLoader(
            training_batches,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

