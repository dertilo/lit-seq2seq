from argparse import ArgumentParser
from pytorch_lightning import Trainer

from chatbot_tutorial_code.chatbot_tutorial import load_and_trim
from lit_seq2seq_module import LitSeq2Seq
from seq2seq_datamodule import Seq2SeqDataModule

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitSeq2Seq.add_model_specific_args(parser)
    args = parser.parse_args()

    args_dict = vars(args)
    dm = Seq2SeqDataModule(**args_dict)
    dm.prepare_data()
    pairs, voc = load_and_trim(dm.corpus_name, dm.datafile)
    args_dict["num_words"] = voc.num_words

    model = LitSeq2Seq(**args_dict)
    args.automatic_optimization = False
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model,datamodule=dm)
