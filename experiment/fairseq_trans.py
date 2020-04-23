from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


class transformer_translator():
    def __init__(self, src_lang='en', tgt_lang='cn', state_name='checkpoint_best.ldc.pt', dict_path='data-bin/ldc.oracle.bpe.en-cn', model_path='data-bin/ldc.oracle.bpe.en-cn', beam_num=10, code_path='experiment/ldc/oracle.bpe/code'):
        self.dict_path = dict_path
        self.model_path = '--path ' + os.path.join(model_path, state_name)
        self.beam_setting = '--beam {}'.format(beam_num)

        self.tokenizer_setting = '--tokenizer moses'
        self.bpe_setting = '--bpe subword_nmt'
        self.bpe_path = '--bpe-codes {}'.format(code_path)
        self.bpe_remove = '--remove-bpe'
        self.print_setting = '--replace-unk'
        self.alignment_setting = '--print-alignment'
        self.detokenized = False

        modified_cls = '{} {} {} {} {} {} {} {} {}'.format(
            self.dict_path,
            self.model_path,
            self.beam_setting,
            self.tokenizer_setting,
            self.bpe_path,
            self.bpe_setting,
            self.bpe_remove,
            self.print_setting,
            self.alignment_setting
        )

        self.modified_args = modified_cls.split(' ')

        self.parser = options.get_generation_parser(interactive=True)
        self.args = options.parse_args_and_arch(self.parser, input_args=self.modified_args)

        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_sentences = 1

        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not self.args.max_sentences or self.args.max_sentences <= self.args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        logger.info(self.args)

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        # Setup self.task, e.g., translation
        self.task = tasks.setup_task(self.args)

        # Load ensemble
        logger.info('loading model(s) from {}'.format(self.args.path))
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(os.pathsep),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()
        
        self.generator = self.task.build_generator(self.models, self.args)

        # Handle tokenization and BPE
        self.tokenizer = encoders.build_tokenizer(self.args)
        self.bpe = encoders.build_bpe(self.args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)
        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )

    def encode_fn(self,x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self,x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def translate(self, src):
        pred = None
        inputs= [src.strip()]
        
        results = []

        start_id = 0
        for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))
        
        # sort output to match input order
        # for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
        sorted_results= sorted(results, key=lambda x: x[0])
        id,src_tokens, hypos = sorted_results[0][0],sorted_results[0][1],sorted_results[0][2]
        if self.src_dict is not None:
            src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
            # print('S-\t{}'.format(src_str))

        # Process top predictions

        # for hypo in hypos[:min(len(hypos), self.args.nbest)]:
        hypo = hypos[:min(len(hypos), self.args.nbest)][0]
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo['tokens'].int().cpu(),
            src_str=src_str,
            alignment=hypo['alignment'],
            align_dict=self.align_dict,
            tgt_dict=self.tgt_dict,
            remove_bpe=self.args.remove_bpe,
        )
        
        pred = hypo_str

        if self.detokenized:
            detok_hypo_str = self.decode_fn(hypo_str)
                
                # score = hypo['score'] / math.log(2)  # convert to base 2
                # # original hypothesis (after tokenization and BPE)
                # print('H-\t{}\t{}'.format( score, hypo_str))
                # # detokenized hypothesis
                # print('D-\t{}\t{}'.format( score, detok_hypo_str))
                # print('P-\t{}'.format(
                #     ' '.join(map(
                #         lambda x: '{:.4f}'.format(x),
                #         # convert from base e to base 2
                #         hypo['positional_scores'].div_(math.log(2)).tolist(),
                #     ))
                # ))
                # if args.print_alignment:
                #     alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                #     print('A-\t{}'.format(
                #         alignment_str
                #     ))
        return src, pred
    
    def pair_check(self, src, tgt):
        t = len(src)
        assert len(tgt) == t, 'Miss tgt in cuurent stream'

    def dco_translate(self, src_path, save_path):
        src = []
        with open(src_path, encoding='utf-8') as f:
            for l in f:
                l = l.strip()
                src.append(l)

        preds = []
        count = 0
        for l in src:
            # time.sleep(0.101)
            _, pred = self.translate(l)
            pred = pred.strip()
            count += 1
            print('-'*30)
            print("Sentences: " + str(count))
            print('Src: '+_)
            print('====>')
            print('Pred: '+pred)
            preds.append(pred)

        self.pair_check(src, preds)

        with open(save_path, mode='w') as f:
            for s in preds:
                f.write(s.strip() + '\n')
        print('Save translation to '+save_path + ' Successfully!')

if __name__ == "__main__":
    srcLang, tgtLang = 'en', 'cn'
    trans = transformer_translator(src_lang=srcLang, tgt_lang=tgtLang)
    source = 'export of high-tech products has frequently been in the spotlight , making a significant contribution to the growth of foreign trade in guangdong .'
    _, pred = trans.translate(source)
    print(pred)