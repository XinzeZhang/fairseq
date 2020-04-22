from fairseq.models.transformer import TransformerModel
en2ch = TransformerModel.from_pretrained(
  'data-bin/ldc.oracle.bpe.en-cn',
  checkpoint_file='checkpoint_best.ldc.pt',
  data_name_or_path='data-bin/ldc.oracle.bpe.en-cn',
  bpe='subword_nmt',
  bpe_codes='experiment/ldc/oracle.bpe/code'
)
pred =  en2ch.translate('hello world',verbose=True,print_alignment=True)
print(pred)