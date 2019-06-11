# Seq2Attn
Official implementation of the Seq2Attn architecture for sequence-to-sequence task.

Paper:
[Transcoding compositionally: using attention to find more generalizable solutions](https://arxiv.org/abs/1906.01234)

Kris Korrel, Dieuwke Hupkes, Verna Dankers, Elia Bruni

## Setup
Requires python >=3.6.

Clone this repo, the parent seq2seq repo and tasks
```
git clone https://github.com/i-machine-think/seq2attn
git clone https://github.com/i-machine-think/machine
git clone https://github.com/i-machine-think/machine-tasks
```
Install `machine` and `seq2attn`
```
pip install --user -e machine
pip install --user -e machine
```

Run example script
```
cd seq2attn
sh seq2attn_example.sh
```
