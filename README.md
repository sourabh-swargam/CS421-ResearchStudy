# Distilling Step-by-Step!

Code for paper [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)

## Environment Setup

- Setup Conda environment (Local Machine):

```
conda create --name distill python=3.10.6 -y
conda activate distill
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/huggingface/transformers@v4.24.0 datasets sentencepiece protobuf==3.20.* tensorboardX
```

- Setup Environment (For Colab)

```
pip install transformers datasets sentencepiece protobuf==3.20.* tensorboardX
pip install accelerate -U
```

- Extract datasets to `datasets/`:

```
unzip datasets.zip
```

## Command Usages

#### Args usages

- `--from_pretrained`: `t5-small`, `t5-base`, `google/flan-t5-small`, `google/flan-t5-base`
- `--dataset`: `esnli`, `anli1`, `cqa`
- `--label_type`:
  - `--label_type gt`: Use GT label for standard training
  - `--label_type llm`: Use LLM predicted label for distillation training
- `--alpha`: Task weight for multi-task training. Loss = alpha _ label_prediction_loss + (1 - alpha) _ rationale_generation_loss
  - `--alpha 0.5`: recommended
- `--batch_size`: Batch size
- `--grad_steps`: Gradient accumulation step
- `--max_input_length`: Maximum input length
- `--eval_steps`: How many steps to evaluate the model during training
- `--max_steps`: Maximum steps for training
- `--run`: Random seed to use
- `--model_type`:
  - `standard`: Standard finetuning (`--label_type gt`) or distillation (`--label_type llm`)
  - `task_prefix`: Distilling step-by-step
- `--parallelize`: Model parallelism (requires a GPU or TPU)

#### Commands to Run the Code

Use the following commands to run the models and observe outputs. In order change the model, dataset, or type of training to be performed, refer the above Args usages to view possible choices. Replacing the string at the appropriate places will make the code run for the specified parameters.

- Standard Training:

```python
python run.py --from_pretrained t5-base --dataset cqa --model_type standard --label_type gt --batch_size 16
```

- Standard distillation:

```python
python run.py --from_pretrained t5-base --dataset cqa --model_type standard --label_type llm --batch_size 16
```

- Distilling step-by-step with `Palm label` and `PaLM rationale`:

```python
python run.py --from_pretrained t5-base --dataset cqa --model_type task_prefix --label_type llm --llm palm --alpha 0.5 --batch_size 16
```
