#!/usr/bin/python3
import sys
import os
import argparse
import logging

import torch
import torch.nn as nn
import pytorch_transformers as M

parser = argparse.ArgumentParser(description="Trace NLP models from pytorch-transformers")
parser.add_argument('models', type=str, nargs='*', help="Models to trace.")
parser.add_argument('--print-models', action='store_true', help="Print all the available models names and exit")
parser.add_argument('-a', "--all", action='store_true', help="Export all available models")
parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")
parser.add_argument('-o', "--output-dir", default=".", type=str, help="Output directory for traced models")
parser.add_argument('-p', "--not-pretrained", dest="pretrained", action='store_false',
                    help="Whether the exported models should not be pretrained")
parser.add_argument('-t', '--template', default="", type=str, help="Template name of the model, as specified by pytorch-transformers")
parser.add_argument('--cpu', action='store_true', help="Force models to be exported for CPU device")
parser.add_argument('--input-size', type=int, default=512, help="Length of the input sequence")
parser.add_argument('--vocab', action='store_true', help="Export the vocab.dat file along with the model.")
parser.add_argument('--train', action='store_true', help="Prepare model for training")
parser.add_argument('--num-labels', type=int, default=2, help="For sequence classification only: number of classes")

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)

model_classes = {
    "bert": M.BertModel,
    "bert_masked_lm": M.BertForMaskedLM,
    "bert_classif": M.BertForSequenceClassification,
    "roberta": M.RobertaModel,
    "roberta_masked_lm": M.RobertaForMaskedLM,
    "gpt2": M.GPT2Model,
    "gpt2_lm": M.GPT2LMHeadModel,
}

def get_model_type(mname):
    for key in default_templates:
        if mname == key or mname.startswith(key + "_"):
            return key
    return ""
    

default_templates = {
    "bert": "bert-base-uncased",
    "roberta":"roberta-base",
    "gpt2": "gpt2",
}

tokenizers = {
    "bert": M.BertTokenizer,
    "roberta": M.RobertaTokenizer,
    "gpt2": M.GPT2Tokenizer,
}

if args.all:
    args.models = model_classes.keys()

if args.print_models:
    print("*** Available models ***")
    for key in model_classes:
        print(key)
    sys.exit(0)
elif not args.models:
    sys.stderr.write("Please specify at least one model to be exported\n")
    sys.exit(-1)

device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
logging.info("Device: %s", device)

if args.input_size > 512 or args.input_size <= 0:
    logging.error("This input size is not supported: %d", args.input_size)
    sys.exit(-1)

logging.info("Input size: %d", args.input_size)

# Example inputs
input_ids = torch.ones((1, args.input_size), dtype=torch.long, device=device)
att_mask = torch.ones_like(input_ids)
token_type_ids = torch.zeros_like(input_ids)
position_ids = torch.arange(args.input_size, dtype=torch.long, device=device).unsqueeze(0)

for mname in args.models:
    if mname not in model_classes:
        logging.warn("model %s is unknown and will not be exported", mname)
        continue

    model_type = get_model_type(mname)

    # Find appropriate template
    if args.template:
        mtemplate = args.template
    else:
        mtemplate = default_templates[model_type]

    # Additionnal parameters
    kvargs = dict()
    if mname in ["bert_classif"]:
        kvargs["num_labels"] = args.num_labels
    if model_type in ["bert", "roberta"]:
        kvargs["output_hidden_states"] = True

    # Create the model
    mclass = model_classes[mname]
    logging.info("Model class: %s", mclass.__name__)
    logging.info("Use template '%s'", mtemplate)
    model = mclass.from_pretrained(mtemplate, torchscript=True, **kvargs)

    if not args.pretrained:
        logging.info("Create model from scratch with the same config as the pretrained one")
        model = mclass(model.config)

    model.to(device)
    if not args.train:
        model.eval()
    else:
        model.train()

    # Trace the model with the correct inputs
    if mname in ["bert", "bert_masked_lm", "bert_classif", "roberta", "roberta_masked_lm"]:
        traced_model = torch.jit.trace(model, (input_ids, token_type_ids, att_mask))
    elif mname in ["distilbert", "distilbert_masked_lm"]:
        traced_model = torch.jit.trace(model, (input_ids, att_mask))
    elif mname in ["gpt2", "gpt2_lm"]:
        # change order of positional arguments
        def real_forward(self, i, p):
            return self.p_forward(input_ids=i, position_ids=p)
        setattr(mclass, 'p_forward', mclass.forward)
        setattr(mclass, 'forward', real_forward)
        
        traced_model = torch.jit.trace(model, (input_ids, position_ids))
    else:
        raise ValueError("there is no method to trace this model: %s" % mname)
    
    filename = os.path.join(args.output_dir, mname + 
        ("-" + mtemplate if args.template in mclass.pretrained_model_archive_map else "") +
        ("-pretrained" if args.pretrained else "") + ".pt")
    logging.info("Saving to %s", filename)
    traced_model.save(filename)

    # Export vocab.dat
    if args.vocab:
        tokenizer = tokenizers[model_type].from_pretrained(mtemplate)
        filename = os.path.join(args.output_dir, "vocab.dat")
        
        with open(filename, 'w') as f:
            for i in range(len(tokenizer)):
                word = tokenizer.convert_ids_to_tokens([i])[0]
                f.write(word + "\t" + str(i) + "\n")

        logging.info("Vocabulary saved to %s", filename)

logging.info("Done")
