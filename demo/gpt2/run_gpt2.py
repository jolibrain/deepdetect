import random
import sys
import argparse
from dd_client import DD

parser = argparse.ArgumentParser(description="Use DeepDetect and GPT-2 to generate text")
parser.add_argument("-r", "--repository", required=True, help="Model repository")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--cpu", action='store_true', help="Force model to run on CPU")
parser.add_argument("--input-size", type=int, default=512)
parser.add_argument("--topk", type=int, default=5, help="How many top predictions should be considered to chose the next token.")
parser.add_argument("--temperature", type=float, default=1, help="Temperature of the predictions. The higher, the 'randomer'.")

args = parser.parse_args()

# dd global variables
sname = 'gpt-2'
description = 'Inference with GPT-2'
mllib = 'torch'

dd = DD(args.host, args.port)
dd.set_return_format(dd.RETURN_PYTHON)

# setting up the ML service
model = {'repository':args.repository}
parameters_input = {
    'connector':'txt',
    'ordered_words': True,
    'wordpiece_tokens': True,
    'punctuation_tokens': True,
    'lower_case': False,
    'width': args.input_size
}
parameters_mllib = {'template':'gpt2', 'gpu':True}
parameters_output = {}
dd.put_service(sname,model,description,mllib,
               parameters_input,parameters_mllib,parameters_output)

# generating text
prompt = input("Enter beggining of sentence >>> ")

for i in range(0, 256):
    data = [prompt]
    parameters_input = {'word_start': "Ġ", 'suffix_start': ""}
    parameters_mllib = {}
    parameters_output = {'best':args.topk}
    result = dd.post_predict(sname, data, parameters_input,parameters_mllib,parameters_output)

    # Select result from the returned tokens
    word_probs = list()
    total_probs = 0

    for cls in result['body']['predictions'][0]['classes']:
        word = cls['cat'].replace("Ġ", " ")
        # dede does not support \n character well, so we don't select tokens containing a new line
        if 'Ċ' in word:
            continue

        prob = pow(cls['prob'], args.temperature)
        total_probs += prob
        word_probs.append((word, prob))
    
    selector = random.uniform(0, total_probs)
    total_probs = 0

    for word, prob in word_probs:
        total_probs += prob
        if total_probs > selector:
            selected_word = word
            break

    print(selected_word, sep='', end='')
    sys.stdout.flush()
    prompt += selected_word
