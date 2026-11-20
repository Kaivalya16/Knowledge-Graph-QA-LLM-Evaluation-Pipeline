#!/usr/bin/env python3
"""Few-shot inference for Qwen2.5-7B-Instruct on CWQ QA datasets."""
import argparse, ast, json, random, re
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = '''You are a question-answering system for ComplexWebQuestions (CWQ). Answer the question using your learned knowledge. Return only a valid JSON array of answer strings, for example ["Paris"] or ["Alice", "Bob"]. If there is no answer, return []. Do not include explanations, markdown, or text outside the JSON array.'''

def parse_args():
    p = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct few-shot CWQ inference")
    p.add_argument('--test_file', required=True)
    p.add_argument('--fewshot_file', required=True, help='Training split used for few-shot demonstrations')
    p.add_argument('--output_file', required=True)
    p.add_argument('--model_name', default=MODEL_DEFAULT)
    p.add_argument('--num_shots', type=int, default=4)
    p.add_argument('--load_in_4bit', action='store_true')
    p.add_argument('--max_new_tokens', type=int, default=128)
    p.add_argument('--max_input_length', type=int, default=4096)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()

def load_list(path):
    with open(path, encoding='utf-8') as f: data = json.load(f)
    if isinstance(data, dict):
        for k in ('data','examples','records'):
            if isinstance(data.get(k), list): data = data[k]; break
    if not isinstance(data, list): raise ValueError(f'{path} must contain a JSON array')
    return data

def answers(value):
    if value is None: return []
    if isinstance(value, (list, tuple)): vals = list(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s: return []
        try: parsed = json.loads(s)
        except json.JSONDecodeError:
            try: parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError): parsed = [s]
        vals = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    else: vals = [value]
    return [str(x).strip() for x in vals if x is not None and str(x).strip()]

def extract_answers(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    try:
        x = json.loads(text)
        if isinstance(x, list): return answers(x), True
    except json.JSONDecodeError: pass
    m = re.search(r'\[[\s\S]*?\]', text)
    if m:
        try:
            x = json.loads(m.group())
            if isinstance(x, list): return answers(x), True
        except json.JSONDecodeError: pass
    return ([text] if text else []), False

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(re.sub(r'[^\w\s]', ' ', s).split())

def score(pred, gold):
    p, g = [norm(x) for x in pred], [norm(x) for x in gold]
    p, g = [x for x in p if x], [x for x in g if x]
    remaining, matched = list(g), 0
    for x in p:
        if x in remaining: remaining.remove(x); matched += 1
    precision = matched/len(p) if p else (1.0 if not g else 0.0)
    recall = matched/len(g) if g else (1.0 if not p else 0.0)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'exact_set_match': set(p)==set(g), 'hit_at_1': matched>0,
            'precision': precision, 'recall': recall, 'f1': f1,
            'num_predicted': len(pred), 'num_gold': len(gold), 'num_matched': matched}

def main():
    a = parse_args()
    if a.num_shots < 0: raise ValueError('--num_shots must be >= 0')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; check nvidia-smi and CUDA-enabled PyTorch.')
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    train, test = load_list(a.fewshot_file), load_list(a.test_file)
    if a.limit is not None: test = test[:a.limit]
    test_ids = {str(x.get('id')) for x in test if x.get('id') is not None}
    # Exclude any train examples whose IDs occur in the test set, to avoid leakage.
    eligible = [x for x in train if x.get('id') is None or str(x.get('id')) not in test_ids]
    shots = random.sample(eligible, min(a.num_shots, len(eligible))) if a.num_shots else []
    print('Few-shot examples:', len(shots), '| IDs:', [x.get('id') for x in shots])

    tok = AutoTokenizer.from_pretrained(a.model_name)
    qconfig = None
    if a.load_in_4bit:
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_name, quantization_config=qconfig,
        device_map='auto', torch_dtype=torch.bfloat16)
    model.eval()
    results = []
    for i, ex in enumerate(tqdm(test, desc='Qwen2.5 CWQ few-shot')):
        q = str(ex.get('question', '')).strip(); gold = answers(ex.get('answers', []))
        if not q:
            results.append({'id': ex.get('id', str(i)), 'question': q, 'gold_answers': gold,
                'predicted_answers': [], 'raw_generation': '', 'answer_parse_success': False,
                'error': 'Missing/empty question'})
            continue
        messages = []
        for s in shots:
            sq = str(s.get('question', '')).strip()
            if not sq: continue
            messages.extend([
                {'role':'user','content': SYSTEM_PROMPT+'\n\nQuestion:\n'+sq},
                {'role':'assistant','content': json.dumps(answers(s.get('answers', [])), ensure_ascii=False)}])
        messages.append({'role':'user','content': SYSTEM_PROMPT+'\n\nQuestion:\n'+q})
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors='pt', truncation=True, max_length=a.max_input_length)
        enc = {k:v.to(model.get_input_embeddings().weight.device) for k,v in enc.items()}
        kwargs = {'max_new_tokens':a.max_new_tokens, 'do_sample':a.temperature>0,
                  'pad_token_id':tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                  'eos_token_id':tok.eos_token_id}
        if a.temperature > 0: kwargs.update(temperature=a.temperature, top_p=a.top_p)
        with torch.no_grad(): out = model.generate(**enc, **kwargs)
        raw = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        pred, ok = extract_answers(raw)
        results.append({'id':ex.get('id',str(i)), 'question':q, 'gold_answers':gold,
                        'predicted_answers':pred, 'raw_generation':raw, 'answer_parse_success':ok,
                        **score(pred,gold)})
    valid = [x for x in results if 'error' not in x]; n = len(valid)
    summary = {'model_name':a.model_name, 'fewshot_file':a.fewshot_file, 'test_file':a.test_file,
        'num_shots':len(shots), 'demonstration_ids':[x.get('id') for x in shots],
        'num_examples':len(test), 'num_evaluated':n,
        'parse_success_rate':sum(x['answer_parse_success'] for x in valid)/n if n else 0.0,
        'exact_set_match':sum(x['exact_set_match'] for x in valid)/n if n else 0.0,
        'hit_at_1':sum(x['hit_at_1'] for x in valid)/n if n else 0.0,
        'precision':sum(x['precision'] for x in valid)/n if n else 0.0,
        'recall':sum(x['recall'] for x in valid)/n if n else 0.0,
        'f1':sum(x['f1'] for x in valid)/n if n else 0.0}
    outpath = Path(a.output_file); outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath,'w',encoding='utf-8') as f: json.dump({'summary':summary,'predictions':results},f,indent=2,ensure_ascii=False)
    print(json.dumps(summary, indent=2)); print('Saved:', outpath)

if __name__ == '__main__': main()
