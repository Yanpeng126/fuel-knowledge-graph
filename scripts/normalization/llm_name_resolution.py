"""Resolve ambiguous fuel names using local context and a local language model."""
import argparse
import json
import re
import time
import html
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from urllib.parse import quote
import requests
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
OUTPUT_DIR = Path('results/normalization')
BASE_DICTIONARY_FILE = OUTPUT_DIR / 'fuel_dictionary_base.json'
LLM_CANDIDATES_FILE = OUTPUT_DIR / 'fuel_candidates.json'
OUTPUT_LLM_JUDGED_FILE = OUTPUT_DIR / 'fuel_judgments.json'
OUTPUT_DICTIONARY_FILE = OUTPUT_DIR / 'fuel_dictionary.json'
CACHE_FILE = OUTPUT_DIR / 'pubchem_cache.json'
XML_DIR = Path('data/xml')
QWEN_MODEL_PATH = None
MAX_CONTEXT_CHARS = 3000
MAX_NEW_TOKENS = 256
MAX_RETRY = 2
ACCEPT_CONFIDENCES = {'high'}
PUBCHEM_SLEEP = 0.2
PUBCHEM_TIMEOUT = 20
SUBSCRIPT_MAP = str.maketrans({'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'})
SUPERSCRIPT_MAP = str.maketrans({'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'})

def clean_string(text):
    if text is None:
        return None
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    text = re.sub('[‐-‒–—−]', '-', text)
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = re.sub('[\\u200b\\u200c\\u200d\\ufeff]', '', text)
    text = re.sub('\\s+', ' ', text).strip()
    text = re.sub('\\s*-\\s*', '-', text)
    text = re.sub('\\s*/\\s*', '/', text)
    return text.strip() if text.strip() else None

def norm_key(text):
    text = clean_string(text)
    if not text:
        return None
    text = text.lower()
    text = re.sub('\\s+', ' ', text).strip()
    return text if text else None

def parse_bool(value, default=True):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {'false', 'no', '0'}:
            return False
        if low in {'true', 'yes', '1'}:
            return True
    return default

def load_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def query_pubchem_cid_by_name(name):
    name = clean_string(name)
    if not name:
        return {'status': 'failed', 'cid': None}
    encoded = quote(name, safe='')
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/JSON'
    try:
        r = requests.get(url, timeout=PUBCHEM_TIMEOUT)
    except Exception:
        return {'status': 'failed', 'cid': None}
    if r.status_code != 200:
        return {'status': 'failed', 'cid': None}
    try:
        data = r.json()
    except Exception:
        return {'status': 'failed', 'cid': None}
    cids = data.get('IdentifierList', {}).get('CID', []) or []
    if len(cids) != 1:
        return {'status': 'failed', 'cid': None}
    return {'status': 'success', 'cid': cids[0]}

def query_with_cache(name, query_cache):
    key = norm_key(name)
    if not key:
        return {'status': 'failed', 'cid': None}
    if key in query_cache:
        return query_cache[key]
    result = query_pubchem_cid_by_name(name)
    query_cache[key] = result
    time.sleep(PUBCHEM_SLEEP)
    return result

def get_name_cid(name, query_cache):
    result = query_with_cache(name, query_cache)
    if result.get('status') == 'success' and result.get('cid') is not None:
        return result['cid']
    return None

def resolve_xml_path(xml_file):
    if not xml_file:
        return None
    p = Path(xml_file)
    if p.is_absolute() and p.exists():
        return p
    direct = XML_DIR / p.name
    if direct.exists():
        return direct
    matches = list(XML_DIR.rglob(p.name))
    if matches:
        return matches[0]
    return None

def xml_to_plain_text(xml_path):
    if xml_path is None or not xml_path.exists():
        return ''
    raw = xml_path.read_text(encoding='utf-8', errors='ignore')
    try:
        root = ET.fromstring(raw)
        text = ' '.join(root.itertext())
    except Exception:
        text = re.sub('<[^>]+>', ' ', raw)
    text = html.unescape(text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def split_sentences(text):
    text = re.sub('\\s+', ' ', text).strip()
    if not text:
        return []
    sentences = re.split('(?<=[.!?])\\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in sentences if s.strip()]

def retrieve_local_context(candidate):
    xml_path = resolve_xml_path(candidate.get('xml_file'))
    text = xml_to_plain_text(xml_path)
    if not text:
        return ''
    raw_name = candidate.get('raw_fuel_name')
    aliases = candidate.get('aliases', []) or []
    terms = [raw_name] + aliases
    terms = [clean_string(x) for x in terms if clean_string(x)]
    terms = sorted(set(terms), key=len, reverse=True)
    if not terms:
        return ''
    sentences = split_sentences(text)
    snippets = []
    seen = set()
    for i, sent in enumerate(sentences):
        sent_low = sent.lower()
        matched = False
        for term in terms:
            if term.lower() in sent_low:
                matched = True
                break
        if not matched:
            continue
        start = max(0, i - 1)
        end = min(len(sentences), i + 2)
        snippet = ' '.join(sentences[start:end])
        key = norm_key(snippet)
        if key and key not in seen:
            snippets.append(snippet)
            seen.add(key)
        if len(' '.join(snippets)) > MAX_CONTEXT_CHARS:
            break
    context = '\n\n'.join(snippets)
    context = context[:MAX_CONTEXT_CHARS]
    return context

def load_qwen_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_PATH, torch_dtype=dtype, device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    return (tokenizer, model)

def get_model_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def safe_parse_json(text):
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    text = re.sub('^```(?:json)?\\s*', '', text, flags=re.IGNORECASE)
    text = re.sub('\\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and (end > start):
        return json.loads(text[start:end + 1])
    raise ValueError('Cannot parse LLM output as JSON')

def build_messages(candidate, context):
    user_content = {'raw_fuel_name': candidate.get('raw_fuel_name'), 'aliases': candidate.get('aliases', []) or [], 'doi': candidate.get('doi'), 'xml_file': candidate.get('xml_file'), 'local_context': context}
    system_prompt = '\nYou are normalizing fuel names for a fuel knowledge graph.\n\nYour task is only to infer a canonical fuel concept name for the given raw fuel mention.\n\nRules:\n1. Return only valid JSON.\n2. Do not use markdown.\n3. Only output normalized_name, confidence, and uncertain.\n4. Do not repeat raw_fuel_name, aliases, doi, or xml_file.\n5. Do not invent a precise compound if the evidence is insufficient.\n6. If the mention is a commercial fuel, surrogate fuel, mixture, or abbreviation, normalize it at the fuel-concept level.\n7. If the same abbreviation may mean different things and the context is insufficient, set uncertain=true.\n8. Do not search or match against a large dictionary.\n9. Only use the given raw name, aliases, and local context.\n\nOutput JSON schema:\n{\n  "normalized_name": "",\n  "confidence": "high / medium / low",\n  "uncertain": true\n}\n'.strip()
    user_prompt = json.dumps(user_content, ensure_ascii=False, indent=2)
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

def call_qwen(messages, tokenizer, model):
    import torch
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt')
    input_device = get_model_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text.strip()

def judge_one_candidate(candidate, tokenizer, model):
    context = retrieve_local_context(candidate)
    messages = build_messages(candidate, context)
    for _ in range(MAX_RETRY):
        try:
            output_text = call_qwen(messages, tokenizer, model)
            obj = safe_parse_json(output_text)
            normalized_name = clean_string(obj.get('normalized_name'))
            confidence = clean_string(obj.get('confidence'))
            uncertain = parse_bool(obj.get('uncertain', True), default=True)
            return {'raw_fuel_name': candidate.get('raw_fuel_name'), 'aliases': candidate.get('aliases', []), 'normalized_name': normalized_name, 'confidence': confidence, 'uncertain': uncertain, 'doi': candidate.get('doi'), 'xml_file': candidate.get('xml_file')}
        except Exception:
            time.sleep(1)
    return {'raw_fuel_name': candidate.get('raw_fuel_name'), 'aliases': candidate.get('aliases', []), 'normalized_name': None, 'confidence': 'low', 'uncertain': True, 'doi': candidate.get('doi'), 'xml_file': candidate.get('xml_file')}

def judge_candidates(candidates, tokenizer, model):
    judged = []
    iterator = candidates
    if tqdm is not None:
        iterator = tqdm(candidates, desc='Judging LLM candidates')
    for candidate in iterator:
        judged.append(judge_one_candidate(candidate, tokenizer, model))
    return judged

def deduplicate_names_by_norm_key(names):
    key_to_name = {}
    for name in names:
        name = clean_string(name)
        if not name:
            continue
        key = norm_key(name)
        if not key:
            continue
        if key not in key_to_name:
            key_to_name[key] = name
        else:
            old = key_to_name[key]
            if len(name) < len(old):
                key_to_name[key] = name
    return sorted(key_to_name.values(), key=lambda x: norm_key(x) or x)

def build_name_index(dictionary):
    index = {}
    for i, entry in enumerate(dictionary):
        names = set(entry.get('names', []))
        if entry.get('canonical_name'):
            names.add(entry['canonical_name'])
        for name in names:
            key = norm_key(name)
            if not key:
                continue
            if key not in index:
                index[key] = i
            elif index[key] != i:
                index[key] = -1
    return index

def infer_entry_cid(entry, query_cache):
    names = []
    if entry.get('canonical_name'):
        names.append(entry['canonical_name'])
    names.extend(entry.get('names', []))
    for name in deduplicate_names_by_norm_key(names):
        result = query_with_cache(name, query_cache)
        if result.get('status') == 'success' and result.get('cid') is not None:
            return result['cid']
    return None

def is_potentially_ambiguous_raw_name(name):
    name = clean_string(name)
    if not name:
        return True
    if re.fullmatch('(?:[A-Z][a-z]?\\d*){2,}', name):
        return True
    if re.fullmatch('[A-Z0-9-]{2,12}', name):
        return True
    if re.search('[/+&]', name):
        return True
    return False

def can_add_name_under_cid_rule(name, target_cid, query_cache):
    if target_cid == 'CID_CONFLICT':
        return False
    if target_cid is None:
        return True
    name_cid = get_name_cid(name, query_cache)
    if name_cid is None:
        return True
    return name_cid == target_cid

def expand_dictionary_with_llm_results(base_dictionary, judged_results):
    dictionary = json.loads(json.dumps(base_dictionary, ensure_ascii=False))
    name_index = build_name_index(dictionary)
    if CACHE_FILE.exists():
        with CACHE_FILE.open('r', encoding='utf-8') as f:
            query_cache = json.load(f)
        print(f'Loaded {len(query_cache)} cached PubChem queries from {CACHE_FILE.name}')
    else:
        query_cache = {}
    entry_cid_cache = {}
    stats = Counter()

    def get_entry_cid_cached(entry_idx):
        if entry_idx not in entry_cid_cache:
            entry_cid_cache[entry_idx] = infer_entry_cid(dictionary[entry_idx], query_cache)
        return entry_cid_cache[entry_idx]
    for item in judged_results:
        normalized_name = clean_string(item.get('normalized_name'))
        raw_name = clean_string(item.get('raw_fuel_name'))
        aliases = item.get('aliases', []) or []
        confidence = norm_key(item.get('confidence'))
        uncertain = parse_bool(item.get('uncertain', True), default=True)
        if uncertain:
            stats['skipped_uncertain'] += 1
            continue
        if confidence not in ACCEPT_CONFIDENCES:
            stats['skipped_low_confidence'] += 1
            continue
        if not normalized_name:
            stats['skipped_empty_normalized_name'] += 1
            continue
        normalized_key = norm_key(normalized_name)
        if not normalized_key:
            stats['skipped_invalid_normalized_name'] += 1
            continue
        existing_idx = name_index.get(normalized_key)
        if existing_idx is not None:
            if existing_idx == -1:
                stats['skipped_ambiguous_existing_name'] += 1
                continue
            target_cid = get_entry_cid_cached(existing_idx)
            if target_cid == 'CID_CONFLICT':
                stats['skipped_existing_entry_cid_conflict'] += 1
                continue
            normalized_cid = get_name_cid(normalized_name, query_cache)
            if target_cid is not None and normalized_cid is not None and (target_cid != normalized_cid):
                stats['skipped_normalized_name_cid_conflict'] += 1
                continue
            names_to_add = [normalized_name]
            if raw_name and (not is_potentially_ambiguous_raw_name(raw_name)):
                names_to_add.append(raw_name)
            for alias in aliases:
                alias = clean_string(alias)
                if alias and norm_key(alias) == normalized_key:
                    names_to_add.append(alias)
            for name in names_to_add:
                if can_add_name_under_cid_rule(name, target_cid, query_cache):
                    dictionary[existing_idx].setdefault('names', []).append(name)
            dictionary[existing_idx]['names'] = deduplicate_names_by_norm_key(dictionary[existing_idx].get('names', []))
            stats['merged_into_existing_entry'] += 1
            name_index = build_name_index(dictionary)
            continue
        normalized_cid = get_name_cid(normalized_name, query_cache)
        new_names = [normalized_name]
        if raw_name and (not is_potentially_ambiguous_raw_name(raw_name)):
            raw_cid = get_name_cid(raw_name, query_cache)
            if normalized_cid is None or raw_cid is None or normalized_cid == raw_cid:
                new_names.append(raw_name)
        for alias in aliases:
            alias = clean_string(alias)
            if alias and norm_key(alias) == normalized_key:
                new_names.append(alias)
        new_entry = {'canonical_name': normalized_name, 'names': deduplicate_names_by_norm_key(new_names)}
        dictionary.append(new_entry)
        stats['created_new_entry'] += 1
        name_index = build_name_index(dictionary)
    for entry in dictionary:
        entry['names'] = deduplicate_names_by_norm_key(entry.get('names', []))
    with CACHE_FILE.open('w', encoding='utf-8') as f:
        json.dump(query_cache, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(query_cache)} PubChem queries to cache')
    dictionary = sorted(dictionary, key=lambda x: norm_key(x['canonical_name']) or x['canonical_name'])
    return (dictionary, stats)

def main():
    global OUTPUT_DIR, BASE_DICTIONARY_FILE, LLM_CANDIDATES_FILE, OUTPUT_LLM_JUDGED_FILE, OUTPUT_DICTIONARY_FILE, CACHE_FILE, XML_DIR, QWEN_MODEL_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local model directory')
    parser.add_argument('--xml-dir', type=Path, required=True)
    parser.add_argument('--dictionary-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.dictionary_dir
    BASE_DICTIONARY_FILE = OUTPUT_DIR / 'fuel_dictionary_base.json'
    LLM_CANDIDATES_FILE = OUTPUT_DIR / 'fuel_candidates.json'
    OUTPUT_LLM_JUDGED_FILE = OUTPUT_DIR / 'fuel_judgments.json'
    OUTPUT_DICTIONARY_FILE = OUTPUT_DIR / 'fuel_dictionary.json'
    CACHE_FILE = OUTPUT_DIR / 'pubchem_cache.json'
    XML_DIR, QWEN_MODEL_PATH = (args.xml_dir, args.model)
    for path in [BASE_DICTIONARY_FILE, LLM_CANDIDATES_FILE]:
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    if not XML_DIR.is_dir():
        parser.error('The XML directory does not exist.')
    print(f'Base dictionary:       {BASE_DICTIONARY_FILE}')
    print(f'LLM candidates:        {LLM_CANDIDATES_FILE}')
    print(f'XML directory:         {XML_DIR}')
    print(f'Local Qwen model:      {QWEN_MODEL_PATH}')
    print(f'Output judged file:    {OUTPUT_LLM_JUDGED_FILE}')
    print(f'Output dictionary:     {OUTPUT_DICTIONARY_FILE}')
    base_dictionary = load_json(BASE_DICTIONARY_FILE)
    candidates = load_json(LLM_CANDIDATES_FILE)
    print(f'\nBase dictionary entries: {len(base_dictionary)}')
    print(f'LLM candidates:          {len(candidates)}')
    print('\nLoading Qwen model...')
    tokenizer, model = load_qwen_model()
    print('Running LLM judgment...')
    judged_results = judge_candidates(candidates, tokenizer, model)
    save_json(judged_results, OUTPUT_LLM_JUDGED_FILE)
    print('Expanding dictionary...')
    expanded_dictionary, stats = expand_dictionary_with_llm_results(base_dictionary=base_dictionary, judged_results=judged_results)
    save_json(expanded_dictionary, OUTPUT_DICTIONARY_FILE)
    print('\n================ DONE ================')
    print(f'Base dictionary entries:     {len(base_dictionary)}')
    print(f'Expanded dictionary entries: {len(expanded_dictionary)}')
    print(f'Judged LLM results saved to: {OUTPUT_LLM_JUDGED_FILE}')
    print(f'Expanded dictionary saved to:{OUTPUT_DICTIONARY_FILE}')
    print('\nMerge stats:')
    for k, v in stats.items():
        print(f'  {k}: {v}')
if __name__ == '__main__':
    main()
