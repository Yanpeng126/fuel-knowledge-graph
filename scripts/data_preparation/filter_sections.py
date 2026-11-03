"""Select article sections using the original extraction keyword rules."""
import argparse
import re
from pathlib import Path
from bs4 import BeautifulSoup

def extract_doi_from_xml(xml_text):
    m = re.search('10\\.\\d{4,9}/[^\\s<>\\"]+', xml_text)
    if not m:
        return None
    doi = m.group(0).strip()
    doi = doi.rstrip('.,);]')
    return doi

def clean_text_basic(text):
    text = re.sub('https?://\\S+', ' ', text)
    text = re.sub('\\s+', ' ', text)
    return text.strip()

def add_part(parts, label, text, min_len=5):
    text = clean_text_basic(text)
    if len(text) < min_len:
        return
    item = f'{label}:\n{text}'
    if item not in parts:
        parts.append(item)

def get_first_text(soup, tag_names):
    for tag_name in tag_names:
        for tag in soup.find_all(tag_name):
            txt = tag.get_text(' ', strip=True)
            if txt:
                return txt
    return ''

def get_all_texts(soup, tag_names):
    texts = []
    for tag_name in tag_names:
        for tag in soup.find_all(tag_name):
            txt = tag.get_text(' ', strip=True)
            txt = clean_text_basic(txt)
            if txt and txt not in texts:
                texts.append(txt)
    return texts

def get_section_title(sec):
    title_tags = ['ce:section-title', 'section-title', 'title', 'ce:title']
    for tag_name in title_tags:
        tag = sec.find(tag_name)
        if tag:
            txt = tag.get_text(' ', strip=True)
            if txt:
                return clean_text_basic(txt)
    txt = sec.get_text(' ', strip=True)
    txt = clean_text_basic(txt)
    return txt[:120]

def should_drop_section(title_lower):
    drop_keywords = ['introduction', 'background', 'chemical kinetic model', 'kinetic model', 'model development', 'reaction mechanism', 'mechanism development', 'reaction path', 'reaction pathway', 'sensitivity analysis', 'flux analysis', 'rate of production', 'conclusion', 'conclusions', 'credit authorship', 'crédit authorship', 'author contribution', 'declaration of competing interest', 'conflict of interest', 'acknowledgement', 'acknowledgment', 'supplementary data', 'appendix', 'references']
    return any((k in title_lower for k in drop_keywords))

def should_keep_section(title_lower):
    keep_keywords = ['experimental', 'experiment', 'apparatus', 'method', 'methods', 'measurement', 'measurements', 'materials', 'facility', 'setup', 'reactor', 'burner', 'shock tube', 'rapid compression machine', 'jet-stirred reactor', 'jet stirred reactor', 'plug flow reactor', 'flow reactor', 'counterflow', 'coflow', 'bunsen', 'flat flame', 'spherically expanding', 'constant-volume', 'constant volume', 'combustion chamber', 'channel propagation', 'tube propagation', 'results', 'discussion', 'results and discussion']
    return any((k in title_lower for k in keep_keywords))

def clean_xml_text(xml_text):
    soup = BeautifulSoup(xml_text, 'lxml')
    kept_parts = []
    doi = extract_doi_from_xml(xml_text)
    if doi:
        add_part(kept_parts, 'DOI', doi)
    title = get_first_text(soup, ['dc:title', 'ce:title', 'title'])
    if title:
        add_part(kept_parts, 'TITLE', title)
    abstract_texts = get_all_texts(soup, ['dc:description', 'ce:abstract', 'abstract'])
    for abs_txt in abstract_texts:
        add_part(kept_parts, 'ABSTRACT', abs_txt)
    keyword_texts = get_all_texts(soup, ['dcterms:subject', 'ce:keyword', 'keyword'])
    if keyword_texts:
        add_part(kept_parts, 'KEYWORDS', '; '.join(keyword_texts))
    remove_tags = ['coredata', 'objects', 'scopus-id', 'scopus-eid', 'xocs:meta', 'xocs:item-toc', 'xocs:references', 'xocs:ref-info', 'xocs:funding-list', 'xocs:funding-text', 'xocs:attachment-metadata-doc', 'xocs:article-sharing-framework', 'xocs:self-archiving', 'xocs:copyright-info', 'xocs:open-access', 'xocs:open-research', 'ce:bibliography', 'ce:acknowledgment', 'ce:acknowledgements', 'ce:conflict-of-interest']
    for tag_name in remove_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for sec in soup.find_all(['ce:section', 'section']):
        title = get_section_title(sec).lower()
        if should_drop_section(title):
            sec.decompose()
    figure_tags = ['ce:figure', 'figure']
    for tag_name in figure_tags:
        for fig in soup.find_all(tag_name):
            fig_text = fig.get_text(' ', strip=True)
            fig_text = clean_text_basic(fig_text)
            if fig_text:
                add_part(kept_parts, 'FIGURE', fig_text)
    table_tags = ['ce:table', 'table', 'cals:table', 'tb:table']
    for tag_name in table_tags:
        for tb in soup.find_all(tag_name):
            tb_text = tb.get_text(' ', strip=True)
            tb_text = clean_text_basic(tb_text)
            if tb_text:
                add_part(kept_parts, 'TABLE', tb_text)
    for sec in soup.find_all(['ce:section', 'section']):
        title = get_section_title(sec)
        title_lower = title.lower()
        if should_drop_section(title_lower):
            continue
        if should_keep_section(title_lower):
            sec_text = sec.get_text(' ', strip=True)
            sec_text = clean_text_basic(sec_text)
            if sec_text:
                add_part(kept_parts, f'SECTION - {title}', sec_text)
    if not kept_parts:
        text = soup.get_text('\n')
        text = clean_text_basic(text)
        return text
    final_text = '\n\n'.join(kept_parts)
    final_text = re.sub('https?://\\S+', ' ', final_text)
    final_text = re.sub('\\n\\s*\\n+', '\n\n', final_text)
    final_text = re.sub('[ \\t]+', ' ', final_text)
    return final_text.strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Local XML file or directory')
    parser.add_argument('--output', type=Path, required=True, help='Directory for selected text')
    args = parser.parse_args()
    files = sorted(args.input.glob('*.xml')) if args.input.is_dir() else [args.input]
    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        selected = clean_xml_text(path.read_text(encoding='utf-8'))
        (args.output / (path.stem + '.txt')).write_text(selected, encoding='utf-8')
if __name__ == '__main__':
    main()
