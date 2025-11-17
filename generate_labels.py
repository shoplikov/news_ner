import pandas as pd
import openai
import json
import time
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from difflib import SequenceMatcher
import re
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set it directly

class LabeledChunks(BaseModel):
    """Structured model for labeled text chunks grouped by label type."""
    PER: List[str] = Field(
        default_factory=list,
        description="Адамдардың аты-жөндері (мәтіннен дәл көшірме)"
    )
    ORG: List[str] = Field(
        default_factory=list,
        description="Ұйымдар мен мекемелер атаулары (мәтіннен дәл көшірме)"
    )
    LOC: List[str] = Field(
        default_factory=list,
        description="Орналасқан жерлер мен мекенжайлар (мәтіннен дәл көшірме)"
    )
    CRIME_TYPE: List[str] = Field(
        default_factory=list,
        description="Қылмыс түрлері (мәтіннен дәл көшірме)"
    )
    LAW: List[str] = Field(
        default_factory=list,
        description="Заң баптары мен кодекстер (мәтіннен дәл көшірме)"
    )
    
    @field_validator('PER', 'ORG', 'LOC', 'CRIME_TYPE', 'LAW')
    @classmethod
    def validate_unique_non_empty(cls, v):
        # Remove empty strings and duplicates while preserving order
        seen = set()
        result = []
        for item in v:
            if item and item.strip() and item not in seen:
                seen.add(item)
                result.append(item)
        return result

def create_ner_prompt(text: str) -> str:
    """Create an enhanced prompt for GPT to extract text chunks grouped by label type."""
    prompt = f"""Төмендегі қазақ тіліндегі қылмыс жаңалығынан Named Entity Recognition (NER) жасау керек.

МАҢЫЗДЫ НҰСҚАУЛАР:
1. ТЕК мәтінде НАҚТЫ бар text chunks-ты белгіле
2. Текст chunks-ты ДӘЛМЕ-ДӘЛ көшір (бірде-бір әріп, тыныс белгі өзгертпе)
3. Әрбір label типі бойынша тізім жаса
4. Егер белгілі бір label типі мәтінде жоқ болса, бос тізім қалдыр []
5. Қайталанған chunks-ты қоспа (тек бір рет қос)

ENTITY ТИПТЕРІ МЕН АНЫҚТАМАЛАРЫ:

PER (Person/Адам):
- Нақты адамдардың толық немесе қысқартылған аты-жөндері
- Лауазым атаулары ЕМЕС (мысалы: "полиция қызметкері" - бұл PER емес)
- Тек АДАМНЫҢ АТЫ-ЖӨНІН белгіле
- Мысалдар: "Ержан Оспанов", "Айгүл Нұрсейітова", "А. Нұрсұлтан"

ORG (Organization/Ұйым):
- Мемлекеттік органдар мен ведомстволар
- Компаниялар, ұйымдар, топтар
- БАҚ атаулары
- Мысалдар: "ІІД", "Tengrinews.kz", "Қазпошта", "Қазақстан Республикасының ІІМ", "Алматы полиция департаменті"

LOC (Location/Орналасқан жер):
- Нақты географиялық орындар: қалалар, аудандар, облыстар, елдер
- Көше атаулары, ғимараттар, мекенжайлар
- Бекеттер, шекара пункттері, аймақтар
- Мысалдар: "Алматы қаласы", "Сарыарқа ауданы", "Достық бекеті", "Абай көшесі, 15", "Қазақстан"

CRIME_TYPE (Қылмыс түрі):
- Қылмыстың нақты заңдық атауы немесе сипаттамасы
- Жалпы сөздер емес, нақты қылмыс типтері ғана
- Мысалдар: "контрабанда", "ұрлық", "алаяқтық", "есірткі заттарын заңсыз айналымға енгізу", "бұзақылық"

LAW (Заң/Кодекс):
- Заңнамалық актілердің атаулары
- Кодекстердің баптары мен бөліктері (нақты сандармен)
- Нақты заң сілтемелері
- Мысалдар: "Қылмыстық кодекстің 255-бабы", "ҚР ҚК 234-бабының 3-бөлігі", "ҚК 188-бабы"

ТЕКСТ:
{text}

ТАПСЫРМА: Жоғарыдағы текстен барлық entities-ті тап және label бойынша топта. Мәтіннен дәл сол түрінде көшір."""
    
    return prompt

def find_all_occurrences(text: str, chunk: str) -> List[Tuple[int, int]]:
    """Find all occurrences of a chunk in the text with their positions."""
    positions = []
    start = 0
    while True:
        pos = text.find(chunk, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(chunk)))
        start = pos + 1
    return positions

def fuzzy_find_chunk(text: str, chunk: str, threshold: float = 0.9) -> Optional[Tuple[int, int]]:
    """Try to find a chunk using fuzzy matching if exact match fails."""
    # Normalize whitespace
    chunk_normalized = ' '.join(chunk.split())
    
    # Try to find with normalized whitespace
    normalized_text = ' '.join(text.split())
    if chunk_normalized in normalized_text:
        # Find in original text by matching word boundaries
        words = chunk_normalized.split()
        pattern = r'\s*'.join(re.escape(word) for word in words)
        match = re.search(pattern, text)
        if match:
            return (match.start(), match.end())
    
    # Try fuzzy matching for close matches (handles minor typos)
    chunk_len = len(chunk)
    best_ratio = 0
    best_pos = None
    
    for i in range(len(text) - chunk_len + 1):
        substring = text[i:i + chunk_len]
        ratio = SequenceMatcher(None, chunk, substring).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_pos = (i, i + chunk_len)
    
    return best_pos

def validate_chunks(text: str, labeled_chunks: LabeledChunks) -> Dict[str, List[str]]:
    """Validate that chunks actually exist in the text."""
    validated = {
        'PER': [],
        'ORG': [],
        'LOC': [],
        'CRIME_TYPE': [],
        'LAW': []
    }
    
    for label in ['PER', 'ORG', 'LOC', 'CRIME_TYPE', 'LAW']:
        chunks = getattr(labeled_chunks, label, [])
        for chunk in chunks:
            # Try exact match first
            if chunk in text:
                validated[label].append(chunk)
            else:
                # Try fuzzy matching
                result = fuzzy_find_chunk(text, chunk)
                if result:
                    # Extract the actual text from found position
                    found_text = text[result[0]:result[1]]
                    validated[label].append(found_text)
                    print(f"Warning: Fuzzy matched '{chunk}' -> '{found_text}' for {label}")
                else:
                    print(f"Warning: Chunk '{chunk}' not found in text for {label}, skipping")
    
    return validated

def get_ner_labels(text: str, model: str = "gpt-4.1") -> Dict:
    """Call OpenAI API to get NER labels as grouped chunks using structured output."""
    try:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": """Сен Named Entity Recognition (NER) жүйесісің. 
Қазақ тіліндегі қылмыс жаңалықтарынан entities табасың.

ҚАҒИДАЛАР:
- ТЕК мәтінде бар text chunks-ты белгіле
- Chunks-ты ДӘЛМЕ-ДӘЛ көшір (өзгертпе)
- Әрбір label бойынша тізімге жаз
- Күмәнді жағдайларда chunks қоспа
- Егер label үшін chunks жоқ болса, бос тізім қалдыр
- Қайталанған chunks-ты қоспа"""
                },
                {"role": "user", "content": create_ner_prompt(text)}
            ],
            temperature=0.1,
            response_format=LabeledChunks
        )
        
        result = response.choices[0].message.parsed
        
        # Validate chunks
        if result:
            validated_chunks = validate_chunks(text, result)
            return {"labeled_chunks": validated_chunks}
        
        return {"labeled_chunks": {'PER': [], 'ORG': [], 'LOC': [], 'CRIME_TYPE': [], 'LAW': []}}
    
    except Exception as e:
        print(f"Error processing text: {e}")
        return {"labeled_chunks": {'PER': [], 'ORG': [], 'LOC': [], 'CRIME_TYPE': [], 'LAW': []}}

def save_checkpoint(checkpoint_file: str, results: List[Dict], last_processed_idx: int):
    """Save checkpoint with current results and progress."""
    checkpoint_data = {
        'last_processed_idx': last_processed_idx,
        'results': results
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint saved at row {last_processed_idx + 1}")

def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict], int]:
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        print(f"Resuming from row {checkpoint_data['last_processed_idx'] + 2}")
        return checkpoint_data['results'], checkpoint_data['last_processed_idx']
    return [], -1

def convert_chunks_to_conll(text: str, labeled_chunks: Dict[str, List[str]]) -> List[tuple]:
    """
    Convert text and labeled chunks to CoNLL format (token, label).
    
    Args:
        text: Original text
        labeled_chunks: Dictionary with label types as keys and lists of text chunks as values
    
    Returns:
        List of (token, label) tuples in BIO format
    """
    # Tokenize text
    tokens = text.split()
    
    # Create character to token mapping
    char_to_token = {}
    char_pos = 0
    for idx, token in enumerate(tokens):
        start = text.find(token, char_pos)
        if start != -1:
            for i in range(start, start + len(token)):
                char_to_token[i] = idx
            char_pos = start + len(token)
    
    # Initialize all tokens as 'O' (Outside)
    labels = ['O'] * len(tokens)
    
    # Process each label type and its chunks
    for label, chunks in labeled_chunks.items():
        for chunk in chunks:
            # Find all occurrences of this chunk in the text
            occurrences = find_all_occurrences(text, chunk)
            
            for start_char, end_char in occurrences:
                # Find which tokens this chunk spans
                token_indices = set()
                for char_idx in range(start_char, end_char):
                    if char_idx in char_to_token:
                        token_indices.add(char_to_token[char_idx])
                
                token_indices = sorted(token_indices)
                
                # Apply BIO tagging (only if not already labeled to avoid conflicts)
                if token_indices and labels[token_indices[0]] == 'O':
                    labels[token_indices[0]] = f'B-{label}'
                    for idx in token_indices[1:]:
                        if labels[idx] == 'O':
                            labels[idx] = f'I-{label}'
    
    token_labels = list(zip(tokens, labels))
    return token_labels

def process_csv(input_file: str, output_format: str = 'both', checkpoint_interval: int = 50):
    """
    Process CSV and generate NER dataset with checkpointing.
    
    Args:
        input_file: Path to input CSV with 'id' and 'text' columns
        output_format: 'json', 'conll', or 'both'
        checkpoint_interval: Save checkpoint every N rows (default: 50)
    """
    # Read input CSV
    df = pd.read_csv(input_file)
    
    if 'text' not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    
    if 'id' not in df.columns:
        raise ValueError("CSV must have an 'id' column")
    
    # Setup checkpoint file
    checkpoint_file = input_file.replace('.csv', '_checkpoint.json')
    
    # Load checkpoint if exists
    results, last_processed_idx = load_checkpoint(checkpoint_file)
    
    # Start from next row after checkpoint
    start_idx = last_processed_idx + 1
    
    for idx, row in df.iterrows():
        # Skip already processed rows
        if idx <= last_processed_idx:
            continue
        
        text = row['text']
        row_id = row['id']
        print(f"Processing {idx + 1}/{len(df)} (ID: {row_id})...")
        
        # Get NER labels from GPT (returns labeled chunks)
        ner_result = get_ner_labels(text)
        
        results.append({
            'id': row_id,
            'text': text,
            'labeled_chunks': ner_result.get('labeled_chunks', {
                'PER': [], 'ORG': [], 'LOC': [], 'CRIME_TYPE': [], 'LAW': []
            })
        })
        
        # Save checkpoint every N rows
        if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == len(df):
            save_checkpoint(checkpoint_file, results, idx)
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save in JSON format
    if output_format in ['json', 'both']:
        output_json = input_file.replace('.csv', '_ner.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON format to: {output_json}")
    
    # Save in CoNLL format
    if output_format in ['conll', 'both']:
        output_conll = input_file.replace('.csv', '_ner.conll')
        with open(output_conll, 'w', encoding='utf-8') as f:
            for item in results:
                row_id = item['id']
                text = item['text']
                labeled_chunks = item['labeled_chunks']
                token_labels = convert_chunks_to_conll(text, labeled_chunks)
                
                # Write document ID as a comment
                f.write(f"# id={row_id}\n")
                for token, label in token_labels:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Blank line between documents
        print(f"Saved CoNLL format to: {output_conll}")
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Checkpoint file removed: {checkpoint_file}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    # openai.api_key = "your-api-key-here"
    
    # Process the CSV file (must have 'id' and 'text' columns)
    input_csv = "data/text_articles_crime.csv"  # Your input file
    
    # Generate both JSON and CoNLL formats with checkpointing every 50 rows
    results = process_csv(input_csv, output_format='both', checkpoint_interval=50)
    
    print(f"\nProcessed {len(results)} texts successfully!")
    print("\nOutput formats:")
    print("1. JSON format: Contains id, text and labeled chunks grouped by entity type")
    print("   - Structure: {id: '...', text: '...', labeled_chunks: {PER: [...], ORG: [...], ...}}")
    print("2. CoNLL format: Token-per-line with BIO tags, ready for training")
    print("   - Format: TOKEN\\tLABEL (e.g., 'Алматы\\tB-LOC')")
    print("   - Each document starts with '# id=...' comment line")
    print("\nCheckpointing:")
    print("- Progress is saved every 50 rows to *_checkpoint.json")
    print("- If interrupted, simply run again to resume from last checkpoint")
    print("- Checkpoint file is automatically deleted after successful completion")