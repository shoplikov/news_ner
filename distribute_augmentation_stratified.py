"""
Script to distribute augmentation dataset with stratification by CRIME_TYPE.
"""
import random
from pathlib import Path
from collections import Counter, defaultdict
import math


def read_iob2_sentences(file_path):
    """Read IOB2 file and return list of sentences (each sentence is a list of lines)"""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip():  # Non-empty line
                current_sentence.append(line)
            else:  # Empty line marks sentence boundary
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        
        # Add last sentence if file doesn't end with empty line
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def extract_crime_types(sentence):
    """Extract all CRIME_TYPE entities from a sentence"""
    crime_types = set()
    
    for line in sentence:
        parts = line.split()
        if len(parts) >= 2:
            token, label = parts[0], parts[1]
            # Check if it's a CRIME_TYPE label (B-CRIME_TYPE or I-CRIME_TYPE)
            if 'CRIME_TYPE' in label:
                crime_types.add('HAS_CRIME_TYPE')
    
    # Return a representative label for stratification
    if crime_types:
        return 'HAS_CRIME_TYPE'
    else:
        return 'NO_CRIME_TYPE'


def extract_specific_crime_types(sentence):
    """Extract specific crime type tokens from a sentence for statistics"""
    crime_tokens = []
    current_crime = []
    
    for line in sentence:
        parts = line.split()
        if len(parts) >= 2:
            token, label = parts[0], parts[1]
            
            if label == 'B-CRIME_TYPE':
                # Start of new crime type
                if current_crime:
                    crime_tokens.append(' '.join(current_crime))
                current_crime = [token]
            elif label == 'I-CRIME_TYPE':
                # Continuation of crime type
                current_crime.append(token)
            else:
                # Not a crime type, save current if exists
                if current_crime:
                    crime_tokens.append(' '.join(current_crime))
                    current_crime = []
    
    # Don't forget last crime type
    if current_crime:
        crime_tokens.append(' '.join(current_crime))
    
    return crime_tokens


def stratified_split(sentences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split sentences with stratification by crime type presence.
    
    Args:
        sentences: List of sentences
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_sentences, val_sentences, test_sentences)
    """
    random.seed(random_seed)
    
    # Group sentences by their crime type category
    strata = defaultdict(list)
    for sentence in sentences:
        category = extract_crime_types(sentence)
        strata[category].append(sentence)
    
    print(f"\n📊 Stratification Analysis:")
    for category, sents in strata.items():
        print(f"   {category}: {len(sents)} sentences ({len(sents)/len(sentences)*100:.1f}%)")
    
    # Split each stratum
    train_sentences = []
    val_sentences = []
    test_sentences = []
    
    for category, sents in strata.items():
        # Shuffle within stratum
        random.shuffle(sents)
        
        # Calculate split points
        n = len(sents)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_sentences.extend(sents[:train_end])
        val_sentences.extend(sents[train_end:val_end])
        test_sentences.extend(sents[val_end:])
    
    # Shuffle the combined sets (optional, but good practice)
    random.shuffle(train_sentences)
    random.shuffle(val_sentences)
    random.shuffle(test_sentences)
    
    return train_sentences, val_sentences, test_sentences


def append_sentences_to_file(sentences, file_path):
    """Append sentences to an existing IOB2 file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        # Ensure there's a blank line before appending
        f.write('\n')
        
        for sentence in sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')  # Blank line after each sentence


def analyze_crime_type_distribution(sentences, set_name):
    """Analyze and display crime type distribution in a set"""
    crime_type_counter = Counter()
    sentences_with_crime = 0
    
    for sentence in sentences:
        crime_types = extract_specific_crime_types(sentence)
        if crime_types:
            sentences_with_crime += 1
            for crime in crime_types:
                crime_type_counter[crime] += 1
    
    print(f"\n   {set_name}:")
    print(f"      Sentences with CRIME_TYPE: {sentences_with_crime}/{len(sentences)} ({sentences_with_crime/len(sentences)*100:.1f}%)")
    
    if crime_type_counter:
        print(f"      Top 10 crime types:")
        for crime, count in crime_type_counter.most_common(10):
            print(f"         • {crime}: {count}")


def distribute_augmentation_stratified(augmentation_path, train_path, val_path, test_path, 
                                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                                      random_seed=42):
    """
    Distribute augmentation dataset with stratification by CRIME_TYPE.
    """
    # Check if augmentation file exists
    if not Path(augmentation_path).exists():
        print(f"Error: Augmentation file not found at {augmentation_path}")
        return
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios don't sum to 1.0 (sum={train_ratio + val_ratio + test_ratio})")
    
    print("="*80)
    print("STRATIFIED DISTRIBUTION BY CRIME_TYPE")
    print("="*80)
    
    # Read augmentation sentences
    print(f"\n📖 Reading augmentation data from {augmentation_path}...")
    aug_sentences = read_iob2_sentences(augmentation_path)
    print(f"Found {len(aug_sentences)} sentences in augmentation data")
    
    if len(aug_sentences) == 0:
        print("Error: No sentences found in augmentation file")
        return
    
    # Perform stratified split
    print(f"\n🔀 Performing stratified split with seed={random_seed}...")
    train_sentences, val_sentences, test_sentences = stratified_split(
        aug_sentences,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Display distribution
    total = len(aug_sentences)
    print(f"\n📊 Split Distribution:")
    print(f"   Train: {len(train_sentences)} sentences ({len(train_sentences)/total*100:.1f}%)")
    print(f"   Val:   {len(val_sentences)} sentences ({len(val_sentences)/total*100:.1f}%)")
    print(f"   Test:  {len(test_sentences)} sentences ({len(test_sentences)/total*100:.1f}%)")
    
    # Analyze crime type distribution in each set
    print(f"\n📈 Crime Type Distribution Analysis:")
    analyze_crime_type_distribution(train_sentences, "Train")
    analyze_crime_type_distribution(val_sentences, "Validation")
    analyze_crime_type_distribution(test_sentences, "Test")
    
    # Append to existing files
    print(f"\n💾 Appending to existing files...")
    
    append_sentences_to_file(train_sentences, train_path)
    print(f"   ✅ Added {len(train_sentences)} sentences to {train_path}")
    
    append_sentences_to_file(val_sentences, val_path)
    print(f"   ✅ Added {len(val_sentences)} sentences to {val_path}")
    
    append_sentences_to_file(test_sentences, test_path)
    print(f"   ✅ Added {len(test_sentences)} sentences to {test_path}")
    
    print("\n" + "="*80)
    print("✅ STRATIFIED DISTRIBUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n💡 The crime type distribution is maintained across all sets.")


if __name__ == "__main__":
    # Configuration
    augmentation_path = 'data/clean/IOB2_crime.txt'
    train_path = 'data/clean/IOB2_train.txt'
    val_path = 'data/clean/IOB2_valid.txt'
    test_path = 'data/clean/IOB2_test.txt'
    
    # Distribute with stratification
    distribute_augmentation_stratified(
        augmentation_path=augmentation_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        train_ratio=0.7,      # 70% to training
        val_ratio=0.15,       # 15% to validation
        test_ratio=0.15,      # 15% to test
        random_seed=42        # For reproducibility
    )

