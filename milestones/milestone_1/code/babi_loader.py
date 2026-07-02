# milestone 1 — bAbI loader
# purpose: load bAbI tasks 1-3 from raw text files, explore structure
# thursday 7 — using local text files (no HuggingFace dependency)

import os

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data'
)

def parse_babi_file(filepath):
    """Parse a bAbI task file into a list of problems.
    
    Each problem is a dict with:
      - 'story': list of (sentence_id, text) tuples
      - 'question': the question text
      - 'answer': the correct answer
      - 'supporting_ids': list of sentence IDs that are supporting facts
    """
    problems = []
    current_story = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # split off the line ID
            parts = line.split(' ', 1)
            line_id = int(parts[0])
            text = parts[1]
            
            # new story starts when line_id resets to 1
            if line_id == 1:
                current_story = []
            
            # question line contains a tab
            if '\t' in text:
                question, answer, supporting = text.split('\t')
                supporting_ids = [int(x) for x in supporting.strip().split()]
                
                # get supporting sentences
                supporting_sentences = [
                    s for sid, s in current_story
                    if sid in supporting_ids
                ]
                
                problems.append({
                    'story': [s for _, s in current_story],
                    'question': question.strip(),
                    'answer': answer.strip(),
                    'supporting_ids': supporting_ids,
                    'supporting_sentences': supporting_sentences
                })
            else:
                current_story.append((line_id, text))
    
    return problems

# load all three tasks
print("Loading bAbI tasks from local files...")

task1 = parse_babi_file(os.path.join(DATA_DIR, 'qa1_train.txt'))
task2 = parse_babi_file(os.path.join(DATA_DIR, 'qa2_train.txt'))
task3 = parse_babi_file(os.path.join(DATA_DIR, 'qa3_train.txt'))

print(f"Task 1: {len(task1)} problems")
print(f"Task 2: {len(task2)} problems")
print(f"Task 3: {len(task3)} problems")

# print structure of first problem from each task
for task_name, task in [("Task 1", task1), ("Task 2", task2), ("Task 3", task3)]:
    print(f"\n{'='*50}")
    print(f"{task_name} — problem 1")
    print('='*50)
    p = task[0]
    print(f"Story:                 {p['story']}")
    print(f"Question:              {p['question']}")
    print(f"Answer:                {p['answer']}")
    print(f"Supporting IDs:        {p['supporting_ids']}")
    print(f"Supporting sentences:  {p['supporting_sentences']}")

print("\nbAbI loader confirmed. Structure understood.")
print("Next step: extract activations at supporting sentence positions.")