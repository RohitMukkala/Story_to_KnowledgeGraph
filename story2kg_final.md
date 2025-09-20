```python
import sys
print(sys.executable)
print(sys.version)

```

    C:\Users\__msi__\anaconda3\envs\story2kg_env\python.exe
    3.10.18 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:08:55) [MSC v.1929 64 bit (AMD64)]
    

# Stage 1


```python
import spacy
import re

# Load the model once at the start of your script
nlp = spacy.load("en_core_web_lg")

story_text = """
Hare was hopping through the forest.  
He saw a tortoise slowly walking along the path.
he greeted him fast.
he is handsome.
"""

def preprocess_story(text, nlp_model):
    # --- 1. CLEANING STEP ---
    # Normalize whitespace (replace multiple spaces/newlines with a single space)
    clean_text = re.sub(r'\s+', ' ', text).strip()
    # You could add other cleaning rules here, e.g., for special characters

    # --- 2. LINGUISTIC ANALYSIS STEP ---
    doc = nlp_model(clean_text)
    
    sentences = list(doc.sents)
    words = [[token.text for token in sent if not token.is_space] for sent in doc.sents]
    pos_tags = [[(token.text, token.pos_) for token in sent if not token.is_space] for sent in doc.sents]
    dependencies = [[(token.text, token.dep_, token.head.text) for token in sent if not token.is_space] for sent in doc.sents]
    
    return {
        "sentences": sentences,
        "words": words,
        "pos_tags": pos_tags,
        "dependencies": dependencies
    }

# Example usage: pass the loaded nlp object into the function
output = preprocess_story(story_text, nlp)
print(output)
```

    C:\Users\__msi__\anaconda3\envs\story2kg_env\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

    {'sentences': [Hare was hopping through the forest., He saw a tortoise slowly walking along the path., he greeted him fast., he is handsome.], 'words': [['Hare', 'was', 'hopping', 'through', 'the', 'forest', '.'], ['He', 'saw', 'a', 'tortoise', 'slowly', 'walking', 'along', 'the', 'path', '.'], ['he', 'greeted', 'him', 'fast', '.'], ['he', 'is', 'handsome', '.']], 'pos_tags': [[('Hare', 'PROPN'), ('was', 'AUX'), ('hopping', 'VERB'), ('through', 'ADP'), ('the', 'DET'), ('forest', 'NOUN'), ('.', 'PUNCT')], [('He', 'PRON'), ('saw', 'VERB'), ('a', 'DET'), ('tortoise', 'NOUN'), ('slowly', 'ADV'), ('walking', 'VERB'), ('along', 'ADP'), ('the', 'DET'), ('path', 'NOUN'), ('.', 'PUNCT')], [('he', 'PRON'), ('greeted', 'VERB'), ('him', 'PRON'), ('fast', 'ADV'), ('.', 'PUNCT')], [('he', 'PRON'), ('is', 'AUX'), ('handsome', 'ADJ'), ('.', 'PUNCT')]], 'dependencies': [[('Hare', 'nsubj', 'hopping'), ('was', 'aux', 'hopping'), ('hopping', 'ROOT', 'hopping'), ('through', 'prep', 'hopping'), ('the', 'det', 'forest'), ('forest', 'pobj', 'through'), ('.', 'punct', 'hopping')], [('He', 'nsubj', 'saw'), ('saw', 'ROOT', 'saw'), ('a', 'det', 'tortoise'), ('tortoise', 'nsubj', 'walking'), ('slowly', 'advmod', 'walking'), ('walking', 'ccomp', 'saw'), ('along', 'prep', 'walking'), ('the', 'det', 'path'), ('path', 'pobj', 'along'), ('.', 'punct', 'saw')], [('he', 'nsubj', 'greeted'), ('greeted', 'ROOT', 'greeted'), ('him', 'dobj', 'greeted'), ('fast', 'advmod', 'greeted'), ('.', 'punct', 'greeted')], [('he', 'nsubj', 'is'), ('is', 'ROOT', 'is'), ('handsome', 'acomp', 'is'), ('.', 'punct', 'is')]]}
    

# Stage 2


```python
import spacy
import re
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

# ==============================================================================
# LOAD MODELS ONCE
# ==============================================================================
nlp = spacy.load("en_core_web_lg")

# ==============================================================================
# COREFERENCE RESOLUTION (robust replacement)
# ==============================================================================
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

def resolve_coreferences(text, model_path):
    predictor = Predictor.from_path(model_path)
    prediction = predictor.predict(document=text)

    tokens = prediction["document"]
    clusters = prediction["clusters"]

    # Convert tokens back to text positions
    resolved_text = tokens[:]  # copy

    for cluster in clusters:
        main_mention = " ".join(tokens[cluster[0][0]: cluster[0][1] + 1])
        for mention in cluster[1:]:
            start, end = mention
            # Replace the pronoun directly
            resolved_text[start] = main_mention
            for i in range(start + 1, end + 1):
                resolved_text[i] = ""

    return " ".join([t for t in resolved_text if t != ""])


    # tokens = prediction["document"]
    # clusters = prediction["clusters"]

    # # Build replacement map: (start, end) → main_mention
    # replacements = {}
    # for cluster in clusters:
    #     main_mention = " ".join(tokens[cluster[0][0] : cluster[0][1] + 1])
    #     for mention in cluster[1:]:
    #         replacements[(mention[0], mention[1])] = main_mention

    # resolved_tokens = []
    # skip_until = -1
    # for i, token in enumerate(tokens):
    #     if i < skip_until:
    #         continue
    #     replaced = False
    #     for (start, end), main in replacements.items():
    #         if i == start:
    #             resolved_tokens.append(main)
    #             skip_until = end + 1
    #             replaced = True
    #             break
    #     if not replaced:
    #         resolved_tokens.append(token)

    # return " ".join(resolved_tokens)


# ==============================================================================
# STAGE 1: Preprocess & Analyze Text
# ==============================================================================
def preprocess_story(text, nlp_model):
    clean_text = re.sub(r"\s+", " ", text).strip()
    doc = nlp_model(clean_text)
    return {
        "sentences": list(doc.sents),
        "dependencies": [
            [(token.text, token.dep_, token.head.text) for token in sent if not token.is_space]
            for sent in doc.sents
        ]
    }

# ==============================================================================
# STAGE 2: Scene Annotation (robust subject tracking)
# ==============================================================================
def get_sentence_subject(sentence_dependencies):
    for token, dep, head in sentence_dependencies:
        if dep == "nsubj":
            return token
    return None

def annotate_scenes(stage1_output):
    annotated_sentences = []
    last_subject = None
    sentences_text = [sent.text for sent in stage1_output["sentences"]]
    dependencies = stage1_output["dependencies"]

    for i, sent_text in enumerate(sentences_text):
        current_subject = get_sentence_subject(dependencies[i])

        if i == 0 or (current_subject and current_subject != last_subject):
            tag = "b scene"
        else:
            tag = "i scene"

        annotated_sentences.append(f"{tag} [{sent_text}]")

        if current_subject:
            last_subject = current_subject

    return annotated_sentences

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # story_text = """
    # Hare was hopping through the forest.
    # He saw a tortoise slowly walking along the path.
    # Then he greeted the tortoise warmly.
    # """

    coref_model_path = "C:/Users/__msi__/coref-spanbert-large"

    print("--- Running Coreference Resolution ---")
    resolved_text = resolve_coreferences(story_text, coref_model_path)
    print("Resolved Text:\n", resolved_text)
    print("-" * 40)

    print("--- Running Stage 1 ---")
    stage1_data = preprocess_story(resolved_text, nlp)
    print("Stage 1 Sentences:\n", [s.text for s in stage1_data["sentences"]])
    print("-" * 40)

    print("--- Running Stage 2 ---")
    stage2_output = annotate_scenes(stage1_data)
    print("Final Stage 2 Output:\n", stage2_output)

```

    --- Running Coreference Resolution ---
    

    error loading _jsonnet (this is expected on Windows), treating C:\Users\__msi__\coref-spanbert-large\config.json as plain json
    Some weights of BertModel were not initialized from the model checkpoint at C:/Users/__msi__/spanbert-large-cased and are newly initialized: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

    Resolved Text:
     
     Hare was hopping through the forest .  
     Hare saw a tortoise slowly walking along the path . 
     Hare greeted a tortoise fast . 
     a tortoise is handsome . 
    
    ----------------------------------------
    --- Running Stage 1 ---
    Stage 1 Sentences:
     ['Hare was hopping through the forest .', 'Hare saw a tortoise slowly walking along the path .', 'Hare greeted a tortoise fast .', 'a tortoise is handsome .']
    ----------------------------------------
    --- Running Stage 2 ---
    Final Stage 2 Output:
     ['b scene [Hare was hopping through the forest .]', 'i scene [Hare saw a tortoise slowly walking along the path .]', 'i scene [Hare greeted a tortoise fast .]', 'b scene [a tortoise is handsome .]']
    


```python
def convert_stage2_to_stage3(stage1_output, stage2_output):
    stage3_input = []
    
    for i, scene_str in enumerate(stage2_output, start=1):
        # Remove "b scene" / "i scene" wrappers
        clean_text = re.sub(r'^[bi]\s+scene\s+\[|\]$', '', scene_str).strip()

        # Grab sentence object + deps from Stage 1
        sent = stage1_output["sentences"][i-1]
        deps = stage1_output["dependencies"][i-1]

        tokens = [tok.text for tok in sent if not tok.is_space]
        pos_tags = [tok.pos_ for tok in sent if not tok.is_space]

        stage3_input.append({
            "scene_id": i,
            "text": clean_text,
            "tokens": tokens,
            "pos": pos_tags,
            "dependencies": deps
        })
    
    return stage3_input

# After running Stage 1 + Stage 2
stage3_ready_input = convert_stage2_to_stage3(stage1_data, stage2_output)

import json
print(json.dumps(stage3_ready_input, indent=2))


```

    [
      {
        "scene_id": 1,
        "text": "Hare was hopping through the forest .",
        "tokens": [
          "Hare",
          "was",
          "hopping",
          "through",
          "the",
          "forest",
          "."
        ],
        "pos": [
          "PROPN",
          "AUX",
          "VERB",
          "ADP",
          "DET",
          "NOUN",
          "PUNCT"
        ],
        "dependencies": [
          [
            "Hare",
            "nsubj",
            "hopping"
          ],
          [
            "was",
            "aux",
            "hopping"
          ],
          [
            "hopping",
            "ROOT",
            "hopping"
          ],
          [
            "through",
            "prep",
            "hopping"
          ],
          [
            "the",
            "det",
            "forest"
          ],
          [
            "forest",
            "pobj",
            "through"
          ],
          [
            ".",
            "punct",
            "hopping"
          ]
        ]
      },
      {
        "scene_id": 2,
        "text": "Hare saw a tortoise slowly walking along the path .",
        "tokens": [
          "Hare",
          "saw",
          "a",
          "tortoise",
          "slowly",
          "walking",
          "along",
          "the",
          "path",
          "."
        ],
        "pos": [
          "PROPN",
          "VERB",
          "DET",
          "NOUN",
          "ADV",
          "VERB",
          "ADP",
          "DET",
          "NOUN",
          "PUNCT"
        ],
        "dependencies": [
          [
            "Hare",
            "nsubj",
            "saw"
          ],
          [
            "saw",
            "ROOT",
            "saw"
          ],
          [
            "a",
            "det",
            "tortoise"
          ],
          [
            "tortoise",
            "nsubj",
            "walking"
          ],
          [
            "slowly",
            "advmod",
            "walking"
          ],
          [
            "walking",
            "ccomp",
            "saw"
          ],
          [
            "along",
            "prep",
            "walking"
          ],
          [
            "the",
            "det",
            "path"
          ],
          [
            "path",
            "pobj",
            "along"
          ],
          [
            ".",
            "punct",
            "saw"
          ]
        ]
      },
      {
        "scene_id": 3,
        "text": "Hare greeted a tortoise fast .",
        "tokens": [
          "Hare",
          "greeted",
          "a",
          "tortoise",
          "fast",
          "."
        ],
        "pos": [
          "PROPN",
          "VERB",
          "DET",
          "NOUN",
          "ADV",
          "PUNCT"
        ],
        "dependencies": [
          [
            "Hare",
            "nsubj",
            "greeted"
          ],
          [
            "greeted",
            "ROOT",
            "greeted"
          ],
          [
            "a",
            "det",
            "tortoise"
          ],
          [
            "tortoise",
            "dobj",
            "greeted"
          ],
          [
            "fast",
            "advmod",
            "greeted"
          ],
          [
            ".",
            "punct",
            "greeted"
          ]
        ]
      },
      {
        "scene_id": 4,
        "text": "a tortoise is handsome .",
        "tokens": [
          "a",
          "tortoise",
          "is",
          "handsome",
          "."
        ],
        "pos": [
          "DET",
          "NOUN",
          "AUX",
          "ADJ",
          "PUNCT"
        ],
        "dependencies": [
          [
            "a",
            "det",
            "tortoise"
          ],
          [
            "tortoise",
            "nsubj",
            "is"
          ],
          [
            "is",
            "ROOT",
            "is"
          ],
          [
            "handsome",
            "acomp",
            "is"
          ],
          [
            ".",
            "punct",
            "is"
          ]
        ]
      }
    ]
    

# Stage 3


```python
import re
import spacy
import json
from transformers import pipeline
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import allennlp_models.structured_prediction # Added for SRL

# ==============================================================================
# 1. LOAD MODELS ONCE
# ==============================================================================
print("Loading models...")
# --- Core Models ---
coref_predictor = Predictor.from_path("C:/Users/__msi__/coref-spanbert-large")
ner_pipeline = pipeline("ner", model="C:/Users/__msi__/ner-model-large", aggregation_strategy="simple")
emotion_classifier = pipeline("text-classification", model="C:/Users/__msi__/emotion-model-local", top_k=1)
nlp = spacy.load("en_core_web_sm")

# --- NEW: Event Extraction Model (SRL) ---
srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)
print("Models loaded.")

```

    Loading models...
    

    error loading _jsonnet (this is expected on Windows), treating C:\Users\__msi__\coref-spanbert-large\config.json as plain json
    error loading _jsonnet (this is expected on Windows), treating C:\Users\__msi__\AppData\Local\Temp\tmpg_qnvosh\config.json as plain json
    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    

    Models loaded.
    


```python

# ==============================================================================
# 2. EVENT SCHEMA AND MAPPING
# ==============================================================================
general_event_schema = {
    "Conflict": {
        "triggers": ["fought", "argued", "attacked", "defended", "competed", "defeated"],
        "roles": {"ARG0": "Protagonist","ARG1": "Antagonist","ARGM-MNR": "Outcome"}
    },
    "Journey": {
        "triggers": ["traveled", "went", "journeyed", "arrived", "departed", "fled"],
        "roles": {"ARG0": "Traveler","ARGM-LOC": "Origin","ARGM-DIR": "Destination"}
    },
    "Transaction": {
        "triggers": ["gave", "received", "bought", "sold", "traded", "stole"],
        "roles": {"ARG0": "Giver","ARG1": "Item","ARG2": "Recipient"}
    },
    "Communication": {
        "triggers": ["said", "told", "laughed", "boasted", "asked", "yelled", "whispered"],
        "roles": {"ARG0": "Speaker","ARG1": "Message","ARG2": "Listener"}
    },
    "Perception": {
        "triggers": ["saw", "heard", "watched", "noticed", "observed", "sensed", "smelled"],
        "roles": {"ARG0": "Observer","ARG1": "Phenomenon"}
    },
        "Cognition": {
        "triggers": ["thought", "believed", "knew", "realized", "wondered", "decided", "forgot"],
        "roles": {"ARG0": "Cognizer","ARG1": "Content"}
    },
    "Creation": {
        "triggers": ["built", "made", "created", "wrote", "painted", "designed", "composed"],
        "roles": {"ARG0": "Creator","ARG1": "Creation"}
    },
    "Destruction": {
        "triggers": ["destroyed", "broke", "ruined", "shattered", "demolished", "tore"],
        "roles": {"ARG0": "Destroyer","ARG1": "Object"}
    },
    "Motion": {
        "triggers": ["moved", "ran", "walked", "flew", "swam", "hopped", "crawled"],
        "roles": {"ARG0": "Mover","ARGM-LOC": "Path"}
    },
    "Possession": {
        "triggers": ["had", "owned", "possessed", "held"],
        "roles": {"ARG0": "Owner","ARG1": "Possession"}
    },
    "Life_Event": {
        "triggers": ["born", "died", "married", "graduated", "became king", "crowned"],
        "roles": {"ARG0": "Person","ARGM-LOC": "Location"}
    },
    "Control": {
        "triggers": ["ruled", "controlled", "commanded", "led", "governed"],
        "roles": {"ARG0": "Controller","ARG1": "Domain"}
    },
    "Emotion_Expression": {
        "triggers": ["loved", "hated", "feared", "enjoyed", "cried", "smiled"],
        "roles": {"ARG0": "Experiencer","ARG1": "Stimulus"}
    },
     "Assistance": {
        "triggers": ["helped", "assisted", "saved", "rescued", "supported"],
        "roles": {"ARG0": "Helper","ARG1": "Recipient","ARG2": "Task"}
    },
    "Consumption": {
        "triggers": ["ate", "drank", "consumed", "used"],
        "roles": {"ARG0": "Consumer","ARG1": "Consumable"}
    },
    "Inspection": {
        "triggers": ["investigated", "examined", "inspected", "searched", "looked for"],
        "roles": {"ARG0": "Investigator","ARG1": "Subject"}
    },
    "Social": {
        "triggers": ["met", "gathered", "celebrated", "partied", "dined"],
        "roles": {"ARG0": "Participant_1","ARG1": "Participant_2","ARGM-PRD": "Event"}
    },
    "Transformation": {
        "triggers": ["became", "transformed", "changed into", "turned into"],
        "roles": {"ARG0": "Entity","ARG1": "Final_State"}
    },
    "Causation": {
        "triggers": ["caused", "made", "forced", "led to", "resulted in"],
        "roles": {"ARG0": "Cause","ARG1": "Effect"}
    }
}

# --- Build a fast lemma→schema lookup ---
event_mapping = {}
for event_type, details in general_event_schema.items():
    for trigger_lemma in details["triggers"]:
        event_mapping[trigger_lemma] = {
            "event_type": event_type,
            "role_map": details["roles"]
        }
```


```python
# ==============================================================================
# 3. HYBRID NER FUNCTION (Unchanged)
# ==============================================================================
def extract_entities_solved(sentence_text, known_characters):
    def normalize_text(chunk):
        if len(chunk) > 1 and chunk[0].pos_ == "DET":
            return chunk[1:].text
        return chunk.text

    ner_results = ner_pipeline(sentence_text)
    entities = []
    seen_words = set()

    for ent in ner_results:
        entities.append({
            "entity_group": ent["entity_group"], "word": ent["word"],
            "score": round(ent["score"], 4), "source": "NER"
        })
        for word in ent["word"].split():
            seen_words.add(word.lower())

    doc = nlp(sentence_text)
    for chunk in doc.noun_chunks:
        normalized_word = normalize_text(chunk)
        if chunk.root.text.lower() not in seen_words:
            dep = chunk.root.dep_
            ent_type = "MISC"
            if normalized_word.lower() in known_characters:
                ent_type = "Character"
            elif dep in ["nsubj", "nsubjpass"]:
                ent_type = "Character"
                known_characters.add(normalized_word.lower())
            elif dep in ["dobj"]:
                ent_type = "Object"
            elif dep in ["pobj", "obl"]:
                ent_type = "Location"

            entities.append({
                "entity_group": ent_type, "word": normalized_word,
                "score": 1.0, "source": "Rule"
            })
            for token in chunk:
                seen_words.add(token.text.lower())
    
    entity_map = {}
    type_priority = {"Character": 3, "Person": 2, "Location": 1, "Object": 0, "MISC": -1}
    for ent in entities:
        word_key = ent["word"].lower()
        current_type = ent["entity_group"]
        if word_key in entity_map:
            existing_type = entity_map[word_key]["entity_group"]
            if type_priority.get(current_type, -1) > type_priority.get(existing_type, -1):
                entity_map[word_key] = ent
        else:
            entity_map[word_key] = ent
            
    final_entities = list(entity_map.values())
    tag_mapping = {
        "PER": "Person", "LOC": "Location", "ORG": "Organization",
        "CHAR": "Character", "OBJ": "Object", "MISC": "Miscellaneous"
    }
    
    for ent in final_entities:
        original_group = ent["entity_group"]
        ent["entity_group"] = tag_mapping.get(original_group, original_group)

    return final_entities, known_characters
```


```python
def extract_attributes_improved(scene):
    """
    Extracts attributes and correctly links them to their semantic entity,
    handling cases like "tortoise is handsome" by finding the subject of the verb.
    """
    attributes = {}
    tokens = scene.get("tokens", [])
    dependencies = scene.get("dependencies", [])

    # Create more useful lookups for easier tree traversal
    # 1. Map each token to its head and dependency type
    token_to_head = {t: (h, d) for t, d, h in dependencies}
    # 2. Map each head to its children tokens and their dependency types
    head_to_children = {}
    for t, d, h in dependencies:
        if h not in head_to_children:
            head_to_children[h] = []
        head_to_children[h].append((t, d))

    for token in tokens:
        if token not in token_to_head:
            continue

        head, dep = token_to_head[token]

        # Check if the token is an attribute we care about
        if dep in {"amod", "advmod", "acomp", "xcomp", "oprd"}:
            final_head = head

            # --- THE CORE IMPROVEMENT ---
            # If the attribute is an adjectival complement (acomp) or open clausal
            # complement (xcomp), its head is a verb. We need to find the *subject*
            # of that verb to find the entity being described.
            if dep in {"acomp", "xcomp"}:
                if head in head_to_children:
                    # Search for the nominal subject (nsubj) of the verb
                    for child, child_dep in head_to_children[head]:
                        if child_dep in {"nsubj", "nsubjpass"}:
                            final_head = child
                            break # Found the subject, stop looking

            # Assign the attribute (token) to its true head (final_head)
            if final_head not in attributes:
                attributes[final_head] = []
            attributes[final_head].append(token)

    return attributes
```


```python
# ==============================================================================
# 4. STAGE 3 ANALYSIS (UPDATED Event Extraction)
# ==============================================================================
def analyze_story_for_deep_context(stage2_output):
    story_analysis = []
    known_characters_in_story = set()

    for scene in stage2_output:
        sentence_text = scene["text"]

        # --- 1. Named Entities ---
        entities, known_characters_in_story = extract_entities_solved(
            sentence_text, known_characters_in_story
        )

        doc = nlp(sentence_text)

        scene_data_for_attributes = {
            "tokens": [token.text for token in doc],
            "dependencies": [(token.text, token.dep_, token.head.text) for token in doc]
        }
        # Now, call the new, improved function
        attributes = extract_attributes_improved(scene_data_for_attributes)


        # --- 3. Emotions ---
        emotions = emotion_classifier(sentence_text)

        # --- 4. Events ---
        detected_events = []
        srl_result = srl_predictor.predict(sentence=sentence_text)
        

        
        for verb_info in srl_result['verbs']:
            # find lemma
            verb_token = next((t for t in doc if t.text == verb_info['verb']), None)
            verb_lemma = verb_token.lemma_ if verb_token else verb_info['verb'].lower()

            # collect SRL args
            tags = re.findall(r'\[(.*?)\]', verb_info['description'])
            srl_args = {}
            for tag in tags:
                parts = tag.split(': ', 1)
                if len(parts) == 2:
                    srl_args[parts[0]] = parts[1]

            # map to schema
            if verb_lemma in event_mapping:
                schema_info = event_mapping[verb_lemma]
                event_type = schema_info['event_type']
                role_map = schema_info['role_map']
                custom_args = {}
                for srl_role, text in srl_args.items():
                    if srl_role in role_map:
                        custom_args[role_map[srl_role]] = text
                detected_events.append({
                    "event_method": "SRL-Schema",
                    "event_type": event_type,
                    "trigger": verb_info['verb'],
                    "arguments": custom_args
                })
            else:
                detected_events.append({
                    "event_method": "SRL-Generic",
                    "event_type": "Generic",
                    "trigger": verb_info['verb'],
                    "arguments": srl_args
                  })

        story_analysis.append({
            "scene_id": scene["scene_id"],
            "text": sentence_text,
            "entities": entities,
            "attributes": attributes,
            "emotions": emotions,
            "events": detected_events
        })

    return story_analysis
```


```python
# ==============================================================================
# 5. CLEANER FOR JSON SERIALIZATION
# ==============================================================================
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj
```


```python
# ==============================================================================
# 6. RUN & PRINT (Example Usage)
# ==============================================================================
# stage3_ready_input = [
#     {"scene_id": 1, "text": "Hare was hopping through the forest."},
#     {"scene_id": 2, "text": "Hare saw a tortoise slowly walking along the path."},
#     {"scene_id": 3, "text": "Hare greeted a tortoise."}
# ]

deep_context = analyze_story_for_deep_context(stage3_ready_input)
deep_context_clean = clean_for_json(deep_context)

print(json.dumps(deep_context_clean, indent=2))
```

    [
      {
        "scene_id": 1,
        "text": "Hare was hopping through the forest .",
        "entities": [
          {
            "entity_group": "Person",
            "word": "Hare",
            "score": 0.986299991607666,
            "source": "NER"
          },
          {
            "entity_group": "Location",
            "word": "forest",
            "score": 1.0,
            "source": "Rule"
          }
        ],
        "attributes": {},
        "emotions": [
          {
            "label": "neutral",
            "score": 0.9677174091339111
          }
        ],
        "events": [
          {
            "event_method": "SRL-Generic",
            "event_type": "Generic",
            "trigger": "was",
            "arguments": {
              "V": "was"
            }
          },
          {
            "event_method": "SRL-Generic",
            "event_type": "Generic",
            "trigger": "hopping",
            "arguments": {
              "ARG0": "Hare",
              "V": "hopping",
              "ARGM-DIR": "through the forest"
            }
          }
        ]
      },
      {
        "scene_id": 2,
        "text": "Hare saw a tortoise slowly walking along the path .",
        "entities": [
          {
            "entity_group": "Person",
            "word": "Hare",
            "score": 0.9977999925613403,
            "source": "NER"
          },
          {
            "entity_group": "Character",
            "word": "tortoise",
            "score": 1.0,
            "source": "Rule"
          },
          {
            "entity_group": "Location",
            "word": "path",
            "score": 1.0,
            "source": "Rule"
          }
        ],
        "attributes": {
          "walking": [
            "slowly"
          ]
        },
        "emotions": [
          {
            "label": "neutral",
            "score": 0.9678376317024231
          }
        ],
        "events": [
          {
            "event_method": "SRL-Generic",
            "event_type": "Generic",
            "trigger": "saw",
            "arguments": {
              "ARG0": "Hare",
              "V": "saw",
              "ARG1": "a tortoise slowly walking along the path"
            }
          },
          {
            "event_method": "SRL-Generic",
            "event_type": "Generic",
            "trigger": "walking",
            "arguments": {
              "ARG0": "a tortoise",
              "ARGM-MNR": "slowly",
              "V": "walking",
              "ARG1": "along the path"
            }
          }
        ]
      },
      {
        "scene_id": 3,
        "text": "Hare greeted a tortoise fast .",
        "entities": [
          {
            "entity_group": "Person",
            "word": "Hare",
            "score": 0.9909999966621399,
            "source": "NER"
          },
          {
            "entity_group": "Character",
            "word": "tortoise",
            "score": 1.0,
            "source": "Rule"
          }
        ],
        "attributes": {
          "greeted": [
            "fast"
          ]
        },
        "emotions": [
          {
            "label": "neutral",
            "score": 0.9602752327919006
          }
        ],
        "events": [
          {
            "event_method": "SRL-Generic",
            "event_type": "Generic",
            "trigger": "greeted",
            "arguments": {
              "ARG0": "Hare",
              "V": "greeted",
              "ARG1": "a tortoise",
              "ARGM-MNR": "fast"
            }
          }
        ]
      },
      {
        "scene_id": 4,
        "text": "a tortoise is handsome .",
        "entities": [
          {
            "entity_group": "Character",
            "word": "tortoise",
            "score": 1.0,
            "source": "Rule"
          }
        ],
        "attributes": {
          "tortoise": [
            "handsome"
          ]
        },
        "emotions": [
          {
            "label": "admiration",
            "score": 0.9384015798568726
          }
        ],
        "events": [
          {
            "event_method": "SRL-Generic",
            "event_type": "Generic",
            "trigger": "is",
            "arguments": {
              "ARG1": "a tortoise",
              "V": "is",
              "ARG2": "handsome"
            }
          }
        ]
      }
    ]
    

# Stage 4 


```python
# ========== 1. IMPORTS ==========
import os
import re
import json
import spacy
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from typing import Optional

# ========== 2. STAGE 1: PRE-PROCESSING FUNCTION ==========
def preprocess_story(text: str, nlp_model) -> dict:
    """Cleans text and processes it with spaCy to get sentences."""
    clean_text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp_model(clean_text)
    return {"sentences": list(doc.sents)}

# ========== 3. STAGE 4: SUMMARIZATION CLASS (SIMPLIFIED) ==========
class SceneSummarizer:
    """
    A class to perform abstractive summarization on scene text.
    """
    def __init__(self, model_type: str = "bart", local_path: Optional[str] = None):
        """Initializes the summarizer with a local transformer model."""
        self.model_type = model_type

        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"❌ Local model path not found: {local_path}")

        if model_type == "bart":
            self.abstractive_model = pipeline("summarization", model=local_path)
        elif model_type == "t5":
            model = T5ForConditionalGeneration.from_pretrained(local_path)
            tokenizer = T5Tokenizer.from_pretrained(local_path)
            self.abstractive_model = (model, tokenizer)
        else:
            raise ValueError("model_type must be 'bart' or 't5'")

    def summarize(self, text: str, max_len: int = 60, min_len: int = 15) -> str:
        """Generates a new, abstractive summary by understanding the text."""
        if self.model_type == "bart":
            summary = self.abstractive_model(
                text, max_length=max_len, min_length=min_len, do_sample=False
            )
            return summary[0]["summary_text"]
        elif self.model_type == "t5":
            model, tokenizer = self.abstractive_model
            input_text = "summarize: " + text
            inputs = tokenizer.encode(
                input_text, return_tensors="pt", max_length=1024, truncation=True
            )
            summary_ids = model.generate(
                inputs, max_length=max_len, min_length=min_len, length_penalty=2.0, num_beams=4
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return ""

# ========== 4. MAIN PIPELINE EXECUTION ==========
if __name__ == "__main__":
    # --- Define initial text and load NLP model ---
    story_text = """
    Hare was hopping through the forest.
    He saw a tortoise slowly walking along the path.
    he greeted him fast.
    he is handsome.
    """
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    # --- STAGE 1: Pre-processing ---
    print("Running Stage 1 pre-processing...")
    stage1_output = preprocess_story(story_text, nlp)
    full_scene_text = " ".join([sent.text for sent in stage1_output['sentences']])

    # --- STAGE 4: Summarization ---
    print("Running summarization...")
    # IMPORTANT: Update this path to your local BART model folder
    local_bart_path = r"C:\Users\__msi__\facebook-bart-large-cnn"
    
    try:
        summarizer = SceneSummarizer(model_type="bart", local_path=local_bart_path)
        
        # Generate the abstractive summary
        final_summary = summarizer.summarize(full_scene_text)
        
        # Store the result
        scene_output = {
            "original_text": full_scene_text,
            "summary": final_summary
        }
        
        # Print the final, organized output
        print("\n✅ Final Scene Output:")
        print(json.dumps(scene_output, indent=2))

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
```

    Loading spaCy model...
    Running Stage 1 pre-processing...
    Running summarization...
    

    Your max_length is set to 60, but you input_length is only 29. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=14)
    

    
    ✅ Final Scene Output:
    {
      "original_text": "Hare was hopping through the forest. He saw a tortoise slowly walking along the path. he greeted him fast. he is handsome.",
      "summary": "Hare was hopping through the forest. He saw a tortoise slowly walking along the path. He greeted him fast."
    }
    


```python
def build_final_structure(stage2_output, deep_context_clean, scene_output):
    """
    Combines Stage 2, Stage 3, and Stage 4 outputs into the final structure
    ready for Knowledge Graph construction.
    
    Parameters:
        stage2_output (list): Output list from Stage 2 (scene annotations).
        deep_context_clean (list): Output list from Stage 3 (deep analysis).
        scene_output (dict): Output dict from Stage 4 (original text + summary).
    
    Returns:
        dict: Final structured JSON-like dictionary.
    """
    final_structure = {
        "stage2_scenes": stage2_output,
        "stage3_details": deep_context_clean,
        "summary_output": scene_output
    }
    return final_structure


#============================
#Example Usage
#============================
final_data = build_final_structure(stage2_output, deep_context_clean, scene_output)
print(json.dumps(final_data, indent=2))

```

    {
      "stage2_scenes": [
        "b scene [Hare was hopping through the forest .]",
        "i scene [Hare saw a tortoise slowly walking along the path .]",
        "i scene [Hare greeted a tortoise fast .]",
        "b scene [a tortoise is handsome .]"
      ],
      "stage3_details": [
        {
          "scene_id": 1,
          "text": "Hare was hopping through the forest .",
          "entities": [
            {
              "entity_group": "Person",
              "word": "Hare",
              "score": 0.986299991607666,
              "source": "NER"
            },
            {
              "entity_group": "Location",
              "word": "forest",
              "score": 1.0,
              "source": "Rule"
            }
          ],
          "attributes": {},
          "emotions": [
            {
              "label": "neutral",
              "score": 0.9677174091339111
            }
          ],
          "events": [
            {
              "event_method": "SRL-Generic",
              "event_type": "Generic",
              "trigger": "was",
              "arguments": {
                "V": "was"
              }
            },
            {
              "event_method": "SRL-Generic",
              "event_type": "Generic",
              "trigger": "hopping",
              "arguments": {
                "ARG0": "Hare",
                "V": "hopping",
                "ARGM-DIR": "through the forest"
              }
            }
          ]
        },
        {
          "scene_id": 2,
          "text": "Hare saw a tortoise slowly walking along the path .",
          "entities": [
            {
              "entity_group": "Person",
              "word": "Hare",
              "score": 0.9977999925613403,
              "source": "NER"
            },
            {
              "entity_group": "Character",
              "word": "tortoise",
              "score": 1.0,
              "source": "Rule"
            },
            {
              "entity_group": "Location",
              "word": "path",
              "score": 1.0,
              "source": "Rule"
            }
          ],
          "attributes": {
            "walking": [
              "slowly"
            ]
          },
          "emotions": [
            {
              "label": "neutral",
              "score": 0.9678376317024231
            }
          ],
          "events": [
            {
              "event_method": "SRL-Generic",
              "event_type": "Generic",
              "trigger": "saw",
              "arguments": {
                "ARG0": "Hare",
                "V": "saw",
                "ARG1": "a tortoise slowly walking along the path"
              }
            },
            {
              "event_method": "SRL-Generic",
              "event_type": "Generic",
              "trigger": "walking",
              "arguments": {
                "ARG0": "a tortoise",
                "ARGM-MNR": "slowly",
                "V": "walking",
                "ARG1": "along the path"
              }
            }
          ]
        },
        {
          "scene_id": 3,
          "text": "Hare greeted a tortoise fast .",
          "entities": [
            {
              "entity_group": "Person",
              "word": "Hare",
              "score": 0.9909999966621399,
              "source": "NER"
            },
            {
              "entity_group": "Character",
              "word": "tortoise",
              "score": 1.0,
              "source": "Rule"
            }
          ],
          "attributes": {
            "greeted": [
              "fast"
            ]
          },
          "emotions": [
            {
              "label": "neutral",
              "score": 0.9602752327919006
            }
          ],
          "events": [
            {
              "event_method": "SRL-Generic",
              "event_type": "Generic",
              "trigger": "greeted",
              "arguments": {
                "ARG0": "Hare",
                "V": "greeted",
                "ARG1": "a tortoise",
                "ARGM-MNR": "fast"
              }
            }
          ]
        },
        {
          "scene_id": 4,
          "text": "a tortoise is handsome .",
          "entities": [
            {
              "entity_group": "Character",
              "word": "tortoise",
              "score": 1.0,
              "source": "Rule"
            }
          ],
          "attributes": {
            "tortoise": [
              "handsome"
            ]
          },
          "emotions": [
            {
              "label": "admiration",
              "score": 0.9384015798568726
            }
          ],
          "events": [
            {
              "event_method": "SRL-Generic",
              "event_type": "Generic",
              "trigger": "is",
              "arguments": {
                "ARG1": "a tortoise",
                "V": "is",
                "ARG2": "handsome"
              }
            }
          ]
        }
      ],
      "summary_output": {
        "original_text": "Hare was hopping through the forest. He saw a tortoise slowly walking along the path. he greeted him fast. he is handsome.",
        "summary": "Hare was hopping through the forest. He saw a tortoise slowly walking along the path. He greeted him fast."
      }
    }
    


```python
# robust_kg_loader.py
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase, Transaction, basic_auth

# ---------------------------
# Logging setup (adjustable)
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobustKG")

# ---------------------------
# Helper functions
# ---------------------------
def normalize_text(s: str) -> str:
    """Normalize a string for matching: lowercase, trim, remove extra punctuation, collapse spaces."""
    if s is None:
        return ""
    s = re.sub(r"[“”\"'`]", "", s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip().lower()
    return s

def tokens_of(text: str) -> List[str]:
    """Simple tokenization (split by whitespace) for matching purposes."""
    return [t for t in re.split(r'\W+', text) if t]

def best_match_entity_name(target_text: str, candidate_names: List[str]) -> Optional[str]:
    """
    Try to map a piece of argument text back to an existing canonical entity name.
    Strategies (in order):
      1) exact normalized match
      2) token-subset match (all tokens of candidate appear in target_text)
      3) token overlap max
    Returns the canonical candidate name (not normalized) or None.
    """
    if not target_text or not candidate_names:
        return None
    n_target = normalize_text(target_text)

    # Precompute normalized candidate map: normalized -> original
    norm_to_orig = {}
    for c in candidate_names:
        norm = normalize_text(c)
        if norm:
            norm_to_orig.setdefault(norm, c)

    # 1) exact normalized
    if n_target in norm_to_orig:
        return norm_to_orig[n_target]

    # 2) if candidate tokens are subset of target tokens
    target_tokens = set(tokens_of(n_target))
    if not target_tokens:
        return None

    best_candidate = None
    best_overlap = 0
    for orig in candidate_names:
        cand_norm = normalize_text(orig)
        cand_tokens = set(tokens_of(cand_norm))
        if not cand_tokens:
            continue
        if cand_tokens.issubset(target_tokens):
            return orig  # strong signal
        # compute overlap for fallback
        overlap = len(cand_tokens & target_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = orig

    # Require at least one token overlap to consider (avoid wrong matches)
    if best_overlap >= 1:
        return best_candidate

    return None

# ---------------------------
# Robust KG Loader Class
# ---------------------------
class RobustKnowledgeGraphLoader:
    def __init__(self, uri: str, user: str, password: str, encryption=False):
        """
        Robust KG loader for Neo4j.
          - uri: bolt URI (bolt://host:7687)
          - user, password: credentials
        """
        auth = basic_auth(user, password)
        self.driver = GraphDatabase.driver(uri, auth=auth)
        logger.info("Initialized KG loader for %s", uri)

    def close(self):
        self.driver.close()
        logger.info("Driver closed.")

    # ---- Setup: indexes and constraints (idempotent) ----
    def ensure_schema(self):
        """
        Create helpful constraints/indexes to speed up lookups and ensure uniqueness.
        This function is idempotent -- safe to call repeatedly.
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Story) REQUIRE s.story_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.entity_id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sc:Scene) REQUIRE (sc.scene_uid) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ev:Event) REQUIRE (ev.event_id) IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                logger.debug("Running constraint/index: %s", c)
                session.run(c)
        logger.info("Schema constraints ensured.")

    # ---- Utility to generate safe unique ids ----
    @staticmethod
    def _make_story_id(story_title: str) -> str:
        return normalize_text(story_title).replace(" ", "_")[:200]

    @staticmethod
    def _make_entity_id(story_id: str, name: str) -> str:
        return f"{story_id}::entity::{normalize_text(name)}"

    @staticmethod
    def _make_scene_uid(story_id: str, scene_id: int) -> str:
        return f"{story_id}::scene::{scene_id}"

    @staticmethod
    def _make_event_id(story_id: str, scene_id: int, idx: int) -> str:
        return f"{story_id}::event::{scene_id}::{idx}"

    # ---- Clear only a story (safe) ----
    def clear_story(self, story_id: str):
        """Safely delete nodes and relationships for a single story_id."""
        q = """
        MATCH (s:Story {story_id: $story_id})-[r0*0..]->()
        WITH s
        OPTIONAL MATCH (s)-[:HAS_SCENE]->(sc:Scene)
        OPTIONAL MATCH (sc)-[r]-()
        DETACH DELETE s, sc
        """
        with self.driver.session() as session:
            session.run(q, story_id=story_id)
        logger.info("Cleared story and its scenes for story_id=%s", story_id)

    # ---- Core loader ----
    def load_pipeline_output(self, pipeline_output: Dict, story_title: str = "Fable", clear_story_first: bool = False, dry_run: bool = False) -> Dict:
        """
        Main entrypoint to load pipeline_output into KG.
        - pipeline_output: the dict you produced (stage2_scenes, stage3_details, summary_output)
        - story_title: name for the story (used for scoping)
        - clear_story_first: if True, remove previous nodes for this story_id
        - dry_run: if True, do not write anything, only return what WOULD be done
        Returns a report dict with counts and any mapping issues.
        """
        report = {
            "created_entities": 0,
            "created_scenes": 0,
            "created_events": 0,
            "created_attributes": 0,
            "created_emotions": 0,
            "warnings": []
        }

        self.ensure_schema()
        story_id = self._make_story_id(story_title)

        if clear_story_first:
            if dry_run:
                logger.info("[dry_run] Would clear story: %s", story_id)
            else:
                self.clear_story(story_id)

        # prepare canonical entity set as we load scenes to allow good matching
        canonical_entities = {}  # normalized -> canonical name

        if dry_run:
            logger.info("[dry_run] Starting dry-run; no changes will be committed")

        # We'll process scenes in order present in stage3_details (safe assumption)
        scenes = pipeline_output.get("stage3_details", [])
        stage2_scenes = pipeline_output.get("stage2_scenes", [])

        # Create Story node (MERGE)
        story_summary = pipeline_output.get("summary_output", {}).get("summary", "")
        story_original_text = pipeline_output.get("summary_output", {}).get("original_text", "")

        if not dry_run:
            with self.driver.session() as session:
                session.run(
                    "MERGE (st:Story {story_id: $story_id}) "
                    "SET st.title = $title, st.summary = $summary, st.original_text = $original_text",
                    story_id=story_id, title=story_title, summary=story_summary, original_text=story_original_text
                )
        else:
            logger.info("[dry_run] Would MERGE Story with id=%s", story_id)

        # Pre-scan entities across all scenes to build initial canonical list (helps mapping)
        for s in scenes:
            for ent in s.get("entities", []):
                name = ent.get("word") or ent.get("text") or ""
                if not name:
                    continue
                norm = normalize_text(name)
                if norm and norm not in canonical_entities:
                    canonical_entities[norm] = name

        # Helper to ensure entity node exists and return canonical name
        def ensure_entity(session: Transaction, entity_name: str):
            if not entity_name:
                return None
            canonical = canonical_entities.get(normalize_text(entity_name), entity_name)
            entity_id = self._make_entity_id(story_id, canonical)
            # create/merge entity node with some metadata
            q = """
            MERGE (e:Entity {entity_id: $entity_id})
            ON CREATE SET e.name = $name, e.type = $etype, e.first_seen = timestamp()
            SET e.last_seen = timestamp()
            RETURN e.entity_id AS entity_id, e.name AS name
            """
            params = {"entity_id": entity_id, "name": canonical, "etype": "Unknown"}
            result = session.run(q, **params)
            _ = result.single()
            return canonical

        # Now iterate scenes and load them
        with self.driver.session() as session:
            for idx, scene in enumerate(scenes):
                scene_id = scene.get("scene_id", idx + 1)
                scene_text = scene.get("text", "")
                # scene_str (b/i) might exist in stage2_scenes at same index; fallback to 'unknown'
                scene_str = stage2_scenes[idx] if idx < len(stage2_scenes) else "unknown scene"
                # classify type robustly
                stype = "unknown"
                stype_raw = scene_str.strip().lower()
                if stype_raw.startswith("b scene"):
                    stype = "beginning"
                elif stype_raw.startswith("i scene"):
                    stype = "intermediate"
                elif stype_raw.startswith("e scene"):
                    stype = "ending"
                else:
                    # maybe it contains markers in other forms
                    if "begin" in stype_raw or "b_scene" in stype_raw:
                        stype = "beginning"
                    elif "inter" in stype_raw:
                        stype = "intermediate"

                scene_uid = self._make_scene_uid(story_id, scene_id)
                if dry_run:
                    logger.info("[dry_run] Would create Scene id=%s type=%s text=%s", scene_uid, stype, scene_text)
                    report["created_scenes"] += 1
                else:
                    # create Scene node and link to story
                    session.run(
                        "MATCH (st:Story {story_id: $story_id}) "
                        "MERGE (sc:Scene {scene_uid: $scene_uid}) "
                        "SET sc.scene_id = $scene_id, sc.scene_uid = $scene_uid, sc.type = $stype, sc.text = $text, sc.created = timestamp() "
                        "MERGE (st)-[:HAS_SCENE {order: $order}]->(sc)",
                        story_id=story_id, scene_uid=scene_uid, scene_id=scene_id, stype=stype, text=scene_text, order=idx
                    )
                    report["created_scenes"] += 1

                # Create or ensure Entities for this scene
                scene_entity_names = []
                for ent in scene.get("entities", []):
                    ent_name = ent.get("word") or ent.get("name") or ""
                    if not ent_name:
                        continue
                    # Update canonical list if new
                    norm = normalize_text(ent_name)
                    if norm not in canonical_entities:
                        canonical_entities[norm] = ent_name
                    # ensure node
                    canonical = ensure_entity(session, ent_name)
                    scene_entity_names.append(canonical)
                    report["created_entities"] += 1

                # Map events and create Event nodes
                for ev_idx, event in enumerate(scene.get("events", [])):
                    ev_id = self._make_event_id(story_id, scene_id, ev_idx)
                    trigger = event.get("trigger") or event.get("verb") or "unknown"
                    arguments = event.get("arguments", {}) or {}

                    # create Event node and link it to scene
                    if dry_run:
                        logger.info("[dry_run] Would create Event %s trigger=%s", ev_id, trigger)
                        report["created_events"] += 1
                    else:
                        session.run(
                            "MATCH (sc:Scene {scene_uid: $scene_uid}) "
                            "MERGE (ev:Event {event_id: $event_id}) "
                            "SET ev.trigger = $trigger, ev.method = $method, ev.event_type = $etype, ev.created = timestamp() "
                            "MERGE (sc)-[:CONTAINS_EVENT]->(ev)",
                            scene_uid=scene_uid, event_id=ev_id, trigger=trigger,
                            method=event.get("event_method", "unknown"),
                            etype=event.get("event_type", "Generic")
                        )
                        report["created_events"] += 1

                    # Create relationships from entities to event (map argument text to entities)
                    for role, arg_text in arguments.items():
                        if not arg_text:
                            continue
                        # try to map arg_text to an existing entity canonical name
                        candidate_names = list(canonical_entities.values())
                        mapped = best_match_entity_name(arg_text, candidate_names)
                        # If best match fails, fall back to exact substring search in scene_entity_names
                        if not mapped:
                            for cand in scene_entity_names:
                                if normalize_text(cand) in normalize_text(arg_text):
                                    mapped = cand
                                    break

                        # If still not found, we create a scene-scoped entity (this avoids global conflicts)
                        if not mapped:
                            # fallback name is cleaned arg_text
                            fallback_name = arg_text.strip()
                            # create a new canonical key
                            canonical_entities[normalize_text(fallback_name)] = fallback_name
                            if dry_run:
                                logger.info("[dry_run] Would create fallback Entity for arg_text=%s", arg_text)
                                mapped = fallback_name
                                report["created_entities"] += 1
                            else:
                                canonical = ensure_entity(session, fallback_name)
                                mapped = canonical
                                report["created_entities"] += 1

                        # Now link entity to event with role property
                        if dry_run:
                            logger.info("[dry_run] Would link Entity(%s) -> PARTICIPATED_IN(role=%s) -> Event(%s)", mapped, role, ev_id)
                        else:
                            # match entity by its entity_id (scoped) and link
                            entity_id = self._make_entity_id(story_id, mapped)
                            session.run(
                                "MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) "
                                "MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) "
                                "SET r.role = $role",
                                event_id=ev_id, entity_id=entity_id, role=role
                            )

                # Attributes: attach to entity, optionally scene-scoped property
                for target, attrs in scene.get("attributes", {}).items():
                    if not attrs:
                        continue
                    # map target to canonical entity
                    mapped_target = best_match_entity_name(target, list(canonical_entities.values()))
                    if not mapped_target:
                        mapped_target = target
                        # create entity if missing
                        if not dry_run:
                            ensure_entity(session, mapped_target)
                            report["created_entities"] += 1
                    for attr in attrs:
                        if dry_run:
                            logger.info("[dry_run] Would attach Attribute(%s) to Entity(%s) in Scene(%s)", attr, mapped_target, scene_uid)
                            report["created_attributes"] += 1
                        else:
                            entity_id = self._make_entity_id(story_id, mapped_target)
                            session.run(
                                "MERGE (a:Attribute {name: $attr_name, story_id: $story_id}) "
                                "MERGE (e:Entity {entity_id: $entity_id}) "
                                "MERGE (e)-[:HAS_ATTRIBUTE]->(a) "
                                "ON CREATE SET a.created = timestamp()",
                                attr_name=attr, story_id=story_id, entity_id=entity_id
                            )
                            report["created_attributes"] += 1

                # Emotions: attach emotion nodes and relation to scene (with score)
                for emo in scene.get("emotions", []):
                    label = emo.get("label")
                    score = float(emo.get("score", 0.0))
                    if dry_run:
                        logger.info("[dry_run] Would create Emotion(%s) and link to Scene(%s) score=%s", label, scene_uid, score)
                        report["created_emotions"] += 1
                    else:
                        session.run(
                            "MERGE (em:Emotion {label: $label, story_id: $story_id}) "
                            "WITH em "
                            "MATCH (sc:Scene {scene_uid: $scene_uid}) "
                            "MERGE (sc)-[r:HAS_EMOTION]->(em) "
                            "SET r.score = $score",
                            label=label, story_id=story_id, scene_uid=scene_uid, score=score
                        )
                        report["created_emotions"] += 1

        logger.info("KG load complete. report=%s", report)
        return report
print("Yes")
```

    Yes
    


```python
if __name__ == "__main__":
    # from robust_kg_loader import RobustKnowledgeGraphLoader
    import json

    # Neo4j connection
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASS = "rohitmukkala@sujal"

    # Use the final_data you created
    pipeline_output = final_data

    loader = RobustKnowledgeGraphLoader(URI, USER, PASS)
    try:
        report = loader.load_pipeline_output(
            pipeline_output, 
            story_title="Fable", 
            clear_story_first=True, 
            dry_run=False   # set True if you just want to test
        )
        print("Load report:", json.dumps(report, indent=2))
    finally:
        loader.close()

```

    INFO:RobustKG:Initialized KG loader for neo4j://127.0.0.1:7687
    INFO:RobustKG:Schema constraints ensured.
    INFO:RobustKG:Cleared story and its scenes for story_id=fable
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:neo4j.notifications:Received notification from DBMS server: {severity: INFORMATION} {code: Neo.ClientNotification.Statement.CartesianProduct} {category: PERFORMANCE} {title: This query builds a cartesian product between disconnected patterns.} {description: If a part of a query contains multiple disconnected patterns, this will build a cartesian product between all those parts. This may produce a large amount of data and slow down query processing. While occasionally intended, it may often be possible to reformulate the query that avoids the use of this cross product, perhaps by adding a relationship between the different parts or by using OPTIONAL MATCH (identifier is: (e))} {position: line: 1, column: 1, offset: 0} for query: 'MATCH (ev:Event {event_id: $event_id}), (e:Entity {entity_id: $entity_id}) MERGE (e)-[r:PARTICIPATED_IN {role: $role}]->(ev) SET r.role = $role'
    INFO:RobustKG:KG load complete. report={'created_entities': 17, 'created_scenes': 4, 'created_events': 6, 'created_attributes': 3, 'created_emotions': 4, 'warnings': []}
    INFO:RobustKG:Driver closed.
    

    Load report: {
      "created_entities": 17,
      "created_scenes": 4,
      "created_events": 6,
      "created_attributes": 3,
      "created_emotions": 4,
      "warnings": []
    }
    


```python

```
