import json
import logging
import os
from datetime import datetime
from pathlib import Path

from cltl.brain import logger as brain_logger
from cltl.brain.long_term_memory import LongTermMemory
from cltl.entity_linking.label_linker import LabelBasedLinker
from cltl.triple_extraction import logger as chat_logger
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.spacy_analyzer import spacyAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules

from create_context_capsule import create_context_capsule
from loaders import DailyDialogueLoader

chat_logger.setLevel(logging.ERROR)
brain_logger.setLevel(logging.ERROR)

out_dir = './rdf_files/daily_dialog'


def load_json(file_name, data_dir='./processed_data/daily_dialogue'):
    file_path = os.path.join(data_dir, file_name + '.json')
    assert os.path.exists(file_path), f'File {file_path} does not exist.'

    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    return data


def collect_speakers(conversation):
    speakers = set()
    for speaker in conversation:
        speakers.add(speaker["speaker"])

    return [s for s in speakers]


### Ignore this:
def expand_author(capsule):
    if type(capsule['author']) == str:
        capsule['author'] = {
            'label': capsule['author'],
            'type': ['person'],
            'uri': []
        }

    capsule['subject']['uri'] = []
    capsule['predicate']['uri'] = []
    capsule['object']['uri'] = []
    capsule['timestamp'] = datetime.now()

    return capsule


daily_dialog = DailyDialogueLoader()
daily_dialog.save_to_file('dd_test')
data = load_json('dd_test')

analyzer = spacyAnalyzer()
linker = LabelBasedLinker()

capsules_skipped = 0
for row in data:
    key = list(row.keys())[0]  # obtain the ID for the dialog
    conversation = row[key]
    speakers = collect_speakers(conversation)  # collect the list of speakers
    assert len(speakers) == 2, f'{len(speakers)} speakers found. Can only parse 2 speakers.'

    file_loc = os.path.join(out_dir, key + '/rdf/')
    file_loc = Path(file_loc)
    if not os.path.exists(file_loc):
        os.makedirs(Path(file_loc))  # create directory to store the rdf files if need be

    # Initialize brain, Chat, and context capsule
    brain = LongTermMemory(address="http://localhost:7200/repositories/test",
                           log_dir=file_loc,
                           clear_all=False)
    chat = Chat(speakers[0], speakers[-1])
    context_capsule = create_context_capsule(key)
    brain.capsule_context(context_capsule)

    capsules = []
    for i, turn in enumerate(conversation):
        # switch around speakers every turn
        speakers = [speakers[-1], speakers[0]]
        utterance = turn['text']

        # add utterance to chat and use spacy analyzer to analyze
        chat.add_utterance(utterance)
        subj, obj = speakers
        analyzer.analyze(chat.last_utterance, subj, obj)

        # add capsules to list of capsules
        capsules += utterance_to_capsules(chat.last_utterance)

        # add statement capsules to brain
        for capsule in capsules:
            capsule = expand_author(capsule)
            linker.link(capsule)

            # Debug?
            if type(capsule['perspective']) == dict:
                print('all good in the hood')
            else:
                print('niet good')

            try:
                brain.capsule_statement(capsule)
            except:
                capsules_skipped += 1
                print(f"Capsule skipped. Total {capsules_skipped}")