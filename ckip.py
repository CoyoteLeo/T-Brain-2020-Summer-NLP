from ckiptagger import NER, POS, WS

disable_cuda = False
ws = WS("./data", disable_cuda=disable_cuda)
pos = POS("./data", disable_cuda=disable_cuda)
ner = NER("./data", disable_cuda=disable_cuda)


def get_names(sentence_list):
    word_sentence_list = ws(sentence_list)
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    return [
        [
            word
            for start_pos, end_pos, entity_type, word in entity_sentence
            if entity_type == "PERSON"
        ]
        for entity_sentence in entity_sentence_list
    ]
