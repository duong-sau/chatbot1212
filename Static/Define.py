from enum import Enum


class path_common(Enum):
    work = "C:\\Users\\DuongSau\\Sau\\chatbot1212"
    # work = "W:\\Sau\\Chatboot1211"
    model = work + "\\Model\\Save"
    data = work + "\\Model\\Data"
    mining = data + "\\Mining"
    classification = data + "\\IntentClassification"

    intent_list = mining + "\\intent_list.csv"
    sentence_list = mining + "\\sentence_list.csv"
    intent_group_list = mining + "\\intent_group_list.csv"

    intent = classification + "\\intent_list.csv"
    sentence = classification + "\\sentence_list.csv"
    intent_group = classification + "\\intent_group_list.csv"
    test = classification + "\\test.csv"
    train = classification + "\\train.csv"

    learn_data_pos = classification + "\\POS\\learn_data.csv"
    learn_data_neg = classification + "\\NEG\\learn_data.csv"
    learn_data_hed = classification + "\\HED\\learn_data.csv"

class tag(Enum):
    sau = "@id1211"
    page = "page"
    net_work_root = "div"
    classify1 = "classify1"
    h1 = "h1"
    h2 = "h2"
    p = "p"
    code = "code"


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
