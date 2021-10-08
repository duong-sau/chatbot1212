from enum import Enum


class path_common(Enum):
    work = "C:\\Users\\DuongSau\\Sau\\Chatboot1212"
    # work = "W:\\Sau\\Chatboot1211"
    model = work + "\\Model"
    data = model + "\\Data"
    mining = data + "\\Mining"
    classification = data + "\\IntentClassification"

    intent_list = mining + "\\intent_list.csv"
    sentence_list = mining + "\\sentence_list.csv"
    intent_group_list = mining + "\\intent_group_list.csv"

    intent = classification + "\\intent_list.csv"
    sentence = classification + "\\sentence_list.csv"
    intent_group = classification + "\\intent_group_list.csv"

    learn_data_pos = classification + "\\POS\\learn_data.csv"
    train_pos = classification + "\\POS\\train.csv"
    test_pos = classification + "\\POS\\test.csv"

    learn_data_neg = classification + "\\NEG\\learn_data.csv"
    train_neg = classification + "\\NEG\\train.csv"
    test_neg = classification + "\\NEG\\test.csv"

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
