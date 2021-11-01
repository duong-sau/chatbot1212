class PathCommon:
    work = "D:\\chatbot1211"
    model = work + "\\Model\\CheckPoint"
    data = work + "\\Model\\Data"
    mining = data + "\\Mining"
    classification = data + "\\IntentClassification"

    intent_list = mining + "\\intent_list.csv"
    sentence_list = mining + "\\sentence_list.csv"
    intent_group_list = mining + "\\intent_group_list.csv"
    answer = mining + "\\answer_list.csv"

    intent = classification + "\\intent_list.csv"
    sentence = classification + "\\sentence_list.csv"
    intent_group = classification + "\\intent_group_list.csv"
    test = classification + "\\test.csv"
    train = classification + "\\train.csv"

    learn_data_pos = classification + "\\Positive\\learn_data.csv"
    learn_data_label = classification + "\\LabelClassification\\learn_data.csv"


class Tag:
    classify1 = "classify1"
    h2 = "h2"


class Colors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
