class PathCommon:
    work = "C:\\Users\\Sau\\IdeaProjects\\chatbot1212"
    model = work + "\\Model\\CheckPoint"
    data = work + "\\Model\\Data"
    mining = data + "\\Mining"
    classification = data + "\\LabelClassification"

    label_list = mining + "\\Label_list.csv"
    sentence_list = mining + "\\sentence_list.csv"
    cluster_list = mining + "\\cluster_list.csv"
    answer_list = mining + "\\answer_list.csv"

    label = classification + "\\label_list.csv"
    sentence = classification + "\\sentence_list.csv"
    cluster = classification + "\\cluster_list.csv"
    test = classification + "\\test.csv"
    train = classification + "\\train.csv"
    answer = classification + "\\answer_list.csv"

    learn_data = classification + "\\Positive\\learn_data.csv"


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
