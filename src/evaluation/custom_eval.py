import pandas as pd

from test_pipeline.utils.general_util import read_json


def count_eval(data_file: str, field: str, filter: str):
    db = read_json(data_file)

    entities = [x for x in db if x[field] == filter]
    entities_count = len(entities)

    return entities, entities_count


def count_values_in_col(data_file: str, field: str, filter_value: str):
    df = pd.read_json(data_file)

    # filters if there is a filter value specified
    if filter_value:
        df = df[df[field] == filter_value]

    value_count = df[field].value_counts().to_dict()
    entities_count = df.shape[0]
    entities = df.to_dict("records")

    return entities, entities_count, value_count


if __name__ == "__main__":
    file_path = "/home/krasimir.angelov@nccdi.local/Code/LLM_Atlas/test_pipeline/data_files/15_08_25_dis_data_flat2.json"
    # count_eval(file_path, "originator", "Xander Bastin")
    # count_eval(file_path, "originator", "Charlie Yelland")
    # count_values_in_col(file_path, "ip_sensitivity", "confidential")
    # count_values_in_col(file_path, "originator", "")
