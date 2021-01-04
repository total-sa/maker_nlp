import pandas as pd

def get_target_rows_index(dataframe: pd.DataFrame) -> [int]:
    dataframe['PhraseLength'] = dataframe['Phrase'].str.len()
    target_rows_index = []
    for sentence_id in dataframe.SentenceId.unique():
        sentence_df = dataframe[dataframe.SentenceId == sentence_id]
        paragraph_length = sentence_df.PhraseLength.max()
        target_row = sentence_df[sentence_df.PhraseLength == paragraph_length]
        target_row_index = target_row.index.values[0]
        target_rows_index.append(target_row_index)
    return target_rows_index

def get_paragraph_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    target_rows_index = get_target_rows_index(dataframe)
    target_df = dataframe[dataframe.index.isin(target_rows_index)]
    target_df = target_df[['Phrase', 'Sentiment']]
    target_df = target_df.reset_index().drop(columns=['index'])
    return target_df
