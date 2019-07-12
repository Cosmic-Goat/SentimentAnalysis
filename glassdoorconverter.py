import re
import pandas as pd

from cleantext import CleanText

def convert(df):
    to_return = pd.DataFrame()
    to_return["recommends"] = df.Recommends.replace({'Recommends': 1, "Doesn't Recommend": -1, None: 0})
    to_return["outlook"] = df.Outlook.replace({'Positive Outlook': 1, "Negative Outlook": -1, "Neutral Outlook": 0, None: 0})
    # to_return.Work_type = df.Employee_work_type.replace({})
    to_return["years"] = df.Years_at_company.fillna('0').replace({'more than a year': '1'}).apply(lambda text: re.sub("[^0-9]+", "", text))
    to_return["rating"] = df.Rating_overall.apply(lambda x: x - 3)

    # Clean up text data
    ct = CleanText()
    to_return["pros_clean"] = ct.fit_transform(df.Pros)
    to_return["cons_clean"] = ct.fit_transform(df.Cons)

    return to_return

