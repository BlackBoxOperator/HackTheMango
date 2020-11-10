""" strip the file name """

import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])
del df["image_id"]
df.to_csv(sys.argv[2], index=False)
