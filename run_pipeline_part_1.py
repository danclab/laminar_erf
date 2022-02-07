import sys

import pipeline_01_raw_prep
import pipeline_02_adjust_events
import pipeline_03_ambient_noise
import pipeline_04_pp_noise
import pipeline_05_ica_selection

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

pipeline_01_raw_prep.run(index, json_file)
pipeline_02_adjust_events.run(index, json_file)
pipeline_03_ambient_noise.run(index, json_file)
pipeline_04_pp_noise.run(index, json_file)
pipeline_05_ica_selection.run(index, json_file)