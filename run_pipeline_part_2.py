import sys

import pipeline_07_behaviour
import pipeline_08_epoch
import pipeline_09_epoch_qc
import pipeline_10_autoreject

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

pipeline_07_behaviour.run(index, json_file)
pipeline_08_epoch.run(index, json_file)
pipeline_09_epoch_qc.run(index, json_file)
pipeline_10_autoreject.run(index, json_file)