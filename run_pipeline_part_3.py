import sys
import matlab.engine
import pipeline_11_spm_conversion
import pipeline_12_epoch_average

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

parasite = matlab.engine.connect_matlab()

pipeline_11_spm_conversion.run(index, json_file, parasite)
pipeline_12_epoch_average.run(index, json_file, parasite)