import json
from utils import *


seeds = [960703545, 1277478588, 1936856304, 186872697, 1859168769, 1598189534, 1822174485, 1871883252, 694388766,
         188312339, 773370613, 2125204119, 2041095833, 1384311643, 1000004583, 358485174, 1695858027, 762772169,
         437720306, 939612284]

gamma = [0.1, 0.25, 0.5, 0.9, 0.99]
epsilon = [0.05, 0.1, 0.3, 0.5, 0.7]


def read(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


analysis_results("unit", "stochastic", gamma, epsilon, seeds)
analysis_results("weighted", "stochastic", gamma, epsilon, seeds)
analysis_results("weighted", "deterministic", gamma, epsilon, seeds)
analysis_results("unit", "deterministic", gamma, epsilon, seeds)
analysis_results3("weighted", "stochastic", [200, 600], seeds)
analysis_results3("unit", "deterministic", [200, 600, 1000], seeds)


