
import json
import random
from typing import List
from utils import Provider
from ProviderAgent import process_providers

input_path = 'C:\\testProjects\\my-ai-agent\\06-ReAct-agent\\email_arin.json'
def loadRawData() -> List[Provider]:
    solutions = []

    # get solution info from json
    with open(input_path, encoding='utf-8') as f:
        solutions = json.load(f)
    print(len(solutions))
    # random select 20 solutions
    solutionsSample = random.sample(solutions, 20)

    res: List[Provider] = []
    for s in solutionsSample:
        solution = Provider(
            internalId=s['internalId'],
            providerName=s['solutionName'],
            asn=s['asn'],
            emails=s['emails'],
        )
        res.append(solution)

    return res

solutions = loadRawData()
print(solutions)
res = []

for solution in solutions:
    updatedSolution = process_providers(solution)
    res.append(updatedSolution)

print(res)