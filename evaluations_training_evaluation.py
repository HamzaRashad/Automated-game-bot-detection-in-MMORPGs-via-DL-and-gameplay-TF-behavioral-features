#v 1.14.5
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt

from detector import detector
Detector = detector()

HUMAN_REPLAY_PATH = "../parser/broodwar/csvsCORRECT/human_test"
BOT_REPLAY_PATH = "../parser/broodwar/csvsCORRECT/bot_test"
HUMAN_EVAL_PATH = "../evaluations/human_evals.csv"
BOT_EVAL_PATH = "../evaluations/bot_evals.csv"

def main():

    # CAUTION USING THIS FUNCTION CALL
    # Clears evaluation files for distinct runs of evaluator.py
    clear_eval_files()

    print("Recording predictions on human replay data...")
    recordAccuracy(HUMAN_EVAL_PATH, HUMAN_REPLAY_PATH, False)

    print("Recording predictions on bot replay data...\n")
    recordAccuracy(BOT_EVAL_PATH, BOT_REPLAY_PATH, True)

    evaluateModel()


# Calls predict in detector.py to determine its accuracy on a specific game replay csv
# for all replay files in specified path. Records model accuracy data in evaluation file.
def recordAccuracy(eval_path, replay_path, isBotEval):

    with open(eval_path, mode='w') as evals:
        eval_writer = csv.writer(evals, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # loops through replay CSVs, calls predict in detector.py, 
        # row format: prediction value (scale of 0-1), correctness flag (1=correct, 0=incorrect)
        for root, dir, files in os.walk(replay_path):
            for filename in files:
                file = replay_path + "/" + filename
                # print(filename)                 # UNCOMMENT THIS TO SEE PROGRESS THROUGH FILES

                # predict returns 1 if it is certain game data is bot data
                prediction = Detector.predict(file)
                if prediction != -1:
                    if isBotEval:
                        if prediction > .5:
                            eval_writer.writerow([prediction, '1'])
                        else:
                            eval_writer.writerow([prediction, '0'])
                    else:                        
                        if prediction <= .5:
                            eval_writer.writerow([prediction, '1'])
                        else:
                            eval_writer.writerow([prediction, '0'])

    evals.close()


def countCorrect(eval_path):
    correct = 0
    incorrect = 0

    # loop through recorded evalutaions and update accuracy counts
    with open(eval_path, 'r') as evals:
        reader = csv.reader(evals, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if int(row[1]) == 1:
                correct += 1
            else:
                incorrect += 1

    evals.close()

    return float(correct), float(incorrect)


# Reads both bot and human eval files to compute and output accuracy score
# 1 = correct prediction, 0 = incorrect prediction
def evaluateModel():

    # set true negatives to number of correct human IDs, false postives to incorrect IDs
    trueNegatives, falsePositives = countCorrect(HUMAN_EVAL_PATH)

    # set true positives to number of correct bot IDs, false negatives to incorrect IDs
    truePositives, falseNegatives = countCorrect(BOT_EVAL_PATH)

    # calculate f1 and f2 scores, avoiding divide by zero errors
    recall = truePositives / (truePositives + falseNegatives)

    if (int(truePositives) + int(falsePositives)) == 0:
        precision = 0
    else:
        precision = truePositives / (truePositives + falsePositives)
    if (precision + recall) == 0:
        f1 = 0
        f2 = 0
    else:
        f1 = 2 * (precision * recall)/(precision + recall)
        f2 = 5 * ((precision * recall) / ((precision*4) + recall))


    print("Human correct: " + str(trueNegatives))
    print("Human incorrect: " + str(falsePositives))
    print("Bot correct: " + str(truePositives))
    print("Bot incorrect: " + str(falseNegatives) + "\n")
    print('f1 = ' + str(f1))
    print('f2 = ' + str(f2))

    labels = ['Correct', 'Incorrect']
    botData = [truePositives, falseNegatives]
    humanData = [trueNegatives, falsePositives]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, botData, width, label='Bot')
    rects1 = ax.bar(x + width/2, humanData, width, label='Human')
    
    ax.set_ylabel('Count')
    ax.set_title('Model\'s Identifications')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()


def clear_eval_files():

    # Clears all csv eval entries
    file = open('evaluations/human_evals.csv', mode='w+')
    file.close()

    file = open('evaluations/bot_evals.csv', mode='w+')
    file.close()

main()




