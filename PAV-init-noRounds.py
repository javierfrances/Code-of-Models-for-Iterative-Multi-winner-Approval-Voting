import time
import sys
import re
import math
import copy
import random
from itertools import combinations
import numpy as np
import bisect

#global variable 
max_rounds = 40 #Max number of iterations allowed
threshold = 0.6 #For threshold heuristic
num_voting = 100 
initialUtilityArray = [] #Stores the initial utilities

def create_Approval_Votes_Spatial_Model(C, N, r, faction):
    A = []
    #Create |C| + |V| random points in a two dimensional Euclidean space [0,1]x[0,1] that represent voters and candidates
    candidatesInEuclideanSpace = []
    for cand in C:
        xPos = random.random()
        yPos = random.random()
        candidatesInEuclideanSpace.append([xPos, yPos])
    votersInEuclideanSpace = []    
    factionXpos = random.random()
    factionYpos = random.random()
    for voter in range(len(faction[0]), len(N)):
        xPos = random.random()
        yPos = random.random()
        votersInEuclideanSpace.append([xPos, yPos])
    #Every voter approves all candidates that are inside their radius
    factionAi = []
    for j in range(len(C)):
            candPosX = candidatesInEuclideanSpace[j][0]
            candPosY = candidatesInEuclideanSpace[j][1]
            #Calculate distance between candidate and voter:
            eucledianDistance = math.sqrt(math.pow((candPosX-factionXpos),2)+math.pow((candPosY-factionYpos),2))
            if(eucledianDistance <= r):
                factionAi.append(1)
            else:
                factionAi.append(0)
    for i in range(len(faction[0])):
        A.append(factionAi)
    for i in range(len(N)-len(faction[0])):
        Ai = []
        voterPosX = votersInEuclideanSpace[i][0]
        voterPosY = votersInEuclideanSpace[i][1]
        for j in range(len(C)):
            candPosX = candidatesInEuclideanSpace[j][0]
            candPosY = candidatesInEuclideanSpace[j][1]
            #Calculate distance between candidate and voter:
            eucledianDistance = math.sqrt(math.pow((candPosX-voterPosX),2)+math.pow((candPosY-voterPosY),2))
            if(eucledianDistance <= r):
                Ai.append(1)
            else:
                Ai.append(0)
        A.append(copy.copy(Ai))
   
    return A

def faction_Heuristic_Sequential_Change_Reactive(k, W, C, i, A, faction, initialA, f):
    '''faction is of type tuple: [[list of voters], [vote to change], [list of voters that have changed], [A_f when faction deviated], Expected Utiltity], initialized to:
    EG: [[1,2,3],[],[], [], 0]
    Hay que actualizar faction, y ver que cambia.
    '''
    currentUtilityFaction = calculate_Num_Selected_Approval(initialA[i],W,C)
    f.write("Current utility of the faction: " + str(currentUtilityFaction) + "\n")
    n = len(A)
    newA = copy.deepcopy(A)
    Ai = newA[i][:]
    newVote = newA[i][:]
    #If there is no vote to change to, look for a new one
    if (faction[1] == []):
        f.write("Faction variable: " + str(faction))
        #There is no previous vote to change to.

        #Calculate success probability: 1/sqr(faction size)
        size = len(faction[0]) #Faction[0] is a list of voters (Eg: [1,2]) that belong to the same faction as i
        successProb = 1/math.sqrt(size) 
        #If the faction is very big it might be good to use a threshold?
        f.write("SP: " + str(successProb) + "\n")
        #Calculate wehter voter i succeeds on convincing the whole faction.
        temp = random.randint(1,101)
        f.write("Value drawn: " + str(temp) + "\n")
        #If he does not, end function and return previous vote and winners.
        #if(temp > successProb*100):
         #   f.write("Voter " + str(i) + " can't convince the faction: " + str(faction) + "\n")
          #  return [newVote, False, W]

        #Voter i can convince faction       
        #Look for a new vote to change to:
        maxUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
        f.write("Previous utility: " + str(maxUtility) + "\n")
        changedVote = False #Variable that tells us wether voter i wants to change the vote
        newWinners = copy.deepcopy(W)
        #Otherwise seearch for a new vote, use interative_Through_Candidates_Single_Heuristic for now
        for candidate in C:      
            if(Ai[C.index(candidate)] == 1):
                #f.write("Candidate " + str(candidate) + " is approved by the faction." + "\n")
                newA = copy.deepcopy(A)
                #Change the approval for all the faction
                for j in faction[0]:
                    newA[j][C.index(candidate)] = 0
                #Simmulates PAV and calculates the new winners, but with the updated votes.
                #f.write(newA)
                res = seqPAV_sim(k, newA, C) 
                #f.write("This newA yields the winners: " + str(res[0]))
                newW = res[0]
                #Calculates the new utility with the new winners, but the original preferences. Utility for i is the same as the utility for 
                
                newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            else:
                newA = copy.deepcopy(A)
                for j in faction[0]:
                    newA[j][C.index(candidate)] = 1
                #Simmulates PAV and calculates the new winners, but with the updated vote
                res = seqPAV_sim(k, newA, C) 
                newW = res[0]
                #Calculates the new utility with the new winners, but the original preferences
                newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
                            
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                futureWinners = newW #Not actual real winners, as not the whole faction has changed its vote.
            
        #If there is a change: update faction[1] and faction[2]
        if(changedVote):
            #Calculate the new winners with voter i changing its vote
            f.write("New winners if all voters in the faction change: " + str(futureWinners) + "\n")
            tempA = copy.deepcopy(A)
            tempA[i] = newVote
            aux = seqPAV_sim(k, tempA, C)
            currentWinners = aux[0]
            f.write("Current winners if i is the only change: " + str(currentWinners) + "\n")
            faction[1] = newVote
            faction[2] = [i]
            faction[3] = []
            for voter in range(len(A)):
                if not voter in faction[0]:
                    faction[3].append(A[voter])
            faction[4] = maxUtility
            f.write("New faction variable:" + str(faction) + "\n")
            return [newVote, True, currentWinners]
        else:
            #There is no vote to change to, so faction[1] and faction[2] remains the same.
            f.write("Voter i has not found a new vote to change to.\n")
            return [Ai, False, W]
    else:
        if (faction[4] == -1):
            f.write("Faction wants to undo previous manipulation and change to vote: " + str(faction[1])  + "\n")
            if(A[i] == faction[1] and not i in faction[2]):
                f.write("Voter i did not previously deviate.\n")
                bisect.insort(faction[2], i)
                return [newVote, False, W]
            if (i in faction[2]):
                f.write("Voter i has already changed to this vote.\n")
                if(faction[0] == faction[2]):
                    f.write("Everyone has changed vote: " + str(faction) + "\n")
                    faction[1] = []
                    faction[2] = []
                    faction[3] = []
                    faction[4] = 0
                return [newVote, False, W]
            else:
                #Change vote to faction[1]
                newA[i] = faction[1]
                #Calculate new winners
                res = seqPAV_sim(k, newA, C) 
                newW = res[0]
                #Update faction[2]
                bisect.insort(faction[2], i)
                f.write("Voter has changed to vote, new faction var: " + str(faction) + "\n")
                return [faction[1], True, newW]
        f.write("Faction wants to change to vote: " + str(faction[1])  + "\n")
        #Check if there was a deviation by an outsider:
        currentA_f = []
        for voter in range(len(A)):
            if not voter in faction[0]:
                currentA_f.append(A[voter])
        if(currentA_f == faction[3]):
            f.write("There was no deviation: voter i switches to signaled vote"+ "\n")
            if (i in faction[2]):
                f.write("Voter i has already changed to this vote.\n")
                if(faction[0] == faction[2]):
                    f.write("Everyone has changed vote: " + str(faction) + "\n")
                    faction[1] = []
                    faction[2] = []
                    faction[3] = []
                    faction[4] = 0
                return [newVote, False, W]
            else:
                #Change vote to faction[1]
                newA[i] = faction[1]
                #Calculate new winners
                res = seqPAV_sim(k, newA, C) 
                newW = res[0]
                #Update faction[2]
                bisect.insort(faction[2], i)
                f.write("Voter has changed to vote, new faction var: " + str(faction) + "\n")
                return [faction[1], True, newW]
        elif(not i in faction[2]): #i has not changed to this new vote
            #Another voter (outsider) has deviated, check if signaled vote still yields an increase in utility:
            tempA = copy.deepcopy(A)
            for voter in range(len(A)):
                if voter in faction[0]:
                    tempA[voter] = faction[1]  #A if faction uses signaled vote
            res1 = seqPAV_sim(k, tempA, C) 
            newUtility1 = calculate_Num_Selected_Approval(initialA[i], res1[0], C)
            if(newUtility1 < currentUtilityFaction):

                #Outsider's deviation has lead to decrease in utility: undo previous change
                faction[1] = A[i]
                faction[2] = [i]
                faction[3] = []
                faction[4] = -1 #If faction[4] == -1 voters no to switch automatically to previous vote´
                #Remain with previous faction vote
                f.write("An outsider has deviated and decreased our utiltity, faction goes to previous vote: " + str(A[i]) + "\n")
                f.write("Expected winners if we do not undo the manipulation: "+ str(res1[0]) + "\n" )
                f.write("Expected utility if we do not undo the manipulation: "+ str(newUtility1) + "\n" )
                for voter in range(len(A)):
                    if voter in faction[0]:
                        tempA[voter] = A[i]
                res2 = seqPAV_sim(k, tempA, C) 
                newUtility2 = calculate_Num_Selected_Approval(initialA[i], res2[0], C)
                f.write("Expected winners if we all undo the manipulation: "+ str(res2[0]) + "\n" )
                f.write("Expected utility if we all undo the manipulation: "+ str(newUtility2) + "\n" )
                                    
                return [A[i], False, W]
            else:
                f.write("Outsider's deviation has not affected faction's utility"+ "\n") 
                #switch to signaled vote and update A_f + faction[4]
                newA[i] = faction[1]
                aux = seqPAV_sim(k, newA, C) 
                newW = aux[0]
                bisect.insort(faction[2], i)
                faction[3] = []
                for voter in range(len(A)):
                    if not voter in faction[0]:
                        faction[3].append(A[voter])
                faction[4] = newUtility1
                return[faction[1], True, newW]
        else: #i has already changed to the new vote
            f.write("Voter i has already changed to this vote.\n")
            if(faction[0] == faction[2]):
                f.write("Everyone has changed vote: " + str(faction) + "\n")
                faction[1] = []
                faction[2] = []
                faction[3] = []
                faction[4] = 0
            return [newVote, False, W]

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def PAV_sim(k, A, C):
    n = len(A) #Number of voters
    maxUtility = 0
    firstTime = True
    for combo in combinations(C,k):
        #Calculate the sum of all voter's utilities for this W
        #print('Combination: ' + str(combo))
        utility = 0.0
        for i in range(n): 
            utility += calculate_Utility(i, A, combo, C)
        #print('Utility for this combination: ' + str(utility) + '\n')
        if(firstTime):
            maxUtility = utility
            W = combo
            firstTime = False
        elif(utility > maxUtility):
            maxUtility = utility
            W = combo
    
    return [W, maxUtility] 

def intersection_W_and_Ai(W, Ai, C):
    '''Returns the number of candidates approved in Ai (have a 1) that are in the winner set'''
    res = 0
    for i in range(len(Ai)):
        if(Ai[i] == 1 and C[i] in W):
            res += 1
    return res

def seqPAV_sim(k, A, C):
    #print("Candidates: " + str(C))
    #print("A: " + str(A) + "\n")
    W = []
    n = len(A)
    for j in range(k):
        #print("Round: " + str(j) + "\n")
        highestApprovalWeight = 0
        highestApprovalWeightCand = 0 
        for cand in C:
            if cand not in W:
                #print("Candidate " + str(cand) + " not in W.")
                approvalWeight = 0
                for i in range(n):
                    #Calculate the approval weight for cand from each voter
                    if(A[i][C.index(cand)] == 1):
                        approvalWeight += 1/(1+intersection_W_and_Ai(W,A[i], C))
                #print("Approval Weight: " + str(approvalWeight) + "\n")
                if(approvalWeight > highestApprovalWeight):
                    highestApprovalWeight = approvalWeight
                    highestApprovalWeightCand = cand
        W.append(highestApprovalWeightCand)
        #print(str(W) + "\n")
    return [W, calculate_Total_Utility(W, C, A)]

def calculate_Utility(i, A, W, C):
    utility = 0
    Ai = A[i]
    lengthInt = 0
    for candidate in C:
        if(candidate in W and Ai[C.index(candidate)] == 1): 
            lengthInt += 1
    for j in range(lengthInt):
            utility += 1/(j+1)
    return utility


def calculate_Num_Selected_Approval(Ai, W, C):
    '''Calculates the number of candidates that a voter apporves, that have been selected in the winner committee.'''
    lengthInt = 0
    for candidate in C:
        if(candidate in W and Ai[C.index(candidate)] == 1): 
            lengthInt += 1
    return lengthInt

def calculate_Num_Approvals_For_Candidate(j, A):
    '''Calculates how many voters approve a candidate (position j in candidate list) in a voting profile A.'''
    res = 0
    for vote in A:
        res += vote[j]
    return res

def justified_Representation(W, A, k):
    #Does W provide a justified representation for (A, k)?
    n = len(A)
    for i in range(math.ceil(n/k), n+1):
        for newSet in combinations(A, i):
            #print('N* = ' + str(newSet) + ' with |N*| = ' + str(i))
            intersectionSet = newSet[0]
            #calculate intersection of the new set of voters
            for voter in newSet:
                intersectionSet = [a*b for a,b in zip(voter,intersectionSet)]
            #If intersection of voters is different to 0
            if(1 in intersectionSet):
                #print('∩Ai ≠ {}, for all i that belong to N*: ' + str(intersectionSet))
                #For each voter in the new set see if the intersecition with W is 0 ==> calculate_Num_Selected_Approval(i, A, W, C) = 0
                num_of_candidates_from_newSet_in_W = 0
                for Ai in newSet:
                    #print(calculate_Num_Selected_Approval(Ai, W, C))
                    num_of_candidates_from_newSet_in_W += calculate_Num_Selected_Approval(Ai, W, C)
                #print('Total number of approved candidates of N* in W: ' + str(num_of_candidates_from_newSet_in_W) + '\n')
                if(not num_of_candidates_from_newSet_in_W):
                    print('W does not provide a justified representation for (A,k)')
                    return    

def trial_And_Error_Heuristic(k, W, C, i, A, initialA, f):
    ''''''
    currentUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
    if(initialUtilityArray[i] == 0): #First round ==> store inital utility
        initialUtilityArray[i] = currentUtility
    else:    
        if(currentUtility < initialUtilityArray[i] and A[i] != initialA[i]):
            #If there has been a deviation, and utility has gone down, then go back to original vote.
            tempA = copy.deepcopy(A)
            tempA[i] = initialA[i]
            tempRes = seqPAV_sim(k, tempA, C)
            if(currentUtility < calculate_Num_Selected_Approval(initialA[i], tempRes[0], C)):
                f.write("By going back to the original vote, utility has increased!\n")
                f.write("New winners: " + str(tempRes[0]))
                #exit()
            return [initialA[i], True, tempRes[0]]
    Ai = A[i][:]
    maxUtility = currentUtility
    newWinners = W[:]
    changedVote = False #Variable that tells us wether voter i wants to change the vote

    #Randomly choose a candidate:
    candidatePos = random.randrange(0, len(C), 1)
    f.write("Candidate chosen: " + str(C[candidatePos]) + " has approval value of: " + str(Ai[candidatePos]) + "\n")

    #Invert their approval:
    if(Ai[candidatePos] == 1):
        newA = copy.deepcopy(A)       
        newA[i][candidatePos] = 0
    else:
        newA = copy.deepcopy(A)
        newA[i][candidatePos] = 1

    #Simulate PAV with the new vote.
    res = seqPAV_sim(k, newA, C)
    newW = res[0]
    #Calculates the new utility with the new winners, but the original preferences.
    newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
    #If the new utility is not smaller than the previous utility, change the vote.
    if(newUtility > maxUtility):
        f.write("Changed vote to: " + str(newA[i]) + "\n")
        return [newA[i], True, res[0]]
    #If it is, then return the previous vote and winners.
    f.write("Remained with the same vote: " + str(Ai) + "\n")
    return [Ai, False, W]


def interative_Through_Candidates_Single_Heuristic(k, W, C, i, A, initialA, f):
    '''For each candidate, the voter i checks wether its utility stricktly increases (with respect to the original preferences) by inverting 
    the approval value for said candidate. Only one single approval value can be changed. This is a variation for the trial-and-error heuristic, 
    which does the same but only for one deviation randomly drawn. Returns a tuple consisting of i's new vote, a variable thart states wether 
    the voter i has decided to change its vote and the new winners resulting of that change.'''
    Ai = A[i][:]
    newVote = Ai[:]
    maxUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
    newWinners = W[:]
    changedVote = False #Variable that tells us wether voter i wants to change the vote
    currentUtility = maxUtility
    supersetmanipulation = False
    for candidate in C:            
        if(Ai[C.index(candidate)] == 1):
            newA = copy.deepcopy(A)       
            newA[i][C.index(candidate)] = 0
            #Simmulates PAV and calculates the new winners, but with the updated vote
            res = seqPAV_sim(k, newA, C) 
            newW = res[0]
            #Calculates the new utility with the new winners, but the original preferences
            newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                newWinners = newW  
                supersetmanipulation = False 
        else:
            newA = copy.deepcopy(A)
            newA[i][C.index(candidate)] = 1
            #Simmulates PAV and calculates the new winners, but with the updated vote
            res = PAV_sim(k, newA, C) 
            newW = res[0]
            #Calculates the new utility with the new winners, but the original preferences
            newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
           
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                newWinners = newW
                supersetmanipulation = True
    if(changedVote and supersetmanipulation):
        f.write("Utility has increased from " + str(currentUtility) + " to " + str(maxUtility) + ", if voter deviates to: " + str(newVote) + "\n")
        f.write("New winners after deviation: " + str(newWinners) + "\n")
        f.write("Initial preferences: " + str(initialA[i]) + "\n")
        f.write("Previous voting profile A: " + str(A) + "\n")
        #exit()
    
    return [newVote, changedVote, newWinners]

def interative_Through_Candidates_K_Heuristic(k, W, C, i, A, initialA, f, maxChanges):
    '''A variant of interative_Through_Candidates_Single_Heuristic, but the voter can make up to maxChanges changes in their vote (instead of a
    single one). But this maxChanges value can never be higher than len(C). Returns a tuple consisting of i's new vote, a variable thart states 
    wether the voter i has decided to change its vote and the new winners resulting of that change.'''
    Ai = A[i][:]
    newVote = Ai[:]
    maxUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
    newWinners = W[:]
    changedVote = False #Variable that tells us wether voter i wants to change the vote
    currentUtility = maxUtility
    if(maxChanges>len(C)):
        print("maxChanges value can never be higher than len(C)")
        #exit()
    numberOfChanges = 0
    for j in range(maxChanges):
        possibleChanges = j + 1
        #Iterates throuhg the possible deviations changing j+1 votes:
        combs = combinations(C,possibleChanges)
        for combo in combs:
            #print(combo)
            newA = copy.deepcopy(A)       
            for candidate in combo:
                if(Ai[C.index(candidate)] == 1):
                    newA[i][C.index(candidate)] = 0
                else:
                    newA[i][C.index(candidate)] = 1
            #print("New Ai:      " + str(newA[i]))
            #Simulate PAV using the new A
            res = seqPAV_sim(k, newA, C) 
            newW = res[0]
            #Calculates the new utility with the new winners, but the original preferences (initialA[i])
            newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                numberOfChanges = possibleChanges
                newVote = newA[i]
                newWinners = newW
                f.write(str(combo)+"\n")
    if(changedVote and numberOfChanges > 1):
        f.write("Utility has increased from " + str(currentUtility) + " to " + str(maxUtility) + ", if voter deviates to: " + str(newVote) + "\n")
        f.write("Changes: " + str(numberOfChanges))
        f.write("New winners after deviation: " + str(newWinners) + "\n")
        f.write("Initial preferences: " + str(initialA[i]) + "\n")
        f.write("Previous voting profile A: " + str(A) + "\n")
        #exit()
    return [newVote, changedVote, newWinners]
            
def threshold_Heuristic_First(k, W, C, i, A, initialA, f):
    '''The voter i disapproves the first candidate that he/she finds with a percentage of approval above a certain threshold, as said candidate
    will probably be elected as winners even when i does not approve them. Returns a tuple consisting of i's new vote, a variable that states 
    wether the voter i has decided to change its vote and the new winners resulting of that change.'''
    n = len(A)
    newA = copy.deepcopy(A)
    newVote = newA[i][:]   
    j = -1
    for candidate in C: 
        j += 1
        #print(str(i) + " voted " + str(newVote[j]) + " for candidate " + candidate)
        if(newVote[j] == 1):
            number_of_approvals = calculate_Num_Approvals_For_Candidate(j, A)
            if(number_of_approvals/n >= threshold):
                newVote[j] = 0
                newA[i][j] = 0
                f.write("Candidate above threshold found!\n")
                #print("New A: " + str(newA))
                #print("New vote: " + str(newVote))
                #New winners are calculated.
                res = seqPAV_sim(k, newA, C)
                #print("New winners: " + str(res[0]))
                return [newVote, True, res[0]]
    f.write("No candidate above threshold found.\n")
    #print("Same vote: " + str(newVote))
    return [newVote, False, W]

def threshold_Heuristic_Highest_Votes(k, W, C, i, A, initialA, f):
    '''The voter i disapproves the candidate with the highest votes if it has a percentage of approval above a certain threshold and it is in
    the winner committee, as said candidate will probably be elected as winners even when i does not approve them. Returns a tuple consisting
    of i's new vote, a variable that states whether the voter i has decided to change its vote and the new winners resulting of that change.'''
    
    currentUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
    n = len(A)
    newA = copy.deepcopy(A)
    newVote = newA[i][:]
    maxpos = calculate_Highest_Scoring_Approved_Candidate(W, C, newA, i, f)
    if(maxpos == -1):
        f.write("No candidates can be disapproved.\n")
        return [newVote, False, W]
    elif(votesCandidates[maxpos]/n >= threshold and sum(A[i]) > 1):
        newVote[maxpos] = 0
        newA[i][maxpos] = 0
        f.write("Most approved candidate " + C[maxpos] + " does pass the threshold of " + str(threshold) + ".\n")
        #f.write("New A: " + str(newA))
        #f.write("New vote: " + str(newVote))
        res = seqPAV_sim(k, newA, C)
        #f.write("New winners: " + str(res[0]))
        newUtility = calculate_Num_Selected_Approval(initialA[i], res[0], C)
        if(newUtility < currentUtility and votesCandidates[maxpos] == 2):
            f.write("Utility has decreased from " + str(currentUtility) + " to " + str(newUtility) + ", if voter deviates to: " + str(newVote) + "\n")
            f.write("New winners after deviation: " + str(res[0]) + "\n")
            f.write("Initial preferences: " + str(initialA[i]) + "\n")
            f.write("Previous voting profile A: " + str(A) + "\n")
        return [newVote, True, res[0], res[1]]
    else:
        f.write("Most approved candidate " + C[maxpos] + " does not pass the threshold.\n")
        return [newVote, False, W]
    
def calculate_Highest_Scoring_Approved_Candidate(W, C, A, i, f):
    '''This method finds the candidate approved by a voter i with the highest number of votes in the winner committee. 
    Returns the position of said candidate in the candidate list. -1 is returned if the voter approves no candidates.'''
    #Calculate the approvals each candidate got:
    Atemp = np.array(A)
    votesCandidates = np.sum(Atemp, 0)
    maxvotes = 0
    maxpos = -1
    f.write("Total votes for candidates: " + str(votesCandidates)+ "\n")
    for candidate in W:
        j = C.index(candidate)
        if(A[i][j] == 1):
            if (votesCandidates[j]>maxvotes):
                maxpos = j
                maxvotes = votesCandidates[j]
    if(maxpos != -1):
        f.write("Max voted candidate approved (in the winner set) by " + str(i) + " is " + str(C[maxpos]) + " with " + str(maxvotes) + " votes.\n")
    return maxpos


def faction_Heuristic_Sequential_Change(k, W, C, i, A, faction, initialA, f):
    '''faction is of type tuple: [[list of voteres], [vote to change], [list of voters that have changed]], initialized to:
    EG: [[1,2,3],[],[]]
    Hay que actualizar faction, y ver que cambia.
    '''
    n = len(A)
    newA = copy.deepcopy(A)
    Ai = newA[i][:]
    newVote = newA[i][:]
    if (faction[1] == []):
        f.write("Faction variable: " + str(faction))
        #There is no previous vote to change to.

        #Calculate success probability: 1/sqr(faction size)
        size = len(faction[0]) #Faction[0] is a list of voters (Eg: [1,2]) that belong to the same faction as i
        successProb = 1/math.sqrt(size) 
        #If the faction is very big it might be good to use a threshold?
        f.write("SP: " + str(successProb) + "\n")
        #Calculate wehter voter i succeeds on convincing the whole faction.
        temp = random.randint(1,101)
        f.write("Value drawn: " + str(temp) + "\n")
        #If he does not, end function and return previous vote and winners.
        #if(temp > successProb*100):
         #   f.write("Voter " + str(i) + " can't convince the faction: " + str(faction) + "\n")
          #  return [newVote, False, W]

        #Voter i can convince faction       
        #Look for a new vote to change to:
        maxUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
        f.write("Previous utility: " + str(maxUtility) + "\n")
        changedVote = False #Variable that tells us wether voter i wants to change the vote
        newWinners = copy.deepcopy(W)
        #Otherwise seearch for a new vote, use interative_Through_Candidates_Single_Heuristic for now
        for candidate in C:      
            if(Ai[C.index(candidate)] == 1):
                #f.write("Candidate " + str(candidate) + " is approved by the faction." + "\n")
                newA = copy.deepcopy(A)
                #Change the approval for all the faction
                for j in faction[0]:
                    newA[j][C.index(candidate)] = 0
                #Simmulates PAV and calculates the new winners, but with the updated votes.
                #f.write(newA)
                res = seqPAV_sim(k, newA, C) 
                #f.write("This newA yields the winners: " + str(res[0]))
                newW = res[0]
                #Calculates the new utility with the new winners, but the original preferences. Utility for i is the same as the utility for 
                
                newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            else:
                newA = copy.deepcopy(A)
                for j in faction[0]:
                    newA[j][C.index(candidate)] = 1
                #Simmulates PAV and calculates the new winners, but with the updated vote
                res = seqPAV_sim(k, newA, C) 
                newW = res[0]
                #Calculates the new utility with the new winners, but the original preferences
                newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
                            
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                futureWinners = newW #Not actual real winners, as not the whole faction has changed its vote.
            
        #If there is a change: update faction[1] and faction[2]
        if(changedVote):
            #Calculate the new winners with voter i changing its vote
            f.write("New winners if all voters in the faction change: " + str(futureWinners) + "\n")
            tempA = copy.deepcopy(A)
            tempA[i] = newVote
            aux = seqPAV_sim(k, tempA, C)
            currentWinners = aux[0]
            f.write("Current winners if i is the only change: " + str(currentWinners) + "\n")
            faction[1] = newVote
            faction[2] = [i]
            f.write("New faction variable:" + str(faction) + "\n")
            f.write(str(tempA))
            print(str(faction[0])+" ha cambiado de voto!")
            return [newVote, True, currentWinners]
        else:
            #There is no vote to change to, so faction[1] and faction[2] remains the same.
            f.write("Voter i has not found a new vote to change to.\n")
            return [Ai, False, W]
    else:
        f.write("Faction wants to change to vote: " + str(faction[1])  + "\n")
        if (i in faction[2]):
            f.write("Voter i has already changed to this vote.\n")
            if(faction[0] == faction[2]):
                f.write("Everyone has changed vote: " + str(faction) + "\n")
                faction[1] = []
                faction[2] = []
            return [newVote, False, W]
        #Change vote to faction[1]
        newA[i] = faction[1]
        #Calculate new winners
        res = seqPAV_sim(k, newA, C) 
        newW = res[0]
        #Update faction[2]
        bisect.insort(faction[2], i)
        f.write("Voter has changed to vote, new faction var: " + str(faction) + "\n")
        return [faction[1], True, newW]
        


def voting_iteration(k, A, W, C, maxRounds, heuristics, N, factions, f, deviations):
    totalDeviations = 0
    totalIterations = 0
    
    initialA = copy.deepcopy(A)
    Atemp = A[:]
    Wtemp = W[:]
    f.write("Initial A: " + str(initialA) + "\n")
    f.write("Initial W: " + str(Wtemp) + "\n")
    n = len(A)
    equilibrium = True   
    hasChangedList = [0]*n #If there is a 1 in pos i: Voter i does not want to change its vote. List must be updated to 0s whenther is a change.
    #If after one iteration, all positions have a 1 ==> we have found an equilibrium.         
    for j in range(maxRounds):
        totalIterations += 1
        i = random.choice(N)
        #Atemp and Wtemp need to be updated after each voter changes their vote
        f.write("\nVoter " + str(i) + "'s turn:"+ "\n")
        f.write("---------------"+ "\n")
        f.write("Previous winners: " + str(Wtemp)+ "\n")
        previousVote = Atemp[i][:]
        f.write("Previous vote: " + str(previousVote)+ "\n")            
        if(heuristics[i] == 0):
            f.write("Apply Lazy heuristic."+ "\n")
            changedVote = False
        elif(heuristics[i] == 1):
            f.write("Apply Threshold heuristic."+ "\n")
            currentA = copy.deepcopy(Atemp)
            res = threshold_Heuristic_Highest_Votes(k, Wtemp, C, i, currentA, initialA, f)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2] 
        elif(heuristics[i] == 2):
            f.write("Apply Iterative Single heuristic."+ "\n")
            currentA = copy.deepcopy(Atemp)
            res = interative_Through_Candidates_K_Heuristic(k, Wtemp, C, i, currentA, initialA, f, 1)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2]
        elif(heuristics[i] == 3):
            f.write("Apply T-and-E heuristic."+ "\n")
            currentA = copy.deepcopy(Atemp)
            res = trial_And_Error_Heuristic(k, Wtemp, C, i, currentA, initialA, f)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2]
        else:
            f.write("Apply Factional Reactive heuristic."+ "\n")
            #We need to find the faction of voter i:
            for faction in factions:
                if i in faction[0]:
                    break
            currentA = copy.deepcopy(Atemp)
            res = faction_Heuristic_Sequential_Change_Reactive(k, Wtemp, C, i, currentA, faction, initialA, f)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2]
        if(changedVote):
            totalDeviations += 1
            Atemp[i] = newVote[:]
            #Change the equilibrium variable as for this round there is no equilibrium (a voter has changed its vote)
            #If there exists a new vote with a higher utility change equilibrium to False
            equilibrium = False
            #Update Atemp:
            Wtemp = newWinners[:]
            f.write("Voter " + str(i) + " has changed its vote from: " + str(previousVote) + ", to: " + str(newVote)+ "\n")
            f.write("New A: " + str(Atemp)+ "\n")
            f.write("New W: " + str(Wtemp)+ "\n")
            hasChangedList = [0]*n
            deviations[i] += 1
        else:
            #Voter i didn't change his vote:
            f.write("Voter " + str(i) + " has not changed its vote."+ "\n")
            hasChangedList[i] = 1
            #Atemp[i] = newVote[:]
            #f.write("A:" + str(Atemp))
            #f.write("W:" + str(Wtemp) + "\n")

        if(not 0 in hasChangedList):
            #If no voter wants to change their vote, we have reached an equilibrium and there is no need for further voting iterations.
            f.write("\nEquilibrium found!"+ "\n")
            return [Wtemp, Atemp, totalDeviations, totalIterations]

    f.write("\nLimit of rounds has been reached. Winners from final round: " + str(Wtemp)+ "\n")
    return [Wtemp, Atemp, totalDeviations, totalIterations]

def calculatePAVScore(W, C, A):
    res = 0.0
    n = len(A)
    for i in range(n):
        res += calculate_Utility(i, A, W, C)
    return res

def calculate_Total_Utility(W, C, A):
    res = 0
    n = len(A)
    for i in range(n):
        res += calculate_Num_Selected_Approval(A[i], W, C)
    return res

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_Approval_Votes_Normal_Dist(C, N, factions, factionalVoters, mean, sd):
    res = [[0 for z in range(len(C))] for j in range(n)]
    if (not factions == []):
        for fact in factions:
            faction = fact[0]
            #How many candidates the faction approves:
            value = np.random.normal(mean, sd)
            if(value < 1):
                value = 1
            elif(value > 10):
                value = 10    
            approvedCands = round((value/10)*len(C))
            #Select the j random candidates approved by the faction
            candidatesApproved = random.sample(C, approvedCands)
            vote = [0 for z in range(len(C))]
            for z in candidatesApproved:
                vote[C.index(z)] = 1
            for voter in faction:
                res[voter] = vote
    for i in range(factionalVoters, len(N)):
        #How many candidates does he approve:
        value = np.random.normal(mean, sd)
        if(value < 1):
            value = 1
        elif(value > 10):
            value = 10    
        approvedCands = round((value/10)*len(C))
        #Select the j random candidates approved by the voter
        candidatesApproved = random.sample(C, approvedCands)
        for z in candidatesApproved:
            res[i][C.index(z)] = 1
    return res 

if __name__ == "__main__":

    if(sys.argv[1] == '' or sys.argv[2] == ''):
        exit()
    f=open(sys.argv[1], "a+")
    random.seed(sys.argv[2])
    f.write("RANDOM SEED ==> " + str(sys.argv[2]) + "\n\n")
    print("\nStarting Simulation")
    print("-------------------\n")
    n = int(input("Enter number of voters (n): "))
    k = int(input("Enter committe size (k): "))
    C = []
    lenC = int(input("Enter number of candidates (C): "))
    for l in range(lenC):
        C.append("c" + str(l))
    print("Candidate List: " + str(C))
    max_rounds = int(input("\nEnter maximum number of rounds per game: "))
    num_voting = int(input("Enter number of simulations: "))
    threshold = float(input("Enter threshold for the heuristic [0,1]: "))
    areFactions = input("\nAre there factions? (y/n) ")
    factions = []
    heuristics = []
    factionalVoters = 0
    counter = 0
    
    if(areFactions == 'y'):
        numFactions = int(input("How many? "))
        for i in range(numFactions):
            numVotersInFaction = int(input("Enter number of voters in faction " + str(i) + ": "))
            fact = [list(range(counter, counter + numVotersInFaction))]
            fact.append([]) #list where the vote to change to of the faction is stored.
            fact.append([]) #list of voters that have changed to that vote.
            fact.append([]) #list of votes from outsiders
            fact.append(0) #Expected future utility
            factions.append(fact)
            counter = counter + numVotersInFaction
            heuristics.extend([4 for j in range(numVotersInFaction)])
            factionalVoters += numVotersInFaction
    print(factions)
    if(factionalVoters > n):
        print("ERROR THERE CAN'T BE MORE FACTIONAL VOTERS THAN N!")
        exit()
    
    print("\nHEURISTICS:")
    lazyVoters = int(input("Enter number of lazy voters (" + str(n-(factionalVoters)) +" remaining): "))
    thresholdVoters = int(input("Enter number of threshold voters (" + str(n-(factionalVoters+lazyVoters)) +" remaining): "))
    iterativeVoters = int(input("Enter number of iterative voters (" + str(n-(factionalVoters+lazyVoters+thresholdVoters)) +" remaining): "))
    tandeVoters = int(input("Enter number of T-and-E voters (" + str(n-(factionalVoters+lazyVoters+thresholdVoters+iterativeVoters)) +" remaining): "))
    print()
    voters = factionalVoters+lazyVoters+thresholdVoters+iterativeVoters+tandeVoters

    if(voters > n):
        print("ERROR THERE CAN'T BE MORE VOTERS THAN N!")
        exit()
    
    heuristics.extend([0 for j in range(lazyVoters)])
    heuristics.extend([1 for j in range(thresholdVoters)])
    heuristics.extend([2 for j in range(iterativeVoters)])
    heuristics.extend([3 for j in range(tandeVoters)])

    if(voters < n):
        heuristics.extend([0 for j in range(voters, n)])
    N = [j for j in range(n)]
    f.write("N: " + str(N) + "\n")
    f.write("Factions: " + str(factions) + "\n")
    f.write("Hueristics: " + str(heuristics) + "\n\n")

    totalFinalSW = 0
    totalInitialSW = 0
    totalIterations = 0
    totalDeviations = 0
    totalNumberOfGamesWithSameWC = 0
    totalNumberOfGamesWithDifferentWC = 0
    totalNumberOfGamesWithSameFinalSW = 0
    totalNumberOfGamesWithLessFinalSW = 0
    totalNumberOfGamesWithMoreFinalSW = 0 

    maxDPoA = 0
    maxDPoARound = 0
    startTime = time.time()
    printProgressBar (0, num_voting)
    for test in range(1,num_voting+1):
        deviations = [0 for i in range(n)]
        initialUtilityArray = [0 for i in range(n)]
        f.write("===============================\n")
        f.write(" NEW PAV SIMULATION, NUMBER: " + str(test)+ "\n")
        f.write("===============================\n\n")
        #Create a random A, remember voters that belong to the same faction have the same approval ballot:
        aux = []
        A = create_Approval_Votes_Spatial_Model(C, N, 0.6, factions[0])
        f.write('Initial state of the game:'+ "\n")
        f.write('--------------------------'+ "\n")
        f.write("Initial A: " + str(A)+ "\n")    
        
        temp = seqPAV_sim(k, A, C)
        W = temp[0]
        maxUtility = temp[1]
        #Calculate the approvals each candidate got:
        Atemp = np.array(A)
        votesCandidates = np.sum(Atemp, 0)
        f.write('Candidates: ' + str(C)+ "\n")
        f.write('Initial Winners: ' + str(W)+ "\n")
        f.write('Approvals for each candidate: ' + str(votesCandidates)+ "\n")
        f.write('Total PAV Score for this W: ' + str(maxUtility)+ "\n\n")
        initialW = copy.copy(W)
        initialA = copy.deepcopy(A)
        initialPAVScore = copy.copy(maxUtility)
        initialTotalUtility = calculate_Total_Utility(W, C, initialA)
        #justified_Representation(W, A, k)
        for i in range(n):
            initialUtilityArray[i] = calculate_Num_Selected_Approval(initialA[i], W, C)
        #voting_iteration(k, A, W, C, 5, heuristics)
        f.write("*******************"+ "\n")
        f.write("    Iterations:"+ "\n")
        f.write("*******************"+ "\n\n")
        outcome = voting_iteration(k, A, W, C, max_rounds, heuristics, N, factions,f, deviations)
        #Calculate the PAVScore for this new outcome, but considering the original votes.
        f.write("\n*************************"+ "\n")
        f.write("     Final Results:      "+ "\n")
        f.write("*************************"+ "\n\n")

        f.write("Initial Winners: " + str(initialW)+ "\n")
        f.write("Final Winners:   " + str(outcome[0])+ "\n\n")
        
        f.write("Initial A: " + str(initialA)+ "\n")
        f.write("Final A:   " + str(outcome[1])+ "\n\n")

        f.write("Initial Total Utility: " + str(initialTotalUtility)+ "\n")
        #Calculate the PAVScore for the new set of winners, but considering the original votes.
        finalTotalUtility = calculate_Total_Utility(outcome[0], C, initialA)
        f.write("Final Total Utility:   " + str(finalTotalUtility)+ "\n")
        dPoA = initialTotalUtility/finalTotalUtility
        f.write("Dynamic PoA = (Initial Total Utility)/(Final Total Utility) = " + str(dPoA)+ "\n\n")
        
        totalFinalSW += finalTotalUtility
        totalInitialSW += initialTotalUtility
        totalDeviations += outcome[2]
        totalIterations += outcome[3]

        if(finalTotalUtility == initialTotalUtility):
            totalNumberOfGamesWithSameFinalSW += 1
        elif(finalTotalUtility > initialTotalUtility):
            totalNumberOfGamesWithMoreFinalSW += 1
        else:
            totalNumberOfGamesWithLessFinalSW += 1

        if(initialW == outcome[0]):
            totalNumberOfGamesWithSameWC += 1
        else:
            totalNumberOfGamesWithDifferentWC += 1

        if(dPoA > maxDPoA):
            maxDPoA = dPoA
            maxDPoARound = test
        printProgressBar (test, num_voting)

        finalUtilities = [calculate_Num_Selected_Approval(initialA[i], outcome[0], C) for i in range(n)]
        #FACTION DATA:

        f.write("Faction Initial Utility: " + str(initialUtilityArray[0]) + "\n")
        f.write("Faction Final Utility:   " + str(finalUtilities[0]) + "\n")

        f.write("Initial Utilities: " + str(initialUtilityArray) + "\n")
        f.write("Final Utilities:   " + str(finalUtilities) + "\n")

        for i in range(n):
            if(initialUtilityArray[i] > finalUtilities[i] and deviations[i] > 0):
                f.write("FOUND! Decrease in utility!\n")
                
        
    f.write("============================================\n")
    f.write(" WORST ROUND: " + str(maxDPoARound) + ", with a Dynamic PoA of: " + str(maxDPoA)+ " \n")
    f.write(" Total number of deviations: " + str(totalDeviations) + "\n")
    f.write(" Total number of iterations: " + str(totalIterations) + "\n")
    f.write(" Total final SW:" + str(totalFinalSW) + "\n")
    f.write(" Total intial SW: " + str(totalInitialSW) + "\n")
    f.write(" Games with same WC: " + str(totalNumberOfGamesWithSameWC) + "\n")
    f.write(" Games with different WC: " + str(totalNumberOfGamesWithDifferentWC) + "\n")
    f.write(" Games with same SW: " + str(totalNumberOfGamesWithSameFinalSW) + "\n")
    f.write(" Games with lower SW: " + str(totalNumberOfGamesWithLessFinalSW) + "\n")
    f.write(" Games with more SW: " + str(totalNumberOfGamesWithMoreFinalSW) + "\n")
    f.write("============================================\n")

    endTime = time.time()
    print("\nTime elapsed: " + str(endTime - startTime) + " seconds.")
    f.write("\nTime elapsed: " + str(endTime - startTime) + " seconds.")
    f.close()


    '''
    #EXAMPLE 1:
    A = [[0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]]
    temp = PAV_sim(4, A, C)
    print("Initial Winners: " + str(temp[0]))
    aux = interative_Through_Candidates_Single_Heuristic(4, temp[0], C, 4, copy.deepcopy(A))
    print("New Winners:     " + str(aux[2]))
    print("Initial PAVScore: " + str(temp[1]))
    print("Final PAVScore:   " + str(calculatePAVScore(aux[2], C, A)))
    '''
    '''
    #EXAMPLE 2:
    C = ['Felix', 'Javier', 'David', 'Rafa', 'Carmela', 'Maria', 'Roberto'] #Candidates
    A = [[1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1, 1]]
    #Previous winners: ('Javier', 'David', 'Rafa', 'Carmela', 'Maria', 'Roberto')
    k = 6
    factions = [[0,1]]
    res1 = PAV_sim(k, A, C)
    aux = faction_Heuristic(k, res1[0], C, 1, A, factions[0])
    print("\nPrevious winners: " + str(res1[0]))
    print("New Winners:     " + str(aux[2]) + "\n")
    print("Previous Vote: " + str(A[0]))
    print("New Vote:     " + str(aux[0]) + "\n")
    print("Initial PAVScore: " + str(res1[1]))
    print("Final PAVScore:   " + str(calculatePAVScore(aux[2], C, A)) + "\n")
    '''