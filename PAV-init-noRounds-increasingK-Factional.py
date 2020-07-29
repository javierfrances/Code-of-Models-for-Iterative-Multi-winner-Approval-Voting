import time
import sys
import re
import math
import copy
import random
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import bisect

#Global variables 
max_rounds = 40 #Max number of iterations allowed
threshold = 0.6 #For threshold heuristic
num_voting = 100 
initialUtilityArray = [] #Stores the initial utilities



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
            utility += calculate_PAV_Utility(i, A, combo, C)
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
        highestApprovalWeight = -1
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

def calculate_PAV_Utility(i, A, W, C):
    utility = 0
    Ai = A[i]
    lengthInt = 0
    for candidate in C:
        if(candidate in W and Ai[C.index(candidate)] == 1): 
            lengthInt += 1
    for j in range(lengthInt):
            utility += 1/(j+1)
    return utility

def calculate_Utility(i, A, W, C):
    utility = 0
    Ai = A[i]
    for candidate in W:
        if(Ai[C.index(candidate)] == 1): 
            utility += 1 
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


def num_Voters_Approving_Candidate(A, c):
    '''Calculates the number of voters in A, that approve the candidate in the positionc of the list of candidates.'''
    res = 0
    for i in range(len(A)):
        if(A[i][c] == 1):
            res += 1
    return res

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def seqPhragmen_sim(k, A, C):
    W = []
    restC = copy.copy(C)
    #We will do k rounds, in each round we will add the candidate that minimizes the (new) maximal voter load.
    loads = zerolistmaker(len(A))
    #In the first round we select the candidate that is most approved by voters,this is because it will increase the maximal load the least.
    Atemp = np.array(A)
    sumOfVotes = np.sum(Atemp, 0) 
    maxVotes = max(sumOfVotes)
    maxPos = np.argmax(sumOfVotes)
    loadPerVoter = 1/maxVotes
    #As all loads are 0, the best way of minimizing the load is equally dividing among approving voters.
    for i in range(len(A)):
        if (A[i][maxPos] == 1): 
            loads[i] += loadPerVoter
    restC.pop(maxPos)
    W.append(C[maxPos])
    #In the next k-1 rounds we will introduce the candidate that yields the smallest possible (maybe new) maximal voter load. But from now on we have to
    #consider that voters may have non-zero loads.
    for j in range(1,k):
        scMin = 1000000000
        minCandidate = restC[0]
        for c in restC:
            sc = 0
            aux = 0 #Variable that stores the loads of all the voters that apporve candidate c.
            for i in range(len(A)):
                if(A[i][C.index(c)] == 1):
                    aux += loads[i] #We add the load of round j-1 (as seen in the equation 5, page 408)
            sc = (1 + aux)/sumOfVotes[C.index(c)]
            if (sc < scMin):
                minCandidate = c
                scMin = sc
        #We add minCandidate to the committe, and delete it from restC
        W.append(minCandidate)
        restC.pop(restC.index(minCandidate))
        #Update the loads variable:
        for i in range(len(loads)):
            #If i belongs to the support of c ==> scMin.
            if(A[i][C.index(minCandidate)] == 1):
                loads[i] = scMin
            #Previous load otherwise.

    return[W, 0]

def trial_And_Error_Heuristic(k, W, C, i, A, initialA, f):
    ''''''
    currentUtility = calculate_Num_Selected_Approval(initialA[i], W, C)  
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
    res = seqPhragmen_sim(k, newA, C)
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

    for candidate in C:            
        if(Ai[C.index(candidate)] == 1):
            newA = copy.deepcopy(A)       
            newA[i][C.index(candidate)] = 0
            #Simmulates PAV and calculates the new winners, but with the updated vote
            res = seqPhragmen_sim(k, newA, C) 
            newW = res[0]
            #Calculates the new utility with the new winners, but the original preferences
            newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                newWinners = newW   
        else:
            newA = copy.deepcopy(A)
            newA[i][C.index(candidate)] = 1
            #Simmulates PAV and calculates the new winners, but with the updated vote
            res = seqPhragmen_sim(k, newA, C) 
            newW = res[0]
            #Calculates the new utility with the new winners, but the original preferences
            newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
           
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                newWinners = newW
    #print(str(newVote))
    #print(str(newWinners))
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
    
    if(maxChanges>len(C)):
        print("maxChanges value can never be higher than len(C)")
        exit()

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
            res = seqPhragmen_sim(k, newA, C) 
            newW = res[0]
            #Calculates the new utility with the new winners, but the original preferences (initialA[i])
            newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            if(newUtility > maxUtility):
                changedVote = True
                maxUtility = newUtility
                newVote = newA[i]
                newWinners = newW

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
                #print("New A: " + str(newA))
                #print("New vote: " + str(newVote))
                #New winners are calculated.
                res = seqPhragmen_sim(k, newA, C)
                #print("New winners: " + str(res[0]))
                return [newVote, True, res[0]]
    #print("Same vote: " + str(newVote))
    return [newVote, False, W]

def threshold_Heuristic_Highest_Votes(k, W, C, i, A, initialA, f):
    '''The voter i disapproves the candidate with the highest votes if it has a percentage of approval above a certain threshold, as said candidate
    will probably be elected as winners even when i does not approve them. Returns a tuple consisting of i's new vote, a variable that states 
    wether the voter i has decided to change its vote and the new winners resulting of that change'''
    #If only one voter approved, cannot disapprove him:
    if(sum(A[i]) == 1):
        return[A[i], False, W]
    n = len(A)
    newA = copy.deepcopy(A)
    newVote = newA[i][:]
    maxpos = calculate_Highest_Scoring_Approved_Candidate(W, C, newA, i, f)
    if(maxpos == -1):
        return [newVote, False, W]
    elif(votesCandidates[maxpos]/n >= threshold):
        newVote[maxpos] = 0
        newA[i][maxpos] = 0
        res = seqPhragmen_sim(k, newA, C)
        return [newVote, True, res[0], res[1]]
    else:
        return [newVote, False, W]
    
def calculate_Highest_Scoring_Approved_Candidate(W, C, A, i, f):
    '''This method finds the candidate approved by a voter i with the highest number of votes. 
    Returns the position of said candidate in the candidate list. -1 is returned if the voter approves no candidates.'''
    #Calculate the approvals each candidate got:
    Atemp = np.array(A)
    votesCandidates = np.sum(Atemp, 0)
    maxvotes = 0
    maxpos = -1
    for candidate in C:
        j = C.index(candidate)
        if(A[i][j] == 1):
            if (votesCandidates[j]>maxvotes):
                maxpos = j
                maxvotes = votesCandidates[j]
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
        maxUtility = calculate_Num_Selected_Approval(initialA[i], W, C)
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
                res = seqPhragmen_sim(k, newA, C) 
                #f.write("This newA yields the winners: " + str(res[0]))
                newW = res[0]
                #Calculates the new utility with the new winners, but the original preferences. Utility for i is the same as the utility for 
                
                newUtility = calculate_Num_Selected_Approval(initialA[i], newW, C)
            else:
                newA = copy.deepcopy(A)
                for j in faction[0]:
                    newA[j][C.index(candidate)] = 1
                #Simmulates PAV and calculates the new winners, but with the updated vote
                res = seqPhragmen_sim(k, newA, C) 
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
            tempA = copy.deepcopy(A)
            tempA[i] = newVote
            aux = seqPhragmen_sim(k, tempA, C)
            currentWinners = aux[0]
            faction[1] = newVote
            faction[2] = [i]
            print(str(faction[0])+" ha cambiado de voto!")
            return [newVote, True, currentWinners]
        else:
            #There is no vote to change to, so faction[1] and faction[2] remains the same.
            return [Ai, False, W]
    else:
        f.write("Faction wants to change to vote: " + str(faction[1])  + "\n")
        if (i in faction[2]):
            if(faction[0] == faction[2]):
                f.write("Everyone has changed vote: " + str(faction) + "\n")
                faction[1] = []
                faction[2] = []
            return [newVote, False, W]
        #Change vote to faction[1]
        newA[i] = faction[1]
        #Calculate new winners
        res = seqPhragmen_sim(k, newA, C) 
        newW = res[0]
        #Update faction[2]
        bisect.insort(faction[2], i)
        return [faction[1], True, newW]

def voting_iteration(k, A, W, C, maxRounds, heuristics, N, factions, f, gameA, upToJ):
    initialA = copy.deepcopy(A)
    #Variable that stores the last A where the voter i did not want to make a change, so that Iterative heuristic only computes 
    #if the current state is different from the lastStableState[i]. Initialized to 0.
    lastStableState = [0 for i in range(len(A))]
    checkForLoop = []
    checkForLoopTemp = []
    Atemp = A[:]
    Wtemp = W[:]
    f.write("Initial A: " + str(initialA) + "\n")
    f.write("Initial C: " + str(C) + "\n")
    f.write("Initial W: " + str(Wtemp) + "\n")
    n = len(A)
    equilibrium = True   
    hasChangedList = [0]*n #If there is a 1 in pos i: Voter i does not want to change its vote. List must be updated to 0s whenther is a change.
    #If after one iteration, all positions have a 1 ==> we have found an equilibrium.       
    numberOfDeviations = 0  
    numberOfCycles = 0
    for j in range(maxRounds):
        i = random.choice(N)
        #Atemp and Wtemp need to be updated after each voter changes their vote
        previousVote = Atemp[i][:]          
        if(heuristics[i] == 0):
            changedVote = False
        elif(heuristics[i] == 1):
            currentA = copy.deepcopy(Atemp)
            res = threshold_Heuristic_Highest_Votes(k, Wtemp, C, i, currentA, initialA, f)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2] 
        elif(heuristics[i] == 2):
            currentA = copy.deepcopy(Atemp)
            if(not currentA == lastStableState[i]):
                res = interative_Through_Candidates_Single_Heuristic(k, Wtemp, C, i, currentA, initialA, f)
                newVote = res[0]
                changedVote = res[1]
                newWinners = res[2]
                if(not changedVote):
                    lastStableState[i] = copy.deepcopy(currentA)
                else:
                    aux = copy.deepcopy(currentA)
                    aux[i] = newVote[:] 
                    lastStableState[i] = 0 #Set to 0 as voter might want to change their vote more even if no one else has manipulated. 
            elif(currentA == lastStableState[i]):
                #We are in a state we already know we don't want to change
                changedVote = False
        elif(heuristics[i] == 3):
            currentA = copy.deepcopy(Atemp)
            if(not currentA == lastStableState[i]):
                res = interative_Through_Candidates_K_Heuristic(k, Wtemp, C, i, currentA, initialA, f, upToJ)
                newVote = res[0]
                changedVote = res[1]
                newWinners = res[2]
                if(not changedVote):
                    lastStableState[i] = copy.deepcopy(currentA)
                else:
                    aux = copy.deepcopy(currentA)
                    aux[i] = newVote[:] 
                    lastStableState[i] = 0
            elif(currentA == lastStableState[i]):
                #We are in a state we already know we don't want to change
                changedVote = False
        elif(heuristics[i] == 4):
            currentA = copy.deepcopy(Atemp)
            res = trial_And_Error_Heuristic(k, Wtemp, C, i, currentA, initialA, f)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2]
        else:
            #We need to find the faction of voter i:
            for faction in factions:
                if i in faction:
                    break
            currentA = copy.deepcopy(Atemp)
            res = faction_Heuristic_Sequential_Change(k, Wtemp, C, i, currentA, faction, initialA, f)
            newVote = res[0]
            changedVote = res[1]
            newWinners = res[2]
        if(changedVote):
            numberOfDeviations += 1
            Atemp[i] = newVote[:]
            #Change the equilibrium variable as for this round there is no equilibrium (a voter has changed its vote)
            #If there exists a new vote with a higher utility change equilibrium to False
            equilibrium = False
            #Update Atemp:
            Wtemp = newWinners[:]
            f.write("Voter " + str(i) + " has changed its vote from: " + str(previousVote) + ", to: " + str(newVote)+ "\n")
            f.write("New W: " + str(Wtemp)+ "\n")
            hasChangedList = [0]*n
            #SECTION FOR CHECKING FOR LOOPS IN THE GAME
            #WHAT IF ATEMP APPEARS TWICE??
            
            if(Atemp in gameA):
                numberOfCycles += 1
            gameA.append(copy.deepcopy(Atemp)) 
                       
            
            '''
            if(not checkForLoop == []):
                if(Atemp == checkForLoop[0]):
                    checkForLoop = checkForLoop[1:]
                    if(checkForLoop == []):#We have deleted all the states --> the cycle has been repeated
                        f.write("There is a cycle in this iteration: " + str(checkForLoopTemp) + "\n")
                        checkForLoopTemp = []
                else:
                    checkForLoop = []
                    checkForLoopTemp = []
            else:
                if(Atemp in gameA[:len(gameA)]):
                    f.write("Checking for loop, gameA = " + str(gameA) + "\n")
                    #A node has been visited more than once
                    numberOfCycles +=1
                    checkForLoop = gameA[gameA.index(Atemp)+1:]
                    checkForLoopTemp = gameA[gameA.index(Atemp)+1:]
                    f.write("checkForLoop = " + str(checkForLoop) + "\n")
                else:
                    checkForLoop = []
                    checkForLoopTemp = []
            '''
            
        else:
            #Voter i didn't change his vote:
            f.write("Voter " + str(i) + " has not changed its vote."+ "\n")
            hasChangedList[i] = 1
            #Atemp[i] = newVote[:]
            #f.write("A:" + str(Atemp))
            #f.write("W:" + str(Wtemp) + "\n")
        #SECTION FOR CHECKING FOR LOOPS IN THE GAME
        
        
        if(not 0 in hasChangedList):
            #If no voter wants to change their vote, we have reached an equilibrium and there is no need for further voting iterations.
            f.write("\nEquilibrium found!"+ "\n")
            return [Wtemp, Atemp, j, numberOfDeviations, numberOfCycles]

    f.write("\nLimit of rounds has been reached. Winners from final round: " + str(Wtemp)+ "\n")
    if(numberOfCycles > 0):
                f.write("CHECK FOR CYCLE !!!!!!!!!")
    return [Wtemp, Atemp, j, numberOfDeviations, numberOfCycles]

def calculatePAVScore(W, C, A):
    res = 0.0
    n = len(A)
    for i in range(n):
        res += calculate_PAV_Utility(i, A, W, C)
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

def create_Approval_Votes_Normal_Dist_For_Approving_Voters(C, N, mean, sd):
    res = [[0 for z in range(len(C))] for j in range(n)]
    for i in range(len(C)):
        value = np.random.normal(mean, sd)
        if(value < 0):
            value = 0
        elif(value > 10):
            value = 10    
        approvalScore = round((value/10)*n)
        #Select the j random voters that approve candidate
        votersApprovingCand = random.sample(N, approvalScore)
        #print(votersApprovingCand)
        for z in votersApprovingCand:
            res[z][i] = 1
    return res        


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

if __name__ == "__main__":
    if(sys.argv[1] == '' or sys.argv[2] == ''):
        exit()
    f=open(sys.argv[1], "a+")
    random.seed(sys.argv[2])
    print("\nStarting Simulation")
    print("-------------------\n")
    n = int(input("Enter the value of n: "))
    N = [i for i in range(n)]
    m = int(input("Enter the value of m (m = |C|): "))
    C = []
    for l in range(m):
        C.append("c" + str(l))
    startK = int(input("Enter the initial value of k: "))
    endK = int(input("Enter the final value of k: "))
    stepInK = int(input("Enter the step in k: "))
    if((endK >= m) or (startK >= m)):
        print("All ks must be smaller than m!")
        exit()
    
    max_rounds = int(input("\nEnter maximum number of rounds per game: "))
    num_voting = int(input("Enter number of simulations done per value of n: "))
    threshold = 0

    heuristics = []
    sizeOfFaction = input("Enter percentage of voters that belong to the faction: ").split()
    variablesToChange = [float(item) for item in sizeOfFaction]
    print(variablesToChange)
    averagesDPOALists = []
    averagesTimesLists = []
    averagesSocialWelfareLists = []
    averagesNumberOfDeviationsLists = []
    averagesNumberOfIterationsLists = []
    averagesNumberOfCyclesLists = []
    averageVotersBetterInFinalOutcomeLists = []
    averageInitialSocialWelfareLists = []
    for item in variablesToChange:
        factionVoters = int(item*n)
        for i in range(factionVoters):
            heuristics.append(5)
        for i in range(factionVoters, n):
            heuristics.append(0)
        factions = [[[i for i in range(factionVoters)], [], []]]
        averagesDPOA = []
        averagesTimes = []
        averagesSocialWelfare = []
        averagesNumberOfDeviations = []
        averagesNumberOfIterations = []
        averagesNumberOfCycles = []
        simulations = []
        averageVotersBetterInFinalOutcome = []
        averageInitialSocialWelfare = []
        upToJ = 0
        for simN in range(startK, endK+stepInK, stepInK):
            simulations.append(simN)
            f.write("\nTESTING WITH A WINNER COMMITTEE SIZE OF " + str(simN)+ ":\n")
            f.write("----------------------\n")
            maxDPoA = 0
            maxDPoARound = 0
            totalTime = 0
            totaldPoA = 0
            totalSocialWelfare = 0
            totalNumberOfDeviations = 0
            totalNumberOfIterations = 0
            totalNumberOfCycles = 0
            totalVotersBetterInFinalOutcome = 0
            k = simN
            for test in range(1,num_voting+1):
                f.write("NEW PAV SIMULATION, NUMBER: " + str(test)+ "\n")
                #Create a random A, remember voters that belong to the same faction have the same approval ballot:
                aux = []
                A = create_Approval_Votes_Spatial_Model(C, N, 0.6,factions[0])
                
                gameA = [A] #List that stores all the As to check for loops
                initialA = copy.deepcopy(A)
                temp = seqPhragmen_sim(k, A, C)
                W = temp[0]
                initialUtilityArray = [calculate_Num_Selected_Approval(initialA[i], W, C) for i in range(n)]
                #Calculate the approvals each candidate got:
                Atemp = np.array(A)
                votesCandidates = np.sum(Atemp, 0)
                initialW = copy.copy(W)
                
                initialTotalUtility = calculate_Total_Utility(W, C, initialA)
                
                startTime = time.time()
                outcome = voting_iteration(k, A, W, C, max_rounds, heuristics, N, factions,f, gameA, upToJ)
                endTime = time.time()
                totalTime += (endTime-startTime)

                #Calculate the PAVScore for this new outcome, but considering the original votes.

                f.write("Initial Winners: " + str(initialW)+ "\n")
                f.write("Final Winners:   " + str(outcome[0])+ "\n\n")
                #Calculate the PAVScore for the new set of winners, but considering the original votes.

                f.write("Initial Total Utility: " + str(initialTotalUtility)+ "\n")
                #Calculate the PAVScore for the new set of winners, but considering the original votes.
                finalTotalUtility = calculate_Total_Utility(outcome[0], C, initialA)
                totalSocialWelfare += finalTotalUtility
                totalNumberOfIterations += outcome[2]
                totalNumberOfDeviations += outcome[3]
                totalNumberOfCycles += outcome[4]
                f.write("Final Total Utility:   " + str(finalTotalUtility)+ "\n")
                dPoA = initialTotalUtility/finalTotalUtility
                totaldPoA += dPoA
                f.write("Dynamic PoA = (Initial Total Utility)/(Final Total Utility) = " + str(dPoA)+ "\n")
                if(dPoA > maxDPoA):
                    maxDPoA = dPoA
                    maxDPoARound = test
                printProgressBar (test, num_voting)

                #Checking who is better of:
                finalUtilityArray = [calculate_Num_Selected_Approval(initialA[i], outcome[0], C) for i in range(n)]
                
                for i in range(n):
                    if(finalUtilityArray[i] < initialUtilityArray[i]):
                        totalVotersBetterInFinalOutcome += 1

            averagesDPOA.append(totaldPoA/num_voting)
            averagesNumberOfDeviations.append(totalNumberOfDeviations/num_voting)
            averagesSocialWelfare.append(totalSocialWelfare/num_voting)
            averagesNumberOfIterations.append(totalNumberOfIterations/num_voting)
            averagesNumberOfCycles.append(totalNumberOfCycles/num_voting)
            averageVotersBetterInFinalOutcome.append((totalVotersBetterInFinalOutcome/num_voting)/n*100)
            
            print("\nTime elapsed: " + str(totalTime) + " seconds.")
            averagesTimes.append(totalTime/num_voting)

        averagesDPOALists.append(averagesDPOA)
        averagesTimesLists.append(averagesTimes)
        averagesSocialWelfareLists.append(averagesSocialWelfare)
        averagesNumberOfDeviationsLists.append(averagesNumberOfDeviations)
        averagesNumberOfIterationsLists.append(averagesNumberOfIterations)
        averagesNumberOfCyclesLists.append(averagesNumberOfCycles)
        averageVotersBetterInFinalOutcomeLists.append(averageVotersBetterInFinalOutcome)
    f.close()
    if(not len(variablesToChange) == 0):
        j = -1
        for item in averagesDPOALists:
            j += 1
            plot1 = plt.figure(1)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How average dPOA of a game increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average Dynamic Price of Anarchy')
        plt.legend()

        j = -1
        for item in averagesTimesLists:
            j += 1
            plot2 = plt.figure(2)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How average time of a game increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average time of a game/seconds')
        plt.legend()

        j = -1
        for item in averagesSocialWelfareLists:
            j += 1
            plot3 = plt.figure(3)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How the average Social Welfare of a game increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average Social Welfare')
        plt.legend()

        j = -1
        for item in averagesNumberOfIterationsLists:
            j += 1
            plot4 = plt.figure(4)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How the average number of iterations of a game increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average number of iterations')
        plt.legend()

        j = -1
        for item in averagesNumberOfDeviationsLists:
            j += 1
            plot5 = plt.figure(5)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How the average number of deviations in a game increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average number of deviations')
        plt.legend()

        j = -1
        for item in averagesNumberOfCyclesLists:
            j += 1
            plot6 = plt.figure(6)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How the average number of cycles occuring in a game increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average number of cycles')
        plt.legend()

        j = -1
        for item in averageVotersBetterInFinalOutcomeLists:
            j += 1
            plot7 = plt.figure(7)
            plt.plot(simulations, item, label = variablesToChange[j], marker = 'x')
            plt.xticks(simulations,simulations)
            plt.title('How the average percentage of voters strictly better off with final outcome increases with k:')
            plt.xlabel('Committee size (k)')
            plt.ylabel('Average percentage of voters better off')
        plt.legend()
    plt.show()

