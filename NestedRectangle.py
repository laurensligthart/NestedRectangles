# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:07:48 2023
Author: Laurens Ligthart


Code for the polarization hierarchy presented in arxiv:2405.xxxxx for the
nonnegative rank and the nested rectangles problem.

Given matrix M of size m x n with positive entries, it gives an outer approximation 
to the feasibility problem whether there exist matrices U, V of size m x r and
r x n with positive entries such that

 M = U V.

"""

import numpy as np
from scipy.sparse import csr_array
import itertools as it
import sys
import mosek


def NestedRectangleMatrix(a,b):
    """
    Produces the left-stochastic version of the nested rectangle matrix.
    """
    
    M = .25* np.array([[1-a, 1+a, 1-b, 1+b], 
                  [1+a, 1-a, 1-b, 1+b], 
                  [1+a, 1-a, 1+b, 1-b], 
                  [1-a, 1+a, 1+b, 1-b]])
    
    return M


def LabelToIndex(label, k=3, r=3, ldim1=4, ldim2=4):
    """
    Changes a variable label into a vector index for the LP
    The index is ordered by level
    
    label: label of the variable in format [ [i_1,...,i_k], [j_1,...,j_k] ], 
            i=0 corresponds to the unit
    k: number of copies
    ldim1: first dimension of M matrix
    ldim2: second dimension of M matrix
    """
    
    index = 0
    unitLdim1 = ldim1*r - r + 1 # add unit element, but subtract left stochastic conditions
    unitLdim2 = r*ldim2 - ldim2 + 1 # add unit element, but subtract left stochastic conditions
    
    base = unitLdim1 * unitLdim2 
    
    for a in range(k):
        localIndex = unitLdim2 * label[0][a] + label[1][a]
        index += base**(k-a-1) * localIndex
    
    return index


def IndexToLabel(index, k=3, r=3, ldim1=4, ldim2=4):
    """
    Changes a variable index into a label of the variable in format [ [i_1,...,i_k], [j_1,...,j_k] ]
    
    index: index of the variable
    k: number of copies
    ldim1: first dimension of M matrix
    ldim2: second dimension of M matrix
    """
    
    unitLdim1 = ldim1*r - r + 1 # add unit element, but subtract left stochastic conditions
    unitLdim2 = r*ldim2 - ldim2 + 1 # add unit element, but subtract left stochastic conditions
    
    base = unitLdim1 * unitLdim2 
    new_ind = index
    
    n = []
    
    for l in range(k):
        power = base**(k-1-l)
        nl = np.floor_divide(new_ind, power)
        n.append(nl)
        remainder = new_ind % power
        new_ind = remainder
    
    i_list = []
    j_list = []
    
    for ni in n:
        i_list.append(np.floor_divide(ni, unitLdim2))
        j_list.append(ni % unitLdim2)
    
    label = [i_list, j_list]
    
    return label

    
def CanonicalLabel(label):
    """
    Applies the symmetry relations to reduce a label to its canonical form.
    """
    
    canLab = [sorted(label[0]), sorted(label[1])]
    return canLab


def CreateCanonicalVars(k=3, r=3, ldim1=4, ldim2=4):
    """
    Creates all canonical variables for level k of the hierarchy for the
    nonnegative rank problem for matrices of size ldim1 x r and r x ldim2.
    Returns dictionaries translating between the label and index representations,
    and the number of variables.
    """
    
    unitLdim1 = ldim1*r - r + 1 # add unit element, but subtract left stochastic conditions
    unitLdim2 = r*ldim2 - ldim2 + 1 # add unit element, but subtract left stochastic conditions
    
    canLabels1 = [list(v) for v in it.combinations_with_replacement(range(unitLdim1), k)]
    canLabels2 = [list(v) for v in it.combinations_with_replacement(range(unitLdim2), k)]
    
    canIndexDict = {} #input: label, output: index
    canLabelDict = {} #input: index, output: label
    numvar = 0
    for a in canLabels1:
        for b in canLabels2:
            label = str([a,b])
            canIndexDict[label] = numvar
            canLabelDict[numvar] = str(label)
            numvar += 1
    
    return canIndexDict, canLabelDict, numvar


def StochasticSubU(label, coefficient, maxLoops=10):
    """
    Recursively removes all ill-defined variables with left-stochasticity for U
    from the label representation of a variable.
    """
    
    #if u label is valid, just return the label
    if all(x < 10 for x in label[0]):
        return [label], [coefficient]

    newLabels = []
    coef = []
    
    tempULabels = []
    tempUCoef = []
    
    #check all u values
    for i in range(len(label[0])):
        if label[0][i] >= 10:
            
            #create Id - sum_j u_jk
            for j in range(1,4):
                newULabel = label[0]
                newULabel[i] -= 3
                tempULabels.append(CanonicalLabel([newULabel, label[1]]))
                tempUCoef.append(-coefficient)
            
            newUlabel = label[0]
            newUlabel[i] = 0
            tempULabels.append(CanonicalLabel([newULabel, label[1]]))
            tempUCoef.append(coefficient)
            
            break #only one substitution at a time, recursiveness takes care of the rest
    
    #check if newly created labels are valid, and if not, do the substitution again for each of them
    for i in range(len(tempULabels)):
        if maxLoops > 0:
            endLabels, endCoefs = StochasticSubU(tempULabels[i], tempUCoef[i], maxLoops-1)
            newLabels += endLabels
            coef += endCoefs
        else:
            raise RecursionError
                

    return newLabels, coef


def StochasticSubV(label, coefficient, maxLoops=10):
    """
    Recursively removes all ill-defined variables with left-stochasticity for V
    from the label representation of a variable.
    """
    
    #if v label is valid, just return the label
    if all(x < 9 for x in label[1]):
        return [label], [coefficient]
    
    newLabels = []
    coef = []
    
    tempVLabels = []
    tempVCoef = []
    for i in range(len(label[1])):
        if label[1][i] >= 9:
            
            #create Id - sum_j v_jk
            for j in range(1,3):
                newVLabel = label[1]
                newVLabel[i] -= 4
                tempVLabels.append(CanonicalLabel([label[0], newVLabel]))
                tempVCoef.append(-coefficient) 
            
            newVlabel = label[1]
            newVlabel[i] = 0
            tempVLabels.append(CanonicalLabel([label[0], newVLabel]))
            tempVCoef.append(coefficient)
            
            break #only one substitution at a time, recursiveness takes care of the rest
    
    #check if newly created labels are valid, and if not, do the substitution again for each of them
    for i in range(len(tempVLabels)):
        if maxLoops > 0:
            endLabels, endCoefs = StochasticSubV(tempVLabels[i], tempVCoef[i], maxLoops-1)
            newLabels += endLabels
            coef += endCoefs
        else:
            raise RecursionError
    
    return newLabels, coef

    
def StochasticSub(label, coefficient, maxLoops=10):
    """
    Substitutes the left-stochastic relations for eliminated variables
    """
    
    newLabels = []
    newCoefs = []
    
    try:
        #do u substitution
        Ulabels, Ucoefs = StochasticSubU(label, coefficient, maxLoops)
        
        #for each new label, also do v substitution
        for i in range(len(Ulabels)):
            Vlabels, Vcoefs = StochasticSubV(Ulabels[i], Ucoefs[i], maxLoops)
            newLabels += Vlabels
            newCoefs += Vcoefs
    except RecursionError:
        print("Recursion limit reached on %s"%str(label))
    
    return newLabels, newCoefs


def NewStochasticConstraints(canIndexDict, canLabelDict, r, k, ldim1=12, ldim2=12):    
    """
    Adds constraints in canonical form of the form 
    sum_i=0,1,2 [[0,...], [...]] - [[u_ij, ...], [...]] >= 0
    sum_i=0,1   [[...], [0,...]] - [[...], [v_ij, ...]] >= 0
    that follow from the left-stochasticity of U and V after reduction.
    """
    
    StocRow = []
    StocVal = []
    Stocb = []
    
    # Creates all unique canonical combinations
    uList = [list(comb1) for comb1 in it.combinations_with_replacement(range(ldim1 + 1), k)]
    vList = [list(comb2) for comb2 in it.combinations_with_replacement(range(ldim2 + 1), k)]
    
    total = 0
    for u in uList:
        for v in vList:
            
            # only treat non-trivial cases and skip the rest
            if all(x < 10 for x in u) and all(y < 9 for y in v):
                continue
            
            checkLabel = [u.copy(),v.copy()]
            
            # creates the reduced vector and puts it in canonical form
            labels, coefficients = StochasticSub(checkLabel, 1)
            indexList = [canIndexDict[str(label)] for label in labels]
            
            StocRow.append(indexList)
            StocVal.append(coefficients)
            Stocb.append(0.)
            
            total += 1
            if total % 25000 == 0:
                print("%s stochastic constraints generated"%total)
    
    
    return StocRow, StocVal, Stocb



def MatrixSub(MstocVal, Mrow, Mcol, label, coefficient, r=3, maxLoops=5):
    """
    Replaces [[0, ...], [0, ...]] by 1/M_ij * sum_k [[u_ik, ...], [v_kj, ...]]
    and puts it in reduced symmetrical form.
    """
    
    constraintLabelList = []
    constraintCoefList = []
    
    if label[0][0] == 0 and label[1][0] == 0:
        
        newLabels = []
        newCoef = []
        
        #implicitly assumes at least level 2 of the hierarchy!
        uk = label[0][1:]
        vk = label[1][1:]
        
        # Do the subsitution
        for matrixMul in range(r):
            newLabel = CanonicalLabel([[r*Mrow + matrixMul + 1]+uk, [4*matrixMul + Mcol + 1]+vk])
            newLabels.append(newLabel)
            newCoef.append(1./MstocVal)
        
        
        # Make sure the eliminated variables from the stochastic constraints are replaced
        for fullLabel in range(len(newLabels)):
            reducedLabels, reducedCoefs = StochasticSub(newLabels[fullLabel], newCoef[fullLabel])
            constraintLabelList += reducedLabels
            constraintCoefList += reducedCoefs
    
            
    else:
        print("The following label is considered, but shouldn't be: %s"%label)
    
    
    return constraintLabelList, constraintCoefList
    

def NewMatrixConstraints(canIndexDict, canLabelDict, Mstoc, r, k=3, ldim1=12, ldim2=12):
    """
    Creates the (symmetric) matrix constraints for the entire matrix, including higher order products
    """
    
    MCrow = []
    MCval = []
    MCb = []
    
    
    simpleUList = [list(comb1) for comb1 in it.combinations_with_replacement(range(ldim1-2), k-1)]
    simpleVList = [list(comb2) for comb2 in it.combinations_with_replacement(range(ldim1-3), k-1)]
    
    total = 0
    for u in simpleUList:
        for v in simpleVList:
            
            checkLabel = [[0] + u.copy(), [0] + v.copy()]
            
            for i, j in it.product(range(Mstoc.shape[0]), range(Mstoc.shape[1])):
                # Adding the matrix multiplication
                constraint, coefficients = MatrixSub(Mstoc[i][j], i, j, checkLabel, 1)
                
                # Adding the -1* Id \otimes ...
                constraint.append(checkLabel)
                coefficients.append(-1)
                
                indexList = [canIndexDict[str(label)] for label in constraint]
                
                MCrow.append(indexList)
                MCval.append(coefficients)
                MCb.append(0.)
                total += 1
                if total % 10000 == 0:
                    print("%s matrix constraints generated"%total)
    
    
    return MCrow, MCval, MCb
    

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def LP(c, A_i = [[], []], b_i = [], A_e = [[], []], b_e = [], bound = []):
    """
    Runs an LP for the given objective c, bounds and constraints A_i x >= b_i and A_e x = b_e.
    """
    
    # hard-coded "infinity"
    inf = 1e9
    
    # Start a Mosek environment
    with mosek.Env() as env:
    
        # Create a task object
        with env.Task() as task:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)
            
            numvar = len(c)
            numcon_e = len(A_e[0]) 
            numcon_i = len(A_i[0])
            
            # Bound keys for constraints (first for equalities, then for inequalities)
            bkc_e = [mosek.boundkey.fx] * len(A_e[0]) 
            bkc_i = [mosek.boundkey.lo] * len(A_i[0])
    
            # Bound values for constraints
            blc_e = b_e
            buc_e = b_e
            blc_i = b_i
            buc_i = [inf]*len(b_i)
    
            # Bound keys for variables
            bkx = [mosek.boundkey.ra] * numvar
    
            # Bound values for variables (all variables are associated with probabilities)
            blx = [0.] * numvar
            bux = [1.] * numvar
    
            # Below is the sparse representation of the A matrix stored by row.
            asub_e = A_e[0]
            aval_e = A_e[1]
            asub_i = A_i[0]
            aval_i = A_i[1]
    
            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon_e + numcon_i)
    
            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)
    
            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
    
                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
    
            
            print("Building the equality constraints")
            
            # Set the bounds on constraints.
            # blc[i] <= constraint_i <= buc[i]
            for i in range(numcon_e):
                
                # Input row i of A_e
                try:
                    sparse_ai = csr_array((aval_e[i], ([0]*len(asub_e[i]), asub_e[i])), shape=(1,numvar))
                    sparse_ai.sum_duplicates()
                    sparse_ai.eliminate_zeros()
                    task.putarow(i,                  # Row index.
                                 sparse_ai.indices,            # Column index of non-zeros in row i.
                                 sparse_ai.data)            # Non-zero Values of row i.
                    
                    task.putconbound(i, bkc_e[i], blc_e[i], buc_e[i])
                except Exception as e:
                    print(e)
                    print(i)
            
            print("Building the inequality constraints")
            for i in range(numcon_i):
                
                # Input row i of A_i
                try:
                    sparse_ai = csr_array((aval_i[i], ([0]*len(asub_i[i]), asub_i[i])), shape=(1,numvar))
                    sparse_ai.sum_duplicates()
                    sparse_ai.eliminate_zeros()
                    task.putarow(i + numcon_e,                  # Row index.
                                 sparse_ai.indices,            # Column index of non-zeros in row i.
                                 sparse_ai.data)            # Non-zero Values of row i.
                    
                    task.putconbound(i + numcon_e, bkc_i[i], blc_i[i], buc_i[i])
                except Exception as e:
                    print(i)
                    print(e)
    
            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            
            print("Solving the LP")
            # Solve the problem
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            task.optimize()
            
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)
    
            # Get status information about the solution
            solsta = task.getprosta(mosek.soltype.itr)
            
            
            return solsta
            


def GenerateEqualityConstraints(canIndexDict, canLabelDict, a, b, r=3, k=3):
    """
    Creates the equality constraints for a nested rectangle matrix with values
    a and b.
    """
    
    
    M = NestedRectangleMatrix(a, b)
    
    # Equality constraint list in sparse format, with normalization [[0,...],[0,...]] = 1
    A_e_row = [[0]]
    A_e_val = [[1.]]
    b_e = [1]

    # generate M constraints 
    MConstr = NewMatrixConstraints(canIndexDict, canLabelDict, M, r, k)

    A_e_row += MConstr[0]
    A_e_val += MConstr[1]
    b_e += MConstr[2]
    
    return A_e_row, A_e_val, b_e


def Bisection(alpha, r0, r1, eps, MaxIter, problemDims, posConstr, k=3):
    """
    Does a binary search to detect (in)feasibility along the angle alpha between
    a and b.
    """
    
    print("r0 is", r0, "and r1 is", r1)
    
    canIndexDict, canLabelDict, numvar = problemDims
    
    if r1 - r0 >= eps and MaxIter >= 0:
        rnew = (r0 + r1)/2.
        a = 1 - rnew * np.sin(alpha)
        b = 1 - rnew * np.cos(alpha)
        M = NestedRectangleMatrix(a, b)
        
        c = [0.] * numvar
        
        matConstr = NewMatrixConstraints(canIndexDict, canLabelDict, M, rnew, k)
        A_e_row = [[0]] + matConstr[0]
        A_e_val = [[1.]] + matConstr[1]
        b_e = [1.] + matConstr[2]
        
        solution = LP(c, [posConstr[0], posConstr[1]], posConstr[2], [A_e_row, A_e_val], b_e)
        
        if solution == mosek.prosta.prim_infeas or solution == mosek.prosta.dual_infeas:
            r0 = rnew
        else:
            r1 = rnew
        
        solution = 0
    
    if r1 - r0 >= eps and MaxIter >= 0:
        r0, r1 = Bisection(alpha, r0, r1, eps, MaxIter-1, problemDims, posConstr, k)
    
    return r0, r1

    

def main():
    
    # Pick matrix and make it left stochastic
    rect1 = 0.83
    rect2 = 0.83
    Mstoc = NestedRectangleMatrix(rect1, rect2)
    r = 3
    k = 3
    
    canIndexDict, canLabelDict, numvar = CreateCanonicalVars(k)
    print("%s symmetric variables generated"%numvar)
    
    print(canLabelDict[0])
    
    # Constraint list in sparse format, with normalization [[0,...],[0,...]] = 1
    A_e_row = [[0]]
    A_e_val = [[1.]]
    b_e = [1]
    
    A_i_row = []
    A_i_val = []
    b_i = []
    
    print("%s Normalization constraint generated"%len(A_e_row))
    
    # generate M constraints 
    
    MConstr = NewMatrixConstraints(canIndexDict, canLabelDict, Mstoc, r, k)
        
    A_e_row += MConstr[0]
    A_e_val += MConstr[1]
    b_e += MConstr[2]
    
    print("%s Matrix constraints generated"%len(MConstr[0]))

    # generate left-stochastic constraints
    StocConstr = NewStochasticConstraints(canIndexDict, canLabelDict, r, k)
    A_i_row += StocConstr[0]
    A_i_val += StocConstr[1]
    b_i += StocConstr[2]
    
    print("%s Stochastic constraints generated"%len(StocConstr[0]))
    
    
    # trivial objective function
    c = [0.] * numvar
    
    numcon_e = len(A_e_row)
    numcon_i = len(A_i_row)
    
    print("There are %s variables, %s equality constraints and %s inequality constraints\n" %(len(c),numcon_e, numcon_i))
    
    
    
    try:
        print("Building the LP")
        sol = LP(c, [A_i_row, A_i_val], b_i, [A_e_row, A_e_val], b_e)
        print(sol)
        
        
    except mosek.Error as e:
        print("ERROR: %s" % str(e.errno))
        if e.msg is not None:
            print("\t%s" % e.msg)
            sys.exit(1)
    except:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # return 0



# test()
# main()