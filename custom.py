from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier
import numpy as np
import os
from subprocess import Popen, PIPE
from sphinx.util.osutil import EPIPE, EINVAL
from IPython.display import display_png

positive_infinity = float("inf")
negative_infinity = float("-inf")
real = (negative_infinity,positive_infinity)

def uniform(a,b):
    return (b-a)*np.random.random_sample() + a

def complete(skeleton,length,mini,maxi):
    X = np.zeros(length)
    for i in range(length):
        interval = skeleton.get(i)
        if interval:
            value = uniform(*interval)
        else:
            value = mini
        X[i] = value
    return X

def reconstruct(classifier,feature,shape,mini,maxi):
    U = []
    for estimator,terminalId in zip(classifier.estimators_,feature):
        tree = estimator.tree_
        sk = skeleton(tree,terminalId,mini,maxi)
        U.append(sk)
    uni = union(U)
    new_rep = complete(uni,shape,mini,maxi)
    return new_rep

def intersection(i1,i2):
    """
    Find the intersection of 2 intervals.
    The implenentation here only 
    make sense when i1 and i2 actually
    intersect with each others.

    Examples:
    (This make sense)
    x = (2,10)
    y = (5,20)
    intersection(x,y) = (5,10)

    (This is nonsense)
    x = (0,1)
    y = (2,3)
    intersection(x,y) = (1,2)
    """
    minimum = max(i1[0],i2[0])
    maximum = min(i1[1],i2[1])
    return (minimum,maximum)

def union(l):
    """
    input: a list of sparse array in the form of
           {2:(0,0.4),10:(1,2)} which represent a 
           vector whose second element is between 0 and 0.4
           and its 10-th element is between 1 and 2
    output: a sparse array whose i-th element is the intersection
            of all the i-th element in the list of the input.
    
    Example:
    l = [{1:(0,1),2:(1,2),5:(2,3)},
         {1:(0.5,1),2:(1,5),6:(4,5)}
        ]

    union(l) = {1:(0.5,1),2:(1,2),5:(2,3),6:(4,5)}

    """
    U = {}
    for x in l:
        for y in x.keys():
            feature = y
            interval = x[y]
            U[feature] = intersection(U.get(feature,real),interval)
    return U      

def run_dot(code, options=[], format='png'):
    # mostly copied from sphinx.ext.graphviz.render_dot
    dot_args = ['dot'] + options + ['-T', format]
    if os.name == 'nt':
        # Avoid opening shell window.
        # * https://github.com/tkf/ipython-hierarchymagic/issues/1
        # * http://stackoverflow.com/a/2935727/727827
        p = Popen(dot_args, stdout=PIPE, stdin=PIPE, stderr=PIPE,
                  creationflags=0x08000000)
    else:
        p = Popen(dot_args, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    wentwrong = False
    try:
        # Graphviz may close standard input when an error occurs,
        # resulting in a broken pipe on communicate()
        stdout, stderr = p.communicate(code.encode('utf-8'))
    except (OSError, IOError) as err:
        if err.errno != EPIPE:
            raise
        wentwrong = True
    except IOError as err:
        if err.errno != EINVAL:
            raise
        wentwrong = True
    if wentwrong:
        # in this case, read the standard output and standard error streams
        # directly, to get the error message(s)
        stdout, stderr = p.stdout.read(), p.stderr.read()
        p.wait()
    if p.returncode != 0:
        raise RuntimeError('dot exited with error:\n[stderr]\n{0}'
                           .format(stderr.decode('utf-8')))
    return display_png(stdout,raw=True)

def parentOf(tree,nodeId):
    """
    input  : a tree object, the id of one of its node
    output : a tuple that shows its parent's id and whether
             the node is a left or right children, i.e. 
             ("left"/"right",parentId)
    """
    parentNodeId = np.where(tree.children_left == nodeId)[0]
    if parentNodeId.size == 0:
        parentNodeId = np.where(tree.children_right == nodeId)[0]
        return ("right",parentNodeId)
    else:
        return ("left",parentNodeId)

def lineageOf(tree,nodeId):
    """
    Find out the path from the root node
    to a particular node.
    """
    parent = parentOf(tree,nodeId)
    parentNodeId = parent[1][0]
    if parentNodeId == 0:
        yield parent
    else:
        yield from lineageOf(tree,parentNodeId)
        yield parent

def visualize(tree):
    """
    visualize the tree using dot language.
    """
    dotString = ""
    t = tree
    for i,node in enumerate(t.children_left):
        if node > -1:
            txt = "{} -> {};".format(i,node)
            dotString += txt
    for i,node in enumerate(t.children_right):
        if node > -1:
            txt = "{} -> {};".format(i,node)
            dotString += txt        
    dotString = "digraph G{" + dotString + "}"
    run_dot(dotString)
    
def skeleton(tree,terminalId,mini = negative_infinity,maxi = positive_infinity):
    """
    Given a tree and one of it terminal node,
    find out what kind of data would be
    assigned to this particular terminal node.
    The output is a dictionary whose keys
    are the feature of the the data,
    and its values are the range of possible value
    for that particualr feature.
    """
    l = lineageOf(tree,terminalId)
    d = {tree.feature[x[1]][0]: (mini,tree.threshold[x[1]][0]) if x[0] == "left" else (tree.threshold[x[1]][0],maxi) for x in l}
    return d

def terminal(tree):
    """
    Find out all the terminal nodes of a tree
    """
    terminals = []
    for i,node in enumerate(tree.children_left):
        if node == -1:
            terminals.append(i)
        else:
            pass
    return terminals 
    
    
