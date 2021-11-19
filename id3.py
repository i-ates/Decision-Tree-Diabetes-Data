from node import Node
from copy import deepcopy
import math


def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    if len(examples) == 0:
        return Node(default)
    elif all(examples[0]['class'] == example['class'] for example in examples):
        return Node(getmodeByFeature(examples, 'class'))
    else:
        best = bestInformationGain(examples)
        # if all available splits have zero information gain
        if getInformationGain(examples, best) == 0:
            feature_list = [feature for feature in examples[0].keys()]
            feature_list.remove('class')
            for feature in feature_list:
                if splitExamplesByFeature(examples, feature).keys() > 1:
                    best = feature
                    break
        tree = Node(getmodeByFeature(examples, 'class'), best)
        split_examples = splitExamplesByFeature(examples, best)
        for val, ex_i in split_examples.items():
            tree.children[val] = ID3(ex_i, getmodeByFeature(examples, 'class'))
        return tree


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    list_of_nodes = postOrderList(node)
    prev_accuracy = test(node, examples)[0]
    tree_list = []
    while len(list_of_nodes):
        for n in list_of_nodes:
            n.track = True
            tree_copy = deepcopy(node)
            new_tree = delNode(tree_copy)
            n.track = False
            new_accuracy = test(new_tree, examples)[0]
            if new_accuracy > prev_accuracy:
                tree_list.append((n, new_accuracy))
        if not len(tree_list):
            return node
        else:
            prune_node = max(tree_list, key=lambda x: x[1])
            prune_node[0].track = True
            delNode(node)
            prev_accuracy = test(node, examples)[0]
            tree_list = []
    return node


def postOrderList(node):
    result, stack = [], [(node, False)]
    while stack:
        n, visited = stack.pop()
        if n:
            if visited:
                result.append(n)
            else:
                stack.append((n, True))
                all_keys = list(n.children.keys())
                for key in all_keys[::-1]:
                    stack.append((n.children[key], False))
    return result


def delNode(root):
    stack = [root]
    while len(stack) != 0:
        x = stack.pop()
        for key, child in x.children.items():
            if child.track:
                del x.children[key]
                return root
        for child in x.children.values():
            stack.append(child)
    return root


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    class_list = [example['class'] for example in examples]
    predict_list = []
    for example in examples:
        predict_list.append(evaluate(node, example))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    num_correct = 0
    for i in range(len(class_list)):
        if predict_list[i] == class_list[i]:
            num_correct += 1
            if predict_list[i] == "1":
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if predict_list[i] == "1":
                FP = FP +1
            else:
                FN = FN +1

    return [(num_correct / len(class_list)),TP, TN, FP , FN]


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    current = node
    try:
        while len(current.children) > 0:
            feature = current.split_on
            val = example[feature]
            if val in current.children.keys():
                current = current.children[val]
            else:
                return current.label
    except:
        return  None
    return current.label


def getmodeByFeature(examples, feature):
    '''
    Takes in a list of examples(dictionaries) and an attribute
    and returns the mode of the given attribute
    '''
    feature_list = []
    for instance in examples:
        if instance[feature] != '?':
            feature_list.append(instance[feature])
    return max(feature_list, key=feature_list.count)


'''
def replaceMissingValues(examples):
    for feature in examples[0].keys():
        for instance in examples:
            if instance[feature] == '?':
                instance[feature] = getmodeByFeature(examples, feature)
'''


def getClassCounts(examples):
    total_count = {}
    for example in examples:
        val = example['class']
        if val in total_count.keys():
            total_count[val] += 1
        else:
            total_count[val] = 1
    return total_count


def splitExamplesByFeature(examples, feature):
    result_dict = {}
    for example in examples:
        if feature in example:
            val = example[feature]
            if val in result_dict.keys():
                result_dict[val].append(example)
            else:
                result_dict[val] = [example]
    return result_dict  # dictionary of (examples) list of dictionaries


def getEntropy(examples):
    counts = getClassCounts(examples)
    total = len(examples)
    if total == 0:
        return 0
    entropy = 0
    for cnt in counts.values():
        entropy += (cnt / total) * math.log(cnt / total, 2)
    return -1 * entropy


def getInformationGain(examples, feature):
    parentEntropy = getEntropy(examples)
    split_examples = splitExamplesByFeature(examples, feature)
    entropy_list = []  # list of tuples (entropy, length)
    for ex_i in split_examples.values():
        length = len(ex_i)
        entropy = getEntropy(ex_i)
        entropy_list.append((entropy, length))

    total = len(examples)
    avg_child_entropy = 0
    for tup in entropy_list:
        avg_child_entropy += (tup[1] / total) * tup[0]

    return parentEntropy - avg_child_entropy


def bestInformationGain(examples):
    best_feature = ''
    best_IG = 0
    for feature in examples[0].keys():
        if feature != 'class':
            IG = getInformationGain(examples, feature)
            if IG > best_IG:
                best_IG = IG
                best_feature = feature
    return best_feature







