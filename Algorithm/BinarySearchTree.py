from TreeNode import TreeNode
import queue

# Build a simple tree
left_node = TreeNode(None,None,1)
right_node = TreeNode(None,None,2)
root = TreeNode(left_node,right_node,4)
main_tree = TreeNode(root,root,5)
left_node.setLeftNode(TreeNode(None,None,7))

def traverseTree(node,node_set):
    if node == None:
        return
    #preorder
    traverseTree(node.leftNode,node_set)
    #inorder
    node_set.append(node)
    traverseTree(node.rightNode,node_set)
    #postorder
    return node_set

node_list = traverseTree(root,[])
for node in node_list:
    print (node.value)

def isTheSameTree(node_1,node_2):
    if (node_1 and node_2):
        sameLeftTree = isTheSameTree(node_1.leftNode,node_2.leftNode)
        sameRightTree = isTheSameTree(node_1.rightNode,node_2.rightNode)
        sameValue = node_1.value == node_2.value
        return sameLeftTree and sameRightTree and sameValue
    else:
        return (node_1 == None and node_2 == None)

print(isTheSameTree(root,main_tree))

def hasSubTree(main,sub):
    node_list = traverseTree(main,[])
    print(len(node_list))
    sameTree_list = []
    for node in node_list:
        if node.value == sub.value:
            if isTheSameTree(node,sub):
                sameTree_list.append(node)
    return len(sameTree_list)

print(hasSubTree(main_tree,root))


def hasSubTree2(main,sub):
    node_queue = queue.Queue()
    node_queue.put(main)
    check_list = []
    count_same = 0
    while not node_queue.empty():
        current_node = node_queue.get()
        if current_node.value == sub.value:
            check_list.append(current_node)
        if current_node.leftNode:
            node_queue.put(current_node.leftNode)
        if current_node.rightNode:
            node_queue.put(current_node.rightNode)

    for node in check_list:
        if isTheSameTree(node,sub):
            count_same += 1

    return count_same 

print("using queue method", hasSubTree2(main_tree,root))

def treeDepth(root):
    if not root:
        return 0
    else:
        left_depth = treeDepth(root.leftNode)
        right_depth = treeDepth(root.rightNode)

    if left_depth > right_depth:
        return 1 + left_depth
    else:
        return 1 + right_depth

print("depth",treeDepth(main_tree))

def diameter(root):
    return treeDepth(root.leftNode) + 1 + treeDepth(root.rightNode)

print("diameter",diameter(main_tree))

def diameterpath(root):
    if not root:
        return 0
    else:
        ld = diameterpath(root.leftNode)
        rd = diameterpath(root.rightNode)
    return max(max(ld,rd),diameter(root))

# Build tree for checking
l_leaf = TreeNode(None,None,7)
r_leaf = TreeNode(None,None,6)
one = TreeNode(l_leaf,None,1)
two = TreeNode(None,r_leaf,2)
four = TreeNode(one,two,4)
five = TreeNode(four,None,5)

print("diameterpath:",diameterpath(five))

def longest_path(root):
    if not root:
        return []
    else:
        left_longest_path = longest_path(root.leftNode)
        right_longest_path = longest_path(root.rightNode)
        
        if len(left_longest_path) > len(right_longest_path):
            left_longest_path.append(root.value)
            return left_longest_path
        else:
            right_longest_path.append(root.value)
            return right_longest_path

print(longest_path(five))

# the longest path between any two nodes in the tree
def diameterPath(root):
    return longest_path(root.leftNode) + [root.value] + longest_path(root.rightNode)[::-1]

print(diameterPath(main_tree))


