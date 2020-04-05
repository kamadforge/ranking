class Tree:
    def __init__(self):
        self.root=None

    def insert(self, data):
        if not self.root:
            self.root=Node(data)
        else:
            self.root.insert(data)


class Node:

    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
# Compare the new value with the parent node
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data

# Print the tree
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print( self.data),
        if self.right:
            self.right.PrintTree()

# Use the insert method to add nodes
# root = Node(12)
# root.insert(6)
# root.insert(14)
# root.insert(3)

# root.PrintTree()

#second way
tree=Tree()
tree.insert(3)
tree.insert(10)
tree.insert(12)
tree.insert(5)
tree.insert(6)
tree.insert(11)

tree.root.PrintTree()
