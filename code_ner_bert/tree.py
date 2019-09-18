from collections import deque


class Tree(object):
    def __init__(self, tag=None, parent=None, children=None):
        self.tag = tag
        self.parent = parent
        if children:
            self.children = children
        else:
            self.children = []
        self.index = -1

    def add_child(self, child):
        self.children.append(child)

    def print_yield(self):
        if self.children:
            return ' '.join([child.print_yield() for child in self.children])
        return self.tag

    def get_yield(self):
        if self.children:
            return sum([child.get_yield() for child in self.children], [])
        return [self]

    def get_span(self):
        if self.index != -1:
            return self.index, self.index + 1
        return self.children[0].get_span()[0], self.children[-1].get_span()[1]

    def __repr__(self):
        return self.tag

    @staticmethod
    def parse_tree(parse):
        word_index = 0
        stack = deque()
        tokens = parse.split()
        for token in tokens:
            if token.startswith('('):
                tag = token[1:]
                if len(stack) > 0:
                    parent = stack[0]
                    tree = Tree(tag, parent)
                    parent.add_child(tree)
                else:
                    tree = Tree(tag)
                stack.appendleft(tree)
            elif token.endswith(')'):
                tag = token[:token.find(')')]
                num_close = len(token) - len(tag)
                parent = stack[0]
                tree = Tree(tag, parent)
                tree.index = word_index
                word_index += 1
                parent.add_child(tree)
                for _ in range(num_close):
                    if len(stack) == 1:
                        return stack.pop()
                    stack.popleft()


def find_head_of_np(np):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    top_level_nouns = list(filter(lambda x: x.tag in noun_tags, np.children))
    if len(top_level_nouns) > 0:
        return top_level_nouns[-1].children[0].index
    top_level_nps = list(filter(lambda x: x.tag == 'NP', np.children))
    if len(top_level_nps) > 0:
        return find_head_of_np(top_level_nps[-1])
    leaves = np.get_yield()
    leaf_nouns = list(filter(lambda x: x.parent.tag in noun_tags, leaves))
    if len(leaf_nouns) > 0:
        return leaf_nouns[-1].index
    return leaves[-1].index