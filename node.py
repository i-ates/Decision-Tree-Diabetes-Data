class Node:
    def __init__(self, label, split_on=None, track=False):
        self.label = label
        self.children = {}
        # you may want to add additional fields here...
        self.split_on = split_on
        self.track = track
