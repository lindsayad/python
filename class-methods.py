class Parent():
    def __init__(self):
        self.print_stuff()

class Child(Parent):
    def __init__(self):
        super(Child, self).__init__()

    def print_stuff(self):
        print("printing stuff")

child = Child()
