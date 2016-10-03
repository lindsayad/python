class MyBadInitClass:
    def __init__(self, name):
        self.name = name

    def name_foo(self, arg):
        print(self)
        print(arg)
        print("My name is", self.name)


class MyNewClass:
    def new_foo(self, arg):
        print(self)
        print(arg)


my_new_object = MyNewClass()
my_new_object.new_foo("NewFoo")
my_bad_init_object = MyBadInitClass(name="Test Name")
my_bad_init_object.name_foo("name foo")
