from objectoriented import MyPet
class MyDog(MyPet):
    def __int__(self, name, volume):
        self.name = name
        self.volume = volume
    def bark(self):
        if self.volume > 3:
            print("Woof woof woof")
        else:
            print("Yip yip yip")
happy_dog = MyDog(MyPet)
happy_dog.bark()