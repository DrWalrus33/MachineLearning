import pygame
from MyDog import MyDog
class PetGame:
    def main(self):
        a_dog = MyDog("Rover", 6)
        a_dog.bark()
    if __name__ == "__main__":
        main()