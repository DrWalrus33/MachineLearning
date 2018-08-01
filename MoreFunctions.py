import random

health = 10



hurt = True






def hit_the_player():
    global health
    roll_the_dice = random.randrange(1, 7)
    health -= roll_the_dice
hit_the_player()

print("Health is " + str(health))