#!/usr/bin/env python WAT IS DIT?

soho = ("SOlar", "Heliospheric", "Observatory")

# print(f"The SOHO {', '.join(soho) } telescope is studying and staring at the Sun")

neptuneInnerMoons = [
    "Naiad",
    "Thalassa",
    "Despina",
    "Galatea",
    "Proteus",
    "Hippocamp",
    "Larissa",
]
neptuneOutermoons = [
    "Triton",
    "Halimede",
    "Sao",
    "Psamathe",
    "Laomedeia",
    "Neso",
    "Nereid",
]
""" print ("het aantal binnen en buiten manen van neptune is gelijk") if len(neptuneOutermoons) == len(neptuneInnerMoons) else print ("het aantal binnen en buiten manen van neptune is ongelijk")


 print("%s is the largest moon of Neptune." %(neptuneOutermoons[0]))
"""
neptuneOutermoons.sort()
 #print("%s is the largest moon of Neptune." %(neptuneOutermoons[0]))

#print(f"Neptune is the farthest planet of our solar system and it has { len(neptuneInnerMoons) + len(neptuneOutermoons) } known moons.")

neptuneMoons = []

for neptuneOuterMoon in neptuneOutermoons:
    neptuneMoons.append(neptuneOuterMoon)
for neptuneInnerMoon in neptuneInnerMoons:
    neptuneMoons.append(neptuneInnerMoon)

neptuneMoons.sort()

#print(neptuneMoons)

firstNeptuneMoon = neptuneMoons.pop()
"""
print(f"Popped {firstNeptuneMoon} from neptuneMoons")
print(f"{neptuneMoons}")
"""
lowerMoons = []

for neptuneMoon in neptuneMoons:
    lowerMoons.append(neptuneMoon.lower())

print(lowerMoons)
