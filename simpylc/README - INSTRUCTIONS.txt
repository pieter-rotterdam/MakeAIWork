start_client2.sh - keuze tussen kerastuner en pietercoded_client
pietercoded_client is gebouwd op basis uitkomsten kerastuner 
variabelen uit opdracht zijn gebruikt in kerastuner: 2-5 lagen 8-256 nodes per laag

start_world.sh - ongewijzigd

start_agent.sh - start de auto die ik geprogrammeerd heb. 

korte roadmap van project:
- informatie verzamelen over hardcoded model en met klassen diagram en sequence model de code in kaart brengen
- data verzameld via hardcoded client over lidar en sonar
- track aangepast om te testen met alternatieve data (zie sonar.tracka en sonarsamplesa)
- tweaks gedaan in parameters.py om auto op hogere doelsnelheid te testen
- tutorials neural net volgen en toepassen op eigen probleem
- selfcoded client maken obv hardcoded
- neural net tuner toegepast en model gebouwd

to do: 
- pietercodedclient kerastuner broke, fix it
- testen niet genormaliseerde data vs genormaliseerde data en if statements in stuurhoek
- eigen code review om te zien waar verbeteringen mogelijk zijn
- beter inzicht verkrijgen in normalisatie van data en wat hij terug geeft.
- kijken of uitkomsten beter worden wanneer we focussen op sturen zonder rechtdoor (0) waarden
- plots maken en actiepunten schrijven opv. uitkomsten


HUIDIG MODEL, SAMPLES, TESTUITKOMSTEN 

model: sonarmode1.h5 , weights in model_weights folder

samples: 
sonar.samples.csv

testuitkomsten:
loghcc.txt
logtuner.txt
logtunerimplement.txt
de map tuner25

