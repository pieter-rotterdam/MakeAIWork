import time as tm

running = True

t = 0.0 #time in seconds
dt = 0.1 #change in time
m = 600.0 #car mass in kg
f = 800.0  #force in newton
v = 0.0 #car start speed in m/s
x= 0.0 # displacement in meters

while running:
    t += dt #t = t + dt
    tm.sleep (0.1)
    print ("Time: ", t)

    a = f / m

    dv = a * dt #change speed
    v = v + dv #current speed

    print("v: ", v)

    dx = v * dt
    x = x + dx

    print ("x: ", x)

    if x > 100.0:
       break #stop loop
   
print("Duration: ", t, " seconds")