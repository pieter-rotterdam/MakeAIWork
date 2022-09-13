import time as tm


class World:
    
    #constructor function
    def __init__(self):
        
        print ("init")
        
        self.t = 0
        self.dt = 0.1
        
        self.running = True
        
        
    def bang (self, t):
        
        self.t=t 
        
        while self.running:
            self.t += self.dt
            
            print ("Time: ", self.t)
            tm.sleep(self.dt)
            
    def stop(self):