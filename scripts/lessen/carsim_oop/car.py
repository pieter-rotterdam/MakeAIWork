class Car:                                                                                                              
    
    #constructor creation
    def __init__(self, name, f, m):
        
        self.name = name #variabel zodat je generieke auto kunt neerzetten, ook voor f en m
        self.f = f
        self.m = m
        self.x = 0.0
        self.v = 0.0
        
    def move(self, dt):
        
        a = self.f / self.m

        dv = a * dt #change speed
        self.v = self.v + dv #current speed
        print("v: ", self.v)

        dx = self.v * dt
        self.x = self.x + dx  
        
        return self.x, self.v
        print("x: ", self.x)