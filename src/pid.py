import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, target=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.last_error = 0
        self.integral = 0
        self.dt = 1.0

    def update(self, current_value):
        error = self.target - current_value
        
        P_term = self.Kp * error
        self.integral += error * self.dt
        I_term = self.Ki * self.integral
        derivative = (error - self.last_error) / self.dt if self.dt > 0 else 0
        D_term = self.Kd * derivative
        output = P_term + I_term + D_term
        
        self.last_error = error

        return output

    def set_target(self, target):
        self.target = target

    def set_tunings(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
    def reset(self):
        self.integral = 0
        
        
if __name__ == "__main__":
    p,v,a = 0,0,0
    m = 1.0
    dt = 1.0
    target_p = 10
    
    def forward(p,v,a,f,m,dt=1.0):
        p += v * dt + 0.5 * a * dt**2
        v += dt*a
        a = f/m
        return p,v,a
    
    pidc = PIDController(0.01, 0.001, 0.5, target_p)
    
    p_over_time = [p]
    for _ in range(1000):
        f = pidc.update(p)
        p,v,a = forward(p,v,a,f,m,dt)
        print(f"p: {p}, v: {v}, a: {a}, f: {f}")
        p_over_time.append(p)
        
        
    plt.figure()
    plt.plot(list(range(len(p_over_time))), p_over_time)
    plt.show()
    