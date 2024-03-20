from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
class ExchangeEconomyClass:

    def __init__(self,**kwargs):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.N = 75
        par.p2=1
        par.P=50

        for key, value in kwargs.items():
            setattr(self,key,value) 

    #Funcitons  
    def utility_function_A(self,x1a,x2a):

        '''Utility function for agent A.'''

        return x1a**self.par.alpha*x2a**(1-self.par.alpha)
    
    def utility_function_B(self,x1b,x2b):

        '''Utility function for agent B.'''

        return x1b**self.par.beta*x2b**(1-self.par.beta)  

    def demand_A(self,p1):

        '''This function finds the demand for x1 and x2 for agent A.'''
        
        return self.par.alpha*(p1* self.par.w1A + self.par.p2*self.par.w2A)/p1,(1-self.par.alpha)*(p1* self.par.w1A + self.par.p2*self.par.w2A)/self.par.p2
    
    def demand_B(self,p1):

        '''This function finds the demand for x1 and x2 for agent B.'''

        return self.par.beta*(p1* (1-self.par.w1A) + self.par.p2* (1-self.par.w2A))/p1,(1-self.par.beta)*(p1* (1-self.par.w1A) + self.par.p2*(1-self.par.w2A))/self.par.p2
    
    #Question 1
    def pareto_opt_allocations(self):
       
        '''This function finds the pareto optimal allocations of x1 and x2 for agent A'''

        par = self.par
        x1a_optimal = []
        x2a_optimal = []
        x1a = []
        x2a = []
        for i in range(par.N+1): 
            x1a.append(i/par.N)
            x2a.append(i/par.N)
        for x1 in x1a:
            for x2 in x2a:  
                if (self.utility_function_A(x1,x2) >= self.utility_function_A(par.w1A,par.w2A)) and (self.utility_function_B(1-x1,1-x2) >= self.utility_function_B(1-par.w1A,1-par.w2A)):
                    x1a_optimal.append(x1)
                    x2a_optimal.append(x2)
    
        return x1a_optimal,x2a_optimal

    #Question 2
    def check_market_clearing(self):

        '''This function checks if the market is clearing for a given price p1.'''

        par = self.par
        p1 = []
        for i in range(par.N+1): 
            p1.append(0.5 + 2*i/par.N)
        
        p1 = np.array(p1)
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2,p1

    #Question 3
    def market_clearing_price(self):

        '''This function finds the market clearing price and the relative demand for x1 and x2 for agent A.'''

        par = self.par
        p_initial = 0.5
        obj1 = lambda p1: (self.demand_A(p1)[0] -par.w1A + self.demand_B(p1)[0] - (1-par.w1A))
        result = optimize.root(obj1, p_initial)
        p1_optimal = result.x

        return p1_optimal, self.demand_A(p1_optimal)[0],self.demand_A(p1_optimal)[1],self.utility_function_A(self.demand_A(p1_optimal)[0],self.demand_A(p1_optimal)[1])
    
    #Question 4.a
    def max_utility_price_discrete(self):
            
        '''This function finds the maximum utility of agent A subject to the constraint of discrete and given prices.'''

        par = self.par
        p1_nonnegative = []
        u = []
        p1 = []
        for i in range(par.N+1): 
            p1.append(0.5 + 2*i/par.N)
        for i in p1:
            x1 =1- self.demand_B(i)[0]
            x2 =1- self.demand_B(i)[1]
            if x1<0 or x2<0 or x1>1 or x2>1:
                continue
            else:
                p1_nonnegative.append(i)  
                u.append(self.utility_function_A(x1,x2))
                
        #let's find the price that maximizes the utility
        p1_optimal = p1_nonnegative[np.argmax(u)]
    
        return p1_optimal,p1_nonnegative,u,u[np.argmax(u)],(1- self.demand_B(p1_optimal)[0]),(1-self.demand_B(p1_optimal)[1])

    #Question 4.b
    def max_utility_price_continuous(self):
            
        '''This function finds the maximum utility of agent A subject to the constraint of continuous prices greater than 0.'''

        # Define the objective function
        def obj(p1):
            x1 =1- self.demand_B(p1)[0]
            x2 =1- self.demand_B(p1)[1]
            return -self.utility_function_A(x1,x2)

        # Find the optimal p1
        result = optimize.minimize_scalar(obj,bracket=(0.6,2.5))
        p1_optimal = result.x 
        return p1_optimal,(1- self.demand_B(p1_optimal)[0]),(1-self.demand_B(p1_optimal)[1]),self.utility_function_A(1- self.demand_B(p1_optimal)[0],(1- self.demand_B(p1_optimal)[1]))
    

    #Question 5a
    def market_maker_A_restrictedtoC(self):

        '''This function finds the optimal allocations of x1 and x2 for agent A if the choice set is restricted to C.'''

        utilityfunction = []
        for i,j in zip(self.pareto_opt_allocations()[0],self.pareto_opt_allocations()[1]):
            utilityfunction.append(self.utility_function_A(i,j))  
            max_utility = utilityfunction.index(np.max(utilityfunction))
        return self.pareto_opt_allocations()[0][max_utility],self.pareto_opt_allocations()[1][max_utility],self.utility_function_A(self.pareto_opt_allocations()[0][max_utility],self.pareto_opt_allocations()[1][max_utility])
    
    #Question 5b
    def market_maker_A_unrestrictedtoC(self):

        '''This function finds the optimal allocations of x1 and x2 for agent A if the only constraint is that B must not be worse of than in the initial
            endowment.'''

        par = self.par
        x = self.pareto_opt_allocations()
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_function_B(1-x[0],1-x[1]) - self.utility_function_B(1-par.w1A,1-par.w2A)})
        bounds = [(0, 1), (0, 1)]  # Bounds for x1 and x2
        initial_guess = [0.80,0.89] # Initial guess
        # call solver, use SLSQP
        sol_case2 = optimize.minimize(lambda x: -self.utility_function_A(x[0], x[1]), 
                                      initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return sol_case2.x[0], sol_case2.x[1],self.utility_function_A(sol_case2.x[0], sol_case2.x[1])
    
    #Question 6a
    def social_planner_optimum(self):
            
            '''This function finds the social planner optimum if the only constraint is that x1A AND x2A must be between 0 and 1'''
    
            par = self.par
            x = self.pareto_opt_allocations()
            bounds = [(0, 1), (0, 1)]  # Bounds for x1 and x2
            initial_guess = [0.3,0.3] # Initial guess

            sol_case3 = optimize.minimize(lambda x: -(self.utility_function_A(x[0], x[1])+self.utility_function_B(1-x[0], 1-x[1])),
                                          initial_guess, method='SLSQP', bounds=bounds)  
    
            return sol_case3.x[0], sol_case3.x[1],self.utility_function_A(sol_case3.x[0], sol_case3.x[1])

    
    
    #Question 8

    def edgeworth_box(self):

        '''This function finds the allocations of x1 and x2 for agent A and B in the Edgeworth box given the random draws of w1a and w2a.'''

        par=self.par
        np.random.seed(seed=1234)
        
        p1 = []
        x1 = []
        x2 = []
        i = 0

        while i < par.P: 

            par.w1a = np.random.uniform(0,1,size=1)
            par.w2a = np.random.uniform(0,1,size=1)
            obj = lambda p1: np.sum(self.demand_A(p1)[0]+ self.demand_B(p1)[0]) -1 
            result = optimize.root(obj,0.7, method='hybr')
            p1_optimal = result.x[0]
            p1.append(p1_optimal)
            x1.append(self.demand_A(p1_optimal)[0])
            x2.append(self.demand_A(p1_optimal)[1])
            i +=1
        return p1,x1,x2


    