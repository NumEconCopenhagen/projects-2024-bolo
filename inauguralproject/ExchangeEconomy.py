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

        '''Utility function for agent A.

        Args:
    
        x1a (float between 0 and 1): demand of good 1 by agent A
        x2a (float between 0 and 1): demand of good 2 by agent A
        alpha (float between 0 and 1): preference parameter of agent A: the higher, the greater agent's liking of good 1
        
        Returns:
    
        utility_function_A (float): utility function of agent A 
        '''

        return x1a**self.par.alpha*x2a**(1-self.par.alpha)
    
    def utility_function_B(self,x1b,x2b):

        '''Utility function for agent B.

        Args:
    
        x1a (float between 0 and 1): demand of good 1 by agent B
        x2a (float between 0 and 1): demand of good 2 by agent B
        beta (float between 0 and 1): preference parameter of agent B: the higher, the greater agent's liking of good 1
        
        Returns:
    
        utility_function_B (float): utility function of agent B 
        '''

        return x1b**self.par.beta*x2b**(1-self.par.beta)  

    def demand_A(self,p1,w1A=0.8,w2A=0.3):

        '''This function finds the demand for x1 and x2 for agent A.

        Args:
    
        p1: price of good 1 
        p2: price of good 2
        w1a: initial endowment of good 1 held by agent A
        w2a: initial endowment of good 2 held by agent A
        x1a (float between 0 and 1): demand of good 1 by agent A
        x2a (float between 0 and 1): demand of good 2 by agent A
        alpha (float between 0 and 1): preference parameter of agent A: the higher, the greater agent's liking of good 1
        
        Returns:
    
        demand_A (float): demand of agent A 
        '''
        return self.par.alpha*(p1* w1A + self.par.p2*w2A)/p1,(1-self.par.alpha)*(p1* w1A + self.par.p2*w2A)/self.par.p2
    
    def demand_B(self,p1,w1A=0.8,w2A=0.3):

        '''This function finds the demand for x1 and x2 for agent B.

         Args:
    
        p1: price of good 1 
        p2: price of good 2
        w1a: initial endowment of good 1 held by agent A
        w2a: initial endowment of good 2 held by agent A
        x1a (float between 0 and 1): demand of good 1 by agent A
        x2a (float between 0 and 1): demand of good 2 by agent A
        alpha (float between 0 and 1): preference parameter of agent A: the higher, the greater agent's liking of good 1
        
        Returns:
    
        demand_A (float): demand of agent A 
        '''

        return self.par.beta*(p1* (1-w1A) + self.par.p2* (1-w2A))/p1,(1-self.par.beta)*(p1* (1-w1A) + self.par.p2*(1-w2A))/self.par.p2
    
    #Question 1
    def pareto_opt_allocations(self):
       
        '''This function finds the pareto optimal allocations of x1 and x2 for agent A

         Args:
    
        utility_function_A: utility function of agent A
        utility_function_B: utility function of agent B
        x1a (float between 0 and 1): demand of good 1 by agent A
        x2a (float between 0 and 1): demand of good 2 by agent A
        alpha (float between 0 and 1): preference parameter of agent A: the higher, the greater agent's liking of good 1
    

        Returns:
    
        pareto_opt_allocations (float): pareto optimal allocations of good 1 and 2 for agent A
        '''
    #a.Set up: predefined parameters are recalled and variables to store optimal parameters are created
        par = self.par
        x1a_optimal = []
        x2a_optimal = []
        x1a = []
        x2a = []
    #b.Loop
        #i. iteration over N 
        for i in range(par.N+1): 
            x1a.append(i/par.N)
            x2a.append(i/par.N)
        #ii. iteration over x1a and x2a, i.e. for every x1 in x1a and x2 in x2a
        for x1 in x1a:
            for x2 in x2a:  
                if (self.utility_function_A(x1,x2) >= self.utility_function_A(par.w1A,par.w2A)) and (self.utility_function_B(1-x1,1-x2) >= self.utility_function_B(1-par.w1A,1-par.w2A)): #condition: allocations of both agents ought to yield at least as much utility as that related to their initial endowments   
                    x1a_optimal.append(x1)
                    x2a_optimal.append(x2)
        #iii. return optimal values
        return x1a_optimal,x2a_optimal

    #Question 2
    def check_market_clearing(self):

        '''This function checks if the market is clearing for a given price p1.

        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_A: demand function of agent A
        demand_B: demand function of agent B
        w1A (float between 0 and 1): endowment of good 1 held by agent A
        w2A (float between 0 and 1): endowment of good 2 held by agent A

        Returns:
    
        demand_A (float): demand of agent A 
        '''
    #a. Set up: predefined parameters are recalled and target variable is created as an empty list
        par = self.par
        p1 = []
    #b. Loop
        #i. p1 is assigned a value according to discrete price 
        for i in range(par.N+1): 
            p1.append(0.5 + 2*i/par.N)
        #ii. p1 is converted to a numpy array so as to perform broadcasting
        p1 = np.array(p1)
        #iii. methods self.demand_A and self.demand_B are unpacked: take single input argument p1 and yield a tuple
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)
    #c. Errors are computed
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2,p1

    #Question 3
    def market_clearing_price(self):

        '''This function finds the market clearing price and the relative demand for x1 and x2 for agent A.

        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_A: demand function of agent A
        demand_B: demand function of agent B
        w1A (float between 0 and 1): endowment of good 1 held by agent A
        w2A (float between 0 and 1): endowment of good 2 held by agent A

        Returns:
    
        p1_optimal, demand1_A(p1_optimal), demand2_A(p1_optimal), self.utility_function_A(self.demand_A(p1_optimal)) (float): Market clearing p1, related demands of agent A, related utility of A 
        '''
    #a. Set up: predefined parameters are recalled, initial guess for the market clearing price is specified, the objective function is defined
        par = self.par
        p_initial = 0.5
        obj1 = lambda p1: (self.demand_A(p1[0])[0] -par.w1A + self.demand_B(p1[0])[0] - (1-par.w1A))
    #b. Optimization: function root from scipy optimization module uses an iterative algorithm to refine the initial guess until it converges
        result = optimize.root(obj1, p_initial)
        p1_optimal = result.x[0]
    #c. Results: optimal values of target variables are returned
        return p1_optimal, self.demand_A(p1_optimal)[0],self.demand_A(p1_optimal)[1],self.utility_function_A(self.demand_A(p1_optimal)[0],self.demand_A(p1_optimal)[1])
    
    #Question 4.a
    def max_utility_price_discrete(self):
            
        '''This function finds the maximum utility of agent A subject to the constraint of discrete and given prices.

        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_B: demand function of agent B
        utility_function_A: utility function of agent A


        Returns:
    
        p1_optimal, p1_nonnegative, u, u[argmax(u)], demand1_A(p1_optimal), demand2_A(p1_optimal) (float): price maximising utility of A in specified framework, related demands of good 1 and good 2 by agent A and related maximised utility of agent A
        '''
    #a. Set up: predefined parameters are recalled, empty lists are created to store values of p1, p1_nonnegative and u for intermediate operations
        par = self.par
        p1_nonnegative = []
        u = []
        p1 = []
    #b. Loops: 
    # 1) every N is assigned a price value according to discrete price formula in parenthesis
        for i in range(par.N+1): 
            p1.append(0.5 + 2*i/par.N)
    # 2) every p1 is assigned a value for x1 and x2 according to demand functions (Note: demand_A = 1 - demand_B)
        for i in p1:
            x1 =1- self.demand_B(i)[0]
            x2 =1- self.demand_B(i)[1]
            #i. condition: all demands must be bounded between 0 and 1
            if x1<0 or x2<0 or x1>1 or x2>1:
                continue
            #ii. if condition holds, related p1 is stored in the 'p1_nonnegative' list and related utility is stored in 'u' list
            else:
                p1_nonnegative.append(i)  
                u.append(self.utility_function_A(x1,x2))
                
        #c. Optimization: the price that maximizes A's utility is computed
        p1_optimal = p1_nonnegative[np.argmax(u)]
        
        #d. Results are called
        return p1_optimal,p1_nonnegative,u,u[np.argmax(u)],(1- self.demand_B(p1_optimal)[0]),(1-self.demand_B(p1_optimal)[1])

    #Question 4.b
    def max_utility_price_continuous(self):
            
        '''This function finds the maximum utility of agent A subject to the constraint of continuous prices greater than 0.
        
        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_B: demand function of agent B
        utility_function_A: utility function of agent A
    

        Returns:
    
        p1_optimal, demand1_A(p1_optimal), demand2_A(p1_optimal), utility_function_A[demand1_A(p1_optimal), demand2_A(p1_optimal)] (float): price maximising utility of A in specified framework, related demands of good 1 and good 2 by agent A and related maximised utility of agent A
        '''

        # a. Set up: the objective function is defined
        def obj(p1):
            x1 =1- self.demand_B(p1)[0]
            x2 =1- self.demand_B(p1)[1]
            return -self.utility_function_A(x1,x2)

        #b. Optimization:'minimize_scalar' function from 'optimize' module in scipy is employed to minimize 'obj', a scalar function of one variable
        result = optimize.minimize_scalar(obj,bracket=(0.6,2.5)) #Braket argument specifies function obj has a minimum between the two specified boundaries
        p1_optimal = result.x 

        #c. Results are called
        return p1_optimal,(1- self.demand_B(p1_optimal)[0]),(1-self.demand_B(p1_optimal)[1]),self.utility_function_A(1- self.demand_B(p1_optimal)[0],(1- self.demand_B(p1_optimal)[1]))
    
    #Question 5a
    def market_maker_A_restrictedtoC(self):

        '''This function finds the optimal allocations of x1 and x2 for agent A if the choice set is restricted to C.
        
        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_B: demand function of agent B
        utility_function_A: utility function of agent A


        Returns:
    
        p1_optimal, demand1_A(p1_optimal), demand2_A(p1_optimal), utility_function_A[demand1_A(p1_optimal), demand2_A(p1_optimal)] (float): price maximising utility of A in specified framework, related demands of good 1 and good 2 by agent A and related maximised utility of agent A
        '''
        #a. Set up: a list is created to store values of utility
        utilityfunction = []

        #b. Loop: iteration over each tuple of the set C containing pareto optimal allocations  
        for i,j in zip(self.pareto_opt_allocations()[0],self.pareto_opt_allocations()[1]):
            #i. A's utility is calculated for each tuple
            utilityfunction.append(self.utility_function_A(i,j))  
            #ii. maximal utility among those computed is found
            max_utility = utilityfunction.index(np.max(utilityfunction))
        #c. Results are called
        return self.pareto_opt_allocations()[0][max_utility],self.pareto_opt_allocations()[1][max_utility],self.utility_function_A(self.pareto_opt_allocations()[0][max_utility],self.pareto_opt_allocations()[1][max_utility])
    
    #Question 5b
    def market_maker_A_unrestrictedtoC(self):

        '''This function finds the optimal allocations of x1 and x2 for agent A if the only constraint is that B must not be worse of than in the initial
            endowment.

        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_B: demand function of agent B
        utility_function_B: utility function of agent B
        w1A (float between 0 and 1): endowment of good 1 held by agent A
        w2A (float between 0 and 1): endowment of good 2 held by agent A

        Returns:
    
        sol1_case2, sol2_case2, utility_function_A[sol1_case2, sol2_case2] (float): optimal allocation and utility when A is the market maker and constraints are partially loosened wrt case 5a
        '''
    #a. Set up:
        par = self.par # Predefined parameters are recalled
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_function_B(1-x[0],1-x[1]) - self.utility_function_B(1-par.w1A,1-par.w2A)})
        bounds = [(0, 1), (0, 1)]  # Bounds for x1 and x2
        initial_guess = [0.80,0.89] # Initial guess
    #b. Optimization: solver is called, method SLSQP is used
        sol_case2 = optimize.minimize(lambda x: -self.utility_function_A(x[0], x[1]), 
                                      initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return sol_case2.x[0], sol_case2.x[1],self.utility_function_A(sol_case2.x[0], sol_case2.x[1])
    
    #Question 6a
    def social_planner_optimum(self):
            
        '''This function finds the social planner optimum if the only constraint is that x1A AND x2A must be between 0 and 1

        Args:
    
        pareto_opt_allocations: pareto optimal allocations contained in set C are called
        demand_B: demand function of agent B
        utility_function_A: utility function of agent A
        utility_function_B: utility function of agent B

        Returns:
    
        sol1_case3, sol2_case3, utility_function_A[sol1_case3, sol2_case3] (float): optimal allocation when a social planneer maximises aggregate utility, related maximised utility
        '''
    #a. Set up:
        par = self.par
        bounds = [(0, 1), (0, 1)]  # Bounds for x1 and x2
        initial_guess = [0.3,0.3] # Initial guess
    
    #b. Optimization: solver is called, method SLSQP is used
        sol_case3 = optimize.minimize(lambda x: -(self.utility_function_A(x[0], x[1])+self.utility_function_B(1-x[0], 1-x[1])),
                                          initial_guess, method='SLSQP', bounds=bounds)  
    
    #c. Results are called
        return sol_case3.x[0], sol_case3.x[1],self.utility_function_A(sol_case3.x[0], sol_case3.x[1])
    
    #Question 8

    def edgeworth_box(self):

        '''This function finds the allocations of x1 and x2 for agent A and B in the Edgeworth box given the random draws of w1a and w2a.

        Args:
    
        p1: price of good 1 
        p2: price of good 2
        demand_A: demand function of agent A
        demand_B: demand function of agent B
        w1A (float between 0 and 1): endowment of good 1 held by agent A
        w2A (float between 0 and 1): endowment of good 2 held by agent A
        N: number of observations
        P: number of draws for w1a and w2a

        Returns:
    
        sol1_case3, sol2_case3, utility_function_A[sol1_case3, sol2_case3] (float): optimal allocation when a social planneer maximises aggregate utility, related maximised utility
        '''
    #a. Set up
        par = self.par # Predefined parameters are recalled
        np.random.seed(seed=1234)  # Seed is set to ensure reproducibility
        #empty lists are created to store p1, x1 and x2
        p1 = [] 
        x1 = []
        x2 = []
        i = 0
    #b. Loop
        #i. while loop is initiated: for values of N lower than P=50
        while i < par.P: 
            
            #ii. random values for w1a and w2a are drawn from a uniform distribution
            w1a = np.random.uniform(0, 1, size=1)
            w2a = np.random.uniform(0, 1, size=1)
            #iii. Optimization: market equilibrium allocation is found for each tuble of A's randomly drawn endowment 
            obj = lambda p1: np.sum(self.demand_A(p1,w1A=w1a,w2A=w2a)[0] + self.demand_B(p1,w1A=w1a,w2A=w2a)[0]) - 1 
            result = optimize.root(obj, 0.7, method='hybr')
            p1_optimal = result.x[0]
            p1.append(p1_optimal)
            x1.append(self.demand_A(p1_optimal,w1A=w1a,w2A=w2a)[0])
            x2.append(self.demand_A(p1_optimal,w1A=w1a,w2A=w2a)[1])
            i += 1
    #c. Results are called
        return p1, x1, x2