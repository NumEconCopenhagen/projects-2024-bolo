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
    
        self
        x1a (float between 0 and 1): demand of good 1 by agent A
        x2a (float between 0 and 1): demand of good 2 by agent A
        
        Returns:
    
        utility_function_A (float): utility function of agent A 
        '''

        return x1a**self.par.alpha*x2a**(1-self.par.alpha)
    
    def utility_function_B(self,x1b,x2b):

        '''Utility function for agent B.

        Args:
    
        self
        x1b (float between 0 and 1): demand of good 1 by agent B
        x2b (float between 0 and 1): demand of good 2 by agent B
        
        Returns:
    
        utility_function_B (float): utility function of agent B 
        '''

        return x1b**self.par.beta*x2b**(1-self.par.beta)  

    def demand_A(self,p1,w1A=0.8,w2A=0.3):

        '''This function finds the demand for x1 and x2 for agent A.

        Args:
    
        self 
        p1: price of good 1 
        w1A: initial endowment of good 1 held by agent A
        w2A: initial endowment of good 2 held by agent A
        
        Returns:
    
        demand_A (float): demand of agent A 
        '''
        return self.par.alpha*(p1* w1A + self.par.p2*w2A)/p1,(1-self.par.alpha)*(p1* w1A + self.par.p2*w2A)/self.par.p2
    
    def demand_B(self,p1,w1A=0.8,w2A=0.3):

        '''This function finds the demand for x1 and x2 for agent B.

        Args:
    
        self: parameters of the model and functions defined in the class
        p1: price of good 1 
        w1A: initial endowment of good 1 held by agent A
        w2A: initial endowment of good 2 held by agent A

        Returns:
    
        demand_B (float): demand of agent B 
        '''

        return self.par.beta*(p1* (1-w1A) + self.par.p2* (1-w2A))/p1,(1-self.par.beta)*(p1* (1-w1A) + self.par.p2*(1-w2A))/self.par.p2
    
    #Question 1
    def pareto_opt_allocations(self):
       
        '''This function finds the pareto optimal allocations of x1 and x2 for agent A

        Args: 
        
        self: parameters of the model and functions defined in the class    
         
        Returns:
    
        x1a_optimal (float): optimal allocation of good 1 for agent A
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

    def plot_allocation_diagram(self):

        '''This function plots the allocation diagram for agent A and B.

            Args: 
            
            self: parameters of the model and functions defined in the class

            Returns:

            Allocation diagram for agent A and B
            '''

        par = self.par
        w1bar = 1.0
        w2bar = 1.0
        
        fig = plt.figure(frameon=True, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        ax_A.scatter(par.w1A, par.w2A, marker='s', color='r', label='Endowment')
        
        x1a_pareto_optimal, x2a_pareto_optimal = self.pareto_opt_allocations()
        ax_A.scatter(x1a_pareto_optimal, x2a_pareto_optimal, marker='s', color='lightskyblue', label='Pareto Optimal')

        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))
        
        plt.show()

    #Question 2
    def check_market_clearing(self):

        '''This function checks if the market is clearing for a given price p1.

        Args:
    
        self: parameters of the model and functions defined in the class

        Returns:
    
        eps1, eps2, p1 (float): errors in the market clearing conditions for both goods
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

    def plot_errors(self):
        '''Plot errors in the market clearing conditions for both goods
        
        Args: 
        
        self: parameters of the model and functions defined in the class
        
        Returns:
        
        Plot of errors in the market clearing conditions for both goods.
        '''
        # Check errors and prices in markets
        eps1, eps2, p1 = self.check_market_clearing()
        
        # Plotting errors
        plt.figure(figsize=(10, 6))
        
        plt.plot(p1, eps1, label="Error in market of good 1")
        plt.plot(p1, eps2, label="Error in market of good 2")
        
        plt.xlabel('Price of good 1 (p1)')
        plt.ylabel('Market clearing error')
        plt.title('Error in Markets for good 1 and 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    

    #Question 3
    def market_clearing_price(self):

        '''This function finds the market clearing price and the relative demand for x1 and x2 for agent A.

        Args:
    
        self : parameters of the model and functions defined in the class

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
        return p1_optimal, self.demand_A(p1_optimal)[0],self.demand_A(p1_optimal)[1],self.utility_function_A(self.demand_A(p1_optimal)[0],self.demand_A(p1_optimal)[1]),result

    def plot_market_clearing_errors(self):

        '''Plot errors in the market clearing conditions for both goods and the market clearing price
        
        Args:
        
        self: parameters of the model and functions defined in the class
        
        Returns:
        
        Plot of errors in the market clearing conditions for both goods and the market clearing price.
        '''
    
        # Recall results 
        eps1, eps2, p1 = self.check_market_clearing()
        p1_optimal, _, _, _, results = self.market_clearing_price()
        
        # Plot errors
        plt.figure(figsize=(10, 6))
        plt.plot(p1, eps1, label="Error in market of good 1")
        plt.plot(p1, eps2, label="Error in market of good 2")
        
        plt.xlabel('Price of good 1 (p1)')
        plt.ylabel('Market clearing error')
        plt.scatter(p1[np.argmin(abs(eps1))], np.min(abs(eps1)), color='red', label='Market clearing price - Discrete case')
        plt.scatter(p1_optimal, results.fun, color='green', label='Market clearing price - Continuous case')
        
        plt.title('Error in Market clearing condition')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    #Question 4.a
    def max_utility_price_discrete(self):
            
        '''This function finds the maximum utility of agent A subject to the constraint of discrete and given prices.

        Args:
    
        self: parameters of the model and functions defined in the class

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
    
    def plot_utility_nonnegative_prices(self):
        '''
        Plot utility of agent A against p1_nonnegative prices, highlighting the optimal price

        Args:

        self: parameters of the model and functions defined in the class

        Returns:

        Plot of utility of agent A against p1_nonnegative prices, highlighting the optimal price
        '''
        p1_optimal, p1_nonnegative, u, max_utility, demand1, demand2 = self.max_utility_price_discrete()

        plt.plot(p1_nonnegative, u, label="utility")
        plt.scatter(p1_optimal, max_utility, label="maximal utility", color='red')
        plt.xlabel('p1')
        plt.ylabel('utility')
        plt.title('Utility of Agent A')
        plt.legend()
        plt.grid(True)
        plt.show()

    #Question 4.b
    def max_utility_price_continuous(self):
            
        '''This function finds the maximum utility of agent A subject to the constraint of continuous prices greater than 0.
        
        Args:
    
        self: parameters of the model and functions defined in the class

        Returns:
    
        p1_optimal, demand1_A(p1_optimal), demand2_A(p1_optimal), utility_function_A[demand1_A(p1_optimal), demand2_A(p1_optimal)] (float): price maximising utility of A, when a continuum of prices is considered, 
        related demands of good 1 and good 2 by agent A and related maximised utility of agent A
        '''

        # a. Set up: the objective function is defined
        def obj(p1):
            x1 = 1 - self.demand_B(p1)[0]
            x2 = 1 - self.demand_B(p1)[1]
            return -self.utility_function_A(x1, x2)

        # Define the constraints
        constraints = ({'type': 'ineq', 'fun': lambda p1: p1})

        # Initial guess
        x0 = [0.7]

        # b. Optimization: 'minimize' function from 'optimize' module in scipy is employed to minimize 'obj', a scalar function of one variable
        result = optimize.minimize(obj, x0, constraints=constraints, method='SLSQP')

        # Optimal value of p1
        p1_optimal = result.x[0]

        #c. Results are called
        return p1_optimal,(1- self.demand_B(p1_optimal)[0]),(1-self.demand_B(p1_optimal)[1]),self.utility_function_A(1- self.demand_B(p1_optimal)[0],(1- self.demand_B(p1_optimal)[1]))
    
    #Question 5a
    def market_maker_A_restrictedtoC(self):

        '''This function finds the optimal allocations of x1 and x2 for agent A if the choice set is restricted to C.
        
        Args:
    
        self: parameters of the model and functions defined in the class

        Returns:
    
        p1_optimal, demand1_A(p1_optimal), demand2_A(p1_optimal), utility_function_A[demand1_A(p1_optimal), demand2_A(p1_optimal)] (float): price maximising utility of A when only prices in C are considered,
        related demands of good 1 and good 2 by agent A and related maximised utility of agent A
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
    
        self: parameters of the model and functions defined in the class

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
    
        self:   parameters of the model and functions defined in the class

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

    #Question 7: Plot all pareto allocations in the Edgeworth box
    def plot_edgeworth_box_all(self):

        '''This function plots the Edgeworth box with all optimal allocations.
        
        Args: 
        
        self: parameters of the model and functions defined in the class

        Returns:

        Edgeworth box with all optimal allocations
        '''

        p1_equilibrium,x1a_equilibrium,x2a_equilibrium,u_equilibrium,results= self.market_clearing_price()
        x1a_pareto_optimal,x2a_pareto_optimal= self.pareto_opt_allocations()
        p1_4a_optimal,p1_nonnegative,u,u_4a_optimal,x1a_4a_optimal,x2a_4a_optimal= self.max_utility_price_discrete()
        p1_4b_optimal,x1a_4b_optimal,x2a_4b_optimal,u_4b_optimal= self.max_utility_price_continuous()
        x1a_5a_optimal,x2a_5a_optimal,u_5a_optimal= self.market_maker_A_restrictedtoC()
        x1a_5b_optimal,x2a_5b_optimal, u_5b_optimal= self.market_maker_A_unrestrictedtoC()
        x1a_social_optimum,x2a_social_optimum,u_social_a_optimal= self.social_planner_optimum()        
        
        # Define the dimensions of the Edgeworth box
        w1bar = 1.0
        w2bar = 1.0
        
        # Set up the figure and axes
        fig = plt.figure(frameon=True,figsize=(8,8), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # Define a smaller size for the scatter points
        size = 10

        # Plot the different points in the Edgeworth box
        ax_A.scatter(x1a_equilibrium, x2a_equilibrium, marker='s', color='black', label='3', s=size)
        ax_A.scatter(x1a_4a_optimal, x2a_4a_optimal, marker='s', color='red', label='4a', s=size)
        ax_A.scatter(x1a_4b_optimal, x2a_4b_optimal, marker='s', color='yellow', label='4b', s=size)
        ax_A.scatter(x1a_5a_optimal, x2a_5a_optimal, marker='s', color='green', label='5a', s=size)
        ax_A.scatter(x1a_5b_optimal, x2a_5b_optimal, marker='s', color='blue', label='5b', s=size)
        ax_A.scatter(x1a_social_optimum, x2a_social_optimum, marker='s', color='purple', label='6a', s=size)
        ax_A.scatter(x1a_pareto_optimal, x2a_pareto_optimal, marker='s', color='blue', label='Pareto allocations', alpha=0.09, s=size)
        ax_A.scatter(self.par.w1A, self.par.w2A, marker='s', color='r', label='endowment', s=size)

        # Plot the limits of the Edgeworth box
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))

        plt.show()

    
    #Question 8: plot random draws of w1a and w2a in Edgeworth box

    def equilibrium_allocation_random(self):

        ''' Market equilibrium allocation for each tuble of A's randomly drawn endowment.
        
        Args: 
        
        self: parameters of the model and functions defined in the class

        Returns:
    
        p1, x1, x2 (float): market equilibrium allocation for each tuble of A's randomly drawn endowment
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


    def random_edgeworth_box(self):

        '''This function plots the Edgeworth box with random draws of w1a and w2a.

        Args: 
        
        self: parameters of the model and functions defined in the class

        Returns:

        Edgeworth box based on random draws of w1a and w2a
        '''

        p1_edg,x1_edg,x2_edg = self.equilibrium_allocation_random() # Random draws of w1a and w2a are called
        par = self.par # Predefined parameters are recalled

        w1bar = 1.0
        w2bar = 1.0

        fig = plt.figure(frameon=True, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        ax_A.scatter(x1_edg, x2_edg, marker='s', color='black', label='allocations')

        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))

        plt.show()