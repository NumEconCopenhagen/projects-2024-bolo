import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from types import SimpleNamespace
import numba as nb


class EntryExitModel():
     
    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.symbolic = SimpleNamespace()
        if do_print: print('calling .setup()')
        self.setup()

    def setup(self):
        """ baseline parameters 
        """
        par = self.par
        symbolic = self.symbolic
        par.b = 1
        par.F = 1
        par.c = 0
        par.d = 5
        par.a = par.d - par.c

        symbolic.a = sm.symbols('a')
        symbolic.b = sm.symbols('b')
        symbolic.q_m = sm.symbols('q_m')
        symbolic.q_e = sm.symbols('q_e')
        symbolic.F = sm.symbols('F')

    def profit_e_sym(self):
            
        """ symbolic profit function for the entrant firm 

        Args:
        self: parameters

        Returns:
        profit_e: symbolic profit function for the entrant firm
        """

        symbolic = self.symbolic
        return (symbolic.a - symbolic.b * (symbolic.q_m + symbolic.q_e)) * symbolic.q_e - symbolic.F
    
    def second_stage_of_the_game(self):
        
        """ second stage of the game 

        Args:
        self: parameters

        Returns:
        reaction_e[0]: reaction function for the entrant firm at the second stage
        profit_2nd_stage_e: profit for the entrant firm in the second stage
        Y: threshold quantity of the incumbent firm for the entrant firm to enter the market
        """
        
        symbolic = self.symbolic

        #profit function for the entrant firm
        profit_e = self.profit_e_sym() 
        #reaction function for the entrant firm
        reaction_e = sm.solve(sm.diff(profit_e, symbolic.q_e),symbolic.q_e)
        #threshold quantity of the incumbent firm for the entrant firm to enter the market
        profit_2nd_stage_e=sm.simplify(profit_e.subs(symbolic.q_e,reaction_e[0])) 
        Y = sm.solve(profit_2nd_stage_e,symbolic.q_m)[0]
            
        return reaction_e[0], profit_2nd_stage_e, Y 

    def profit_m_sym(self):

        """ symbolic profit function for the incumbent firm 

        Args:
        self: parameters

        Returns:
        profit_m: symbolic profit function for the incumbent firm
        """

        symbolic = self.symbolic
        profit_m = sm.Piecewise(
            ((symbolic.a - symbolic.b * (symbolic.q_m + symbolic.q_e)) * symbolic.q_m -symbolic.F, symbolic.q_m < self.second_stage_of_the_game()[2]) ,  # This is the first piece of the function. It applies when q_m < Y.
            ((symbolic.a - symbolic.b * (symbolic.q_m)) * symbolic.q_m -symbolic.F, True)  # This is the second piece of the function. It applies otherwise.
        )
        return profit_m

    def first_stage_of_the_game(self,do_print=True):

        """ first stage of the game 

        Args:
        self: parameters

        Returns:
        profit_m: profit function for the incumbent firm
        profit_m_e: profit for the incumbent firm when the entrant firm enters the market
        profit_m_ne: profit for the incumbent firm when the entrant firm does not enter the market
        block_condition: condition for blocking entry
        deterred_condition: condition for deterring entry
        accomodation_condition: condition for accomodating entry
        acc_profit: profit for the incumbent firm when the entrant firm enters the market
        crit_points1: critical points for the incumbent firm when the entrant firm enters the market
        crit_points2: critical points for the incumbent firm when the entrant firm does not enter the market
        """

        symbolic = self.symbolic
        profit_m = self.profit_m_sym()

        #incorporate the reaction function of the entrant firm into the profit function of the incumbent firm
        profit_m = profit_m.subs(symbolic.q_e,self.second_stage_of_the_game()[0])
        
        # optimal quantity for the incumbent firm when the entrant firm enters the market
        expr1, _ = profit_m.args[0]
        deriv1 = sm.diff(expr1, symbolic.q_m)
        crit_points1 = sm.solve(deriv1, symbolic.q_m)
        #profit for the incumbent firm when the entrant firm enters the market
        profit_m_e = expr1.subs(symbolic.q_m, crit_points1[0])
        
        #optimal quantity for the incumbent firm when the entrant firm does not enter the market
        expr2, _ = profit_m.args[1]
        deriv2 = sm.diff(expr2, symbolic.q_m)
        crit_points2 = sm.solve(deriv2, symbolic.q_m)
        #profit for the incumbent firm when the entrant firm does not enter the market
        profit_m_ne = expr2.subs(symbolic.q_m, crit_points2[0])
        
        #entry is blocked under these conditions:
        blocked = sm.solve(sm.Eq(self.second_stage_of_the_game()[2], crit_points1[0]),symbolic.F )  
        block_condition = sm.Ge(symbolic.F,blocked[0])
        
        #entry is deterred under these condtions:
        q_m_hat = sm.solve(expr2 - profit_m_e , symbolic.q_m)[0]  # Find the equilibrium quantity of firm m
        deterred_condition = sm.And(sm.Le((3- 2*sm.sqrt(2))*symbolic.a**2/(32*symbolic.b), symbolic.F), sm.Gt(symbolic.a**2/(16*symbolic.b), symbolic.F))

        #entry is accommodated under these conditions:
        accomodation_condition = sm.And(sm.Lt(0, symbolic.F),sm.Gt((3- 2*sm.sqrt(2))*symbolic.a**2/(32*symbolic.b), symbolic.F))
        
        #the profit when the entry is accomodated is:
        acc_profit = sm.simplify(expr2.subs(symbolic.q_m, self.second_stage_of_the_game()[2]))

        return profit_m, profit_m_e, profit_m_ne, block_condition, deterred_condition, accomodation_condition, acc_profit, crit_points1, crit_points2
    
    def profit_m_inverse(self):

        """ This method is just for graphical purposes

        Args:
        self: parameters

        Returns:
        profit_m_inverse: profit function for the incumbent firm with inverted condition needed to do the dot line in the graph
        """

        symbolic = self.symbolic
    
        profit_m_inverse = sm.Piecewise(
            ((symbolic.a - symbolic.b * (symbolic.q_m + symbolic.q_e)) * symbolic.q_m -symbolic.F, symbolic.q_m > self.second_stage_of_the_game()[2]),  # This is the first piece of the function. It applies when q_m < Y.
            ((symbolic.a - symbolic.b * (symbolic.q_m)) * symbolic.q_m -symbolic.F, True)  # This is the second piece of the function. It applies otherwise.
        )
        profit_m_inverse = (profit_m_inverse.subs(symbolic.q_e, self.second_stage_of_the_game()[0])) # profit of firm m in the first stage 
        return profit_m_inverse
    
    def plot_profit_m(self):

        """ plot the profit function for the incumbent firm 

        Args:
        self: parameters
        
        Returns:    
        plot: plot of the profit function for the incumbent firm
        """

        symbolic = self.symbolic
        par = self.par
        F_vals = [0.01, 1, 2.5]  # List of F values
        q_m_values = np.linspace(0, par.a/par.b, 10000)
    
        # Create the figure
        plt.figure(figsize=(15, 5)) 
        for i, F_val in enumerate(F_vals, 1):
            profit_m, profit_m_e, profit_m_ne, block_condition, deterred_condition, accomodation_condition, acc_profit ,crit_points1, crit_points2= self.first_stage_of_the_game(do_print=False)
            Y = self.second_stage_of_the_game()[2]

            # Convert the symbolic expression into a numerical function
            profit_m_e_func =sm.lambdify(symbolic.q_m, profit_m.subs({symbolic.a: par.a, symbolic.b: par.b, symbolic.F: F_val}))
            profit_m_e_func_inverse= sm.lambdify(symbolic.q_m, self.profit_m_inverse().subs({symbolic.a: par.a, symbolic.b: par.b, symbolic.F: F_val}))
            q_m_func = sm.lambdify(symbolic.q_m, symbolic.q_m.subs({symbolic.a: par.a, symbolic.b: par.b, symbolic.F: F_val}))
            Y_val = Y.subs({symbolic.a: par.a, symbolic.b: par.b, symbolic.F: F_val})
            
            # Generate the corresponding y-values
            profit_values = profit_m_e_func(q_m_func(q_m_values))
            profit_values_1 = profit_m_e_func_inverse(q_m_func(q_m_values))
            
            # Create the subplot
            plt.subplot(1, 3, i)  # 1 row, 3 columns, i-th plot

            # Plot the profit function
            plt.plot(q_m_values, profit_values, color="black")
            plt.plot(q_m_values, profit_values_1, color="black", linestyle='--')

            # Evaluate the critical points values
            crit_points_values = [cp.evalf(subs={symbolic.a: par.a, symbolic.b: par.b, symbolic.F: F_val}) for cp in crit_points1]
            crit_points_values_1 = [cp.evalf(subs={symbolic.a: par.a, symbolic.b: par.b, symbolic.F: F_val}) for cp in crit_points2]
            profit_at_crit_points = [profit_m_e_func(cp) for cp in crit_points_values]
            profit_at_crit_points_1 = [profit_m_e_func(Y_val) for cp in crit_points_values_1]

            # Plot the critical points
            plt.scatter(crit_points_values, profit_at_crit_points, color='red', label='Monopoly profit')
            plt.scatter(Y_val, profit_m_e_func(Y_val), color='green', label='Y')

            # Add vertical lines
            plt.axvspan(q_m_values[0], Y_val, ymin=0, ymax=1, color='lightblue', alpha=0.5)

            # Add labels
            plt.xlabel('q_m')
            plt.ylabel('profit_m_e')
            plt.title(f'Profit as a function of the quantity of firm m (F={F_val})')
            if i == 1:
                plt.annotate(f'E does enter', xy=(Y_val, profit_m_e_func(Y_val)), xytext=(Y_val, profit_m_e_func(Y_val)  + 0.5), 
                ha='left', va='bottom', bbox=dict(boxstyle='square', pad=0.003, facecolor='lightblue'));
            elif i ==2:
                 plt.annotate(f'E does not enter', xy=(Y_val, profit_m_e_func(Y_val)), xytext=(Y_val, profit_m_e_func(Y_val)  + 0.01), 
                ha='left', va='bottom', bbox=dict(boxstyle='square', pad=0.003, facecolor='lightblue'));
            elif i ==3:
                 plt.annotate(f'E does not enter', xy=(Y_val, profit_m_e_func(Y_val)), xytext=(Y_val, profit_m_e_func(Y_val)  - 1.5), 
                ha='left', va='bottom', bbox=dict(boxstyle='square', pad=0.003, facecolor='lightblue'));
            plt.tight_layout()
        plt.show()

    def plot_profit_m_2(self):
            
            """ Firm M's equilibrium profit as a function of F

            Args:
            self: parameters

            Returns:
            plot: a plot of the equilibrium profit of the incumbent firm as a function of F.
            """
    
            symbolic = self.symbolic
            par = self.par
            profit_m_e = self.first_stage_of_the_game()[1]
            profit_m_ne = self.first_stage_of_the_game()[2]
            profit_of_firm_1 = sm.Piecewise(
                (profit_m_e, self.first_stage_of_the_game()[5]), 
                (self.first_stage_of_the_game()[6], self.first_stage_of_the_game()[4]),
                (profit_m_ne, self.first_stage_of_the_game()[3]),  
            )

            profit_of_firm_1_lam = sm.lambdify((symbolic.F), profit_of_firm_1.subs({symbolic.a: par.a, symbolic.b: par.b}))
            #The following part is just for graphical purposes without economic reasoning 
            first_profit_lam = sm.lambdify((symbolic.F), profit_m_e.subs({symbolic.a: par.a, symbolic.b: par.b}))
            last_profit_lam = sm.lambdify((symbolic.F), profit_m_ne.subs({symbolic.a: par.a, symbolic.b: par.b}))

            # Create the figure
            F = np.linspace(0,(par.a**2)/(16*par.b)+0.4, 100000)
            profit_values = profit_of_firm_1_lam(F)
            first_profit_values = first_profit_lam(F)
            last_profit_values = last_profit_lam(F)
            plt.plot(F, profit_values,color="black")
            plt.plot(F, first_profit_values, color= "black", linestyle='--')    
            ##add text to the plot
            mask1 = (F > 0.14) & (F < 1.30)
            mask2 = (F > 1.30) & (F < 2)
            mask3 = (F > 0) & (F < 0.14)
            plt.plot(F[mask1], profit_values[mask1], 'orange', lw=2.5, label='Entry is deterred (ED)')
            plt.plot(F[mask2], profit_values[mask2], 'red', lw=2.5, label='Entry is blocked (EB)')
            plt.plot(F[mask3], first_profit_values[mask3], 'green', lw=2.5, label='Entry is accommodated (EA)')
            plt.legend(loc='upper right')
            plt.plot(F[mask3], first_profit_values[mask3], 'green', lw=2.5)
            plt.plot(F, last_profit_values, label='Profit of firm M with entry blocked', color="black", linestyle='--')
            plt.text(0.4, 4.6, 'ED', style='italic', fontsize=12, color='orange')
            plt.text(1.75, 4.6, 'EB', style='italic', fontsize=12, color='red')
            plt.text(0.07, 2.5, 'EA', style='italic', fontsize=12, color='green')
            plt.xlabel('F')
            plt.ylabel('Profit of firm M')
            plt.title('Profit of firm M as a function of F')
            plt.show()

    def demand(self,q_m,q_e):

        """ Demand function 

        Args:
        self: parameters
        q_m: quantity of the incumbent firm
        q_e: quantity of the entrant firm

        Returns:
        demand: demand function
        """

        par = self.par

        if q_m + q_e < par.d/par.b:
            return par.d - par.b*(q_m + q_e)
        else:
            return 0       

    def cost(self,q):

        """ Cost function 

        Args:
        self: parameters
        q: quantity

        Returns:
        cost: cost function
        """

        par = self.par

        return par.c*q + par.F 
    
    def profit_e(self,q_m,q_e):

        """ Profit function for the entrant firm 

        Args:
        self: parameters
        q_m: quantity of the incumbent firm
        q_e: quantity of the entrant firm

        Returns:
        profit: profit function for the entrant firm
        """

        return self.demand(q_m,q_e) * q_e - self.cost(q_e)


    def profit_m(self,q_m):

        """ profit function for the incumbent firm 

        Args:
        q_m: quantity of the incumbent firm
        self: parameters

        Returns:
        profit: profit function for the incumbent firm
        """

        par = self.par
        
        x_entrant = self.entrant_opt(q_m)
        if q_m < self.Y():
            return self.demand(q_m,x_entrant) * q_m - self.cost(q_m)
        else:
            return  self.demand(q_m,0) * q_m - self.cost(q_m)
            

    def entrant_opt(self,q_m): 

        """ optimal entrant quantity 

        Args:
        self: parameters
        q_m: quantity of the incumbent firm

        Returns:
        optimal_q: optimal quantity for the entrant firm
        """

        par = self.par

        def obj(q_e):
            return - self.profit_e(q_m,q_e)
        
        sol =  sol = optimize.minimize(obj, x0=[0.2], bounds=[(0, (par.d-par.c)/par.b)],constraints={'type':'ineq', 'fun': lambda q_e: (par.d-par.c)/par.b - q_e + q_m}, method="SLSQP", tol=1e-10)
        optimal_q = sol.x[0]

        #I put epsilon instead of 0 because of the numerical error
        epsilon = 1e-8
        if self.profit_e(q_m,optimal_q) > epsilon:
            return sol.x[0]
        else:
            return 0
    
    def Y(self):

        """ threshold quantity of the incumbent firm for the entrant firm to enter the market

        Args:
        self: parameters

        Returns:
        soll.root[0]: threshold quantity of the incumbent firm for the entrant firm to enter the market""" 
        def obj(q_m):
            return -self.profit_e(q_m,self.entrant_opt(q_m))
        soll = optimize.root_scalar(obj, x0=[2.8], method='newton')
        return soll.root[0] 
    
    
    def m_opt(self,multiple_start=False,initial_condition_norandom=True,initial_condition_random=False,do_print=True):

        """ Optimal quantity for the incumbent firm

        Args:
        self: parameters
        multiple_start: multiple starting points
        initial_condition_norandom: initial condition without random
        initial_condition_random: initial condition with random

        Returns:
        optimal_q: optimal quantity for the incumbent firm
        q2: optimal quantity for the entrant firm
        """
        
        par = self.par
        
        def obj(q_m):
            return -self.profit_m(q_m)
        
        if multiple_start:
            np.random.seed(168)
            x0s = np.random.uniform(0, (par.d-par.c)/par.b, 50)
            xs = np.empty(50)
            fs = np.empty(50)
            for i,x0 in enumerate(x0s):
                sol = optimize.minimize(obj, x0=[x0],bounds=[(0, (par.d-par.c)/par.b)],constraints={'type':'ineq', 'fun': lambda q_m: (par.d-par.c)/par.b -  self.entrant_opt(q_m) + q_m},method='SLSQP', tol=1e-10)
                xs[i] = sol.x[0]
                fs[i] = -sol.fun
            optimal_q = xs[np.argmax(fs)]  
        elif initial_condition_norandom:
            sol = optimize.minimize(obj, x0=[self.Y()],bounds=[(0, (par.d-par.c)/par.b)],constraints={'type':'ineq', 'fun': lambda q_m: (par.d-par.c)/par.b -  self.entrant_opt(q_m) + q_m},method='SLSQP', tol=1e-10)

            optimal_q = sol.x[0] 
        elif initial_condition_random:
            np.random.seed(16)
            x0 = np.random.uniform(0, (par.d-par.c)/par.b)
            sol = optimize.minimize(obj, x0=[1],bounds=[(0, (par.d-par.c)/par.b)],constraints={'type':'ineq', 'fun': lambda q_m: (par.d-par.c)/par.b -  self.entrant_opt(q_m) + q_m},method='SLSQP', tol=1e-10)
            optimal_q = sol.x[0]
        
        q2 = self.entrant_opt(optimal_q)

        if do_print:
            print(f'Optimal quantity for the incumbent firm: {optimal_q}')
            print(f'Equilibium profit for the incumbent firm: {self.profit_m(optimal_q)}')
            print(f'Optimal quantity for the entrant firm: {q2}') 
            print(f'Equilibium profit for the entrant firm: {self.profit_e(optimal_q,q2)}')
        return optimal_q, q2


    def plot_numerical_solution(self):

        """ plot the numerical solution of the model to see the equilibrium quantities and profits

        Args:
        self: parameters
        
        Returns:
        plot: plot of the numerical solution of the model
        """

        par = self.par

        # Create a grid of F values
        Fs = np.linspace(0,2, 20)

        
        # Initialize lists to store the results
        q_m = []
        q_e = []
        profit_m = []

        for F in Fs:

            # Update the parameter F in the model
            par.F = F
            # Obtain the optimized values
            opt_values = self.m_opt(do_print=False)
            # Append the results to the respective lists
            q_m.append(opt_values[0])
            q_e.append(opt_values[1])
            profit_m.append(self.profit_m(opt_values[0]))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a new figure with 1 row and 2 columns
        # Plot profit_m against F on the first subplot
        axs[0].plot(Fs, profit_m)
        axs[0].set_xlabel('F')
        axs[0].set_ylim(1,6.5) 
        axs[0].set_ylabel('Profit of M')
        axs[0].set_title('Profit of M as a function of F')
        axs[0].grid(True)

        # Plot q_m and q_e against F on the second subplot
        axs[1].plot(Fs, q_m, label='M')
        axs[1].plot(Fs, q_e, label='E')
        axs[1].set_xlabel('F')
        axs[1].set_ylabel('Quantity')
        axs[1].set_title('Quantity M, E as a function of F')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()  # Adjust the layout so that plots do not overlap
        plt.show()  # Display the plots
        
     


    def profit_m_ext(self,q_m): 

        """ Profit function for the incumbent firm

        Args:
        self: parameters
        q_m: quantity of the incumbent firm

        Returns:
        profit: profit function for the incumbent firm
        """
        
        x_entrant = self.entrant_opt_ext(q_m)
        
        return self.demand(q_m,x_entrant) * q_m - self.cost(q_m)
    

    def entrant_opt_ext(self,q_m):  # entrant optimization

        """ Optimal quantity for the entrant firm

        Args:
        self: parameters
        q_m: quantity of the incumbent firm

        Returns:
        sol.x[0]: optimal quantity for the entrant firm
        """
        
        par = self.par
        def obj(q_e):
            return - self.profit_e(q_m,q_e)

        sol = optimize.minimize(obj, x0=[1], bounds=[(0, (par.d-par.c)/par.b)],constraints={'type':'ineq', 'fun': lambda q_e: (par.d-par.c)/par.b - q_e + q_m}, method="SLSQP", tol=1e-10)
        return sol.x[0]

    def m_opt_ext(self):

        """ Optimal quantity for the incumbent firm

        Args:
        self: parameters
       

        Returns:
        optimal_q: optimal quantity for the incumbent firm
        entrant_opt_ext(optimal_q): optimal quantity for the entrant firm
        """
        
        par = self.par
        def obj(q_m):
            return -self.profit_m_ext(q_m)
        sol = optimize.minimize(obj, x0=[1],bounds=[(0, (par.d-par.c)/par.b)],constraints={'type':'ineq', 'fun': lambda q_m: (par.d-par.c)/par.b - self.entrant_opt_ext(q_m) + q_m},method='SLSQP', tol=1e-10)
        optimal_q = sol.x[0] 

        print(f'Optimal quantity for the leader firm: {optimal_q}')
        print(f'Optimal quantity for the follower firm: {self.entrant_opt_ext(optimal_q)}')
        return optimal_q, self.entrant_opt_ext(optimal_q)
