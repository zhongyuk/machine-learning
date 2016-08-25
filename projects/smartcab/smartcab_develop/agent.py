import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

sim_start = {}
sim_destination = {}
sim_success = {}
sim_end_t = {}
sim_deadline = {}
sim_net_reward = {}
sim_penalty = {}

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, omega=1, gamma=.8, epsilon=.2): #
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.omega = omega # power for construct polynomial learning rate
        self.alpha =  None # polynomial learning rate = 1/(t^omega); form a constant learning rate: set omega to 0 and multiple 1/(t^omega) by a constant factor
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.initialize_Q() # initialize Q_hat(state, action) = 0.0 for all states and actions
        self.prev_state = None
        self.prev_index = None
        self.data_key = (self.omega, self.gamma, self.epsilon)
        self.penalty = 0.0

    
    def initialize_Q_geoloc(self):
        # data structure design: Q_hat[state] = [Q_values]
        # Q_hat is a dictionary mapping states, actions to Q_values
        # Q_values is list, directly mapping to self.env.valid_actions
        self.Q_hat = {}
        x = self.env.grid_size[0] #8
        y = self.env.grid_size[1] #6
        loc_x = range(-x+1,x)
        loc_y = range(-y+1,y)
        for lx in loc_x:
            for ly in loc_y:
                Q_hat_key = (lx, ly)
                Q_hat_val = [0.0] * len(self.env.valid_actions)
                self.Q_hat[Q_hat_key] = Q_hat_val

    def initialize_Q(self):
        # data structure design: Q_hat[state] = [Q_val for action 1, Q_val for action 2, Q_val for action 3, Q_val for action 4]
        self.Q_hat = {}
        valid_waypoint = [None, 'left', 'right', 'forward']
        valid_light = ['green', 'red']
        for light in valid_light:
            for oncoming in valid_waypoint:
                for left in valid_waypoint:
                    for right in valid_waypoint:
                        for valid_nxt_waypoint in valid_waypoint:
                            valid_state = (light, oncoming, left, right, valid_nxt_waypoint)
                            self.Q_hat[valid_state] = [0.0] * len(self.env.valid_actions)


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.alpha = None
        self.prev_state = None
        self.prev_index = None
        sim_start[self.data_key].append(self.env.agent_states[self]['location'])
        sim_destination[self.data_key].append(self.env.agent_states[self]['destination'])
        sim_deadline[self.data_key].append(self.env.agent_states[self]['deadline'])
        sim_net_reward[self.data_key].append(self.env.trial_data['net_reward'])
        sim_penalty[self.data_key].append(self.penalty)
        self.penalty = 0.0
    
    
    def choose_action(self, state):
        # simulated-annealking like approach
        rand_seed = np.random.seed(998372)
        coin = np.random.binomial(1, 1-self.epsilon)
        if (coin > 0) and (sum(self.Q_hat[state]) > 0.0): # take the action which argmax Q_hat(state, action)
            max_Q_val = max(self.Q_hat[state])
            index = self.Q_hat[state].index(max_Q_val)
            action = self.env.valid_actions[index]
        else: # take a random action
            index = random.randint(0,len(self.env.valid_actions)-1)
            action = self.env.valid_actions[index]
        return action, index
    

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) #
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.alpha = 1./((t+1)**self.omega) # learning rate
        
        location = self.env.agent_states[self]['location']
        destination = self.env.agent_states[self]['destination']
        #self.state = (location[0]-destination[0], location[1]-destination[1])
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        # Pure randomly takeing actions
        #index = random.randint(0,3)
        #action = self.env.valid_actions[index]
        action, index = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.penalty += reward
        
        # Record simulated trail data
        if self.env.trial_data['success'] == 1:
            sim_success[self.data_key].append(True)
            sim_end_t[self.data_key].append(self.env.agent_states[self]['deadline'])
        else:
            if self.env.enforce_deadline and self.env.agent_states[self]['deadline'] <= 0:
                sim_success[self.data_key].append(False)
                sim_end_t[self.data_key].append(self.env.agent_states[self]['deadline'])

        # TODO: Learn policy based on state, action, reward
        self.Q_hat[self.state][index] = self.alpha * reward # receive immediate reward
        
        if (self.prev_state!=None) and (self.prev_index!=None): # check if previous state exists
            # update previous state, action pair's discounted future Q value
            self.Q_hat[self.prev_state][self.prev_index] += self.alpha * self.gamma * self.Q_hat[self.state][index]
        
        self.prev_state = self.state
        self.prev_index = index

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run(Omega, Gamma, Epsilon):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, omega=Omega, gamma=Gamma, epsilon=Epsilon)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

def tune_parameter():
    Omega = [1., .7, .8, .9, 2.0]
    Gamma = [.5, .6, .7, .8, .9]
    Epsilon = [.1, .2, .3, .4, .5]
    start = []
    destination = []
    success = []
    end_time = []
    deadline = []
    parameters = []
    penalty = []
    net_reward = []
    for o in Omega:
        for g in Gamma:
            for e in Epsilon:
                key = (o, g, e)
                sim_start[key] = []
                sim_destination[key] = []
                sim_success[key] = []
                sim_end_t[key] = []
                sim_deadline[key] = []
                sim_penalty[key] = []
                sim_net_reward[key] = []
                run(Omega=o, Gamma=g, Epsilon=e)

                start += sim_start[key]
                destination += sim_destination[key]
                success += sim_success[key]
                end_time += sim_end_t[key]
                deadline += sim_deadline[key]
                penalty += sim_penalty[key]
                net_reward += sim_net_reward[key]
                parameters += [str(key)]*100
    # debug purpose
    #print "start length ", len(start)
    #print "destination length ", len(destination)
    #print "success length ", len(success)
    #print "end_time length ", len(end_time)
    #print "deadline length ", len(deadline)
    df = pd.DataFrame({'start':start, 'destination':destination, 'success':success, 'end_time':end_time,
                       'deadline': deadline, 'penalty':penalty, 'net_reward':net_reward, 'parameters':parameters})
    df.to_csv('data.csv')

    # testing purpose
    #print "-"*30+"start"+"-"*30
    #print sim_start
    #print "-"*30+"destination"+"-"*30
    #print sim_destination
    #print "-"*25+"reached destination"+"-"*25
    #print sim_success
    #print "-"*25+"time to reach destination"+"-"*25
    #print sim_end_t



if __name__ == '__main__':
    #tune_parameter() # uncomment this line for parameter tunning
    tuned_omega = .7
    tuned_gamma = .6
    tuned_epsilon = .3
    key = (tuned_omega, tuned_gamma, tuned_epsilon)
    sim_start[key] = []
    sim_destination[key] = []
    sim_success[key] = []
    sim_end_t[key] = []
    sim_deadline[key] = []
    sim_penalty[key] = []
    sim_net_reward[key] = []
    run(Omega=tuned_omega, Gamma=tuned_gamma, Epsilon=tuned_epsilon)

    success_rate = sum(sim_success[key])/100.
    avg_penalty = sum(sim_penalty[key])/100.
    avg_reward = sum(sim_net_reward[key])/100.
    time_cost = np.array(sim_deadline[key]) - np.array(sim_end_t[key])
    avg_time_cost = time_cost.mean()
    print "+"*64
    print "Success Rate of SmartCab trained by optimal parameter setting: %.2f" %success_rate
    print "Received Average Penalty of SmartCab trained by optimal parameter setting: ", avg_penalty
    print "Received Average Net Reward of SmartCab trained by optimal parameter setting: ", avg_reward
    print "Average Time Cost for SmartCab trained by optimal parameter setting: ", avg_time_cost
    print '+'*64
