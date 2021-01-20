from random import randint
from random import random
import itertools
import math
import copy

import matplotlib.pyplot as plt



###############################################################################################
##################          BKT FUNCTIONS         #############################################
###############################################################################################

#The BKT update probability of mastery given observation function  - P(b|o)
def update_b(obs,belief,task):
	b_new = belief[:]
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			if (obs[i] == 0):
				b_new[i] = belief[i]*(task.subskills[i].p_slip) / BKT_p_inc(belief[i],task.subskills[i])
			if (obs[i] == 1):
				b_new[i] = belief[i]*(1-task.subskills[i].p_slip) / BKT_p_corr(belief[i],task.subskills[i])
	return b_new
	
# Probability of correct observation given the probability of mastery
def BKT_p_corr(b,sk):
	know_notSlip = b*(1-sk.p_slip)
	notKnow_guess = (1-b)*sk.p_guess
	return (know_notSlip + notKnow_guess)
	
# Probability of incorrect observation given the probability of mastery
def BKT_p_inc(b,sk):
	know_slip = b*(sk.p_slip)
	notKnow_notGuess = (1-b)*(1-sk.p_guess)
	return know_slip + notKnow_notGuess
	

	
###############################################################################################
##################          BKT-POMDP FUNCTIONS         #######################################
###############################################################################################

discount = 0.9
max_it = 1

#The probability of a specific observation, given the belief of the person
def P_obs(obs,belief,task):
	p = 1
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			if (obs[i] == 0):
				p_incorrect = BKT_p_inc(belief[i],task.subskills[i])
				p = p*p_incorrect
			if (obs[i] == 1):
				p_correct = BKT_p_corr(belief[i],task.subskills[i])
				p = p*p_correct
	return p

#Chooses the best action given a belief of their skills
def V_function(it, belief):
	max_r = -9999
	max_action = []
	for task in all_tasks:
		r = Q_function(belief,task,it)
		if (r>max_r):
			max_r = r
			max_action = task
	return max_r, max_action

#estimates the reward of choosing a particular task given the persons belief		
def Q_function(belief,task,it):
	#base condition
	if (it==max_it):
			return 0
	
	total = 0		
	q = 0
	for obs in task.observations:
		belief_new = update_b(obs,belief,task)
		v_r,v_a = V_function(it+1,belief)
		next_r = R_obs(belief, belief_new) + discount*v_r
		q = q + P_obs(obs,belief,task)*next_r
		total += P_obs(obs,belief,task)
	return (q)
	
		

###############################################################################################
##################          SKILLS, TASKS, AND PEOPLE         #################################
###############################################################################################



# Class for Skill. Uses the average probability of guessing and slipping for all participants
class SubSkill():
	def __init__(self, name, p_guess, p_slip):
		self.name = name
		self.p_guess = p_guess
		self.p_slip = p_slip
		
#Class for Task. It uses the details of the 12 snap circuit tasks.
class Task():
	def __init__(self, name,subskills,count_subskills,action):
		self.name = name
		self.subskills = subskills
		self.observations = create_obs(count_subskills,action)
		self.action = action
		#print action

#Class for Person. Uses the real observations and skill masteries of the participants				
class Person():
	def __init__(self, name, skills, obs):
		self.name = name
		self.belief = [0.5] * n_subskills
		self.skills = skills
		self.obs = obs
		
#Creates all the possible variations of observations for a particular task		
def create_obs(length,action):
	lst = list(itertools.product([0, 1], repeat=length))
	all_obs = []
	for i in lst:
		ind = 0
		obs = [-1] * n_subskills
		for el in range(0,n_subskills):
			if (action[el] == 1):
				obs[el] = i[ind]
				ind = ind + 1
			else:
				obs[el] = -1
		all_obs.append(obs)
	return all_obs
	



	
###############################################################################################
##################          MEAUSRES AND REWARD       #########################################
###############################################################################################

#calculates distance to persons actual skills - this defines how accurate the current belief estimate is
def distance_to_persons_skills(person, belief):
	real = person.skills
	distance = 0
	for i in range(0, n_subskills):
		# ~ print person.name
		d = abs(real[i] - belief[i])
		distance += d
	#print distance
	return distance
	
# The number of skills that the system has high certainty over its mastery
def number_skills_certain(belief):
	r = 0
	for b in belief:
		if b > 0.95 or b < 0.05:
			r+=1
	return r
			
#A measure of how confident the system is over the belief vector. It rewards beliefs that are further from 0.5.
def distance_to_medium(belief):
	distance = 0
	for i in range(0, n_subskills):
		if belief[i] > 0.5:
			distance += 1 - belief[i]
		else:
			distance += belief[i]
	#print distance
	return distance
	
#calculates the reward using KLD
def R_obs(previous_belief, future_belief):
	
	sum_previous=0
	sum_future=0
	for i in range(0, n_subskills):
		if (future_belief[i] == 0):
			future_belief[i] = 0.0000001
		if (future_belief[i] == 1):
			future_belief[i] = 0.9999999
			
		if (previous_belief[i] == 0):
			previous_belief[i] = 0.0000001
		if (previous_belief[i] == 1):
			previous_belief[i] = 0.9999999
			
		value_previous = previous_belief[i] * math.log(previous_belief[i] / 0.5) + (1-previous_belief[i]) * math.log((1-previous_belief[i]) / 0.5)
		value_future = future_belief[i] * math.log(future_belief[i] / 0.5) + (1-future_belief[i]) * math.log((1-future_belief[i]) / 0.5)
		sum_previous += value_previous
		sum_future += value_future
	reward = sum_future - sum_previous
	return reward
	
###############################################################################################
##################          AUXILIARY        ##################################################
###############################################################################################

#print the belief vector
def print_belief(belief):
	b = []
	for v in belief:
		if  v > 0.99:
			b.append(1)
		if v < 0.01:
			b.append(0)
		else:
			b.append(round(v, 2))
	
#finds the element location in a list
def find_el(the_list, the_el):
	i = 0
	for it in the_list:
		if (it.name == the_el.name):
			return i
		i += 1
	return -1

###############################################################################################
##################          RANDOM BASELINE        ############################################
###############################################################################################

#Chooses the 12 tasks in order given the random policy. At each time step updates the beliefs given the chosen task.	
def condition_random(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	n_actions = n_tasks
	already_done = []
	for i in range(0, n_actions):
		ok_n = False
		while (not ok_n):
			j = randint(0, n_actions-1)
			if (j not in already_done):
				already_done.append(j)
				ok_n = True
		max_action = all_tasks[j]
		obs = person.obs[j]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		person.belief = new_belief	
	return r_v, r_c,r_d
	
		
###############################################################################################
##################          HAND CRAFTED        ###############################################
###############################################################################################

#values the tasks combination with the least recently tested skills.
def value_of_action(task, person):
	v = 0
	for i in range(0, n_subskills):
		if (task.action[i] == 1):
			v += 1
		if (person.belief[i] == 0.5):
			v += 1
	return v
	
def value_recent(ts_since_tested, action):
	v = 0
	for i in range(0, len(ts_since_tested)):
		if (action[i] == 1):
			v += ts_since_tested[i]
	return v
		
#chooses the next action based on the handcrafted policy
def handcrafted_choose(person, ts_since_tested):
	
	best_t = None
	best_v = -1
	
	for task in all_tasks:
		v = value_recent(ts_since_tested, task.action)
		if (v > best_v):
			best_v = v
			best_t = task
	return best_t

#updates the values of each skill after a task is chosen		
def update_ts_since_tested(ts_since_tested, action):
	for i in range(0, len(ts_since_tested)):
		if (action[i] == 1):
			ts_since_tested[i] = 0
		else:
			ts_since_tested[i] += 1
		
#Chooses the 12 tasks in order given the Hand-crafted policy. At each time step updates the beliefs given the chosen task.
def condition_handcrafted(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	ts_since_tested = [0] * n_subskills
	for i in range(0, n_actions):
		max_action = handcrafted_choose(person, ts_since_tested)
		update_ts_since_tested(ts_since_tested, max_action.action)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c,r_d
	


###############################################################################################
##################          PERFECT BASELINE        ###########################################
###############################################################################################

#The probability of an observation for the optimal condition. The optimal condition knows what skills the user has mastered or not. 
#Therefore it can creater better estimates of the proability of an observations
#However there still is some randomness as the policy does not know whether they will slip or not.
def get_perfect_obs(person, task):
	obs = []
	for i in range(0, n_subskills):
		if (task.action[i] == 1):
			can_do = person.skills[i]
			obs.append(can_do)
		else:
			obs.append(-1)
	return obs
	
#Goes through all tasks and chooses the task that will bring the belief closest to the true state	
def perfect_choose(person):
	best_i = -1
	min_d = 99
	for i in range(0, len(all_tasks)):
		obs = get_perfect_obs(person, all_tasks[i])
		new_belief = update_b(obs,person.belief,all_tasks[i])
		d = distance_to_persons_skills(person, new_belief)
		if (d < min_d):
			min_d = d
			best_i = i
	return all_tasks[best_i]
			
	

#Chooses the 12 tasks in order given the Optimal policy. At each time step updates the beliefs given the chosen task.
def condition_perfect(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	for i in range(0, n_actions):
		max_action = perfect_choose(person)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c,r_d
	


###############################################################################################
##################          BKT - POMDP        ################################################
###############################################################################################


#Chooses the 12 tasks in order given the BKT-POMDP policy. At each time step updates the beliefs given the chosen task.	
def condition_bktpomdp(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_d = []
	r_d.append(0.5*n_subskills)
	r_v.append(0.5*n_subskills)
	for i in range(0,n_actions):
		max_r, max_action = V_function(0,person.belief)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_v.append(round(reward, 2))
		r_c.append(number_skills_certain(new_belief))
		r_d.append(round(dist,2))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c,r_d

###############################################################################################


n_subskills = 18 # There are technically only 14 skills (6 tasks * (choosing piece + placing piece)+ 2 directionaility). But we create it as a vector of 18 skills of which 4 will always be empty. 
n_tasks = 12 #12 different tasks were created
n_rounds = 23 # There are 23 participants. So each one counts as one round
n_actions = 12 #The participant will complete all 12 tasks.



###############################################################################################
##################          SPECIFICS ON TASKS, SKILLS AND PARTICIPANTS         ###############
###############################################################################################


#Creates each of the 14 skills using their name, their guess and their slip probabilities.
p_sw = SubSkill("p_sw", 0.3, 0.03) 
p_led = SubSkill("p_led", 0.3, 0.05) #When the guess prob was high, it was set to 0.3
p_res = SubSkill("p_res", 0.30, 0.16)
p_mc = SubSkill("p_mc", 0.3, 0.05)
p_sp = SubSkill("p_sp", 0.3, 0.04)
p_pr = SubSkill("p_pr", 0.13, 0.04)

l_sw = SubSkill("l_sw", 0.26, 0.13)
l_led = SubSkill("l_led", 0.30, 0.17)
l_res = SubSkill("l_res", 0.24, 0.20)
l_mc = SubSkill("l_mc", 0.12, 0.09)
l_sp = SubSkill("l_sp", 0.1, 0.16)
l_pr = SubSkill("l_pr", 0.19, 	0.07)

d_sw = SubSkill("d_sw", 0.0, 0.0) # Not used. This is just a filler
d_led = SubSkill("d_led", 0.27, 0.18)
d_res = SubSkill("d_res", 0.0, 0.0) # Not used. This is just a filler
d_mc = SubSkill("d_mc", 0.11, 0.09)
d_sp = SubSkill("d_sp", 0.0, 0.0) # Not used. This is just a filler
d_pr = SubSkill("d_pr", 0.0, 0.0) # Not used. This is just a filler

#All the 12 tasks action vectors. Each task used a combination of different skills
task1 = [1,1,1,0,0,0, 1,1,1,0,0,0, 0,1,0,0,0,0]
task2 = [0,1,1,0,0,0, 0,1,1,0,0,0, 0,1,0,0,0,0]
task3 = [1,1,0,0,0,1, 1,1,0,0,0,1, 0,1,0,0,0,0]
task4 = [0,1,0,0,0,1, 0,1,0,0,0,1, 0,1,0,0,0,0]
task5 = [0,1,0,1,0,0, 0,1,0,1,0,0, 0,1,0,1,0,0]
task6 = [1,1,0,1,1,0, 1,1,0,1,1,0, 0,1,0,1,0,0]
task7 = [0,0,1,1,1,0, 0,0,1,1,1,0, 0,0,0,1,0,0]
task8 = [0,0,0,1,1,1, 0,0,0,1,1,1, 0,0,0,1,0,0]
task9 = [0,0,0,1,1,0, 0,0,0,1,1,0, 0,0,0,1,0,0]
task10 = [1,1,0,1,0,1, 1,1,0,1,0,1, 0,1,0,1,0,0]
task11 = [1,0,1,1,1,0, 1,0,1,1,1,0, 0,0,0,1,0,0]
task12 = [0,1,1,0,0,1, 0,1,1,0,0,1, 0,1,0,0,0,0]

#All the 12 tasks skills. Each task used a combination of different skills
task1_sk = [p_sw, p_led, p_res, None, None, None, l_sw, l_led, l_res, None, None, None, None, p_led, None, None, None, None] 
task2_sk = [None, p_led, p_res, None, None, None, None, l_led, l_res, None, None, None, None, p_led, None, None, None, None]
task3_sk = [p_sw, p_led, None, None, None, p_pr, l_sw, l_led, None, None, None, l_pr, None, p_led, None, None, None, None]
task4_sk = [None, p_led, None, None, None, p_pr, None, l_led, None, None, None, l_pr, None, p_led, None, None, None, None]
task5_sk = [None, p_led, None, p_mc, None, None, None, l_led, None, l_mc, None, None, None, p_led, None, p_mc, None, None]
task6_sk = [p_sw, p_led, None, p_mc, p_sp, None, l_sw, l_led, None, l_mc, l_sp, None, None, p_led, None, p_mc, None, None]
task7_sk = [None, None, p_res, p_mc, p_sp, None, None, None, l_res, l_mc, l_sp, None, None, None, None, p_mc, None, None]
task8_sk = [None, None, None, p_mc, p_sp, p_pr, None, None, None, l_mc, l_sp, l_pr, None, None, None, p_mc, None, None]
task9_sk = [None, None, None, p_mc, p_sp, None, None, None, None, l_mc, l_sp, None, None, None, None, p_mc, None, None]
task10_sk = [p_sw, p_led, None, p_mc, None, p_pr, l_sw, l_led, None, l_mc, None, l_pr, None, p_led, None, p_mc, None, None]
task11_sk = [p_sw, None, p_res, p_mc, p_sp, None, l_sw, None, l_res, l_mc, l_sp, None, None, None, None, p_mc, None, None]
task12_sk = [None, p_led, p_res, None, None, p_pr, None, l_led, l_res, None, None, l_pr, None, p_led, None, None, None, None]

#Creates the 12 tasks. Composed of the name, the skills, the number of skills, and the action vector.
t1 = Task("task1", task1_sk,7 , task1)
t2 = Task("task2", task2_sk,5 , task2)
t3 = Task("task3", task3_sk,7 , task3)
t4 = Task("task4", task4_sk,5 , task4)
t5 = Task("task5", task5_sk,6 , task5)
t6 = Task("task6", task6_sk,10 , task6)
t7 = Task("task7", task7_sk,7 , task7)
t8 = Task("task8", task8_sk,7 , task8)
t9 = Task("task9", task9_sk,5 , task9)
t10 = Task("task10", task10_sk,10 , task10)
t11 = Task("task11", task11_sk,9 , task11) 
t12 = Task("task12", task12_sk,7 , task12)

#All of the participants observations. These were the real observations in the study

# ~ p23_skills = [1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0]
p23_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,1,-1,0,1,-1,1,0,-1,-1,1,-1,1,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1],
[1,0,-1,1,-1,1,1,0,-1,1,-1,1,-1,0,-1,1,-1,-1],
[1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p22_skills = [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
p22_obs = [[1,0,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,0,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[0,0,-1,-1,-1,1,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,0,-1,-1,-1,1,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,0,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[0,1,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,0,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,0,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,1,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p21_skills = [1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
p21_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,0,-1,1,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,0,1,0,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,0,1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,1,-1,1,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,1,1,1,-1,0,-1,0,1,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p20_skills = [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p20_obs = [[1,0,0,-1,-1,-1,1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,0,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,0,-1,-1,-1,1,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,0,-1,-1,-1,1,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,0,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,1,-1,0,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,0,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,0,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p19_skills = [1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0]
p19_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,0,1,-1,0,0,-1,0,1,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,0,1,1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,0,-1,1,0,0,-1,0,-1,1,-1,0,-1,0,-1,-1],
[1,-1,1,0,1,-1,1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p18_skills = [1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0]
p18_obs = [[1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,1,-1,0,0,-1,1,1,-1,-1,0,-1,1,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,0,-1,1,-1,-1],
[1,-1,1,1,1,-1,0,-1,0,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p17_skills = [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p17_obs = [[1,1,0,-1,-1,-1,0,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,0,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,1,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,0,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p16_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p16_obs = [[1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[0,1,-1,-1,-1,1,0,1,-1,-1,-1,1,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,1,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,1,-1,-1,-1,0,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,0,-1,-1,0,-1,1,0,-1,-1,0,-1,1,-1,-1,-1,-1]]

# ~ p15_skills = [1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
p15_obs = [[1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[0,1,-1,-1,-1,1,0,1,-1,-1,-1,1,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,0,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,0,-1,-1,-1,0,1,0,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,0,0,-1,-1,-1,1,0,0,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,1,-1,1,0,0,-1,1,-1,0,-1,0,-1,1,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,1,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,0,-1,-1,-1,-1]]

# ~ p14_skills = [1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,1,0,0]
p14_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,1,1,0,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,0,-1,1,-1,-1],
[1,-1,1,1,1,-1,0,-1,1,1,0,-1,-1,-1,-1,1,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,0,-1,-1,-1,-1]]

# ~ p13_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p13_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,0,1,1,-1,-1,-1,0,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,0,-1,1,1,-1,1,0,-1,0,1,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,0,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,0,-1,1,1,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p12_skills = [1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
p12_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,0,-1,1,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,0,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,0,-1,1,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,0,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,0,-1,-1,-1,-1]]

# ~ p11_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p11_obs = [[1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,0,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,0,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,1,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,0,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p10_skills = [1,1,1,1,0,1,1,1,0,1,0,1,0,0,0,1,0,0]
p10_obs = [[1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,0,-1,1,1,-1,1,0,-1,-1,0,-1,1,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,0,1,-1,-1,-1,1,0,1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,0,-1,1,-1,-1],
[1,-1,1,1,0,-1,0,-1,0,1,0,-1,-1,-1,-1,1,-1,-1],
[-1,1,0,-1,-1,1,-1,1,0,-1,-1,1,-1,0,-1,-1,-1,-1]]

# ~ p9_skills = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p9_obs = [[1,1,0,-1,-1,-1,1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,0,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[0,1,-1,1,0,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[0,1,-1,0,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,0,1,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p8_skills = [1,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0]
p8_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,0,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,0,-1,-1,-1,1,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,1,-1,0,0,-1,1,0,-1,-1,0,-1,1,-1,-1],
[-1,-1,1,0,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,0,1,1,-1,-1,-1,0,0,1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,1,-1,1,0,1,-1,1,-1,1,-1,1,-1,1,-1,-1],
[1,-1,1,0,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p7_skills = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p7_obs = [[0,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[0,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,1,-1,0,0,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,0,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,0,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[0,1,-1,1,-1,1,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[0,-1,0,1,1,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,0,0,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p6_skills = [1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0]
p6_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,1,0,1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,-1],
[1,-1,1,1,1,-1,1,-1,0,1,0,-1,-1,-1,-1,1,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p5_skills = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p5_obs = [[1,0,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,0,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,0,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,0,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,0,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,1,1,0,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,0,1,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p4_skills = [1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0]
p4_obs = [[1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1],
[1,1,-1,1,0,-1,1,1,-1,1,0,-1,-1,0,-1,1,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1],
[1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,-1],
[1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,1,0,-1,-1,1,-1,1,0,-1,-1,1,-1,1,-1,-1,-1,-1]]


# ~ p3_skills = [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p3_obs = [[1,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1],
[1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[1,1,-1,1,0,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,1,1,0,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,0,0,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,1,-1,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,0,-1,0,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,1,-1,-1,0,-1,0,0,-1,-1,0,-1,0,-1,-1,-1,-1]]

# ~ p2_skills = [1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
p2_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,-1],
[0,1,-1,1,1,-1,0,0,-1,0,0,-1,-1,0,-1,0,-1,-1],
[-1,-1,0,1,1,-1,-1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,0,0,0,-1,-1,-1,0,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,-1,-1],
[1,1,-1,0,-1,1,1,0,-1,0,-1,0,-1,0,-1,0,-1,-1],
[1,-1,0,1,1,-1,1,-1,0,0,0,-1,-1,-1,-1,0,-1,-1],
[-1,1,0,-1,-1,1,-1,1,0,-1,-1,1,-1,1,-1,-1,-1,-1]]

# ~ p1_skills = [1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0]
p1_obs = [[1,1,0,-1,-1,-1,1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[-1,1,0,-1,-1,-1,-1,1,0,-1,-1,-1,-1,1,-1,-1,-1,-1],
[1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1],
[-1,1,-1,1,-1,-1,-1,0,-1,1,-1,-1,-1,0,-1,1,-1,-1],
[1,1,-1,1,1,-1,0,1,-1,1,0,-1,-1,1,-1,1,-1,-1],
[-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1],
[-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1],
[1,0,-1,1,-1,1,1,0,-1,1,-1,1,-1,0,-1,1,-1,-1],
[1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1],
[-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1]]



#The real skill states of participants. A skill was considered mastered if they achieved got the skill correct over 70% of the time.
p1_skills = [1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0]
p2_skills = [1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
p3_skills = [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p4_skills = [1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0]
p5_skills = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p6_skills = [1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,0,0]
p7_skills = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p8_skills = [1,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0]
p9_skills = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p10_skills = [1,1,1,1,0,1,1,1,0,1,0,1,0,0,0,1,0,0]
p11_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p12_skills = [1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
p13_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p14_skills = [1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,0]
p15_skills = [1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0]
p16_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p17_skills = [1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p18_skills = [1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,0]
p19_skills = [1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0]
p20_skills = [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
p21_skills = [1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
p22_skills = [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
p23_skills = [1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,0]


# Populates participants in the person class. Composed of the name, their mastery, and the observations
p1 = Person("p1", p1_skills, p1_obs)
p2 = Person("p2", p2_skills, p2_obs)
p3 = Person("p3", p3_skills, p3_obs)
p4 = Person("p4", p4_skills, p4_obs)
p5 = Person("p5", p5_skills, p5_obs)
p6 = Person("p6", p6_skills, p6_obs)
p7 = Person("p7", p7_skills, p7_obs)
p8 = Person("p8", p8_skills, p8_obs)
p9 = Person("p9", p9_skills, p9_obs)
p10 = Person("p10", p10_skills, p10_obs)
p11 = Person("p11", p11_skills, p11_obs)
p12 = Person("p12", p12_skills, p12_obs)
p13 = Person("p13", p13_skills, p13_obs)
p14 = Person("p14", p14_skills, p14_obs)
p15 = Person("p15", p15_skills, p15_obs)
p16 = Person("p16", p16_skills, p16_obs)
p17 = Person("p17", p17_skills, p17_obs)
p18 = Person("p18", p18_skills, p18_obs)
p19 = Person("p19", p19_skills, p19_obs)
p20 = Person("p20", p20_skills, p20_obs)
p21 = Person("p21", p21_skills, p21_obs)
p22 = Person("p22", p22_skills, p22_obs)
p23 = Person("p23", p23_skills, p23_obs)




###############################################################################################
##################          TESTS         #####################################################
###############################################################################################

all_skills_main = [p_sw, p_led, p_res, p_mc, p_sp, p_pr,l_sw, l_led, l_res, l_mc, l_sp, l_pr,d_sw, d_led, d_res, d_mc, d_sp, d_pr]
all_tasks_main = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12]
all_people_main = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23]

#lists to save measures data
all_random = []
all_bktpomdp = []
all_handcrafted = []
all_perfect = []

all_random_known = []
all_bktpomdp_known = []
all_handcrafted_known = []
all_perfect_known = []

all_random_dist = []
all_bktpomdp_dist = []
all_handcrafted_dist = []
all_perfect_dist = []

#run for the number of rounds. Each round will use one of the participants data and run all 4 conditions on it
for p in all_people_main:
	print (p.name)
	person_random = copy.deepcopy(p)
	all_tasks = copy.deepcopy(all_tasks_main)
	random_reward, random_known, random_distance = condition_random(person_random)
	all_random.append(random_reward)
	all_random_known.append(random_known)
	all_random_dist.append(random_distance)
	
	person_bktpomdp = copy.deepcopy(p)
	all_tasks = copy.deepcopy(all_tasks_main)
	bktpomdp_reward, bktpomdp_known, bktpomdp_distance = condition_bktpomdp(person_bktpomdp)
	all_bktpomdp.append(bktpomdp_reward)
	all_bktpomdp_known.append(bktpomdp_known)
	all_bktpomdp_dist.append(bktpomdp_distance)

	person_handcrafted = copy.deepcopy(p)
	all_tasks = copy.deepcopy(all_tasks_main)
	handcrafted_reward, handcrafted_known, handcrafted_distance = condition_handcrafted(person_handcrafted)
	all_handcrafted.append(handcrafted_reward)
	all_handcrafted_known.append(handcrafted_known)
	all_handcrafted_dist.append(handcrafted_distance)

	person_perfect = copy.deepcopy(p)
	all_tasks = copy.deepcopy(all_tasks_main)
	perfect_reward, perfect_known, perfect_distance = condition_perfect(person_perfect)
	all_perfect.append(perfect_reward)
	all_perfect_known.append(perfect_known)
	all_perfect_dist.append(perfect_distance)


# Gets the values at timesteps 10,20 and 30 to check significance between conditions
	
# ~ list20_perfect = []
# ~ list20_bktpomdp = []
# ~ list20_hand = []
# ~ list20_random = []
# ~ list10_perfect = []
# ~ list10_bktpomdp = []
# ~ list10_hand = []
# ~ list10_random = []
# ~ list30_perfect = []
# ~ list30_bktpomdp = []
# ~ list30_hand = []
# ~ list30_random = []
# ~ for i in range(0, len(all_perfect)):
	# ~ list20_perfect.append(round(all_perfect[i][5],2))
	# ~ list20_hand.append(round(all_handcrafted[i][5],2))
	# ~ list20_bktpomdp.append(round(all_bktpomdp[i][5],2))
	# ~ list20_random.append(round(all_random[i][5],2))
	# ~ list10_perfect.append(round(all_perfect[i][3],2))
	# ~ list10_hand.append(round(all_handcrafted[i][3],2))
	# ~ list10_bktpomdp.append(round(all_bktpomdp[i][3],2))
	# ~ list10_random.append(round(all_random[i][3],2))
	# ~ list30_perfect.append(round(all_perfect[i][7],2))
	# ~ list30_hand.append(round(all_handcrafted[i][7],2))
	# ~ list30_bktpomdp.append(round(all_bktpomdp[i][7],2))
	# ~ list30_random.append(round(all_random[i][7],2))
	
# ~ known = open("real.txt", "w")
# ~ known.write(str(list10_perfect)+"\n")
# ~ known.write(str(list10_bktpomdp)+"\n")
# ~ known.write(str(list10_hand)+"\n")
# ~ known.write(str(list10_random)+"\n\n")

# ~ known.write(str(list20_perfect)+"\n")
# ~ known.write(str(list20_bktpomdp)+"\n")
# ~ known.write(str(list20_hand)+"\n")
# ~ known.write(str(list20_random)+"\n\n")

# ~ known.write(str(list30_perfect)+"\n")
# ~ known.write(str(list30_bktpomdp)+"\n")
# ~ known.write(str(list30_hand)+"\n")
# ~ known.write(str(list30_random)+"\n")
# ~ known.close()

#Sums up and averages for all the rounds
sum_random = [sum(x) for x in zip(*all_random)]
sum_bktpomdp = [sum(x) for x in zip(*all_bktpomdp)]
sum_handcrafted = [sum(x) for x in zip(*all_handcrafted)]
sum_perfect = [sum(x) for x in zip(*all_perfect)]

sum_random_known = [sum(x) for x in zip(*all_random_known)]
sum_bktpomdp_known = [sum(x) for x in zip(*all_bktpomdp_known)]
sum_handcrafted_known = [sum(x) for x in zip(*all_handcrafted_known)]
sum_perfect_known = [sum(x) for x in zip(*all_perfect_known)]


sum_random_dist = [sum(x) for x in zip(*all_random_dist)]
sum_bktpomdp_dist = [sum(x) for x in zip(*all_bktpomdp_dist)]
sum_handcrafted_dist = [sum(x) for x in zip(*all_handcrafted_dist)]
sum_perfect_dist = [sum(x) for x in zip(*all_perfect_dist)]

average_random2 = [((x / n_rounds)-2) for x  in sum_random] 
average_bktpomdp2 = [((x / n_rounds)-2) for x  in sum_bktpomdp] 
average_handcrafted2 = [((x / n_rounds)-2) for x  in sum_handcrafted] 
average_perfect2 = [((x / n_rounds)-2) for x  in sum_perfect] 


average_random_known = [float(x) / n_rounds for x  in sum_random_known]
average_bktpomdp_known = [float(x) / n_rounds for x  in sum_bktpomdp_known]
average_handcrafted_known = [float(x) / n_rounds for x  in sum_handcrafted_known]
average_perfect_known = [float(x) / n_rounds for x  in sum_perfect_known]

average_random_dist2 = [7-((x / n_rounds)-2) for x  in sum_random_dist] 
average_bktpomdp_dist2 = [7-((x / n_rounds)-2) for x  in sum_bktpomdp_dist] 
average_handcrafted_dist2 = [7-((x / n_rounds)-2) for x  in sum_handcrafted_dist] 
average_perfect_dist2 = [7-((x / n_rounds)-2) for x  in sum_perfect_dist] 

#generates the figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 18})

#Shows how accurate the belief vector is compared to the true skill vector
plt.figure(figsize=(8,5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Distance to True Skill State')
plt.plot(average_bktpomdp2, color='red')
plt.plot(average_random2, color='green')
plt.plot(average_handcrafted2, color='blue')
plt.plot(average_perfect2, color='yellow')
plt.axis([0, n_actions, 0, 0.5*n_subskills - 2])
plt.xlabel('Action Number')
plt.ylabel("Distance of Belief b to True State S")
plt.show()

#Shows how certain the belief vector is. Distance from 0.5
plt.figure(figsize=(8,5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Average Skill Confidence')
plt.plot(average_bktpomdp_dist2, color='red')
plt.plot(average_random_dist2, color='green')
plt.plot(average_handcrafted_dist2, color='blue')
plt.plot(average_perfect_dist2, color='yellow')
plt.axis([0, n_actions, 0, 0.5*n_subskills - 2])
# ~ plt.axis([0, n_actions, 0.5*n_subskills - 2,0])
plt.xlabel('Action Number')
plt.ylabel("Confidence across all skills")
plt.show()

#Shows the number of skills which it has very high certainty over.
plt.figure(figsize=(8,5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Average Skill Certainty')
plt.plot(average_bktpomdp_known, color='red')
plt.plot(average_random_known, color='green')
plt.plot(average_handcrafted_known , color='blue')
plt.plot(average_perfect_known, color='yellow')
plt.axis([0, n_actions, 0, n_subskills - 4])
plt.xlabel('Action Number')
plt.ylabel("Number of Skills with Certainty")
plt.show()







