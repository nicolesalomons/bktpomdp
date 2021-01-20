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
##################          RANDOM GENERATION OF TASKS AND SKILLS         #####################
###############################################################################################

#create random different skills, each with their own probability of guessing and slipping
class SubSkill():
	def __init__(self, name):
		self.name = name
		invalid_n = 1
		while (invalid_n):
			value = random()
			if (value < 0.30):
				self.p_guess = value
				invalid_n = 0
		invalid_n = 1
		while (invalid_n):
			value = random()
			if (value < 0.30):
				self.p_slip = value
				invalid_n = 0
		
#creates random task, which has between 1 and 5 subskills		
class Task():
	def __init__(self, name,subskills,count_subskills,action):
		self.name = name
		self.subskills = subskills
		self.observations = create_obs(count_subskills,action)
		self.action = action
		
#creates all the possible observations given a task	
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

#Creates a list of subskills
def create_random_subskills():
	all_subskills = []
	names = []
	for i in range(0, n_subskills):
		names.append("sk_" + str(i))
	for n in names:
		sk = SubSkill(n)
		all_subskills.append(sk)
	return all_subskills
	
#Creates a list of tasks
def create_random_tasks(subskills):
	all_tasks = []
	names = []
	for i in range(0, n_tasks):
		names.append("t_" + str(i))
	for i in range(n_tasks):
		task_subskills = [None] * n_subskills
		action =  [0] * n_subskills
		#create tasks between 1 and 5 subskills
		n_subskills_in_task = randint(1, 5)
		valid_sk = 0
		while (valid_sk < n_subskills_in_task):
			subskill = randint(0, n_subskills-1)
			if (subskills[subskill] not in task_subskills):
				task_subskills[subskill] = (subskills[subskill])
				action[subskill] = 1
				valid_sk += 1
		current_task = Task(names[i], task_subskills,n_subskills_in_task,action)
		all_tasks.append(current_task)
	return all_tasks
			
	

###############################################################################################
##################          RANDOM GENERATION OF USER        ##################################
###############################################################################################

	
#Generates the observation for a user for a particular task. It generates it using the probability of guessing and slipping
#We use this function to make sure that all conditions get the same observation.
def get_obs(person, task):
	obs = []
	for i in range(0, n_subskills):
		if (task.action[i] == 1):
			can_do = person.skills[i]
			if (can_do == 1):
				probability = 1 - task.subskills[i].p_slip
				obs.append(decision(probability))
			else:
				probability = task.subskills[i].p_guess
				obs.append(decision(probability))
		else:
			obs.append(-1)
	return obs
	
# creates the list of observations for the user.
def create_obs_person_skills(person, all_tasks):
	obs = []
	for t in all_tasks:
		o = get_obs(person, t)
		obs.append(o)
	return obs
	
#generates a random person. 
class Person():
	def __init__(self, name, all_tasks):
		self.name = name
		self.belief = [0.5] * n_subskills
		self.skills = create_random_person_capabilities()
		self.obs = create_obs_person_skills(self, all_tasks)
	
# The person is assigned for each of the skills randomly whether its mastered or not	
def create_random_person_capabilities():
	person_skills = []
	for i in range(0,n_subskills):
		capable = randint(0, 1)
		person_skills.append(capable)
	return person_skills
		

###############################################################################################
##################          UPDATE USING Q AND V FUNCTIONS         ############################
# These functions can be used when wanting to look more than 1 step ahead #####################
###############################################################################################

discount = 0.9
max_it = 1

#When wanting to look more than 1 step ahead in BKT-POMDP, call the V function instead of bktpomdp_choose()
# The it will define how many iterations you want to look ahead.
def V_function(it, belief):
	max_r = -9999
	max_action = []
	for task in all_tasks:
		r = Q_function(belief,task,it)
		if (r>max_r):
			max_r = r
			max_action = task
	return max_r, max_action

		
def Q_function(belief,task,it):
	#base condition
	if (it==max_it):
			return 0	
	q = 0
	for obs in task.observations:
		belief_new = update_b(obs,belief,task)
		v_r,v_a = V_function(it+1,belief)
		next_r = R_obs(belief, belief_new) + discount*v_r
		q = q + P_obs(obs,belief,task)*next_r
	return (q)
	
	
	
###############################################################################################
##################          MEASURES AND REWARD FUNCTIONS        ##############################
###############################################################################################

#calculates distance to persons actual skills - this defines how accurate the current belief estimate is
def distance_to_persons_skills(person, belief):
	real = person.skills
	distance = 0
	for i in range(0, n_subskills):
		d = abs(real[i] - belief[i])
		distance += d
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
	print (b)
	
#finds the element location in a list
def find_el(the_list, the_el):
	i = 0
	for it in the_list:
		if (it.name == the_el.name):
			return i
		i += 1
	return -1
	
# return 1 with the parameters probability, else returns 0.
def decision(probability):
	v = random() < probability
	if (v == True):
		return 1
	else:
		return 0


###############################################################################################
##################          RANDOM BASELINE        ############################################
###############################################################################################

#Chooses the 40 tasks in order given the random policy. At each time step updates the beliefs given the chosen task.
def condition_random(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	n_actions = n_tasks
	r_d = []
	r_d.append(0.5*n_subskills)
	for i in range(0, n_actions):
		max_action = all_tasks[i]
		obs = person.obs[i]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		person.belief = new_belief	
	return r_v, r_c, r_d
	
	

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
		
#Chooses the 40 tasks in order given the Hand-crafted policy. At each time step updates the beliefs given the chosen task.
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
	return r_v, r_c, r_d
	


###############################################################################################
##################          OPTIMAL BASELINE        ###########################################
###############################################################################################
	
#The probability of an observation for the optimal condition. The optimal condition knows what skills the user has mastered or not. 
#Therefore it can creater better estimates of the proability of an observations
#However there still is some randomness as the policy does not know whether they will slip or not.
def probability_obs_perfect(task, person, obs):
	p = 1
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			if (obs[i] == 0):
				p_incorrect = BKT_p_inc(person.skills[i],task.subskills[i])
				p = p*p_incorrect
			if (obs[i] == 1):
				p_correct = BKT_p_corr(person.skills[i],task.subskills[i])
				p = p*p_correct
	return p
	


#Goes through all tasks and chooses the task that will bring the belief closest to the true state		
def perfect_choose(person):
	best_i = -1
	min_d = 99
	max_d = -1000
	for i in range(0, len(all_tasks)):
		r_task = 0
		task = all_tasks[i]
		for obs in task.observations:
			p_obs = probability_obs_perfect(task, person, obs)
			new_belief = update_b(obs,person.belief,task)
			d = distance_to_persons_skills(person, new_belief) # minimize distance
			m = distance_to_medium(new_belief) #minimize dist to medium
			r = R_obs(person.belief, new_belief) # maximize reward
			n = number_skills_certain(new_belief) # maximize number of skills
			r_task += p_obs * d
		if (r_task < min_d):
			min_d = r_task
			best_i = i
	return all_tasks[best_i]
	
	
#Chooses the 40 tasks in order given the Optimal policy. At each time step updates the beliefs given the chosen task.
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
	return r_v, r_c, r_d
	


###############################################################################################
##################          BKT - POMDP        ################################################
###############################################################################################

#The probability of each observation. The probabilities are based on the estimates of the current belief.
def probability_obs_bktpomdp(task, person, obs):
	p = 1
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			if (obs[i] == 0):
				p_incorrect = BKT_p_inc(person.belief[i],task.subskills[i])
				p = p*p_incorrect
			if (obs[i] == 1):
				p_correct = BKT_p_corr(person.belief[i],task.subskills[i])
				p = p*p_correct

	return p
	
# Goes through all tasks and chooses the one that will give the highest KLD reward.
def bktpomdp_choose(person):
	best_i = -1
	min_d = 99
	max_d = -1000
	for i in range(0, len(all_tasks)):
		r_task = 0
		task = all_tasks[i]
		for obs in task.observations:
			belief_new = update_b(obs,person.belief,task)
			p_obs = probability_obs_bktpomdp(task, person, obs) 
			reward  = R_obs(person.belief, belief_new)
			r_task += p_obs * reward

		if (r_task > max_d):
			max_d = r_task
			best_i = i
	return all_tasks[best_i]
	
	
#Chooses the 40 tasks in order given the BKT-POMDP policy. At each time step updates the beliefs given the chosen task.
def condition_bktpomdp(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	for i in range(0,n_actions):
		max_action = bktpomdp_choose(person)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		r_c.append(number_skills_certain(new_belief))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c, r_d


###############################################################################################
##################          TESTS         #####################################################
###############################################################################################

n_subskills = 20 #number of different skills to test
n_tasks = 200 #number of unique tasks to create
n_rounds = 100 #The number of times to run the program. 
n_actions = 40 # The number of actions to take during each round.

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

#run for the number of rounds. Each round will create a random user and run all 4 conditions on it.
for i in range(0,n_rounds):
	print (i)
	all_subskills = create_random_subskills()
	all_tasks_main = create_random_tasks(all_subskills)
	all_tasks = copy.deepcopy(all_tasks_main)
	person = Person("personName", all_tasks)
	
	person_random = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	random_reward, random_known, random_distance = condition_random(person_random)
	all_random.append(random_reward)
	all_random_known.append(random_known)
	all_random_dist.append(random_distance)

	person_bktpomdp = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	bktpomdp_reward, bktpomdp_known, bktpomdp_distance = condition_bktpomdp(person_bktpomdp)
	all_bktpomdp.append(bktpomdp_reward)
	all_bktpomdp_known.append(bktpomdp_known)
	all_bktpomdp_dist.append(bktpomdp_distance)

	person_handcrafted = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	handcrafted_reward, handcrafted_known, handcrafted_distance = condition_handcrafted(person_handcrafted)
	all_handcrafted.append(handcrafted_reward)
	all_handcrafted_known.append(handcrafted_known)
	all_handcrafted_dist.append(handcrafted_distance)

	person_perfect = copy.deepcopy(person)
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
	# ~ list20_perfect.append(round(all_perfect[i][20],2))
	# ~ list20_hand.append(round(all_handcrafted[i][20],2))
	# ~ list20_bktpomdp.append(round(all_bktpomdp[i][20],2))
	# ~ list20_random.append(round(all_random[i][20],2))
	# ~ list10_perfect.append(round(all_perfect[i][10],2))
	# ~ list10_hand.append(round(all_handcrafted[i][10],2))
	# ~ list10_bktpomdp.append(round(all_bktpomdp[i][10],2))
	# ~ list10_random.append(round(all_random[i][10],2))
	# ~ list30_perfect.append(round(all_perfect[i][30],2))
	# ~ list30_hand.append(round(all_handcrafted[i][30],2))
	# ~ list30_bktpomdp.append(round(all_bktpomdp[i][30],2))
	# ~ list30_random.append(round(all_random[i][30],2))
	
# ~ known = open("average.txt", "w")
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


average_random2 = [((x / n_rounds)) for x  in sum_random] 
average_bktpomdp2 = [((x / n_rounds)) for x  in sum_bktpomdp] 
average_handcrafted2 = [((x / n_rounds)) for x  in sum_handcrafted] 
average_perfect2 = [((x / n_rounds)) for x  in sum_perfect] 


average_random_known = [float(x) / n_rounds for x  in sum_random_known]
average_bktpomdp_known = [float(x) / n_rounds for x  in sum_bktpomdp_known]
average_handcrafted_known = [float(x) / n_rounds for x  in sum_handcrafted_known]
average_perfect_known = [float(x) / n_rounds for x  in sum_perfect_known]


average_random_dist2 = [10-((x / n_rounds)) for x  in sum_random_dist] 
average_bktpomdp_dist2 = [10-((x / n_rounds)) for x  in sum_bktpomdp_dist] 
average_handcrafted_dist2 = [10-((x / n_rounds)) for x  in sum_handcrafted_dist] 
average_perfect_dist2 = [10-((x / n_rounds)) for x  in sum_perfect_dist] 

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
plt.axis([0, n_actions, 0, 0.5*n_subskills])
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
plt.axis([0, n_actions, 0, 0.5*n_subskills])
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
plt.axis([0, n_actions, 0, n_subskills])
plt.xlabel('Action Number')
plt.ylabel("Number of Skills with Certainty")
plt.show()










