from random import randint
from random import random
import itertools
import math
import copy

import matplotlib.pyplot as plt


###############################################################################################
##################          BKT FUNCTIONS         #############################################
###############################################################################################


#updates the belief using the BKT framework
def individual_update_b(b,o,s,g):
	b_new = 0
	if (o == 0):
		b_new = b*s / (b*s + (1-b)*(1-g))
	if (o == 1):
		b_new = b*(1-s) / (b*(1-s) + (1-b)*g)
	return b_new
	
#For each tested skill it updates the belief and then has a probability of learning it
def update_b_learning(belief,obs,task):
	b_new = belief[:]
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			b_ind = individual_update_b(belief[i], obs[i], task.subskills[i].p_slip, task.subskills[i].p_guess)
			b_new[i] = b_ind + (1-b_ind)*task.subskills[i].p_transit
	return b_new
	
	
def BKT_p_corr(b,sk):
	know_notSlip = b*(1-sk.p_slip)
	notKnow_guess = (1-b)*sk.p_guess
	return (know_notSlip + notKnow_guess)
	
def BKT_p_inc(b,sk):
	know_slip = b*(sk.p_slip)
	notKnow_notGuess = (1-b)*(1-sk.p_guess)
	return know_slip + notKnow_notGuess


###############################################################################################
##################          RANDOM SELECTION OF TASKS         #################################
###############################################################################################

#create random number of skills with guess, slip, and learning probabilities
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
		invalid_n = 1
		while (invalid_n):
			value = random()
			if (value > 0.15 and value < 0.3):
				self.p_transit = value
				invalid_n = 0

#creates a random task				
class Task():
	def __init__(self, name,subskills,count_subskills,action):
		self.name = name
		self.subskills = subskills
		self.observations = create_obs(count_subskills,action)
		self.action = action
		#print action
		
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
##################          INFORMATION FOR RANDOM PERSON        ##############################
###############################################################################################


	
#creates an observation for the person given their mastery and guess/slip probabilities	
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
	
#creates a random person
class Person():
	def __init__(self, name, all_tasks):
		self.name = name
		self.belief = [0.05] * n_subskills
		self.skills = create_random_person_capabilities()
		self.obs = []#create_obs_person_skills(self, all_tasks)

#the person starts off without having mastery of any of the skills	
def create_random_person_capabilities():
	person_skills = []
	for i in range(0,n_subskills):
		person_skills.append(0)
	return person_skills
	
#if the person has not mastered the skill. When they practice during a task they have a chance of learning it
def update_learning(person, max_action):
	for i in range(0,n_subskills):
		if (max_action.action[i] == 1):
			if (person.skills[i] == 0):
				person.skills[i] = decision(max_action.subskills[i].p_transit)
			


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
		belief_new = update_b_learning(belief,obs,task) 
		v_r,v_a = V_function(it+1,belief)
		next_r = R_obs_teaching(belief, belief_new) + discount*v_r
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
	
#The distance of the belief vector to full mastery over all skills. Represents how close the system believes it is to have mastered all skills.s
def distance_to_one(belief):
	distance = 0
	for i in range(0, n_subskills):
		distance += 1 - belief[i]
	return distance
	
#Caluclates the number of skills that the person has mastered.
def r_skills1(skills):
	sum_s = 0
	for i in range (0, len(skills)):
		if (skills[i] == 1):
			sum_s +=1
	return sum_s
	
#Reward function for BKT-POMDP. Rewards tasks that it estimates will increase the belief closer to 1.
def R_obs_teaching(past_belief,future_belief):
	sum_b = 0
	for i in range(0, n_subskills):
		if (future_belief[i] > past_belief[i]):
			sum_b += (future_belief[i] - past_belief[i])
		else:
			sum_b += 0
	return sum_b
	
#The optimal policy chooses actions that have the sum of the probabilities of learning it is the highest.
def R_teaching(state):
	sum_state = 0
	for i in range (0,len(state)):
		if (state[i] < .50):
			sum_state+=state[i]
	return sum_state
	
	
###############################################################################################
##################          AUXILIARY        ##################################################
###############################################################################################

#prints the belief vector
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
	r_d = []
	r_o = [10]
	r_d.append(0.5*n_subskills)
	r_s = [0]
	for i in range(0, n_actions):
		max_action = all_tasks[i]
		update_learning(person, max_action)
		obs = get_obs(person, max_action)
		new_belief = update_b_learning(person.belief,obs,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		r_o.append(distance_to_one(new_belief))
		r_s.append(r_skills1(person.skills))
		person.belief = new_belief	
		#print_belief(person.belief)
	return r_v, r_c, r_d, r_o, r_s
	
	

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
	r_o = [10]
	r_s = [0]
	r_d.append(0.5*n_subskills)
	ts_since_tested = [0] * n_subskills
	for i in range(0, n_actions):
		max_action = handcrafted_choose(person, ts_since_tested)
		update_ts_since_tested(ts_since_tested, max_action.action)
		ind = find_el(all_tasks_main, max_action)
		update_learning(person, max_action)
		obs = get_obs(person, max_action)
		new_belief = update_b_learning(person.belief,obs,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		r_o.append(distance_to_one(new_belief))
		r_s.append(r_skills1(person.skills))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c, r_d, r_o, r_s
	


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
	

#Goes through all tasks and chooses the task that is likely to teach the most to the user.	
def perfect_choose(person):
	best_i = -1
	min_d = 99
	max_d = -1000
	for i in range(0, len(all_tasks)):
		r_task = 0
		task = all_tasks[i]
		prev_skills = person.skills[:]
		for j in range(0,n_subskills):
			if (task.action[j] == 1):
				if (prev_skills[j] == 0):
					person.skills[j] = all_subskills[j].p_transit #insteal of looking at unmastered skill as unmastered. It looks at them as having p_transit probability of knowing
		for obs in task.observations:
			belief_new = update_b_learning(person.belief,obs,task) 
			p_obs = probability_obs_perfect(task, person, obs) 
			reward = R_teaching(person.skills)
			r_task += p_obs * reward
		person.skills = prev_skills

		if (r_task > max_d):
			max_d = r_task
			best_i = i

	return all_tasks[best_i]
	


#Chooses the 40 tasks in order given the Optimal policy. At each time step updates the beliefs given the chosen task.
def condition_perfect(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_o = [10]
	r_s = [0]
	r_d.append(0.5*n_subskills)
	for i in range(0, n_actions):
		max_action = perfect_choose(person)
		ind = find_el(all_tasks_main, max_action)
		update_learning(person, max_action)
		obs = get_obs(person, max_action)

		new_belief = update_b_learning(person.belief,obs,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		r_o.append(distance_to_one(new_belief))
		r_s.append(r_skills1(person.skills))
		person.belief = new_belief	
		all_tasks.remove(max_action)
		
	return r_v, r_c, r_d, r_o, r_s
	

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
	
# Goes through all tasks and chooses the one that it estimates using the belief vector will bring it closes to 1.
def bktpomdp_choose(person):
	best_i = -1
	min_d = 99
	max_d = -1000
	for i in range(0, len(all_tasks)):
		r_task = 0
		task = all_tasks[i]
		for obs in task.observations:
			belief_new = update_b_learning(person.belief,obs,task) 			
			p_obs = probability_obs_bktpomdp(task, person, obs) 
			reward = R_obs_teaching(person.belief, belief_new)
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
	r_o = [10]
	r_s = [0]
	r_d.append(0.5*n_subskills)
	for i in range(0,n_actions):
		max_action = bktpomdp_choose(person)
		ind = find_el(all_tasks_main, max_action)
		update_learning(person, max_action)
		obs = get_obs(person, max_action)

		new_belief = update_b_learning(person.belief,obs,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		r_c.append(number_skills_certain(new_belief))
		r_o.append(distance_to_one(new_belief))
		r_s.append(r_skills1(person.skills))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c, r_d, r_o,r_s

###############################################################################################
##################          TESTS         #####################################################
###############################################################################################

n_subskills = 20 
n_tasks = 200 
n_rounds = 10
n_actions = 40 


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

all_random_1dist = []
all_bktpomdp_1dist = []
all_handcrafted_1dist = []
all_perfect_1dist = []

all_random_total = []
all_bktpomdp_total = []
all_handcrafted_total = []
all_perfect_total = []

#run for the number of rounds. Each round will create a random user and run all 4 conditions on it.
for i in range(0,n_rounds):
	print (i)
	all_subskills = create_random_subskills()
	all_tasks_main = create_random_tasks(all_subskills)
	all_tasks = copy.deepcopy(all_tasks_main)
	person = Person("Nicole", all_tasks)
	
	person_random = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	random_reward, random_known, random_distance, random_1dist, random_total = condition_random(person_random)
	all_random.append(random_reward)
	all_random_known.append(random_known)
	all_random_dist.append(random_distance)
	all_random_1dist.append(random_1dist)
	all_random_total.append(random_total)
	

	person_bktpomdp = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	bktpomdp_reward, bktpomdp_known, bktpomdp_distance, bktpomdp_1dist, bktpomdp_total = condition_bktpomdp(person_bktpomdp)
	all_bktpomdp.append(bktpomdp_reward)
	all_bktpomdp_known.append(bktpomdp_known)
	all_bktpomdp_dist.append(bktpomdp_distance)
	all_bktpomdp_1dist.append(bktpomdp_1dist)
	all_bktpomdp_total.append(bktpomdp_total)

	person_handcrafted = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	handcrafted_reward, handcrafted_known, handcrafted_distance, handcrafted_1dist, handcrafted_total = condition_handcrafted(person_handcrafted)
	all_handcrafted.append(handcrafted_reward)
	all_handcrafted_known.append(handcrafted_known)
	all_handcrafted_dist.append(handcrafted_distance)
	all_handcrafted_1dist.append(handcrafted_1dist)
	all_handcrafted_total.append(handcrafted_total)

	person_perfect = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	perfect_reward, perfect_known, perfect_distance, perfect_1dist, perfect_total = condition_perfect(person_perfect)
	all_perfect.append(perfect_reward)
	all_perfect_known.append(perfect_known)
	all_perfect_dist.append(perfect_distance)
	all_perfect_1dist.append(perfect_1dist)
	all_perfect_total.append(perfect_total)
	

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
# ~ for i in range(0, len(all_perfect_1dist)):
	# ~ list20_perfect.append(round(all_perfect_total[i][20],2))
	# ~ list20_hand.append(round(all_handcrafted_total[i][20],2))
	# ~ list20_bktpomdp.append(round(all_bktpomdp_total[i][20],2))
	# ~ list20_random.append(round(all_random_total[i][20],2))
	# ~ list10_perfect.append(round(all_perfect_total[i][10],2))
	# ~ list10_hand.append(round(all_handcrafted_total[i][10],2))
	# ~ list10_bktpomdp.append(round(all_bktpomdp_total[i][10],2))
	# ~ list10_random.append(round(all_random_total[i][10],2))
	# ~ list30_perfect.append(round(all_perfect_total[i][30],2))
	# ~ list30_hand.append(round(all_handcrafted_total[i][30],2))
	# ~ list30_bktpomdp.append(round(all_bktpomdp_total[i][30],2))
	# ~ list30_random.append(round(all_random_total[i][30],2))
	
# ~ known = open("distance1.txt", "w")
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


sum_random_1dist = [sum(x) for x in zip(*all_random_1dist)]
sum_bktpomdp_1dist = [sum(x) for x in zip(*all_bktpomdp_1dist)]
sum_handcrafted_1dist = [sum(x) for x in zip(*all_handcrafted_1dist)]
sum_perfect_1dist = [sum(x) for x in zip(*all_perfect_1dist)]


sum_random_total = [sum(x) for x in zip(*all_random_total)]
sum_bktpomdp_total = [sum(x) for x in zip(*all_bktpomdp_total)]
sum_handcrafted_total = [sum(x) for x in zip(*all_handcrafted_total)]
sum_perfect_total = [sum(x) for x in zip(*all_perfect_total)]

average_random2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_random] 
average_bktpomdp2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_bktpomdp] 
average_handcrafted2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_handcrafted] 
average_perfect2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_perfect] 

average_random_dist2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_random_dist] 
average_bktpomdp_dist2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_bktpomdp_dist] 
average_handcrafted_dist2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_handcrafted_dist] 
average_perfect_dist2 = [(n_subskills/2)-((x / n_rounds)) for x  in sum_perfect_dist] 

average_random_known = [float(x) / n_rounds for x  in sum_random_known]
average_bktpomdp_known = [float(x) / n_rounds for x  in sum_bktpomdp_known]
average_handcrafted_known = [float(x) / n_rounds for x  in sum_handcrafted_known]
average_perfect_known = [float(x) / n_rounds for x  in sum_perfect_known]

average_random_total = [float(x) / n_rounds for x  in sum_random_total]
average_bktpomdp_total = [float(x) / n_rounds for x  in sum_bktpomdp_total]
average_handcrafted_total = [float(x) / n_rounds for x  in sum_handcrafted_total]
average_perfect_total = [float(x) / n_rounds for x  in sum_perfect_total]

average_random_1dist = [(n_subskills/2)-((x / n_rounds)) for x  in sum_random_1dist] 
average_bktpomdp_1dist = [(n_subskills/2)-((x / n_rounds)) for x  in sum_bktpomdp_1dist] 
average_handcrafted_1dist = [(n_subskills/2)-((x / n_rounds)) for x  in sum_handcrafted_1dist] 
average_perfect_1dist = [(n_subskills/2)-((x / n_rounds)) for x  in sum_perfect_1dist] 


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 18})

#The following two plots will initially have a dip, as people learn 
#new skills and the belief has not caught up yet. 
# ~ plt.figure(figsize=(8,5))
# ~ plt.gcf().subplots_adjust(bottom=0.15)
# ~ plt.title('Average Skill Correctness')
# ~ plt.plot(average_bktpomdp2, color='red')
# ~ plt.plot(average_random2, color='green')
# ~ plt.plot(average_handcrafted2, color='blue')
# ~ plt.plot(average_perfect2, color='yellow')
# ~ plt.axis([0, n_actions, 0, 0.5*n_subskills])
# ~ plt.xlabel('Action Number')
# ~ plt.ylabel("Similarity to True Skills")
# ~ plt.show()

# ~ plt.figure(figsize=(8,5))
# ~ plt.gcf().subplots_adjust(bottom=0.15)
# ~ plt.title('Average Skill Certainty')
# ~ plt.plot(average_bktpomdp_known, color='red')
# ~ plt.plot(average_random_known, color='green')
# ~ plt.plot(average_handcrafted_known , color='blue')
# ~ plt.plot(average_perfect_known, color='yellow')
# ~ plt.axis([0, n_actions, 0, n_subskills])
# ~ plt.xlabel('Action Number')
# ~ plt.ylabel("Number of Skills with Certainty")
# ~ plt.show()

plt.figure(figsize=(8,5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Average Number of Mastered Skills')
plt.plot(average_bktpomdp_total, color='red')
plt.plot(average_random_total, color='green')
plt.plot(average_handcrafted_total , color='blue')
plt.plot(average_perfect_total, color='yellow')
plt.axis([0, n_actions, 0, n_subskills])
plt.xlabel('Action Number')
plt.ylabel("Number of Mastered Skills")
plt.show()


















