from random import randint
from random import random
import itertools
import math
import copy

import matplotlib.pyplot as plt


###############################################################################################
##################          BKT-POMDP FUNCTIONS         #######################################
###############################################################################################

def update_b(obs,belief,task):
	b_new = belief[:]
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			if (obs[i] == 0):
				# ~ print belief[i]
				# ~ print task.subskills[i].p_slip
				# ~ print BKT_p_inc(belief[i],task.subskills[i])
				# ~ print "------"
				b_new[i] = belief[i]*(task.subskills[i].p_slip) / BKT_p_inc(belief[i],task.subskills[i])
			if (obs[i] == 1):
				b_new[i] = belief[i]*(1-task.subskills[i].p_slip) / BKT_p_corr(belief[i],task.subskills[i])
	return b_new
	
def BKT_p_corr(b,sk):
	know_notSlip = b*(1-sk.p_slip)
	notKnow_guess = (1-b)*sk.p_guess
	return (know_notSlip + notKnow_guess)
	
def BKT_p_inc(b,sk):
	know_slip = b*(sk.p_slip)
	notKnow_notGuess = (1-b)*(1-sk.p_guess)
	return know_slip + notKnow_notGuess
	
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
	
def R_obs(previous_belief, future_belief):
	
	sum_previous=0
	sum_future=0
	for i in range(0, n_subskills):
		#print future_belief[i]
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
	#print reward
	return reward
		
	#calculates Kullback-Leibner divergence
	#returns a reward/cost value between -1 and 1
	
	#worst case scenario is uniform distribution
	# ~ size_b = len(previous_belief)
	# ~ unif_b = []
	# ~ for i in range(0, size_b):
		# ~ unif_b.append(1.0/size_b) 
		
	# ~ worst_b = [0.5] * n_subskills
		
	# ~ sum_previous=0
	# ~ #previous belief KLD	
	# ~ for i in range(0, n_subskills):
		# ~ if (previous_belief[i] == 0):
			# ~ div = 0.0
		# ~ else:
			# ~ div = previous_belief[i] * math.log(previous_belief[i] / worst_b[i])
		# ~ sum_previous = sum_previous + div	
	
	# ~ #potential belief KLD
	# ~ sum_future = 0
	# ~ for i in range(0, n_subskills):
		# ~ if (future_belief[i] == 0):
			# ~ div = 0.0
		# ~ else:
			# ~ div = future_belief[i] * math.log(future_belief[i] / worst_b[i])
		# ~ sum_future = sum_future + div
		
	# ~ #information gain / information loss
	# ~ reward = sum_future - sum_previous
	
	# ~ print sum_previous
	# ~ print sum_future
	# ~ print "------"

	
	# ~ return reward

###############################################################################################
##################          RANDOM SELECTION OF TASKS         #################################
###############################################################################################

#create random selection of actions and tasks

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
				
class Task():
	def __init__(self, name,subskills,count_subskills,action):
		self.name = name
		self.subskills = subskills
		self.observations = create_obs(count_subskills,action)
		self.action = action
		#print action
		
def create_obs(length,action):
	#print length
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

def create_random_subskills():
	all_subskills = []
	names = []
	for i in range(0, n_subskills):
		names.append("sk_" + str(i))
	#names = ['sk_a', 'sk_b', 'sk_c','sk_d','sk_e','sk_f','sk_g','sk_h','sk_i','sk_j']
	for n in names:
		sk = SubSkill(n)
		all_subskills.append(sk)
	return all_subskills
	

def create_random_tasks(subskills):
	all_tasks = []
	names = []
	for i in range(0, n_tasks):
		names.append("t_" + str(i))
	#names = ['t_a','t_b','t_c','t_d','t_e','t_f','t_g','t_h','t_i','t_j','t_k','t_l','t_m','t_n','t_o','t_p','t_q','t_r','t_s','t_t']
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

def get_perfect_obs(person, task):
	obs = []
	for i in range(0, n_subskills):
		if (task.action[i] == 1):
			can_do = person.skills[i]
			obs.append(can_do)
		else:
			obs.append(-1)
	return obs
	
def get_obs(person, task):
	obs = []
	for i in range(0, n_subskills):
		if (task.action[i] == 1):
			can_do = person.skills[i]
			if (can_do == 1):
				probability = 1 - task.subskills[i].p_slip
				obs.append(decision(probability))
				# ~ obs.append(1)
			else:
				probability = task.subskills[i].p_guess
				obs.append(decision(probability))
				# ~ obs.append(0)
		else:
			obs.append(-1)
	return obs
	
def decision(probability):
	v = random() < probability
	if (v == True):
		return 1
	else:
		return 0
    
def create_obs_person_skills(person, all_tasks):
	obs = []
	for t in all_tasks:
		o = get_obs(person, t)
		obs.append(o)
	return obs
	
class Person():
	def __init__(self, name, all_tasks):
		self.name = name
		self.belief = [0.5] * n_subskills
		self.skills = create_random_person_capabilities()
		self.obs = create_obs_person_skills(self, all_tasks)
		
def create_random_person_capabilities():
	person_skills = []
	for i in range(0,n_subskills):
		capable = randint(0, 1)
		person_skills.append(capable)
	return person_skills
		

###############################################################################################
##################          UPDATE USING Q AND V FUNCTIONS         ############################
###############################################################################################

discount = 0.9
max_it = 1

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
	
	# ~ total = 0		
	q = 0
	for obs in task.observations:
		belief_new = update_b(obs,belief,task)
		v_r,v_a = V_function(it+1,belief)
		next_r = R_obs(belief, belief_new) + discount*v_r
		q = q + P_obs(obs,belief,task)*next_r
		# ~ total += P_obs(obs,belief,task)
		# ~ q += probability_obs_perfect(task, person, obs)*next_r
	return (q)
	
def V_function_print(it,belief):
	max_r = -9999
	max_action = []
	for task in all_tasks:
		r = Q_function(belief,task,it)
		# ~ print str(task.action) + " -  " + str(r) 
		#print task.action
		#print "------"
		# ~ print r
		if (r>max_r):
			max_r = r
			max_action = task
	# ~ print max_r
	# ~ print "###################"
	return max_r, max_action
	
	
###############################################################################################
##################          MEASURES        ###################################################
###############################################################################################

#calculates distance to persons actual skills

def distance_to_persons_skills(person, belief):
	real = person.skills
	distance = 0
	for i in range(0, n_subskills):
		d = abs(real[i] - belief[i])
		distance += d
	#print distance
	return distance
	
def number_skills_certain(belief):
	r = 0
	for b in belief:
		if b > 0.95 or b < 0.05:
			r+=1
	return r
			
def distance_to_medium(belief):
	distance = 0
	for i in range(0, n_subskills):
		if belief[i] > 0.5:
			distance += 1 - belief[i]
		else:
			distance += belief[i]
	#print distance
	return distance		
	
	
	
###############################################################################################
##################          AUXILIARY        ##################################################
###############################################################################################


def print_belief(belief):
	b = []
	for v in belief:
		if  v > 0.99:
			b.append(1)
		if v < 0.01:
			b.append(0)
		else:
			b.append(round(v, 2))
	print b
	
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

def condition_random(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	#person = Person("Nicole")
	n_actions = n_tasks
	r_d = []
	r_d.append(0.5*n_subskills)
	for i in range(0, n_actions):
		max_action = all_tasks[i]
		obs = person.obs[i]
		new_belief = update_b(obs,person.belief,max_action)
		#reward += R_obs(person.belief, new_belief)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		person.belief = new_belief	
		#print_belief(person.belief)
	return r_v, r_c, r_d
	
	

###############################################################################################
##################          HAND CRAFTED        ###############################################
###############################################################################################

#one point for not have tested before
#one point for the number of skills tested

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
		
	
def handcrafted_choose(person, ts_since_tested):
	best_t = None
	best_v = -1
	for task in all_tasks:
		v = value_recent(ts_since_tested, task.action)
		if (v > best_v):
			best_v = v
			best_t = task
	return best_t
		
def update_ts_since_tested(ts_since_tested, action):
	for i in range(0, len(ts_since_tested)):
		if (action[i] == 1):
			ts_since_tested[i] = 0
		else:
			ts_since_tested[i] += 1
		
def condition_handcrafted(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	ts_since_tested = [0] * n_subskills
	for i in range(0, n_actions):
		
		# ~ print ts_since_tested

		max_action = handcrafted_choose(person, ts_since_tested)
		update_ts_since_tested(ts_since_tested, max_action.action)
		# ~ print max_action.action
		# ~ print "-------"

		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		#reward += R_obs(person.belief, new_belief)
		reward = distance_to_persons_skills(person, new_belief)
		dist = distance_to_medium(new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		r_d.append(round(dist,2))
		person.belief = new_belief	
		all_tasks.remove(max_action)
	return r_v, r_c, r_d
	


###############################################################################################
##################          PERFECT BASELINE        ###########################################
###############################################################################################
	
def probability_obs_perfect(task, person, obs):
	p = 1
	p2 = 1
	for i in range(0,n_subskills):
		if (task.action[i] == 1):
			if (obs[i] == 0):
				p_incorrect = BKT_p_inc(person.skills[i],task.subskills[i])
				p2_incorrect = BKT_p_inc(person.belief[i],task.subskills[i])
				p = p*p_incorrect
				p2 = p2*p2_incorrect
				# ~ print "v_i" + str(p_incorrect)
			if (obs[i] == 1):
				p_correct = BKT_p_corr(person.skills[i],task.subskills[i])
				p2_correct = BKT_p_corr(person.belief[i],task.subskills[i])
				p2 = p2*p2_correct
				p = p*p_correct
				# ~ print "v_c" + str(p_correct)
	# ~ print person.skills
	# ~ print person.belief
	# ~ print obs
	# ~ print p
	# ~ print p2
	# ~ print "-----"
	return p
	


		
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
		# ~ print r_task
		# ~ if (r_task > max_d):
			# ~ max_d = r_task
			# ~ best_i = i
	# ~ print max_d
	# ~ print "--------------------"
	return all_tasks[best_i]
	
	
def condition_perfect(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	for i in range(0, n_actions):
		max_action = perfect_choose(person)
		# ~ max_r, max_action = V_function_perfect(0,person.belief,person)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		# ~ print i
		# ~ print obs
		# ~ print person.skills
		# ~ print "perfect"
		# ~ print ind
		# ~ print probability_obs_perfect(max_action, person, obs)
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

	
def condition_bktpomdp(person):
	#person = Person("Nicole")
	# ~ n_actions = n_tasks
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	r_d = []
	r_d.append(0.5*n_subskills)
	for i in range(0,n_actions):
		max_r, max_action = V_function_print(0,person.belief)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		# ~ print "bktpomdp"
		# ~ print probability_obs_perfect(max_action, person, obs)
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
##################          CONDITIONS WITH REPETITION         ################################
###############################################################################################

def rep_condition_random(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	# ~ n_actions = n_tasks
	for i in range(0, n_actions):
		action_id = randint(0, n_tasks-1)
		max_action = all_tasks[action_id]
		obs = person.obs[action_id]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		person.belief = new_belief	
	return r_v, r_c

def rep_condition_handcrafted(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	ts_since_tested = [0] * n_subskills
	for i in range(0, n_actions):
		max_action = handcrafted_choose(person, ts_since_tested)
		update_ts_since_tested(ts_since_tested, max_action.action)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		person.belief = new_belief	
		#all_tasks.remove(max_action)
	return r_v, r_c

def rep_condition_perfect(person):
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	for i in range(0, n_actions):
		max_action = perfect_choose(person)
		ind = find_el(all_tasks_main, max_action)
		obs = get_perfect_obs(person, max_action)
		
		# ~ obs = person.obs[ind]
		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		r_c.append(number_skills_certain(new_belief))
		r_v.append(round(reward, 2))
		person.belief = new_belief	
		#all_tasks.remove(max_action)
	return r_v, r_c
	
def rep_condition_bktpomdp(person):
	#person = Person("Nicole")
	#n_actions = n_tasks
	reward = 0
	r_v = []
	r_c = [0]
	r_v.append(0.5*n_subskills)
	for i in range(0,n_actions):
		max_r, max_action = V_function_print(0,person.belief)
		ind = find_el(all_tasks_main, max_action)
		obs = person.obs[ind]

		new_belief = update_b(obs,person.belief,max_action)
		reward = distance_to_persons_skills(person, new_belief)
		r_v.append(round(reward, 2))
		r_c.append(number_skills_certain(new_belief))
		person.belief = new_belief	
		#all_tasks.remove(max_action)
	return r_v, r_c
###############################################################################################
##################          TESTS         #####################################################
###############################################################################################

n_subskills = 20
n_tasks = 200
n_rounds = 500
n_actions = 40
# ~ n_actions = 20
# ~ all_subskills = create_random_subskills()
# ~ all_tasks_main = create_random_tasks(all_subskills)


# ~ all_tasks = copy.deepcopy(all_tasks_main)
# ~ person = Person("Nicole", all_tasks)


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

for i in range(0,n_rounds):
	print i
	all_subskills = create_random_subskills()
	all_tasks_main = create_random_tasks(all_subskills)
	all_tasks = copy.deepcopy(all_tasks_main)
	person = Person("Nicole", all_tasks)
	
	person_random = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	random_reward, random_known, random_distance = condition_random(person_random)
	# ~ random_reward, random_known = rep_condition_random(person_random)
	all_random.append(random_reward)
	all_random_known.append(random_known)
	all_random_dist.append(random_distance)

	person_bktpomdp = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	bktpomdp_reward, bktpomdp_known, bktpomdp_distance = condition_bktpomdp(person_bktpomdp)
	# ~ bktpomdp_reward, bktpomdp_known = rep_condition_bktpomdp(person_bktpomdp)
	all_bktpomdp.append(bktpomdp_reward)
	all_bktpomdp_known.append(bktpomdp_known)
	all_bktpomdp_dist.append(bktpomdp_distance)

	person_handcrafted = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	handcrafted_reward, handcrafted_known, handcrafted_distance = condition_handcrafted(person_handcrafted)
	# ~ handcrafted_reward, handcrafted_known = rep_condition_handcrafted(person_handcrafted)
	all_handcrafted.append(handcrafted_reward)
	all_handcrafted_known.append(handcrafted_known)
	all_handcrafted_dist.append(handcrafted_distance)

	person_perfect = copy.deepcopy(person)
	all_tasks = copy.deepcopy(all_tasks_main)
	perfect_reward, perfect_known, perfect_distance = condition_perfect(person_perfect)
	# ~ perfect_reward, perfect_known = rep_condition_perfect(person_perfect)
	all_perfect.append(perfect_reward)
	all_perfect_known.append(perfect_known)
	all_perfect_dist.append(perfect_distance)
	


f_average = open("average.txt", "w")
f_dist = open("dist.txt", "w")
f_known = open("known.txt", "w")
f_average.write(str(all_random))
f_average.write(str(all_handcrafted))
f_average.write(str(all_bktpomdp))
f_average.write(str(all_perfect))
f_dist.write(str(all_random_dist))
f_dist.write(str(all_handcrafted_dist))
f_dist.write(str(all_bktpomdp_dist))
f_dist.write(str(all_perfect_dist))
f_known.write(str(all_random_known))
f_known.write(str(all_handcrafted_known))
f_known.write(str(all_bktpomdp_known))
f_known.write(str(all_perfect_known))
f_average.close()
f_dist.close()
f_known.close()

# ~ print str(all_random)

quit()

	

random_20 = []
hand_20 = []
bktpomdp_20 = []
optimal_20 = []

k_random_20 = []
k_hand_20 = []
k_bktpomdp_20 = []
k_optimal_20 = []

d_random_20 = []
d_hand_20 = []
d_bktpomdp_20 = []
d_optimal_20 = []
for i in range(0, len(all_random)):
	random_20.append(10-(all_random[i][20]))
	hand_20.append(10-(all_handcrafted[i][20]))
	bktpomdp_20.append(10-(all_bktpomdp[i][20]))
	optimal_20.append(10-(all_perfect[i][20]))
	
	k_random_20.append((all_random_known[i][20]))
	k_hand_20.append((all_handcrafted_known[i][20]))
	k_bktpomdp_20.append((all_bktpomdp_known[i][20]))
	k_optimal_20.append((all_perfect_known[i][20]))
	
	d_random_20.append(10-(all_random_dist[i][20]))
	d_hand_20.append(10-(all_handcrafted_dist[i][20]))
	d_bktpomdp_20.append(10-(all_bktpomdp_dist[i][20]))
	d_optimal_20.append(10-(all_perfect_dist[i][20]))
	
print random_20
print hand_20
print bktpomdp_20
print optimal_20
print "---------------------------"

print d_random_20
print d_hand_20
print d_bktpomdp_20
print d_optimal_20
print "---------------------------"

print k_random_20
print k_hand_20
print k_bktpomdp_20
print k_optimal_20
print "---------------------------"
	
	
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

# ~ average_random = [(x / n_rounds)-2 for x  in sum_random] 
# ~ average_bktpomdp = [(x / n_rounds)-2 for x  in sum_bktpomdp] 
# ~ average_handcrafted = [(x / n_rounds)-2 for x  in sum_handcrafted] 
# ~ average_perfect = [(x / n_rounds)-2 for x  in sum_perfect] 

average_random2 = [10-((x / n_rounds)) for x  in sum_random] 
average_bktpomdp2 = [10-((x / n_rounds)) for x  in sum_bktpomdp] 
average_handcrafted2 = [10-((x / n_rounds)) for x  in sum_handcrafted] 
average_perfect2 = [10-((x / n_rounds)) for x  in sum_perfect] 


average_random_known = [float(x) / n_rounds for x  in sum_random_known]
average_bktpomdp_known = [float(x) / n_rounds for x  in sum_bktpomdp_known]
average_handcrafted_known = [float(x) / n_rounds for x  in sum_handcrafted_known]
average_perfect_known = [float(x) / n_rounds for x  in sum_perfect_known]

# ~ average_random_dist = [((x / n_rounds)-2) for x  in sum_random_dist] 
# ~ average_bktpomdp_dist = [((x / n_rounds)-2) for x  in sum_bktpomdp_dist] 
# ~ average_handcrafted_dist = [((x / n_rounds)-2) for x  in sum_handcrafted_dist] 
# ~ average_perfect_dist = [((x / n_rounds)-2) for x  in sum_perfect_dist] 

average_random_dist2 = [10-((x / n_rounds)) for x  in sum_random_dist] 
average_bktpomdp_dist2 = [10-((x / n_rounds)) for x  in sum_bktpomdp_dist] 
average_handcrafted_dist2 = [10-((x / n_rounds)) for x  in sum_handcrafted_dist] 
average_perfect_dist2 = [10-((x / n_rounds)) for x  in sum_perfect_dist] 


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 18})


plt.figure(figsize=(6,5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Average Skill Correctness')
plt.plot(average_bktpomdp2, color='red')
plt.plot(average_random2, color='green')
plt.plot(average_handcrafted2, color='blue')
plt.plot(average_perfect2, color='yellow')
plt.axis([0, n_actions, 0, 0.5*n_subskills])
plt.xlabel('Action Number')
plt.ylabel("Similarity to True Skills")
plt.show()

plt.figure(figsize=(6,5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.title('Average Skill Confidence')
plt.plot(average_bktpomdp_dist2, color='red')
plt.plot(average_random_dist2, color='green')
plt.plot(average_handcrafted_dist2, color='blue')
plt.plot(average_perfect_dist2, color='yellow')
plt.axis([0, n_actions, 0, 0.5*n_subskills])
# ~ plt.axis([0, n_actions, 0.5*n_subskills - 2,0])
plt.xlabel('Action Number')
plt.ylabel("Confidence across all skills")
plt.show()

plt.figure(figsize=(6,5))
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





#print (person.belief)
#max_r, max_action = V_function_print(0,person.belief)
#print max_action






















