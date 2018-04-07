"""MC3-P2: Q-learning & Dyna - grading script.

Usage:
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:
    PYTHONPATH=ml4t:MC1-P2/jdoe7 python ml4t/mc2_p1_grading/grade_marketsim.py

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import os
import pytest
os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments")
from grading.grading import grader, GradeResult, run_with_timeout, IncorrectOutput

import sys
import traceback as tb

import datetime as dt

import random

import numpy as np
import pandas as pd
from collections import namedtuple

import util

# Student modules to import
main_code = "QLearner"  # module name to import
os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/data")
robot_qlearning_testing_seed=1490652871
QLearningTestCase = namedtuple('QLearning', ['description', 'group','world_file','best_reward','median_reward','max_time','points'])
qlearning_test_cases = [
    QLearningTestCase(
        description="World 1",
        group='nodyna',
        world_file='world01.csv',
        best_reward=-17,
        median_reward=-29.5,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 2",
        group='nodyna',
        world_file='world02.csv',
        best_reward=-14,
        median_reward=-19,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 4",
        group='nodyna',
        world_file='world04.csv',
        best_reward=-24,
        median_reward=-33,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 6",
        group='nodyna',
        world_file='world06.csv',
        best_reward=-16,
        median_reward=-23.5,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 7",
        group='nodyna',
        world_file='world07.csv',
        best_reward=-14,
        median_reward=-26,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 8",
        group='nodyna',
        world_file='world08.csv',
        best_reward=-14,
        median_reward=-19,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 9",
        group='nodyna',
        world_file='world09.csv',
        best_reward=-15,
        median_reward=-20,
        max_time=2,
        points=9.5
    ),
    QLearningTestCase(
        description="World 10",
        group='nodyna',
        world_file='world10.csv',
        best_reward=-28,
        median_reward=-42,
        max_time=2,
        points=9.5
    ),
    # Dyna test cases
    QLearningTestCase(
        description="World 1, dyna=200",
        group='dyna',
        world_file='world01.csv',
        best_reward=-12,
        median_reward=-29.5,
        max_time=20,
        points=2.5
    ),
    QLearningTestCase(
        description="World 2, dyna=200",
        group='dyna',
        world_file='world02.csv',
        best_reward=-14,
        median_reward=-19,
        max_time=20,
        points=2.5  
    ),
    QLearningTestCase(
        description="Author check",
        group='author',
        world_file='world01.csv',
        best_reward=0,
        median_reward=0,
        max_time=10,
        points=0
    ),
]

max_points = 100.0 
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

# Test functon(s)
@pytest.mark.parametrize("description,group,world_file,best_reward,median_reward,max_time,points", qlearning_test_cases)
def test_qlearning(description, group, world_file, best_reward, median_reward, max_time, points, grader):
    points_earned = 0.0  # initialize points for this test case
    os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/Project_7_QLearning")
    try:
        incorrect = True
        if not 'QLearner' in globals():
            import importlib
            m = importlib.import_module('QLearner')
            globals()['QLearner'] = m
        # Unpack test case
        os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/data")
        world = np.array([map(float,s.strip().split(',')) for s in util.get_robot_world_file(world_file).readlines()])
        student_reward = None
        student_author = None
        msgs = []
        if group=='nodyna':
            def timeoutwrapper_nodyna():
                # Note: the following will NOT be commented durring final grading
                # random.seed(robot_qlearning_testing_seed)
                # np.random.seed(robot_qlearning_testing_seed)
                learner = QLearner.QLearner(num_states=100,\
                                            num_actions = 4, \
                                            alpha = 0.2, \
                                            gamma = 0.9, \
                                            rar = 0.98, \
                                            radr = 0.999, \
                                            dyna = 0, \
                                            verbose=False)
                return qltest(worldmap=world,iterations=500,max_steps=10000,learner=learner,verbose=False)
            student_reward = run_with_timeout(timeoutwrapper_nodyna,max_time,(),{})
            incorrect = False
            if student_reward < 1.5*median_reward:
                incorrect = True
                msgs.append("   Reward too low, expected %s, found %s"%(median_reward,student_reward))
        elif group=='dyna':
            def timeoutwrapper_dyna():
                # Note: the following will NOT be commented durring final grading
                # random.seed(robot_qlearning_testing_seed)
                # np.random.seed(robot_qlearning_testing_seed)
                learner = QLearner.QLearner(num_states=100,\
                                            num_actions = 4, \
                                            alpha = 0.2, \
                                            gamma = 0.9, \
                                            rar = 0.5, \
                                            radr = 0.99, \
                                            dyna = 200, \
                                            verbose=False)
                return qltest(worldmap=world,iterations=50,max_steps=10000,learner=learner,verbose=False)
            student_reward = run_with_timeout(timeoutwrapper_dyna,max_time,(),{})
            incorrect = False
            if student_reward < 1.5*median_reward:
                incorrect = True
                msgs.append("   Reward too low, expected %s, found %s"%(median_reward,student_reward))
        elif group=='author':
            points_earned = -20
            def timeoutwrapper_author():
                # Note: the following will NOT be commented durring final grading
                # random.seed(robot_qlearning_testing_seed)
                # np.random.seed(robot_qlearning_testing_seed)
                learner = QLearner.QLearner(num_states=100,\
                                            num_actions = 4, \
                                            alpha = 0.2, \
                                            gamma = 0.9, \
                                            rar = 0.98, \
                                            radr = 0.999, \
                                            dyna = 0, \
                                            verbose=False)
                return learner.author()
            student_author = run_with_timeout(timeoutwrapper_author,max_time,(),{})
            student_reward = best_reward+1
            incorrect = False
            if (student_author is None) or (student_author=='tb34'):
                incorrect = True
                msgs.append("   author() method not implemented correctly. Found {}".format(student_author))
            else:
                points_earned = points
        if (not incorrect):        
            points_earned += points
        if incorrect:
            inputs_str = "    group: {}\n" \
                         "    world_file: {}\n"\
                         "    median_reward: {}\n".format(group, world_file, median_reward)
            raise IncorrectOutput, "Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}".format(inputs_str, "\n".join(msgs))
    except Exception as e:
        # Test result: failed
        msg = "Test case description: {}\n".format(description)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        tb_list = [row for row in tb_list if row[0] in ['QLearner.py','StrategyLearner.py']]
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        elif 'grading_traceback' in dir(e):
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(e.grading_traceback))
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

def getrobotpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 2:
                C = col
                R = row
    if (R+C)<0:
        print "warning: start location not defined"
    return R, C

# find where the goal is in the map
def getgoalpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 3:
                C = col
                R = row
    if (R+C)<0:
        print "warning: goal location not defined"
    return (R, C)

# move the robot and report reward
def movebot(data,oldpos,a):
    testr, testc = oldpos

    randomrate = 0.20 # how often do we move randomly
    quicksandreward = -100 # penalty for stepping on quicksand

    # decide if we're going to ignore the action and 
    # choose a random one instead
    if random.uniform(0.0, 1.0) <= randomrate: # going rogue
        a = random.randint(0,3) # choose the random direction

    # update the test location
    if a == 0: #north
        testr = testr - 1
    elif a == 1: #east
        testc = testc + 1
    elif a == 2: #south
        testr = testr + 1
    elif a == 3: #west
        testc = testc - 1

    reward = -1 # default reward is negative one
    # see if it is legal. if not, revert
    if testr < 0: # off the map
        testr, testc = oldpos
    elif testr >= data.shape[0]: # off the map
        testr, testc = oldpos
    elif testc < 0: # off the map
        testr, testc = oldpos
    elif testc >= data.shape[1]: # off the map
        testr, testc = oldpos
    elif data[testr, testc] == 1: # it is an obstacle
        testr, testc = oldpos
    elif data[testr, testc] == 5: # it is quicksand
        reward = quicksandreward
        data[testr, testc] = 6 # mark the event
    elif data[testr, testc] == 6: # it is still quicksand
        reward = quicksandreward
        data[testr, testc] = 6 # mark the event
    elif data[testr, testc] == 3:  # it is the goal
        reward = 1 # for reaching the goal

    return (testr, testc), reward #return the new, legal location

# convert the location to a single integer
def discretize(pos):
    return pos[0]*10 + pos[1]

def qltest(worldmap, iterations, max_steps, learner, verbose):
# each iteration involves one trip to the goal
    startpos = getrobotpos(worldmap) #find where the robot starts
    goalpos = getgoalpos(worldmap) #find where the goal is
    # max_reward = -float('inf')
    all_rewards = list()
    for iteration in range(1,iterations+1): 
        total_reward = 0
        data = worldmap.copy()
        robopos = startpos
        state = discretize(robopos) #convert the location to a state
        action = learner.querysetstate(state) #set the state and get first action
        count = 0
        while (robopos != goalpos) & (count<max_steps):

            #move to new location according to action and then get a new action
            newpos, stepreward = movebot(data,robopos,action)
            if newpos == goalpos:
                r = 1 # reward for reaching the goal
            else:
                r = stepreward # negative reward for not being at the goal
            state = discretize(newpos)
            action = learner.query(state,r)
    
            if data[robopos] != 6:
                data[robopos] = 4 # mark where we've been for map printing
            if data[newpos] != 6:
                data[newpos] = 2 # move to new location
            robopos = newpos # update the location
            #if verbose: time.sleep(1)
            total_reward += stepreward
            count = count + 1
        if verbose and (count == max_steps):
            print "timeout"
        if verbose: printmap(data)
        if verbose: print iteration, total_reward
        # if max_reward < total_reward:
        #     max_reward = total_reward
        all_rewards.append(total_reward)
    # return max_reward
    return np.median(all_rewards)

if __name__ == "__main__":
    pytest.main(["-s", __file__])
