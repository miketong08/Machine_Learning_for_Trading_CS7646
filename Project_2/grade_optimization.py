"""MC1-P2: Optimize a portfolio - grading script.

Usage:
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:
    PYTHONPATH=ml4t:MC1-P2/jdoe7 python ml4t/mc1_p2_grading/grade_optimization.py

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pytest
from grading.grading import grader, GradeResult, run_with_timeout, IncorrectOutput

import os
import sys
import traceback as tb

import numpy as np
import random
import pandas as pd
import datetime
from collections import namedtuple

from util import get_data

main_code = "optimization"

def str2dt(strng):
    year,month,day = map(int,strng.split('-'))
    return datetime.datetime(year,month,day)

# Test cases
OptimizationTestCase = namedtuple('OptimizationTestCase', ['inputs', 'outputs', 'description','seed'])
optimization_test_cases = [
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2010-01-01'),
            end_date=str2dt('2010-12-31'),
            symbols=['GOOG', 'AAPL', 'GLD', 'XOM']
        ),
        outputs=dict(
            allocs=[ 0.10612267,  0.00777928,  0.54377087,  0.34232718],
            benchmark=0.00828691718086 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="Wiki example 1",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2004-01-01'),
            end_date=str2dt('2006-01-01'),
            symbols=['AXP', 'HPQ', 'IBM', 'HNZ']
        ),
        outputs=dict(
            allocs=[ 0.29856713,  0.03593918,  0.29612935,  0.36936434],
            benchmark = 0.00706292107796 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="Wiki example 2",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2004-12-01'),
            end_date=str2dt('2006-05-31'),
            symbols=['YHOO', 'XOM', 'GLD', 'HNZ']
        ),
        outputs=dict(
            allocs=[ 0.05963382,  0.07476148,  0.31764505,  0.54795966],
            benchmark=0.00700653270334 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="Wiki example 3",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2005-12-01'),
            end_date=str2dt('2006-05-31'),
            symbols=['YHOO', 'HPQ', 'GLD', 'HNZ']
        ),
        outputs=dict(
            allocs=[ 0.10913451,  0.19186373,  0.15370123,  0.54530053],
            benchmark=0.00789501806472 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="Wiki example 4",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2005-12-01'),
            end_date=str2dt('2007-05-31'),
            symbols=['MSFT', 'HPQ', 'GLD', 'HNZ']
        ),
        outputs=dict(
            allocs=[ 0.29292607,  0.10633076,  0.14849462,  0.45224855],
            benchmark=0.00688155185985 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="MSFT vs HPQ",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2006-05-31'),
            end_date=str2dt('2007-05-31'),
            symbols=['MSFT', 'AAPL', 'GLD', 'HNZ']
        ),
        outputs=dict(
            allocs=[ 0.20500321,  0.05126107,  0.18217495,  0.56156077],
            benchmark=0.00693253248047 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="MSFT vs AAPL",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2011-01-01'),
            end_date=str2dt('2011-12-31'),
            symbols=['AAPL', 'GLD', 'GOOG', 'XOM']
        ),
        outputs=dict(
            allocs=[ 0.15673037,  0.51724393,  0.12608485,  0.19994085],
            benchmark=0.0096198317644 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="Wiki example 1 in 2011",
        seed=1481094712
    ),
    OptimizationTestCase(
        inputs=dict(
            start_date=str2dt('2010-06-01'),
            end_date=str2dt('2011-06-01'),
            symbols=['AAPL', 'GLD', 'GOOG']
        ),
        outputs=dict(
            allocs=[ 0.21737029,  0.66938007,  0.11324964],
            benchmark=0.00799161174614 # BPH: updated from reference solution, Sunday 3 Sep 2017
        ),
        description="Three symbols #1: AAPL, GLD, GOOG",
        seed=1481094712
    ),
]
abs_margins = dict(sum_to_one=0.02, alloc_range=0.02, alloc_match=0.1, sddr_match=0.05)  # absolute margin of error for each component
points_per_component = dict(sum_to_one=2.0, alloc_range=2.0, alloc_match=4.0, benchmark_match=4.0)  # points for each component, for partial credit
points_per_test_case = 8 
seconds_per_test_case = 5  # execution time limit

# Grading parameters (picked up by module-level grading fixtures)
max_points = float(len(optimization_test_cases) * points_per_test_case)
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

# Test functon(s)
@pytest.mark.parametrize("inputs,outputs,description,seed", optimization_test_cases)
def test_optimization(inputs, outputs, description, seed, grader):
    """Test find_optimal_allocations() returns correct allocations.

    Requires test inputs, expected outputs, description, and a grader fixture.
    """

    points_earned = 0.0  # initialize points for this test case
    try:
        # Try to import student code (only once)
        if not main_code in globals():
            import importlib
            # * Import module
            nprs_func = np.random.seed; rs_func = random.seed
            np.random.seed = fake_seed; random.seed = fake_seed;
            mod = importlib.import_module(main_code)
            globals()[main_code] = mod
            np.random.seed = nprs_func
            random.seed = rs_func

        # Unpack test case
        start_date = inputs['start_date']
        end_date = inputs['end_date']
        symbols = inputs['symbols']  # e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']

        def timeoutwrapper_optimize():
            np.random.seed(seed); random.seed(seed)
            nprs_func = np.random.seed; rs_func = random.seed
            np.random.seed = fake_seed; random.seed = fake_seed
            s_allocs, s_cr, s_adr, s_sddr, s_sr = optimization.optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=False)
            s_allocs = np.float32(s_allocs)
            try:
                assert(not(np.isnan(s_allocs).any()))
            except AssertionError as ae:
                raise RuntimeError('NaN values in returned allocations! Check the return type of optimize_portfolio(...)')
            np.random.seed = nprs_func
            random.seed = rs_func
            return s_allocs
        student_allocs = run_with_timeout(timeoutwrapper_optimize,seconds_per_test_case,(),{})

        # Verify against expected outputs and assign points
        incorrect = False
        msgs = []
        correct_allocs = outputs['allocs']
        benchmark_value = outputs['benchmark']

        # * Check sum_to_one: Allocations sum to 1.0 +/- margin
        sum_allocs = np.sum(student_allocs)
        if abs(sum_allocs - 1.0) > abs_margins['sum_to_one']:
            incorrect = True
            msgs.append("    sum of allocations: {} (expected: 1.0)".format(sum_allocs))
            student_allocs = student_allocs / sum_allocs  # normalize allocations, if they don't sum to 1.0
        else:
            points_earned += points_per_component['sum_to_one']

        points_per_alloc_range = points_per_component['alloc_range'] / len(correct_allocs)
        for symbol, alloc in zip(symbols,student_allocs):
            if alloc < -abs_margins['alloc_range'] or alloc > (1.0+abs_margins['alloc_range']):
                incorrect = True
                msgs.append("    {} - allocation out of range: {} (expected [0.0, 1.0)".format(symbol,alloc))
            else:
                points_earned += points_per_alloc_range
        student_allocs_sddr = alloc2sddr(student_allocs,inputs)
        if student_allocs_sddr/benchmark_value - 1.0 > abs_margins['sddr_match']:
            incorrect = True
            msgs.append("    Sddr too large: {} (expected < {} + {})".format(student_allocs_sddr, benchmark_value, benchmark_value*abs_margins['sddr_match']))
        else:
            points_earned += points_per_component['benchmark_match']

        if incorrect:
            inputs_str = "    start_date: {}\n" \
                         "    end_date: {}\n" \
                         "    symbols: {}\n".format(start_date, end_date, symbols)
            raise IncorrectOutput, "Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}".format(inputs_str, "\n".join(msgs))
    except Exception as e:
        # Test result: failed
        msg = "Test case description: {}\n".format(description)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        tb_list = [row for row in tb_list if row[0] == 'optimization.py']
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

def alloc2sddr(allocs,inputs):
    syms = inputs['symbols']
    sd = inputs['start_date']
    ed = inputs['end_date']
    dates = pd.date_range(sd,ed)
    prices_all = get_data(syms,dates)
    prices = prices_all[syms]
    pv = ((prices/prices.ix[0,:])*allocs).sum(axis=1)
    return ((pv/pv.shift(1))-1)[1:].std()

def fake_seed(*args,**kwargs):
    pass

if __name__ == "__main__":
    pytest.main(["-s", __file__])
