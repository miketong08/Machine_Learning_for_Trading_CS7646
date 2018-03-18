"""MLT - Grading Helper

Utility functions for processing downloaded student submissions, running test cases.
"""

import os
import sys
import argparse
import textwrap
from dateutil import parser, tz

def parse_helper_args(parent_parsers=[]):
    """Setup and parse command-line args for grading helper."""
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
            Process submissions in <subs_dir>, each under <user_id>/<quiz>/.

            E.g.:
                python helper.py --subs_dir MC1-HW1_2015-08-27 --quiz MC1-HW1
                python helper.py --quiz MC1-P1 -u jdoe42 -u tbone21
        '''),
        parents=parent_parsers)
    argparser.add_argument('-s', '--subs_dir', default='./', help='directory containing submissions (default: current)')
    argparser.add_argument('-q', '--quiz', required=True, help='quiz title on Udacity node (= sub-dir under each user)')
    argparser.add_argument('-u', '--user_id', required=False, action='append', dest='user_ids', metavar='USER_ID', help='only process this user ID (repeat arg for multiple)')
    argparser.add_argument('-t', '--tz_out', required=False, default='EST5EDT', help='timezone for output timestamps (default: EST5EDT)')
    args = argparser.parse_args()
    #print args  # [debug]

    # Post-process args
    args.tz_out = tz.gettz(args.tz_out)  # timezone object for output timestamps
    return args


def get_metadata(quiz_dir, filename='.metadata', sep='|'):
    """Read metadata from a file and return as a dict."""
    metadata_filepath = os.path.join(quiz_dir, filename)
    with open(metadata_filepath, 'r') as metadata_file:
        metadata = {}
        for line in metadata_file:
            parts = line.strip().split(sep)
            key = parts[0].strip()
            value = parts[1].strip()
            if key == 'submission_time':
                try:
                    value = parser.parse(value)
                except ValueError as e:
                    print >> sys.stderr, "[WARNING] get_metadata(): Unable to parse datetime string \"{}\": {}".format(value, str(e))
            metadata[key] = value
        return metadata


def process_submissions(args, func=None, func_args=[], func_kwargs={}):
    """Fetch all indicated submissions and apply func to each.

    Call: func(args, user_id, quiz_dir, metadata, *func_args, **func_kwargs)
    """
    # Callback func must be defined, and accept at least 4 params
    if func is None:
        return

    # Fetch submissions (user IDs)
    submissions = os.listdir(args.subs_dir)
    if args.user_ids:  # limit to desired list of IDs (if non-empty)
        submissions = [s for s in submissions if s in args.user_ids]

    # For reach submission, read metadata, apply func
    for user_id in submissions:
        quiz_dir = os.path.join(args.subs_dir, user_id, args.quiz)
        # TODO: assert quiz_dir is a directory; if not, skip
        metadata = get_metadata(quiz_dir)
        func(args, user_id, quiz_dir, metadata, *func_args, **func_kwargs)


def print_submission_time(args, user_id, quiz_dir, metadata):
    # Print submission time in desired output timezone
    submission_time = metadata['submission_time']
    print "{},{}".format(user_id, submission_time.astimezone(args.tz_out))


def print_if_late(args, user_id, quiz_dir, metadata, deadline=parser.parse("2015-08-24 23:55:00.000-04:00")):
    # Print submission time if submitted late    
    submission_time = metadata['submission_time']
    if submission_time > deadline:
        print "{},{}".format(user_id, submission_time.astimezone(args.tz_out))


def main():
    """Default grading helper script."""
    
    # Parse command-line args
    args = parse_helper_args()

    # Process submissions (examples):-

    # * Print submission times
    #process_submissions(args, print_submission_time)

    # * Count late submissions
    counts = {'total': 0, 'late': 0}
    def count_if_late(args, user_id, quiz_dir, metadata, deadline=None):
        counts['total'] += 1
        submission_time = metadata['submission_time']
        if submission_time > deadline:
            counts['late'] += 1
    process_submissions(args, count_if_late, func_kwargs={'deadline': parser.parse("2015-08-24 23:55:00.000-04:00")})
    print "Late submissions: {} out of {}".format(counts['late'], counts['total'])

    # * Print late submissions
    process_submissions(args, print_if_late, func_kwargs={'deadline': parser.parse("2015-08-24 23:55:00.000-04:00")})


if __name__ == "__main__":
    main()
