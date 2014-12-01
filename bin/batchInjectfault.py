#! /usr/bin/env python3

"""

%(prog)s is a wrapper for LLFI profile command to run profile command through all the work directories generated by \'batchInstrument\'. Each work directory should have an input.yaml file which only contains one software failure model defined. All the software failure modes should be defined in the master input.yaml under current (base) directory.

Usage: %(prog)s <source IR file> <arguemnts>

Prerequisite:
You need to run \'batchInstrument\' first, then run %(prog)s under the same directory, the directory that contains multiple sub directories for different software faults. Same as \'batchInstrument\', %(prog)s is only applicable when multiple software failure modes are defined in input.yaml.
"""

import sys, os, shutil
import yaml
import subprocess

prog = os.path.basename(sys.argv[0])
script_path = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, '../config'))
import llvm_paths

injectfault_script = os.path.join(script_path, 'injectfault')
# basedir and options are assigned in parseArgs(args)
basedir = ""
options = []

def parseArgs(args):
	global basedir
	global options
	cwd = os.getcwd()
	for arg in args:
		option = arg
		if os.path.isfile(arg):
			basedir = os.path.realpath(os.path.dirname(arg))
			option = os.path.basename(arg)
		options.append(option)
	os.chdir(basedir)

def usage(msg = None):
  retval = 0
  if msg is not None:
    retval = -1
    msg = "ERROR: " + msg
    print(msg, file=sys.stderr)
  print(__doc__ % globals(), file=sys.stderr)
  sys.exit(retval)

def phraseMasterYaml():
	master_yaml_dict = {}
	model_list = []
	try:
		with open('input.yaml', 'r') as master_yaml_file:
			master_yaml_dict = yaml.load(master_yaml_file)
	except:
		print ("ERROR: Unable to find input.yaml or load the input.yaml under current directory")
		print (basedir)
		sys.exit(-1)
	try:
		model_list = list(master_yaml_dict['compileOption']['instSelMethod'][0]['customInstselector']['include'])
	except:
		print ("ERROR: this wrapper script is not applicable on the input.yaml under current directory. Please note this script is only applicable on input.yaml files with multiple software failure models defined.")
		print (basedir)
		sys.exit(-1)
	return master_yaml_dict, model_list

def callInjectfault(model_list, *argv):
	num_failed = 0
	for model in model_list:
		workdir = os.path.join(basedir, "llfi-"+model)
		try:
			os.chdir(workdir)
		except:
			print ("ERROR: Unable to change to directory:", workdir)
			sys.exit(-1)
		faultinjection_exe_name = argv[0]
		if faultinjection_exe_name.endswith('.ll'):
			faultinjection_exe_name = faultinjection_exe_name.split('.ll')[0]
		elif faultinjection_exe_name.endswith('.bc'):
			faultinjection_exe_name = faultinjection_exe_name.split('.bc')[0]
		faultinjection_exe_name = faultinjection_exe_name + '-faultinjection.exe'
		command = [injectfault_script]
		command.extend(['./llfi/'+faultinjection_exe_name])
		command.extend(argv[1:])
		print ("\nRun injectfault command:", ' '.join(command))
		try:
			o = subprocess.check_output(command, stderr=sys.stderr)
		except subprocess.CalledProcessError:
			print ("injectfault:", model, " failed!")
			num_failed += 1
		else:
			print (o.decode())
			print ("injectfault:", model, " successed!")
		os.chdir(basedir)
	return num_failed

def main(*argv):
	global options
	parseArgs(argv)
	master_yaml_dict, model_list = phraseMasterYaml()
	r = callInjectfault(model_list, *options)
	return r

if __name__ == "__main__":
	if len(sys.argv[1:]) < 1 or sys.argv[1] == '--help' or sys.argv[1] == '-h':
		usage()
		sys.exit(0)
	else:
		argv = sys.argv[1:]
	r = main(*argv)
	sys.exit(r)