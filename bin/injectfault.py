#! /usr/bin/env python3

"""

%(prog)s takes a fault injection executable and executes it

Usage: %(prog)s --CLI/--GUI <fault injection executable> <the same options that you use to run the excutable before>

       %(prog)s --help(-h): show help information

Prerequisite:
0. You need to specify --CLI or --GUI depending on whether you're invoking it from the command line or from the GUI.
1. You need to be at the parent directory of the <fault injection executable> to invoke %(prog)s. This is to make it easier for LLFI to track the outputs generated by <fault injection executable>
2. (prog)s only checks recursively at the current directory for possible outputs, if your output is not under current directory, you need to store that output by yourself
3. You need to put your input files (if any) under current directory
4. You need to have 'input.yaml' under your current directory, which contains appropriate options for LLFI.
"""

# This script injects faults the program and produces output
# This script should be run after the profiling step

import sys, os, subprocess
import yaml
import time
import random
import shutil
from subprocess import TimeoutExpired

runOverride = False
optionlist = []
defaultTimeout = 500
fi_max_multiple_default = 100

# basedir is assigned in parseArgs(args)
basedir = ""
prog = os.path.basename(sys.argv[0])
fi_exe = ""

options = {
  "verbose": False,
}

def usage(msg = None):
  retval = 0
  if msg is not None:
    retval = 1
    msg = "ERROR: " + msg
    print(msg, file=sys.stderr)
  print(__doc__ % globals(), file=sys.stderr)
  sys.exit(retval)


def parseArgs(args):
  global optionlist, fi_exe
  if args[0] == "--help" or args[0] == "-h":
    usage()
  fi_exe = os.path.realpath(args[0])
  basedir = basedir = os.path.abspath(os.path.dirname(os.path.dirname(fi_exe)))
  optionlist = args[1:]

  # remove the directory prefix for input files, this is to make it easier for the program
  # to take a snapshot
  for index, opt in enumerate(optionlist):
    if os.path.isfile(opt):
      if os.path.realpath(os.path.dirname(opt)) != basedir:
        usage("File %s passed through option is not under current directory" % opt)
      else:
        optionlist[index] = os.path.basename(opt)

  if basedir != os.getcwd():
    print("Change directory to:", basedir)
    os.chdir(basedir)


def checkInputYaml():
  global doc
  global defaultTimeout
  #Check for input.yaml's presence
  yamldir = os.path.dirname(os.path.dirname(fi_exe))
  try:
    f = open(os.path.join(basedir, 'input.yaml'),'r')
  except:
    usage("No input.yaml file in the parent directory of fault injection executable")
    exit(1)

  #Check for input.yaml's correct formmating
  try:
    doc = yaml.load(f)
  except:
    f.close()
    usage("input.yaml is not formatted in proper YAML (reminder: use spaces, not tabs)")
    exit(1)
  finally:
    f.close()

  if "kernelOption" in doc:
    for opt in doc["kernelOption"]:
      if opt=="forceRun":
        runOverride = True
        print("Kernel: Forcing run")
  if "defaultTimeout" in doc:
    defaultTimeout = int(doc["defaultTimeout"])
    assert defaultTimeout > 0, "The timeOut option must be greater than 0"
  else:
    print("Default timeout is set to " + str(defaultTimeout) + " by default.")


def print_progressbar(idx, nruns):
  pct = (float(idx) / float(nruns))
  WIDTH = 50
  bar = "=" *  int(pct * WIDTH)
  bar += ">"
  bar += "-" * (WIDTH - int(pct * WIDTH))
  print(("\r[%s] %.1f%% (%d / %d)" % (bar, pct * 100, idx, nruns)), end='\n')
  sys.stdout.flush()


################################################################################
def config():
  global inputdir, outputdir, errordir, stddir, llfi_stat_dir
  # config
  llfi_dir = os.path.dirname(fi_exe)
  inputdir = os.path.join(llfi_dir, "prog_input")
  outputdir = os.path.join(llfi_dir, "prog_output")
  errordir = os.path.join(llfi_dir, "error_output")
  stddir = os.path.join(llfi_dir, "std_output")
  llfi_stat_dir = os.path.join(llfi_dir, "llfi_stat_output")

  if not os.path.isdir(outputdir):
    os.mkdir(outputdir)
  if not os.path.isdir(errordir):
    os.mkdir(errordir)
  if not os.path.isdir(inputdir):
    os.mkdir(inputdir)
  if not os.path.isdir(stddir):
    os.mkdir(stddir)
  if not os.path.isdir(llfi_stat_dir):
    os.mkdir(llfi_stat_dir)


################################################################################
def execute( execlist, timeout):
  global outputfile
  global return_codes
  print(' '.join(execlist))
  #get state of directory
  dirSnapshot()
  p = subprocess.Popen(execlist, stdout = subprocess.PIPE)
  outputFile = open(outputfile, "wb")
  program_timed_out = False
  start_time = 0
  elapsetime = 0

  #communicate() will block until program exits or until timeout is reached
  try:
    start_time = time.time()
    (p_stdout,p_stderr) = p.communicate(timeout=timeout)
    elapsetime = int(time.time() - start_time + 1)
  except TimeoutExpired: #Child process timed out
    p.kill() #Need to kill the process and then clean up commmunication
    (p_stdout,p_stderr) = p.communicate(timeout=timeout)
    program_timed_out = True

  moveOutput()
  if program_timed_out:
    print("\tParent : Child timed out. Cleaning up ... ")
  else:
    print("\t program finish", p.returncode)
    print("\t time taken", elapsetime,"\n")
  outputFile = open(outputfile, "wb")

  if program_timed_out:
    outputFile.write(
    bytes("\n\n ### Process killed by LLFI for timing out ###\n","UTF-8"))

  outputFile.write(p_stdout)

  if program_timed_out:
    outputFile.write(
    bytes("\n\n ### Process killed by LLFI for timing out ###\n","UTF-8"))

  outputFile.close()
  replenishInput() #for cases where program deletes input or alters them each run

  # Keep a dict of all return codes received.
  if program_timed_out:
    if "TO" in return_codes:
      return_codes["TO"] += 1
    else:
      return_codes["TO"] = 1
  else:
    if p.returncode in return_codes:
      return_codes[p.returncode] += 1
    else:
      return_codes[p.returncode] = 1

  if program_timed_out:
    return "timed-out"
  else:
    return str(p.returncode)



################################################################################
def storeInputFiles():
  global inputList
  inputList=[]
  ##========Consider comma as separator of arguments ==================================
  temp_optionlist = []
  for item in optionlist:
    if item.count(',') == 0:
      temp_optionlist.append(item)
    else:
      temp_optionlist.extend(item.split(','))
  ##===================================================================================
  for opt in temp_optionlist:
    if os.path.isfile(opt):#stores all files in inputList and copy over to inputdir
      shutil.copy2(opt, os.path.join(inputdir, opt))
      inputList.append(opt)

################################################################################
def replenishInput():#TODO make condition to skip this if input is present
  for each in inputList:
    if not os.path.isfile(each):#copy deleted inputfiles back to basedir
      shutil.copy2(os.path.join(inputdir, each), each)

################################################################################
def moveOutput():
  #move all newly created files
  newfiles = [_file for _file in os.listdir(".")]
  for each in newfiles:
    if each not in dirBefore:
      fileSize = os.stat(each).st_size
      if fileSize == 0 and each.startswith("llfi"):
        #empty library output, can delete
        print(each+ " is going to be deleted for having size of " + str(fileSize))
        os.remove(each)
      else:
        flds = each.split(".")
        newName = '.'.join(flds[0:-1])
        newName+='.'+run_id+'.'+flds[-1]
        if newName.startswith("llfi"):
          os.rename(each, os.path.join(llfi_stat_dir, newName))
        else:
          os.rename(each, os.path.join(outputdir, newName))

################################################################################
def dirSnapshot():
  #snapshot of directory before each execute() is performed
  global dirBefore
  dirBefore = [_file for _file in os.listdir(".")]

################################################################################
def readCycles():
  global totalcycles
  profinput= open("llfi.stat.prof.txt","r")
  while 1:
    line = profinput.readline()
    if line.strip():
      if line[0] == 't':
        label, totalcycles = line.split("=")
        break
  profinput.close()

################################################################################
def checkValues(key, val, var1 = None,var2 = None,var3 = None,var4 = None):
  #preliminary input checking for fi options
  #also checks for fi_bit usage by non-kernel users
  #optional var# are used for fi_bit's case only
  if key =='run_number':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val)>0, key+" must be greater than 0 in input.yaml"

  elif key == 'fi_type':
    pass

  ##======== Add number of corrupted bits QINING @MAR 13th========
  elif key == 'fi_num_bits':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >=1, key+" must be greater than or equal to 1 in input.yaml"
  ##==============================================================

  ##======== Add second corrupted regs QINING @MAR 27th===========
  elif key == "window_len":
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >=0, key+" must be greater than or equal to zero in input.yaml"
  ##==================================================================

  ##BEHROOZ: Add max number of target locations
  elif key == "fi_max_multiple":
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >0, key+" must be greater than zero in input.yaml"
    assert int(val) <=int(fi_max_multiple_default), key+" must be smaller than or equal to "+str(fi_max_multiple_default)+ " in input.yaml"
  ##==============================================================

  ##BEHROOZ: Add multiple corrupted regs
  elif key == "window_len_multiple":
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >0, key+" must be greater than zero in input.yaml"
  elif key == "window_len_multiple_startindex":
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >0, key+" must be greater than zero in input.yaml"
  elif key == "window_len_multiple_endindex":
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >0, key+" must be greater than zero in input.yaml"

  ##==============================================================

  elif key == 'fi_cycle':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    ##BEHROOZ: I changed the below line to the current one to fix the fi_cycle
    assert int(val) > 0, key+" must be greater than 0 in input.yaml"
    #assert int(val) >= 0, key+" must be greater than or equal to 0 in input.yaml"
    assert int(val) <= int(totalcycles), key +" must be less than or equal to "+totalcycles.strip()+" in input.yaml"

  elif key == 'fi_index':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >= 0, key+" must be greater than or equal to 0 in input.yaml"

  elif key == 'fi_reg_index':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >= 0, key+" must be greater than or equal to 0 in input.yaml"

  elif key == 'fi_bit':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >= 0, key+" must be greater than or equal to 0 in input.yaml"
    if runOverride:
      pass
    elif var1 != None and var1 > 1 and (var2 or var3) and var4:
      user_input = input("\nWARNING: Injecting into the same cycle(index), bit multiple times "+
                  "is redundant as it would yield the same result."+
                  "\nTo turn off this warning, please see Readme "+
                  "for kernel mode.\nDo you wish to continue anyway? (Y/N)\n ")
      if user_input.upper() =="Y":
        pass
      else:
        exit(1)

  elif key == 'fi_random_seed':
    assert isinstance(val, int)==True, key+" must be an integer in input.yaml"
    assert int(val) >= 0, key+" must be greater than or equal to 0 in input.yaml"

################################################################################
def main(args):
  global optionlist, outputfile, totalcycles,run_id, return_codes
  global defaultTimeout

  parseArgs(args)
  checkInputYaml()
  config()

  # get total num of cycles
  readCycles()
  storeInputFiles()

  #Set up each config file and its corresponding run_number
  try:
    rOpt = doc["runOption"]
  except:
    print("ERROR: Please include runOption in input.yaml.")
    exit(1)

  if not os.path.isfile(fi_exe):
    print("ERROR: The executable "+ fi_exe+" does not exist.")
    print("Please build the executables with create-executables.\n")
    exit(1)
  else:
    print("======Fault Injection======")
    for ii, run in enumerate(rOpt):
      # Maintain a dict of all return codes received and print summary at end
      return_codes = {}

      # Put an empty line between configs
      if ii > 0:
        print("")
      print("---FI Config #"+str(ii)+"---")

      if "numOfRuns" not in run["run"]:
        print("ERROR: Must include a run number per fi config in input.yaml.")
        exit(1)

      if "timeOut" in run["run"]:
         timeout = int(run["run"]["timeOut"])
         assert timeout > 0, "The timeOut option must be greater than 0"
      else:
         timeout = defaultTimeout
         print("Run with default timeout " + str(timeout))

      run_number=run["run"]["numOfRuns"]
      checkValues("run_number", run_number)

      # check for verbosity option, set at the FI run level
      if "verbose" in run["run"]:
        options["verbose"] = run["run"]["verbose"]

      # reset all configurations
      if 'fi_type' in locals():
        del fi_type
      if 'fi_cycle' in locals():
        del fi_cycle
      if 'fi_index' in locals():
        del fi_index
      if 'fi_reg_index' in locals():
        del fi_reg_index
      if 'fi_bit' in locals():
        del fi_bit
      ##======== Add number of corrupted bits QINING @MAR 13th========
      if 'fi_num_bits' in locals():
        del fi_num_bits
      ##==============================================================
      ##======== Add second corrupted regs QINING @MAR 27th===========
      if 'window_len' in locals():
        del window_len
      ##==============================================================
      if 'fi_random_seed' in locals():
        del fi_random_seed
      ##==============================================================
      ##BEHROOZ: Add max number of target locations
      if 'fi_max_multiple' in locals():
        del fi_max_multiple
      ##==============================================================
      ##BEHROOZ: Add multiple corrupted regs
      if 'window_len_multiple' in locals():
        del window_len_multiple
      if 'window_len_multiple_startindex' in locals():
        del window_len_multiple_startindex
      if 'window_len_multiple_endindex' in locals():
        del window_len_multiple_endindex
      ##==============================================================
      #write new fi config file according to input.yaml
      if "fi_type" in run["run"]:
        fi_type=run["run"]["fi_type"]
        if fi_type == "SoftwareFault" or fi_type == "AutoInjection" or fi_type == "Automated":
          try:
            cOpt = doc["compileOption"]
            injectorname = cOpt["instSelMethod"][0]["customInstselector"]["include"][0]
          except:
            print("\n\nERROR: Cannot extract fi_type from instSelMethod. Please check the customInstselector field in input.yaml\n")
          else:
            fi_type = injectorname
        checkValues("fi_type",fi_type)
      ##======== Add number of corrupted bits QINING @MAR 13th========
      if "fi_num_bits" in run["run"]:
        fi_num_bits=run["run"]["fi_num_bits"]
        checkValues("fi_num_bits", fi_num_bits)
      ##==============================================================
      ##======== Add second corrupted regs QINING @MAR 27th===========
      if 'window_len' in run["run"]:
        window_len=run["run"]["window_len"]
        checkValues("window_len", window_len)
      ##==============================================================
      ##BEHROOZ: Add max number of target locations
      if 'fi_max_multiple' in run["run"]:
        fi_max_multiple=run["run"]["fi_max_multiple"]        
        checkValues("fi_max_multiple", fi_max_multiple)
        if ('fi_max_multiple' in locals()) and 'window_len' in locals():
          print(("\nERROR: window_len and fi_max_multiple cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
          exit(1)
      ##==============================================================
      ##BEHROOZ: Add multiple corrupted regs
      if 'window_len_multiple' in run["run"]:
        window_len_multiple=run["run"]["window_len_multiple"]
        checkValues("window_len_multiple", window_len_multiple)
        if ('window_len_multiple' in locals()):
          if ('window_len' in run["run"]):
            print(("\nERROR: window_len and window_len_multiple cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
          elif ('window_len_multiple_startindex' in run["run"]):
            print(("\nERROR: window_len_multiple_startindex and window_len_multiple cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
          elif ('window_len_multiple_endindex' in run["run"]):
            print(("\nERROR: window_len_multiple_endindex and window_len_multiple cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
      if 'window_len_multiple_startindex' in run["run"]:
        window_len_multiple_startindex=run["run"]["window_len_multiple_startindex"]
        checkValues("window_len_multiple_startindex", window_len_multiple_startindex)
        if ('window_len_multiple_startindex' in locals()):
          if ('window_len' in run["run"]):
            print(("\nERROR: window_len and window_len_multiple_startindex cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
          elif ('window_len_multiple' in run["run"]):
            print(("\nERROR: window_len_multiple_startindex and window_len_multiple cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
          elif ('window_len_multiple_endindex' not in run["run"]):
            print(("\nERROR: window_len_multiple_startindex should come with window_len_multiple_endindex."
               " Please specify both."))
            exit(1)
      if 'window_len_multiple_endindex' in run["run"]:
        window_len_multiple_endindex=run["run"]["window_len_multiple_endindex"]
        checkValues("window_len_multiple_endindex", window_len_multiple_endindex)
        if ('window_len_multiple_endindex' in locals()):
          if('window_len' in run["run"]):
            print(("\nERROR: window_len and window_len_multiple_endindex cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
          elif('window_len_multiple' in run["run"]):
            print(("\nERROR: window_len_multiple_endindex and window_len_multiple cannot be specified"
               " at the same time in the input.yaml file. Please choose one."))
            exit(1)
          elif('window_len_multiple_startindex' not in run["run"]):
            print(("\nERROR: window_len_multiple_endindex should come with window_len_multiple_startindex."
               " Please specify both."))
            exit(1)
      ##==============================================================
      if "fi_cycle" in run["run"]:
        fi_cycle=run["run"]["fi_cycle"]
        checkValues("fi_cycle",fi_cycle)
      if "fi_index" in run["run"]:
        fi_index=run["run"]["fi_index"]
        checkValues("fi_index",fi_index)
      if "fi_reg_index" in run["run"]:
        fi_reg_index=run["run"]["fi_reg_index"]
        checkValues("fi_reg_index",fi_reg_index)
      if "fi_bit" in run["run"]:
        fi_bit=run["run"]["fi_bit"]
        checkValues("fi_bit",fi_bit,run_number,fi_cycle,fi_index,fi_reg_index)
      if "fi_random_seed" in run["run"]:
        fi_random_seed=run["run"]["fi_random_seed"]
        checkValues("fi_random_seed",fi_random_seed)

      if ('fi_cycle' not in locals()) and 'fi_index' in locals():
        print(("\nINFO: You choose to inject faults based on LLFI index, "
               "this will inject into every runtime instruction whose LLFI "
               "index is %d\n" % fi_index))
      ##BEHROOZ:
      if ('window_len_multiple' in locals() or 'window_len_multiple_startindex' in locals() or 'window_len_multiple_endindex' in locals()):
        if('fi_max_multiple' not in locals()):
          print(("\nINFO: You choose a window length for multiple bit-flip injection, "
               "however you have not specified the maximum number of locations."
               " Thus, the maximum number of locations will be chosen as " +str(fi_max_multiple_default)+ ".\n"))
          fi_max_multiple = int(fi_max_multiple_default)

      if ('window_len_multiple' not in locals() and 'window_len_multiple_startindex' not in locals()) and 'fi_max_multiple' in locals():
        print(("\nINFO: You choose the maximum number of multiple bit injection, "
               "however you have not specified the window length for multiple bit-flip injection."
               " Thus, the window size will be chosen equal to the total number of cycles-1= "
               + str(int(totalcycles)-1)+ ".\n"))
        window_len_multiple = int(totalcycles) - 1
      ##======================================================
      need_to_calc_fi_cycle = True
      if ('fi_cycle' in locals()) or 'fi_index' in locals():
        need_to_calc_fi_cycle = False

      # fault injection
      for index in range(0, run_number):
        run_id = str(ii)+"-"+str(index)
        outputfile = stddir + "/std_outputfile-" + "run-"+run_id
        errorfile = errordir + "/errorfile-" + "run-"+run_id
        execlist = [fi_exe]

        if('fi_cycle' not in locals() and 'fi_random_seed' in locals()):
          random.seed(fi_random_seed)

        if need_to_calc_fi_cycle:
          ##BEHROOZ: I changed the below line to the current one to fix the fi_cycle
          fi_cycle = random.randint(1, int(totalcycles))
          ##fi_cycle = random.randint(0, int(totalcycles) - 1)

        ficonfig_File = open("llfi.config.runtime.txt", 'w')        
        
        if 'fi_cycle' in locals():
          ficonfig_File.write("fi_cycle="+str(fi_cycle)+'\n')
        elif 'fi_index' in locals():
          ficonfig_File.write("fi_index="+str(fi_index)+'\n')

        if 'fi_type' in locals():
          ficonfig_File.write("fi_type="+fi_type+'\n')
        if 'fi_reg_index' in locals():
          ficonfig_File.write("fi_reg_index="+str(fi_reg_index)+'\n')
        if 'fi_bit' in locals():
          ficonfig_File.write("fi_bit="+str(fi_bit)+'\n')
        ##======== Add number of corrupted bits QINING @MAR 13th========
        if 'fi_num_bits' in locals():
          ficonfig_File.write("fi_num_bits="+str(fi_num_bits)+'\n')
        ##==============================================================
        ##======== Add second corrupted regs QINING @MAR 27th===========
        if 'window_len' in locals():
          ##BEHROOZ: I changed the below line to the current one to fix the fi_cycle          
          fi_second_cycle = min(fi_cycle + random.randint(1, int(window_len)), int(totalcycles))
          #fi_second_cycle = min(fi_cycle + random.randint(1, int(window_len)), int(totalcycles) - 1)
          ficonfig_File.write("fi_second_cycle="+str(fi_second_cycle)+'\n')
        ##==================================================================
        ##BEHROOZ: Add max number of target locations
        if ('fi_max_multiple' in locals()):
          win_start_index = 1
          win_end_index = 1
          if('window_len_multiple' in locals()):
            win_end_index = int(window_len_multiple)
          elif('window_len_multiple_startindex' in locals() and 'window_len_multiple_endindex' in locals()):
            win_start_index = window_len_multiple_startindex
            win_end_index = window_len_multiple_endindex
            if(win_start_index > win_end_index):
              print(("\nERROR: In the yaml file, the window_len_multiple_startindex cannot be bigger than window_len_multiple_endindex!"))
              exit(1)
          #The line below has been substituted with the one below it. This way the maximum number injection is not selected randomly and is
          #equal to the value specified by the user   
          ##selected_num_of_injection = random.randint(1, int(fi_max_multiple))
          ficonfig_File.write("fi_max_multiple="+str(fi_max_multiple)+'\n')
          selected_num_of_injection = fi_max_multiple
          ##======The -1 here is because we have already selected the first location by choosing the fi-cycle
          ##===== and here we are looking for the remaining cycles.=================
          fi_next_cycle = fi_cycle
          for index_multiple in range(1, int(selected_num_of_injection)):
            fi_next_cycle = min(fi_next_cycle + random.randint(win_start_index, win_end_index), int(totalcycles))
            ficonfig_File.write("fi_next_cycle="+str(fi_next_cycle)+'\n')
            if fi_next_cycle == int(totalcycles):
              break
        ##==================================================================
        ficonfig_File.close()

        # print run index before executing. Comma removes newline for prettier
        # formatting
        execlist.extend(optionlist)
        ret = execute(execlist, timeout)
        if ret == "timed-out":
          error_File = open(errorfile, 'w')
          error_File.write("Program hang\n")
          error_File.close()
        elif int(ret) < 0:
          error_File = open(errorfile, 'w')
          error_File.write("Program crashed, terminated by the system, return code " + ret + '\n')
          error_File.close()
        elif int(ret) > 0:
          error_File = open(errorfile, 'w')
          error_File.write("Program crashed, terminated by itself, return code " + ret + '\n')
          error_File.close()

        # Print updates, print the number of injections finished
        print_progressbar(index+1, run_number)

      #print_progressbar(run_number, run_number)
      print("") # progress bar needs a newline after 100% reached
      # Print summary
      if options["verbose"]:
        print("========== SUMMARY ==========")
        print("Return codes: (code:\toccurance)")
        for r in list(return_codes.keys()):
          print(("  %3s: %5d" % (str(r), return_codes[r])))

################################################################################

if __name__=="__main__":
  if len(sys.argv) == 1:
    usage('Must provide the fault injection executable and its options')
    exit(1)
  main(sys.argv[1:])
