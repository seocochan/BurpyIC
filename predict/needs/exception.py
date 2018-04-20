import sys
import traceback

def config_except_info():
    e_type, e_value, e_traceback = sys.exc_info()
    
    # Extract stack traces as tuples
    e_traceback = traceback.extract_tb(e_traceback)

    stack_trace = list()

    for t in e_traceback:
        stack_trace.append("File : %s, line : %d, Function : %s, Message : %s \n" % (t[0], t[1], t[2], t[3]))
    
    trace_string = "\n"
    
    for stack in stack_trace:
        trace_string = trace_string + str(stack)

    exception = dict()
    exception["Exception type"] = str(e_type)
    exception["Exception message"] = str(e_value)    
    exception["Stack trace"] = trace_string

    return exception

def print_except_info(dict):
    print("======================= Error =======================")
    print("Exception type - %s" % dict["Exception type"])
    print("Exception message - %s" % dict["Exception message"])
    print("Stack trace - %s" % dict["Stack trace"])
    print("=====================================================")
    
