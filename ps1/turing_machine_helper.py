def turing_machine(tape, program, verbose=False, return_states=False, return_symbols=False):
    """Run a given Turing machine with the given input.

    The program, accepted as the second argument, should accept two parameters:

      * state: A string representing the current state of the Turing machine.
        For example, the initial state that the Turing machine starts in is
        a string "initial state".
      * symbol: The current symbol being read on the tape, for example,
        "a" or "b". If there is nothing to read, then the symbol will be an
        empty string.

    The program should return a 3-tuple of (new state, new symbol, action):

      * new state: The next state of the Turing machine, as determined by
        your program.
      * new symbol: A new symbol to write on the tape *before* the Turing
        machine executes the action.
      * action: The action that the Turing machine should take after writing
        the new symbol. This should be a string, and should be either "left"
        (move left), "right" (move right), or "halt" (stop executing).

    The program is always run starting from the beginning (index 0) of the tape.
    If the Turing machine is at index 0, and the program returns an action of
    "left", then the tape will be extended to the left. If the Turing machine is
    at the last index of the tap, and the program returns an action of "right",
    then the tape will be extended to the right. In both cases, the symbol that
    is read after extending the tape is always an empty string.
    
    Parameters
    ----------
    tape: list
        A list of symbols representing the tape
    program: function
        The Turing machine to be run. Should accept two paramters:
        the current state of the Turing machine, and the current symbol
        being read.
    verbose: boolean (optional)
        Whether to print out the current state of the Turing machine
        after each step.
        
    Returns
    -------
    The symbol being read by the Turing machine when it halted.
        
    """
    
    if return_states and return_symbols:
        raise ValueError("return_states and return_symbols cannot both be true")
    
    # start in the intial state with the head over the first position
    state = "initial state"
    head = 0
    action = ""
    
    # keep track of states and symbols
    states = set([state])
    symbols = set()

    # helper function to print out the current state of the
    # Turing machine
    def print_tape(tape, head, state):
        print("state: " + str(state))
        tape = [str(x) if x != "" else " " for x in tape]
        print(" ".join(tape))
        print((" " * head * 2) + "^")
    
    # don't actually loop forever, to prevent infinite for loops
    max_steps = 1000
    for i in range(max_steps):
        if verbose:
            print_tape(tape, head, state)
        
        # run the next step of the program, which tells us the new
        # state, the new symbol, and what action to take
        state, tape[head], action = program(state, tape[head])
        states.add(state)
        symbols.add(tape[head])
        
        # if the action is left, then make sure there is enough
        # tape on the left size
        if action == "left":
            if head == 0:
                tape.insert(0, "")
            else:
                head -= 1
                
        # if the action is right, then make sure there is enough
        # tape on the right side
        elif action == "right":
            if head == (len(tape) - 1):
                tape.append("")
            head += 1
            
        # if the action is halt, then stop
        elif action == "halt":
            break
            
        # unrecognized action
        else:
            raise ValueError("invalid action: " + str(action))

    if verbose:
        print_tape(tape, head, state)

    # return the last symbol
    if i == (max_steps - 1):
        raise RuntimeError("the turing machine did not halt in {} steps".format(max_steps))
    elif return_states:
        return states
    elif return_symbols:
        return symbols
    else:
        return tape[head]