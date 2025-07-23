

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def get_table_text(problem):
    if "table" in problem.keys():
        table = problem['table']
        title = problem['table_title']
        if title and len(title) > 0:
            table = f"[TITLE]: {title}\n{table}"
        return table
    else:
        return None




def get_question_text(problem, option_inds):
    if "question" in problem.keys():
        question = problem['question']
    elif "QUESTION" in problem.keys():
        question = problem['QUESTION']
    elif "problem" in problem.keys():
        question = problem['problem']
    elif "input" in problem.keys():
        question = problem['input']
    else:
        raise ValueError("Problem has no QUESTION")
    if "unit" in problem.keys():
        unit = problem['unit']
        if unit and len(unit) > 0:
            question = f"{question} (Unit: {unit})"
    if "options" in problem.keys():
        if "question" in problem.keys():
            options = problem['options']
            question = f"{question}\nOptions: {options}"
        else:
            choices = problem['options']
            if choices and len(choices) > 0:
                choice_list = []
                for i, c in enumerate(choices):
                    choice_list.append("({}) {}".format(option_inds[i], c))
                options = " ".join(choice_list)
                # print(options)
                question = f"{question}\nOptions: {options}"
    if "choice" in problem.keys():
        choices = problem['choices']
        if choices and len(choices) > 0:
            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append("({}) {}".format(option_inds[i], c))
            options = " ".join(choice_list)
            # print(options)
            question = f"{question}\nOptions: {options}"


    return question


def get_answer(problem):
    if "answer" in problem.keys():
        return problem['answer']
    elif "final_decision" in problem.keys():
        return problem['final_decision']
    elif "output" in problem.keys():
        return problem['output']


def get_solution_text(problem):
    if "solution" in problem.keys():
        # \\n: GPT-3 can generate the solution with more tokens
        solution = problem['solution'].replace("\n", "\\n")
    elif "CONTEXTS" in problem.keys():
        solution = ""
        for i in problem['CONTEXTS']:
            solution += i
        solution = solution.replace("\n", "\\n")
    else:
        solution = None

    return solution


def create_one_example(format, table, question, answer, solution, test_example=True):

    input_format, output_format = format.split("-") # e.g., "TQ-A"

    elements = {"Q": f"Question: {question}",
                "T": f"Table: {table}",
                "S": f"Statement: {solution}",
                "A": f"Answer: The answer is {answer}.",
                "AS": f"Answer: The answer is {answer}. BECAUSE: {solution}",
                "SA": f"Answer: {solution} The answer is {answer}."}

    # Input
    input = "\n".join(elements[label] for label in input_format)

    # Output
    if test_example:
        output = "Answer:"
    else:
        if output_format == "":
            output = ""
        else:
            output = elements[output_format]

    # Prompt text
    text = input + "\n" + output
    text = text.replace("  ", " ").strip()

    return text


def build_prompt(problems, shot_pids, test_pid, args, include_test=True):

    examples = []
    pids = shot_pids + [test_pid]

    # n-shot training examples
    for pid in pids:
        if 'medqa' in args.data_root_test:
            pid = int(pid)
        problem = problems[pid]
        if 'new_samples' in problem.keys():
            example = problem['new_samples']
        else:
            table = get_table_text(problem)
            question = get_question_text(problem, args.option_inds)
            answer = get_answer(problem)
            solution = get_solution_text(problem)

            if pid == test_pid:
                assert pid not in shot_pids
                if include_test:
                    example = create_one_example(args.prompt_format, table, question, answer, solution,
                                                 test_example=True)
                else:
                    continue
            else:
                example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=False)


        examples.append(example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input


def create_example_from_pid(pid, problems, args, test=False):
    problem = problems[pid]
    table = get_table_text(problem)
    question = get_question_text(problem, args.option_inds)
    answer = get_answer(problem)
    solution = get_solution_text(problems[pid])

    if test:
        example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=True)
    else:
        example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=False)

    return example
