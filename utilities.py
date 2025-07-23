import re
import random
import numpy as np
import os
import json
random.seed(123)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval
def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0
def read_json_all(path: str):
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    jason_data = []
    for index, js in enumerate(json_files):
        jason_data.append(json.load(open(os.path.join(path, js))))


    return jason_data

def extract_prediction(output, options, option_inds):
    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r"\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?", r"\1/\2", output)

    output = re.sub(r"(?<![AP]\.M)\.$", "", output)
    output = re.sub(r"(?<=\d)[\=](?=[\-\$\d])", " = ", output)
    output = re.sub(r"\u2212", "-", output)

    ## Multi-choice questions
    if options:
        patterns = [
            r'^\(([A-Za-z])\)$',  # "(b)", "(B)"
            r'^([A-Za-z])$',  # "b", "B"
            r'^([A-Za-z]). ',  # "b", "B"
            r'[Th]he answer is ([A-Z])',  # "The answer is B"
            r'^\(([A-Za-z])\) [\s\S]+$',  # "(A) XXXXX"
            r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$',  # "The answer is (B) XXXXX."
        ]

        # have "X" in the output
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                pred = res[0].upper()  # e.g., "B"
                if pred in option_inds:
                    ind = option_inds.index(pred)  # 1
                    if ind >= len(options):
                        ind = random.choice(range(len(options)))
                    predition = options[ind]
                    return predition

        # find the most similar options
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
        predition = options[max_idx]
        return predition

    else:
        ## free_text QA problems, numeric answer
        patterns = [
            # r'^\([A-Za-z]\) ([\s\S]+)$', # "(A) XXXXX"
            # r'[Th]he answer is \([A-Za-z]\) ([\s\S]+)$', # "The answer is (B) XXXXX."
            r'[Th]he answer is ([\s\S]+)$',  # "The answer is XXXXX.",
            r'[Th]he table shows that ([\d\$\.\,\/\:]+) ',
            r' = ([\d\$\.\,\/\:]+)',  # "= $1.40"
            r'(?<= be| is) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "will be $1.40"
            r'(?<= are| was) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r'(?<= were) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                predition = res[-1].strip()
                if predition.endswith(".") and ".M." not in predition:
                    predition = predition[:-1]
                return predition

    return output


def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)
    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)
        try:
            if result is not None:
                number = int(text)
            elif "/" in text:
                nums = text.split("/")
                number = round(float(nums[0]) / float(nums[1]), 3)
            else:
                number = round(float(text), 3)
            number = str(number)
            number = re.sub(r"\.[0]+$", "", number)
            return number
        except:
            return text
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text




def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string