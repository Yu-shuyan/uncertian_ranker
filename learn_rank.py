import os
import sys
import math
import json
import argparse
import random
import time
import torch
import openai

import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from tools import utils
from base_prompt import *
from model import *
from utilities import extract_prediction, normalize_answer, _strip_string, read_json_all, last_boxed_only_string
from algorithm import init_algorithm
#from tenacity import retry, wait_chain, wait_fixed
from retriever.bm25_retriever import bm25_retrieve
from openai import AzureOpenAI
import random
import copy
from tenacity import retry, wait_chain, wait_fixed
import os
from retriever.bm25_retriever import bm25_retrieve
os.environ['CURL_CA_BUNDLE'] = ''

# OpenAI Endpoint details
OPENAI_ENDPOINT = "https://openai-llm-frontdoor-hma7evbthrd4cugn.a01.azurefd.net"
OPENAI_DEPLOYMENT_MODEL = "gpt-4-32k-beta"
OPENAI_AZURE_API_VERSION = "2023-12-01-preview"
APIM_KEY = "*****************"
# client = AzureOpenAI(
#     api_key="xxx",
#     # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
#     azure_endpoint=OPENAI_ENDPOINT,
#     azure_deployment=OPENAI_DEPLOYMENT_MODEL,
#     api_version=OPENAI_AZURE_API_VERSION,
#     http_client=httpx.Client(verify=False),
#     default_headers={
#         'Authorization': f'Bearer {get_llm_access_token()}',
#         'Content-Type': 'application/json',
#         'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
#     })
@retry(wait=wait_chain(*[wait_fixed(10) for i in range(3)] +
                       [wait_fixed(30) for i in range(2)] +
                       [wait_fixed(35)]))
def completion_with_backoff(**kwargs):
    import os

    from openai import AzureOpenAI
    import httpx
    from llm_idam_token_generator.idam_token_generator import get_llm_access_token
    os.environ["APP_CLIENT_ID"] = "long-tail-knowledge-app"
    os.environ["APP_CLIENT_SECRET"] = "zyAeZTrIEJli3cqyVD2jvTDia6Ua"


    client = AzureOpenAI(
        api_key="xxx",
        # This is not playing any role, but required as per OpenAI sdk. So any random could be passed.
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_DEPLOYMENT_MODEL,
        api_version=OPENAI_AZURE_API_VERSION,
        http_client=httpx.Client(verify=False),
        default_headers={
            'Authorization': f'Bearer {get_llm_access_token()}',
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': f'{APIM_KEY}'
        })
    return client.chat.completions.create(**kwargs)











sys.path.append("../")
#openai.api_key = os.getenv("OPENAI_API_KEY")




def load_data(args):
    if 'pubmed' in args.data_root:
        problems = json.load(open(os.path.join(args.data_root)))
        problems_test = json.load(open(os.path.join(args.data_root_test)))
        pids_test = list(problems_test.keys())
        pids = list(problems.keys())
    elif 'medqa' in args.data_root:
        problems_test = [json.loads(line) for line in open(args.data_root_test, 'r')]
        problems_train = [json.loads(line) for line in open(args.data_root, 'r')]
        problems = problems_test + problems_train
        pids_test = list(i for i in range(len(problems_test)))
        pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
        if args.data_root_vali is not None:
            problems_vali = [json.loads(line) for line in open(args.data_root_vali, 'r')]
            problems += problems_vali
            pids_vali = list(i for i in range(len(problems_test) + len(problems_train),
                                              len(problems_test) + len(problems_train) + len(problems_vali)))
    elif args.data_root in ['ethos-national_origin', 'tweet_eval-stance_climate', 'kilt_nq', 'kilt_trex']:
        with open('../data/metaicl/task_data_splits.json', 'r', encoding='utf-8') as json_file:
            train_test_split = json.load(json_file)
        problems_test = train_test_split[args.data_root]['test']
        problems_train = train_test_split[args.data_root]['train']
        problems = problems_test + problems_train
        pids_test = list(i for i in range(len(problems_test)))
        pids = list(i for i in range(len(problems_test), len(problems_test) + len(problems_train)))
    else:
        raise Exception("The dataset does not exist!")
    print("train sample number:", len(pids))
    if args.train_number + args.cand_number < len(pids):
        samples = random.sample(pids, args.train_number + args.cand_number)  # random sample
    else:
        samples = pids
    train_pids = samples[:args.train_number]
    random.shuffle(train_pids)
    cand_pids = samples[args.train_number:]
    if args.data_root_vali is not None:
        train_pids = random.sample(pids_vali, args.train_number)
        random.shuffle(train_pids)

    if args.cand_ckpt:
        cand_pids = torch.load(args.cand_ckpt)
        #remove overlap
        train_pids = [i for i in train_pids if i not in cand_pids]
    if args.train_ckpt:
        samples_id = torch.load(args.train_ckpt)
        if args.train_ckpt2 is not None:
            samples_id2 = torch.load(args.train_ckpt2)
            samples_id = list(set(samples_id+samples_id2))

        samples_id = [i for i in samples_id if i not in pids_test]
        random.shuffle(samples_id)
        if args.train_number > 0:
            train_pids = samples_id[:args.train_number]
        else:
            train_pids = samples_id
        cand_pids = random.sample([i for i in pids if i not in train_pids and i not in pids_test],args.cand_number)
    cand_file = f"cand_results/cand_{args.label}_{args.train_number}_{args.cand_number}.pt"
    torch.save(cand_pids, cand_file)
    vali_file = f"cand_results/validation_{args.label}_{args.train_number}_{args.cand_number}.pt"
    torch.save(train_pids, vali_file)
    print(f"successfully save {cand_file} and {vali_file}")



    # import ipdb; ipdb.set_trace()
    print("cand_pids num", len(cand_pids))
    print("train_pids num", len(train_pids))
    return problems, cand_pids, train_pids


def get_gpt_output(prompt, args):
    if "gpt4" in args.engine:

        if 'pubmed' in args.data_root_test:
            user_prompt = "Please answer yes or no or maybe for the question."
        elif 'medqa' in args.data_root_test or args.data_root_test in ['ethos-national_origin', 'tweet_eval-stance_climate', 'kilt_nq', 'kilt_trex']:
            user_prompt = "Please choose from all the options follow the given example."
        else:
            user_prompt = "Follow the given examples and answer the question following the same format."
    #     response = client.chat.completions.create(
    #         model=OPENAI_DEPLOYMENT_MODEL,
    #         # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    #         messages=[{
    #             "role": "system", "content": prompt
    #         },
    #             {
    #     "role": "user", "content": user_prompt
    # }],
    #         temperature=0.0, max_tokens=args.max_tokens, top_p=1, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)
        response = completion_with_backoff(
            model=OPENAI_DEPLOYMENT_MODEL,
            # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
            messages=[{
                "role": "system", "content": prompt
            },
                {
                    "role": "user", "content": user_prompt
                }],
            temperature=0.0, max_tokens=args.max_tokens, top_p=1, frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty)
        output = response.choices[0].message.content
        if output is not None:
            if output.startswith("\n\n"):
                output = output[2:]
            output = output.split("\n")[0]
    else:
        response = openai.Completion.create(engine=args.engine,
                                            prompt=prompt,
                                            temperature=args.temperature,
                                            max_tokens=args.max_tokens,
                                            top_p=args.top_p,
                                            frequency_penalty=args.frequency_penalty,
                                            presence_penalty=args.presence_penalty,
                                            stop=["\n"])
        output = response["choices"][0]["text"].strip()


    return output




def get_batch_reward_loss(batch_i, scores, Cand_pids, pid_batch, option_batch, unit_batch, label_batch, args):
    shot_number_sum = 0

    batch_loss = 0
    batch_reward = 0
    # max_score_correct = []
    # max_score_wrong = []
    #count = 0
    ## loop over the training examples
    max_score = []
    for i in range(len(scores)):

        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-4 and evaluate the prediction
        cand_prob = scores[i, :].clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1
        # sample shot_pids from the cand_prob distribution
        #print("cand_prob", cand_prob)
        if args.preselection:
            cand_pids = Cand_pids[i]
        else:
            cand_pids = Cand_pids
        cids = np.random.choice(range(len(cand_pids)), args.shot_number, p=cand_prob, replace=False)
        #exit(0)


        # reverse shot_pids so more relevant prompt will be put closer to the question
        cids = cids[::-1]
        # calculate the total number of shots for the query sample based on args.score_th, the candidate samples are saved in cids
        cids = [cid for cid in cids if scores[i, cid] > args.score_th]


        shot_pids = [cand_pids[cid] for cid in cids]
        # print(f"shot_pids: {shot_pids}")
        print("reduced shot number", len(shot_pids))
        shot_number_sum += sum([s for s in range(len(shot_pids)+1)])

        log_prob = 0
        _reward = 0


        if args.adjust_batch >= 0:
            flag = 0
            original_shot_pids = copy.deepcopy(shot_pids)
            #query from 0-shot to len(shot_pids)-shot
            for j in range(len(shot_pids) + 1):
                if j == 0:
                    shot_pids = []
                else:
                    shot_pids = original_shot_pids[:j]
                # generate the prompt input
                prompt = build_prompt(problems, shot_pids, pid_batch[i], args)
                # get the output from GPT-4
                output = get_gpt_output(prompt, args)
                if output:
                    # extract the prediction from the output
                    prediction = extract_prediction(output, option_batch[i], args.option_inds)
                    # normalize the number in the text
                    prediction_norm = normalize_answer(prediction, unit_batch[i])
                else:
                    prediction = output
                    prediction_norm = output


                log_prob = 0
                if j > 0:
                    for cid in cids[:j]:
                        log_prob += torch.log(scores[i, cid])


                if prediction_norm and label_batch[i]:
                    if label_batch[i].lower() in prediction_norm.lower():
                        _reward = 1
                        if j > 0:
                            batch_loss -= _reward * torch.log(scores[i, cids[j - 1]])

                    else:
                        previous_reward = _reward
                        _reward = -1
                        if j > 0:
                            if previous_reward == 1:
                                print(f"answer gets wrong for {j} shot samples.")
                                if args.case_study:
                                    logger.write("query question:")
                                    logger.write(problems[pid_batch[i]])
                                    logger.write(f"query ground truth answer: {label_batch[i]}")
                                    logger.write(f"predicted answer: {prediction_norm}")
                                    print("last shot_pids", shot_pids[-1])
                                    logger.write(f"This ICL sample make the answer wrong: {problems[shot_pids[-1]]}")
                                    logger.write("Previous correct ICL sample:")
                                    for jj in range(len(shot_pids) - 1):
                                        logger.write(problems[shot_pids[jj]])
                                # start_idx = max(j - 1, 1)
                                if j - 1 < len(cids):
                                    max_score.append(max(scores[i, cids[j-1:]]))
                                # args.score_th = max(scores[i, cids[j:]])
                                batch_loss -= _reward * torch.log(scores[i, cids[j - 1]])
                                flag = 1

                                break
                            else:
                                batch_loss -= _reward * torch.log(scores[i, cids[j - 1]])
                else:
                    previous_reward = _reward
                    _reward = -1
                    if j > 0:
                        if previous_reward == 1:
                            print(f"answer gets wrong for {j} shot samples.")
                            if args.case_study:
                                logger.write("query question:")
                                logger.write(problems[pid_batch[i]])
                                logger.write(f"query ground truth answer: {label_batch[i]}")
                                logger.write(f"predicted answer: {prediction_norm}")

                                logger.write(f"This ICL sample make the answer wrong: {problems[shot_pids[j-1]]}")
                                logger.write("Previous correct ICL sample:")
                                for jj in range(j-1):
                                    logger.write(problems[shot_pids[jj]])

                            # start_idx = max(j - 1, 1)
                            if j - 1 < len(cids):
                                max_score.append(max(scores[i, cids[j-1:]]))
                            batch_loss -= _reward * torch.log(scores[i, cids[j - 1]])

                            flag = 1

                            break
                        else:
                            batch_loss -= _reward * torch.log(scores[i, cids[j - 1]])


        else:
            # generate the prompt input
            prompt = build_prompt(problems, shot_pids, pid_batch[i], args)
            # print("prompt", prompt)
            # get the output from GPT-3
            output = get_gpt_output(prompt, args)
            if 'MATH' in args.data_root:
                prediction = remove_boxed(last_boxed_only_string(output))
                if not prediction:
                    prediction = extract_prediction(output, option_batch[i], args.option_inds)
                prediction_norm = _strip_string(prediction)
            else:
                if output:
                    # extract the prediction from the output
                    prediction = extract_prediction(output, option_batch[i], args.option_inds)
                    # normalize the number in the text
                    prediction_norm = normalize_answer(prediction, unit_batch[i])
                else:
                    prediction = output
                    prediction_norm = output

            log_prob = 0
            for cid in cids:
                log_prob += torch.log(scores[i, cid])
            # print(f"log_prob: {log_prob}")
            if prediction_norm:
                if label_batch[i].lower() in prediction_norm.lower():
                    _reward = 1
                else:
                    _reward = -1
            else:
                _reward = -1


            batch_loss -= _reward * log_prob

        batch_reward += _reward


   #update threshold for the score
    if args.adjust_batch >= 0:
        if len(max_score) > 0 and batch_i % args.adjust_batch == 0:
            args.score_th = sum(max_score) / len(max_score)
        print("updated score_th", args.score_th)
    logger.write(f"shot_number_sum for {batch_i} is {shot_number_sum}")

    return cids, batch_reward, batch_loss


def policy_gradient_train(algorithm, problems, train_pids, cand_pids, cand_examples, args):


    optimizer = algorithm.optimizer

    train_samples, train_labels, units, options = [], [], [], []
    if args.preselection:
        Cand_example = []
        Cand_pids = []
        original_cand_pids = copy.deepcopy(cand_pids)
    for pid in train_pids:
        if 'medqa' in args.data_root:
            pid = int(pid)
        train_example_pid = create_example_from_pid(
            pid, problems, args, test=True)
        train_samples.append(train_example_pid)  # Set test=True to avoid answer being added to the training input.
        if args.preselection:
            Cand_example.append(bm25_retrieve(train_example_pid, cand_examples, n=args.select_number))
            pid_idx = bm25_retrieve(train_example_pid, cand_examples, n=args.select_number, return_index=True)
            Cand_pids.append([original_cand_pids[c] for c in pid_idx])



        if 'pubmed' in args.data_root:
            answer_norm = problems[pid]['final_decision']
        elif "output" in problems[pid].keys():
            answer_norm = problems[pid]['output']
        else:
            if "unit" in problems[pid].keys():
                unit = problems[pid]['unit']
            else:
                unit = None
            answer_norm = normalize_answer(problems[pid]['answer'], unit)
        if 'choices' in problems[pid].keys():
            option = problems[pid]['choices']
        elif "options" in problems[pid].keys():
            if 'medqa' in args.data_root_test:
                option = []
                for o in problems[pid]['options'].keys():
                    option.append(problems[pid]['options'][o])
            else:
                option = problems[pid]['options']

        else:
            option = None
        if 'unit' in problems[pid].keys():
            unit = problems[pid]['unit']
        else:
            unit = None
        train_labels.append(answer_norm)
        units.append(unit)
        options.append(option)

    num_batch = math.ceil(len(train_samples) / args.batch_size)

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    if args.resume_epoch > 0:
        reward_history_json = json.load(open(os.path.join(args.ckpt_root, args.history_ckpt)))
        total_reward_history.extend(reward_history_json['total_reward_history'])
        total_loss_history.extend(reward_history_json['total_loss_history'])
        #args.score_th = reward_history_json['score_th'][0]
        #print("initial score_th is", args.score_th)

    STOP_FLAG = False

    for epoch in range(args.resume_epoch, args.epochs+args.resume_epoch):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            label_batch = train_labels[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            pid_batch = train_pids[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            unit_batch = units[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            option_batch = options[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]

            # We need to encode cands again every time we update the network
            if args.preselection:
                cand_batch = Cand_example[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
                embedding_cands = [algorithm.predict(cand_batch[c]) for c in range(len(cand_batch))]  # len(train_batch) x len(cand_examples) x embedding_size
                cand_pids = Cand_pids[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            else:

                embedding_cands = algorithm.predict(cand_examples)  # len(cand_examples) x embedding_size
            embedding_ctxt = algorithm.predict(train_batch)  # len(train_batch) x embedding_size
            if args.preselection:
                scores = torch.stack([torch.mm(embedding_ctxt[c].unsqueeze(0), embedding_cands[c].t()) for c in range(len(cand_batch))], dim=0).squeeze()  # len(train_batch) x len(cand_examples)
            else:
                scores = torch.mm(embedding_ctxt, embedding_cands.t())  # len(train_batch) x len(cand_examples)
            # print(f"unnormed scores: {scores}")

            scores = F.softmax(scores, dim=1)  # len(train_batch) x len(cand_examples)


            cids, reward, loss = get_batch_reward_loss(batch_i, scores, cand_pids, pid_batch, option_batch, unit_batch,
                                                       label_batch, args)

            logger.write(f"cids for sample[-1] in batch: {cids}")
            logger.write(f"Cand prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            # linear layer has Weight and bias
            # prev_param = list(policy_model.linear.parameters())[0].clone()
            # print(f"prev_param: {prev_param.data}")
            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # for each iteration/batch
                total_train_reward += reward
                total_train_loss += loss.item()

                reward_history.append(reward)
                loss_history.append(loss.item())

                if np.isnan(loss.item()):
                    STOP_FLAG = True
                    break
            else:
                total_train_reward += reward
                total_train_loss += 0

                reward_history.append(reward)
                loss_history.append(0)





        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(algorithm.model.linear.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(algorithm.model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")


        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(algorithm.model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")




        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
            "score_th": [args.score_th.detach().numpy().tolist()],
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # print cache info
        #logger.write(call_gpt3.cache_info())
        logger.write("============================================\n")

        if STOP_FLAG:
            break

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(algorithm.model.linear.state_dict(), ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='', help='Data root for train.')
    parser.add_argument('--data_root_vali', type=str, default=None, help='Data root for validation. If this one is left as None, we will random partition some data from training data as validation data.')
    parser.add_argument('--data_root_test', type=str, default='', help='Data root for test.')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list,
                        default=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                                 "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])

    # User options
    parser.add_argument('--label', type=str, default='exp0', help='Name for this run.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='Q-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A', 'SQ-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=5, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-4 settings
    parser.add_argument('--engine', type=str, default='gpt4', choices=['text-davinci-002', 'ada', 'gpt-3.5-turbo-1106', 'gpt4'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased', 'gpt2-xl'])
    parser.add_argument('--train_number', type=int, default=-1, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=1000, help='Number of candidate prompts.')
    parser.add_argument('--cand_ckpt', type=str, default=None, help="cand_pids root.")
    parser.add_argument('--train_ckpt', type=str, default=None, help="cand_pids root.")
    parser.add_argument('--train_ckpt2', type=str,
                        default=None,
                        help="cand_pids root2.")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--resume_epoch', type=int, default=0, help='The resumed epoch.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint root for the trained retriever')
    parser.add_argument('--history_ckpt', type=str, default=None, help='Checkpoint root for the history results json file.')

    # Method
    parser.add_argument('--algorithm', type=str, default='ERM', choices=['ERM'])
    parser.add_argument('--adapt_bn', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.0, help='hyperparameter for MMD regularization.')

    #rank
    parser.add_argument('--score_th', type=float, default=0, help='Initialize threshold for retriever score.')
    parser.add_argument('--adjust_batch', type=int, default=1, help='Adjust score_th every adjust_bach.')
    parser.add_argument('--preselection', action='store_true', help='Preselect candidate sample from a large pool of training samples.')
    parser.add_argument('--case_study', action='store_true',
                        help='case study for uncertain samples.')
    parser.add_argument('--select_number', type=int, default=50, help='select_number from preselection pool.')
    args = parser.parse_args()
    args.meta_batch_size = 1

    if 'pubmed' in args.data_root:
        args.ckpt_path = os.path.join(args.ckpt_root, 'pubmed')
    elif 'medqa' in args.data_root:
        args.ckpt_path = os.path.join(args.ckpt_root, 'medqa')
    elif args.data_root in ['ethos-national_origin', 'tweet_eval-stance_climate', 'kilt_nq', 'kilt_trex']:
        args.ckpt_path = os.path.join(args.ckpt_root, args.data_root)

    else:
        raise Exception("The dataset does not exist!")
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    args.ckpt_path = os.path.join(args.ckpt_path, args.algorithm)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True


    ## problems, test question ids, candidate prompt pids, RL training pids
    problems, cand_pids, train_pids = load_data(args)
    algorithm = init_algorithm(args)

    #exit(0)

    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt_root, args.ckpt)

        if os.path.exists(ckpt_path):
            algorithm.model.linear.load_state_dict(torch.load(ckpt_path))
            print("Policy model loaded")
        else:
            print(f"The ckpt path for [{ckpt_path}] does not exist!")  # CHECK
            #exit()



    else:
        print(f"!!! Load the pre-traind model instead!")  # CHECK


    ## construct candidate examples
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)
        cand_examples.append(example)




    #import ipdb; ipdb.set_trace()
    # exit()

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    policy_gradient_train(algorithm, problems, train_pids, cand_pids, cand_examples, args)
