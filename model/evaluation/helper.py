def get_predicted_answers(output, tokens):
    extracted_answers = []
    last_idx = None
    current_answer = ''
    for idx, label in enumerate(output):
        if label == 2 and len(current_answer) > 0: # add to existing answer, only if there is an existing answer..
            current_answer += tokens[idx] +' '
        elif label == 1:
            # append the current answer
            if len(current_answer) > 0:
                extracted_answers.append(current_answer)
            current_answer = tokens[idx] +' '
        elif label == 0:
            # extracted_answers.append(current_answer)
            current_answer = ''
    print('Outputs: ', extracted_answers)


def get_labeled_answers(labels, tokens):
    labels_stats = []
    for idx, label in enumerate(labels):
        if label == 1 and idx+1 < len(labels):
            next_label = labels[idx+1]
            count = 2
            while idx+count < len(labels) and next_label in [2, -100]:
                next_label = labels[idx+count]
                count += 1
            labels_stats.append((idx, count-1))
    print(labels_stats)
    all_answers = []
    for ans in labels_stats:
        ans_start = ans[0]
        answer_text = []
        for i in range(ans[1]):
            answer_text.append(tokens[ans_start+i])
        answer = ' '.join(answer_text)
        all_answers.append(answer)
    print('Answers: ', all_answers)


def get_token_segments(labels):
    labels_stats = []
    for idx, label in enumerate(labels):
        if label == 1 and idx+1 < len(labels):
            next_label = labels[idx+1]
            count = 2
            while idx+count < len(labels) and next_label in [2, -100]:
                next_label = labels[idx+count]
                count += 1
            labels_stats.append((idx, count-1))
    return labels_stats





