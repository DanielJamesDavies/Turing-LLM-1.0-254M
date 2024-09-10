import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

num_return_sequences = 11
max_length = 128
dataset_path = "./dataset"
original_dataset_path = "../phi3-mini/dataset"



character_count = 0
start_time = time.time()

torch.random.manual_seed(12)
device = torch.cuda.current_device()
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)



def save_txt(file_path, content):
    open(file_path, "w", encoding='utf-8').write(content)



def display_character_count(character_count):
    print("")
    print(f"Character Count:       {character_count:,}")
    print(f"Predicted Token Count: {(character_count/5):,}")
    print(f"Time Elapsed:          {time.time() - start_time}")
    print("")



def run():
    original_dataset_contents = os.listdir(original_dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    for folder in original_dataset_contents:
        current_path_original = original_dataset_path + "/" + folder
        path_contents_original = os.listdir(current_path_original)
        
        # Subjects
        if folder == "subjects":
            os.makedirs(dataset_path + "/subjects", exist_ok=True)
            items_txt = ""
            if "items.txt" in path_contents_original:
                items_txt = open(current_path_original + "/items.txt", 'r', encoding='utf-8').read()
                save_txt(dataset_path + "/subjects/items.txt", items_txt)
            
            for subject_index, subject in enumerate(items_txt.split("\n")):
                display_character_count(character_count)
                depth = 0
                path = dataset_path + "/" + folder + "/items/" + str(subject_index)
                path_original = original_dataset_path + "/" + folder + "/items/" + str(subject_index)
                os.makedirs(path, exist_ok=True)
                process_subject_item(path, path_original, subject, subject, depth)
            

            
def process_subject_item(path, path_original, item, subject, depth):
    global character_count
    
    os.makedirs(path, exist_ok=True)
    path_contents = os.listdir(path)
    path_original_contents = os.listdir(path_original)
    
    # Get or Generate
    desc_aug_txts_to_gen = []
    for i in range(6):
        if f"desc_{str(i)}.txt" in path_original_contents:
            if f"desc_{str(i)}.txt" in path_contents:
                aug_txt = open(path + f"/desc_{str(i)}.txt", 'r', encoding='utf-8').read()
                character_count += len(aug_txt)
            else:
                if i not in desc_aug_txts_to_gen:
                    desc_aug_txts_to_gen.append(i)
                
            # os.makedirs(path + f"/desc_{str(i)}", exist_ok=True)
            # desc_path_contents = os.listdir(path + f"/desc_{str(i)}")
            # for j in range(6):
            #     if f"aug_{str(j)}.txt" in desc_path_contents:
            #         aug_txt = open(path + f"/desc_{str(i)}/aug_{str(j)}.txt", 'r', encoding='utf-8').read()
            #         character_count += len(aug_txt)
            #     else:
            #         if i not in desc_aug_txts_to_gen:
            #             desc_aug_txts_to_gen.append(i)
                    
    generated_character_count = generate_aug_txt(desc_aug_txts_to_gen, path, path_original, item, subject, depth)
    character_count += generated_character_count

    display_character_count(character_count)
    
    # Process Child Items
    items_txt = ""
    if "items.txt" in path_original_contents:
        items_txt = open(path_original + "/items.txt", 'r', encoding='utf-8').read()
    
    if len(items_txt) != 0:
        for child_index, child in enumerate(items_txt.split("\n")):
            process_subject_item(path + "/items/" + str(child_index), path_original + "/items/" + str(child_index), child, subject, depth + 1)
            
            
            
def generate_aug_txt(desc_aug_txts_to_gen, path, path_original, item, subject, depth):
    generated_character_count = 0
    
    # batch_size = 3
    # batches = [desc_aug_txts_to_gen[i:i + batch_size] for i in range(0, len(desc_aug_txts_to_gen), batch_size)]
    
    # Open desc_num.txt files
    desc_txts = []
    split_desc_txts = []
    jobs = []
    for desc_num in desc_aug_txts_to_gen:
        desc_txt = open(path_original + f"/desc_{str(desc_num)}.txt", 'r', encoding='utf-8').read()
        desc_txts.append(desc_txt)
        split_desc_txts.append(get_split_desc_txt(desc_txt))
        
        split_desc_txt = get_split_desc_txt(desc_txt)
        
        for text_index, text in enumerate(split_desc_txt):
            jobs.append({ "desc_num": desc_num, "text_index": text_index, "text": text })
    
    # Get Job Batches
    batches = create_job_batches(jobs)
    
    print("Jobs Count: ", len(jobs))
    print(f"Batch Count: {len(batches)}")
    print(f"Batch Sizes: {', '.join([str(len(batch)) for batch in batches])}")
    print("")
    
    # Generate Augmented Texts
    augmented_texts = []
    
    for desc_num in desc_aug_txts_to_gen:
        augmented_texts.append([])
    
    for batch in batches:
        desc_txt_print_text = ", ".join(f"desc_{job['desc_num']}.{job['text_index']}" for job in batch)
        print(f"Generating augmented texts for {desc_txt_print_text} for {item} (Subject: {subject}, Depth: {depth})")
        
        texts = [job["text"] for job in batch]
        outputs = paraphrase(texts)
        
        for output_index, output in enumerate(outputs):
            desc_num = batch[output_index]["desc_num"]
            text_index = batch[output_index]["text_index"]
            for sequence_index, sequence in enumerate(output):
                while len(augmented_texts[desc_num]) - 1 < text_index:
                    augmented_texts[desc_num].append([])
                augmented_texts[desc_num][text_index].append(sequence)
    
    
    end_of_aug_text = "<|endofaugtext|>"
    for desc_num in desc_aug_txts_to_gen:
        new_content = ""
        
        # Add each single sequence to new_content
        for sequences in augmented_texts[desc_num]:
            for sequence in sequences:
                new_content += sequence + end_of_aug_text
        
        # Add each full sequence to new_content
        new_aug_texts = []
        for sequences_index, sequences in enumerate(augmented_texts[desc_num]):
            for sequence_index, sequence in enumerate(sequences):
                if sequences_index == 0:
                    new_aug_texts.append(sequence)
                else:
                    new_aug_texts[sequence_index] += "\n\n" + sequence
                    
        new_content += str(end_of_aug_text).join(new_aug_texts)
        generated_character_count += len(new_content)
        
        # Save aug
        save_txt(path + f"/desc_{str(desc_num)}.txt", new_content)
    
    return generated_character_count



def get_split_desc_txt(desc_txt):
    split_desc_txt = []
    
    def filter_desc_txt_split_by_line(line):
        if len(line.strip()) == 0:
            return False
        return True
        
    desc_txt_split_by_line = list(filter(filter_desc_txt_split_by_line, desc_txt.split("\n")))
    
    for i, paragraph in enumerate(desc_txt_split_by_line):
        if i == 0:
            split_desc_txt.append(paragraph)
        else:
            is_not_last_paragraph = i != len(desc_txt_split_by_line) - 1
            does_finish_with_full_stop = paragraph.strip()[-1] == "."
            last_paragraph_was_over_500_chars = len(desc_txt_split_by_line[i-1]) > 500
            
            if len(paragraph) > 500:
                split_desc_txt.append(paragraph)
            elif is_not_last_paragraph and does_finish_with_full_stop and last_paragraph_was_over_500_chars:
                split_desc_txt.append(paragraph)
            else:
                split_desc_txt[-1] += "\n" + paragraph
                
    return split_desc_txt



def create_job_batches(jobs):
    batches = []
    max_batch_size = 12
    remainder = len(jobs) % max_batch_size
    
    batches = [jobs[i:i + max_batch_size] for i in range(0, len(jobs), max_batch_size)]
    
    if remainder < max_batch_size - 1 and len(batches) > 1:
        min_batch_size = max_batch_size - 1
        curr_batch_index = len(batches) - 2
        
        while len(batches[-1]) < min_batch_size and min_batch_size > 1:
            job = batches[curr_batch_index].pop()
            batches[-1].append(job)
            
            if curr_batch_index == 0:
                min_batch_size += -1
                curr_batch_index = len(batches) - 2
            else:
                curr_batch_index += -1
    
    return batches



def paraphrase(texts):
    input_ids = tokenizer([f"paraphrase: {text}" for text in texts], return_tensors="pt", padding="longest", truncation=True, max_length=max_length).input_ids.to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            temperature=0.7,
            repetition_penalty=10.0,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            num_beams=11,
            num_beam_groups=11,
            max_length=max_length,
            diversity_penalty=3.0
        )
    end_time = time.time()
    seconds = end_time - start_time
    
    tokens_generated_count = 0
    
    for output in outputs:
        tokens_generated_count += len(output)

    generation_rate = tokens_generated_count / seconds

    print("")
    print(f"Generation Rate: {generation_rate:.4f} tok/sec")
    print("")
    
    if generation_rate < 800:
        torch.cuda.empty_cache()

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    output_texts = []
    for output_index, output_text in enumerate(res):
        if output_index % num_return_sequences == 0:
            output_texts.append([])
        output_texts[-1].append(output_text)

    return output_texts


   
run()
