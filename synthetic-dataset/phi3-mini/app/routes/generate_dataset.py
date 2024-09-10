import os
import time
from flask import Blueprint, request, jsonify
from app.llm import LLM
from collections import OrderedDict

generate_dataset_bp = Blueprint('generate_dataset_bp', __name__)

character_count = 0
start_time = time.time()
time_elapsed_offset = 0

@generate_dataset_bp.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    global start_time
    start_time = time.time()
    data = request.get_json()
    
    if "depth" not in data:
        return jsonify({"error": "Request must contain depth"}), 400
    max_depth = data["depth"]
    
    textbook_subjects_count = 3
    if "" in data:
        textbook_subjects_count = data["textbook_subjects_count"]
    
    os.makedirs("./dataset", exist_ok=True)
    
    # Subjects
    subjects = "Physics\nMathematics\nComputer Science\nChemistry\nBiology\nPhilosophy\nEnglish\nCreative Writing\nLanguages\nArt\nMusic\nEngineering\nEconomics\nBusiness\nGeography\nHistory\nCulinary Arts\nCommunication"
    create_subjects(subjects)
    current_path = "./dataset/subjects"
    depth = 0
    generate_data_for_item("subjects", -1, current_path, depth, max_depth)
    
    # Public Figures
    generate_data_for_public_figures(subjects)
    
    # Short Stories
    generate_data_for_short_stories(subjects)
    
    # Textbooks
    generate_data_for_textbooks(subjects.split("\n")[:textbook_subjects_count])
    
    return jsonify({ 'message': 'Success' })



def save_txt(file_path, content):
    open(file_path, "w", encoding='utf-8').write(content)



def create_subjects(subjects):
    os.makedirs("./dataset/subjects", exist_ok=True)
    save_txt('./dataset/subjects/items.txt', subjects)
    
    

def display_character_count(character_count):
    print("")
    print(f"Character Count:       {character_count:,}")
    print(f"Predicted Token Count: {(character_count/5):,}")
    print(f"Time Elapsed:          {time.time() - start_time + time_elapsed_offset}")
    print("")



# --------------------------------------------------
# Subject Items
# --------------------------------------------------



def generate_data_for_item(item, subject, current_path, depth, max_depth):
    global character_count
    
    # Search path
    path_contents = os.listdir(current_path)

    # Open or generate items.txt
    items_txt = ""
    if depth < max_depth:
        if "items.txt" in path_contents:
            items_txt = open(current_path + "/items.txt", 'r', encoding='utf-8').read()
        else:
            # Generate items.txt
            if current_path != "./dataset/subjects":
                parent_items_txt = open("/".join(current_path.split("/")[:-2]) + "/items.txt", 'r', encoding='utf-8').read()
            else:
                parent_items_txt = ""
            items_txt = generate_items_txt(item, subject, depth, parent_items_txt)
            save_txt(current_path + "/items.txt", items_txt)
    
    # Create item folders
    if depth < max_depth:
        if "items" not in path_contents:
            os.makedirs(current_path + "/items", exist_ok=True)
        items_path_contents = os.listdir(current_path + "/items")
        for index, new_item in enumerate(items_txt.split("\n")):
            if index not in items_path_contents:
                os.makedirs(current_path + "/items/" + str(index), exist_ok=True)

    # Open or generate public_figures.txt
    if depth < max_depth and depth < 2:
        item_public_figure_txts_to_gen = []
        for index, new_item in enumerate(items_txt.split("\n")):
            child_path_contents = os.listdir(current_path + "/items/" + str(index))
            if "public_figures.txt" not in child_path_contents:
                item_public_figure_txts_to_gen.append({ "index": index, "item": new_item })
        if len(item_public_figure_txts_to_gen) != 0:
            generate_public_figures_txt(item_public_figure_txts_to_gen, subject, depth, current_path)
    
    # Open or generate desc.txt
    if current_path != "./dataset/subjects":
        item_desc_txts_to_gen = []
        for prompt_index in range(6):
            if f"desc_{str(prompt_index)}.txt" in path_contents:
                item_desc_txt = open(current_path + f"/desc_{str(prompt_index)}.txt", 'r', encoding='utf-8').read()
                character_count += len(item_desc_txt)
            else:
                item_desc_txts_to_gen.append(prompt_index)
        if len(item_desc_txts_to_gen) != 0:
            all_generated_texts_char_count = generate_item_desc_txts(item, subject, depth, current_path, item_desc_txts_to_gen)
            character_count += all_generated_texts_char_count
    
    # Generate items
    if depth < max_depth:
        for index, new_item in enumerate(items_txt.split("\n")):
            display_character_count(character_count)
            print("Depth: " + str(depth) + " | Item: " + str(new_item))
            if depth == 0:
                subject = new_item
            generate_data_for_item(new_item, subject, current_path + "/items/" + str(index), depth+1, max_depth)



def generate_items_txt(item, subject, depth, parent_items_txt):
    llm = LLM()
    
    prompt = "Using your wealth of knowledge, please provide a numbered list of general sub-items relating to <|label|>. If the item has no sub-items, write the text, 'NA'. You only write the names separated by a line break. Never include parentheses."
    if depth == 1:
        prompt = prompt.replace("<|label|>", str(item))
        prompt += f" Please be concise and mention core topics in {item}. Never mention copyrighted content or trademarks. Never write the text 'sub-items'."
        prompt += " Start with the most important. Write 18 sub-items. To end the list write 19 and end."
    else:
        prompt = prompt.replace("<|label|>", str(item) + ", in the field of " + str(subject))
        prompt += " Please be as exhaustive as possible but never mention copyrighted content or trademarks. Never write the text 'sub-items'."
        prompt += " Write 22 sub-items. To end the list write 23."
        
    if depth == 1:
        prompt += f" Line-seperated list of core subjects of {item}."
    elif depth == 2:
        prompt += " Sub-items are general and obvious. Line-seperated list."
    else:
        prompt += " Line-seperated list."
        
    prompt += "Structure: 1. Sub-item\n2. Sub-item\n3. Sub-item"
    
    messages = [ {"role": "user", "content": prompt } ]
    
    max_length = 200
    skip_special_tokens = False
    
    print(f"Generating list of items for {item} ({subject}) |", end=" ")
    generated_text, _ = llm.generate(messages, max_length, skip_special_tokens)
    
    if generated_text.lower().replace(" ", "") == "na<|end|>":
        return []

    decoded_list = [". ".join(e.split(". ")[1:]).strip().rstrip(".") for e in generated_text.split("\n")]
    
    if decoded_list[-1][-7:] != "<|end|>":
        decoded_list.pop()
    else:
        decoded_list[-1] = decoded_list[-1][:-7]
        
    new_decoded_list = []
    parent_items = parent_items_txt.split("\n")
    for new_item in decoded_list:
        if new_item not in parent_items:
            new_decoded_list.append(new_item)
    
    new_decoded_list = list(dict.fromkeys(new_decoded_list))
    
    if depth == 1:
        new_decoded_list = new_decoded_list[:18]
    else:
        new_decoded_list = new_decoded_list[:22]
        
    def check_na(e):
        if str(e).lower().strip() == "na":
            return False
        return True
    
    new_decoded_list = filter(check_na, new_decoded_list)
    
    new_decoded_list = "\n".join(new_decoded_list)
            
    return new_decoded_list



def generate_item_desc_txts(item, subject, depth, current_path, item_desc_txts_to_gen):
    llm = LLM()
    
    prompts = [
        {
            "type": "normal",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a description about <|label|>.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a description about <|label|>."
        },
        {
            "type": "formal",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a brilliant explanation on the topic of <|label|>. The explanation will be formal and use perplexity.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a brilliant explanation on the topic of <|label|>. Please be formal and use perplexity."
        },
        {
            "type": "dry-humour",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a description about <|label|>. The description will be mainly informative with a little amount of dry humour where appropriate.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a description on the topic of <|label|>. Please be mainly informative with a little amount of dry humour where appropriate. The first sentence should be normal and without humour."
        },
        {
            "type": "agreeable",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a description about <|label|>. The description will be agreeable and kind.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a description on the topic of <|label|>. Please be agreeable and kind."
        },
        {
            "type": "report",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a report on the topic of <|label|>. Do not mention this is a report. Never include a references section.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a report on the topic of <|label|>. Do not mention this is a report."
        },
        {
            "type": "timeline",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a timeline on the topic of <|label|>. Feel free to add eras or years in headings.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a report of a timeline on the topic of <|label|>. Do not mention this is a report. Feel free to add eras or years. Write any eras or years mentioned in the header."
        }
    ]
    heading_prompt_all = "Never write more than 8 headings. Never write anything innapropriate in a work setting. Please be as exhaustive as possible but it is vital that you never mention copyrighted content or trademarks. Never include references to works. Write the list of headings in a line-seperated numbered list, with '\n' between the headings. Structure: 1 Heading\n2 Heading"
    section_prompt_all = "Remember this is a small section for a larger text, only include an introduction if it is the introduction. Never write more than 150 words, end with '\\n', but without the quotes. Never write a note to me. Never write anything innapropriate in a work setting. Please be as exhaustive as possible but it is vital that you never mention copyrighted content or trademarks. Never include references to works. Never include markdown. Do not start in a generic way. Only include one heading. Structure: Heading\nText"
    
    print(f"Generating desc.txt Headings for {item} ({subject}) |", end=" ")
    all_headings = generate_item_desc_txt_headings(item, subject, depth, item_desc_txts_to_gen, llm, prompts, heading_prompt_all)
    
    all_generated_texts_char_count = 0
    
    for i in range(len(all_headings)):
        headings = []
        for line in all_headings[i].split("\n"):
            if len(line.strip()) != 0:
                headings.append(line.strip())
        
        if i == 1 and depth < 2:
            llm.reset()

        prompt_index = item_desc_txts_to_gen[i]
        print(f"Generating desc_{prompt_index}.txt for {item} ({subject}) | ", end="")
        prompt = prompts[prompt_index]["section"] + " " + section_prompt_all
        sections = generate_item_desc_txt_sections(item, subject, depth, llm, prompt, prompt_index, headings)
        sections = ["\n".join(section.strip().split("\n")).strip().rstrip("-") for section in sections]
        prev_sections = sections.copy()
        sections = ["\n".join(section.strip().lstrip("\\n").strip().split("\n")).strip().rstrip("-") for section in sections]
        sections = [section.rstrip(")").rstrip("\\n").rstrip("(\\").strip() for section in sections]

        new_sections = []
        for j, section in enumerate(sections):
            new_section = '\n'.join(line for line in section.split('\n')[1:] if line != section.split('\n')[0]).strip().lstrip("\\n").strip()
            if new_section.startswith(headings[j].strip()):
                new_section = new_section[len(headings[j].strip()):]
            if new_section.startswith("Text: "):
                new_section = new_section[len("Text: "):]
            new_sections.append(new_section.strip())
        
        try:
            for j in range(len(new_sections)):
                if new_sections[j].strip()[0].isdigit():
                    print("")
                    for line in prev_sections[j].split("\n")[:3]:
                        print("'", line)
                    print("")
        except:
            print("", end="")
        
        text = "\n\n".join(new_sections)
        text = text.strip()
        save_txt(current_path + f"/desc_{str(prompt_index)}.txt", text)
        all_generated_texts_char_count += len(text)
    
    return all_generated_texts_char_count



def generate_item_desc_txt_headings(item, subject, depth, item_desc_txts_to_gen, llm, prompts, heading_prompt_all):
    batch = []
    
    for i in item_desc_txts_to_gen:
        prompt = prompts[i]["heading"] + " " + heading_prompt_all
        if depth == 0:
            prompt = prompt.replace("<|label|>", str(item))
        else:
            prompt = prompt.replace("<|label|>", str(item) + ", in the field of " + str(subject))
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    max_length = 300
    skip_special_tokens = True
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens)

    return generated_texts



def generate_item_desc_txt_sections(item, subject, depth, llm, prompt, prompt_index, headings):
    batch = []
    
    for heading in headings:
        section_prompt = str(prompt)
        section_prompt = section_prompt.replace("<|heading|>", str(heading))
        if depth == 0:
            section_prompt = section_prompt.replace("<|label|>", str(item))
        else:
            section_prompt = section_prompt.replace("<|label|>", str(item) + ", in the field of " + str(subject))
        messages = [ {"role": "user", "content": section_prompt } ]
        batch.append(messages)
    
    max_length = 300
    skip_special_tokens = True
    reset_on_low_tok_sec = True
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec)

    return generated_texts



# --------------------------------------------------
# Public Figures
# --------------------------------------------------



def generate_public_figures_txt(items, subject, depth, current_path):
    llm = LLM()
    
    batch = []
    
    for item in items:
        prompt = "Using your wealth of knowledge, please provide a numbered list of public figures relating to <|label|>."
        if depth == 1 or str(subject) == "-1":
            prompt = prompt.replace("<|label|>", "the subject of " + str(item["item"]))
        else:
            prompt = prompt.replace("<|label|>", str(item["item"]) + ", in the field of " + str(subject))
            
        prompt += "Only write names, separated by a line break. Never include parentheses or notes. Never explain why you wrote a name."
        prompt += " Never mention copyrighted content or trademarks. Never write names of evil people. Never write names of communists, facists, or other political extremists. Never write the text 'public figures'."
        prompt += f" Start with the most important person related to {item['item']}. Write 20 public figures. To end the list write 21 and end. Please be exhaustive. Only include relevant figures. Never write titles. Always number items. Never use roman numerals."
        prompt += f" Write a line-seperated numbered list of full names of public figures of {item['item']}."
        prompt += " Structure: 1. Full Name\n2. Full Name\n3. Full Name"
        
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    print(f"Generating list of public figures for {subject} |", end=" ")
    
    max_length = 200
    skip_special_tokens = False
    reset_on_low_tok_sec = True
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec)
    
    for index, generated_text in enumerate(generated_texts):
        if generated_text.lower().replace(" ", "") == "na<|end|>":
            continue
            
        def check_filter_1(e):
            if len(str(e).lower().strip()) == 0:
                return False
            if len(str(e).lower().strip()) >= 50:
                return False
            return True
        
        decoded_list = generated_text.split("\n")
        decoded_list = filter(check_filter_1, decoded_list)
        
        if generated_text.strip()[0].isdigit():
            decoded_list = [". ".join(e.split(". ")[1:]).strip().rstrip(".") for e in decoded_list]
        else:
            decoded_list = [e.strip().rstrip(".") for e in decoded_list]
        
        if decoded_list[-1][-7:] != "<|end|>":
            decoded_list.pop()
        else:
            decoded_list[-1] = decoded_list[-1][:-7]
            
        decoded_list = list(dict.fromkeys(decoded_list))
        
        decoded_list = decoded_list[:20]
            
        def check_filter_2(e):
            if len(str(e).lower().strip()) == 0:
                return False
            if len(str(e).lower().strip()) >= 50:
                return False
            if str(e).lower().strip() == "na":
                return False
            if str(e).lower().strip() == "end":
                return False
            if str(e).lower().strip() == "end.":
                return False
            if str(e).lower().strip() == "19":
                return False
            if str(e).lower().strip() == "19.":
                return False
            if str(e).lower().strip() == "end of list.":
                return False
            return True
        
        decoded_list = filter(check_filter_2, decoded_list)
        
        new_decoded_list = []
        for decoded_list_item in decoded_list:
            if "(" not in decoded_list_item:
                new_decoded_list.append(decoded_list_item)
            else:
                new_decoded_list.append(decoded_list_item.split("(")[0])
        
        new_decoded_list = filter(check_filter_2, new_decoded_list)
        new_decoded_list = "\n".join(new_decoded_list)
        
        save_txt(current_path + "/items/" + str(items[index]["index"]) + "/public_figures.txt", new_decoded_list)



def generate_data_for_public_figures(subjects):
    global character_count
    
    print("")
    print("")
    print("")
    print("generate_data_for_public_figures")
    print("")
    print("")
    print("")
    
    os.makedirs("./dataset/public_figures", exist_ok=True)
    for i, subject in enumerate(subjects.split("\n")):
        os.makedirs("./dataset/public_figures/" + str(i), exist_ok=True)
        os.makedirs("./dataset/public_figures/" + str(i) + "/public_figures", exist_ok=True)
        
        figures_subject_path_contents = os.listdir("./dataset/public_figures/" + str(i))
        if "public_figures.txt" not in figures_subject_path_contents:
            public_figures = get_item_public_figures("./dataset/subjects/items/" + str(i))
            public_figures = "\n".join(public_figures[:84])
            save_txt("./dataset/public_figures/" + str(i) + "/public_figures.txt", public_figures)
    
        # For each public figure, generate descriptions
        public_figures_txt = open("./dataset/public_figures/" + str(i) + "/public_figures.txt", 'r', encoding='utf-8').read()
        for figure_index, figure in enumerate(public_figures_txt.split("\n")):
            current_path = "./dataset/public_figures/" + str(i) + "/public_figures/" + str(figure_index)
            os.makedirs(current_path, exist_ok=True)
            path_contents = os.listdir(current_path)
            
            figure_desc_txts_to_gen = []
            for prompt_index in range(2):
                if f"desc_{str(prompt_index)}.txt" in path_contents:
                    item_desc_txt = open(current_path + f"/desc_{str(prompt_index)}.txt", 'r', encoding='utf-8').read()
                    character_count += len(item_desc_txt)
                else:
                    figure_desc_txts_to_gen.append(prompt_index)
            if len(figure_desc_txts_to_gen) != 0:
                all_generated_texts_char_count = generate_public_figure_desc_txts(figure, subject, current_path, figure_desc_txts_to_gen)
                character_count += all_generated_texts_char_count

            display_character_count(character_count)



def get_item_public_figures(current_path):
    public_figures = []
    
    path_contents = os.listdir(current_path)
    if "public_figures.txt" in path_contents:
        public_figures_txt = open(current_path + "/public_figures.txt", 'r', encoding='utf-8').read()
        public_figures = public_figures + public_figures_txt.split("\n")
        
    if "items" in path_contents:
        items_public_figures = []
        items_path_contents = os.listdir(current_path + "/items")
        for item_folder in items_path_contents:
            result = get_item_public_figures(current_path + "/items/" + str(item_folder))
            for index, result_figure in enumerate(result):
                if len(items_public_figures) < index + 1:
                    items_public_figures.append([])
                items_public_figures[index].append(result_figure)
        for figures in items_public_figures:
            public_figures = public_figures + figures
    
    new_public_figures = []
    for public_figure in public_figures:
        public_figure = public_figure.lstrip("-").strip()
        if public_figure.strip()[:2].lower() == "dr":
            new_public_figures.append(public_figure.strip()[2:].strip())
        elif public_figure.strip()[:3].lower() == "sir":
            new_public_figures.append(public_figure.strip()[3:].strip())
        else:
            new_public_figures.append(public_figure)
            
    public_figures = list(OrderedDict.fromkeys(new_public_figures))
            
    def check_filter(e):
        if len(str(e).lower().strip()) == 0:
            return False
        if len(str(e).lower().strip()) >= 50:
            return False
        if str(e).lower().strip() == "na":
            return False
        if str(e).lower().strip() == "end":
            return False
        if str(e).lower().strip() == "end.":
            return False
        if str(e).lower().strip() == "19":
            return False
        if str(e).lower().strip() == "19.":
            return False
        if str(e).lower().strip() == "end of list.":
            return False
        return True
    
    public_figures = filter(check_filter, public_figures)
    
    public_figures = list(public_figures)

    return public_figures




def generate_public_figure_desc_txts(figure, subject, current_path, figure_desc_txts_to_gen):
    llm = LLM()
    
    prompts = [
        {
            "type": "formal",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a brilliant explanation on the topic of <|label|>. The explanation will be formal and use perplexity.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a brilliant explanation on the topic of <|label|>. Please be formal and use perplexity."
        },
        {
            "type": "dry-humour",
            "heading": "Using your wealth of knowledge, please provide a list of headings for a description about <|label|>. The description will be mainly informative with a little amount of dry humour where appropriate.",
            "section": "Using your wealth of knowledge, please provide a section with the heading \"<|heading|>\". The section is deep within a description on the topic of <|label|>. Please be mainly informative with a little amount of dry humour where appropriate. The first sentence should be normal and without humour."
        }
    ]
    heading_prompt_all = "Never write more than 8 headings. Never write anything innapropriate in a work setting. Please be as exhaustive as possible but it is vital that you never mention copyrighted content or trademarks. Never include references to works. Write the list of headings in a line-seperated numbered list, with '\n' between the headings. Structure: 1 Heading\n2 Heading"
    section_prompt_all = "Remember this is a small section for a larger text, only include an introduction if it is the introduction. Never write more than 150 words, end with '\\n', but without the quotes. Never write a note to me. Never write anything innapropriate in a work setting. Please be as exhaustive as possible but it is vital that you never mention copyrighted content or trademarks. Never include references to works. Never include markdown. Do not start in a generic way. Only include one heading. Only write what you know to be true. Structure: Heading\nText"
    
    print(f"Generating desc.txt Headings for {figure} ({subject}) |", end=" ")
    all_headings = generate_public_figure_desc_txt_headings(figure, subject, figure_desc_txts_to_gen, llm, prompts, heading_prompt_all)
    
    all_generated_texts_char_count = 0
    
    for i in range(len(all_headings)):
        headings = []
        for line in all_headings[i].split("\n"):
            if len(line.strip()) != 0:
                headings.append(line.strip())
        
        if i == 1:
            llm.reset()

        prompt_index = figure_desc_txts_to_gen[i]
        print(f"Generating desc_{prompt_index}.txt for {figure} ({subject}) | ", end="")
        prompt = prompts[prompt_index]["section"] + " " + section_prompt_all
        sections = generate_public_figure_desc_txt_sections(figure, subject, llm, prompt, prompt_index, headings)
        sections = ["\n".join(section.strip().split("\n")).strip().rstrip("-") for section in sections]
        prev_sections = sections.copy()
        sections = ["\n".join(section.strip().lstrip("\\n").strip().split("\n")).strip().rstrip("-") for section in sections]
        sections = [section.rstrip(")").rstrip("\\n").rstrip("(\\").strip() for section in sections]

        new_sections = []
        for j, section in enumerate(sections):
            new_section = '\n'.join(line for line in section.split('\n')[1:] if line != section.split('\n')[0]).strip().lstrip("\\n").strip()
            if new_section.startswith(headings[j].strip()):
                new_section = new_section[len(headings[j].strip()):]
            if new_section.startswith("Text: "):
                new_section = new_section[len("Text: "):]
            new_sections.append(new_section.strip())
        
        try:
            for j in range(len(new_sections)):
                if new_sections[j].strip()[0].isdigit():
                    print("")
                    for line in prev_sections[j].split("\n")[:3]:
                        print("'", line)
                    print("")
        except:
            print("", end="")
        
        text = "\n\n".join(new_sections)
        text = text.strip()
        save_txt(current_path + f"/desc_{str(prompt_index)}.txt", text)
        all_generated_texts_char_count += len(text)
    
    return all_generated_texts_char_count



def generate_public_figure_desc_txt_headings(figure, subject, figure_desc_txts_to_gen, llm, prompts, heading_prompt_all):
    batch = []
    
    for i in figure_desc_txts_to_gen:
        prompt = prompts[i]["heading"] + " " + heading_prompt_all
        prompt = prompt.replace("<|label|>", str(figure) + ", relating to the field of " + str(subject))
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    max_length = 300
    skip_special_tokens = True
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens)

    return generated_texts



def generate_public_figure_desc_txt_sections(figure, subject, llm, prompt, prompt_index, headings):
    batch = []
    
    for heading in headings:
        section_prompt = str(prompt)
        section_prompt = section_prompt.replace("<|heading|>", str(heading))
        section_prompt = section_prompt.replace("<|label|>", str(figure) + ", relating to the field of " + str(subject))
        messages = [ {"role": "user", "content": section_prompt } ]
        batch.append(messages)
    
    max_length = 300
    skip_special_tokens = True
    reset_on_low_tok_sec = True
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec)

    return generated_texts



# --------------------------------------------------
# Short Stories
# --------------------------------------------------



def generate_data_for_short_stories(subjects):
    global character_count
    
    print("")
    print("")
    print("")
    print("generate_data_for_short_stories")
    print("")
    print("")
    print("")
    
    os.makedirs("./dataset/short_stories", exist_ok=True)
    subject_short_stories_txt_to_gen = []
    for i, subject in enumerate(subjects.split("\n")):
        os.makedirs("./dataset/short_stories/" + str(i), exist_ok=True)
        os.makedirs("./dataset/short_stories/" + str(i) + "/short_stories", exist_ok=True)
        os.makedirs("./dataset/short_stories/" + str(i) + "/items", exist_ok=True)
        path_contents = os.listdir("./dataset/short_stories/" + str(i))
        if "short_stories.txt" not in path_contents:
            subject_short_stories_txt_to_gen.append({ "index": i, "item": subject })
        
    # Generate short_stories.txt
    subject = "-1"
    depth = 1
    current_path = "./dataset/short_stories/"
    if len(subject_short_stories_txt_to_gen) != 0:
        print("Generating short_stories.txt for " + current_path)
        generate_short_stories_txt(subject_short_stories_txt_to_gen, subject, depth, current_path)
        
    # For each subject
    for i, subject in enumerate(subjects.split("\n")):
        depth = 1
        parent_item = subject
        current_path = "./dataset/short_stories/" + str(i) + "/"
        short_stories_txt = open(current_path + "short_stories.txt", 'r', encoding='utf-8').read()
        
        subject_short_stories_text_txt_to_gen = []
        short_stories_path_contents = os.listdir(current_path + "short_stories")
        for index, item in enumerate(short_stories_txt.split("\n")):
            if f"{str(index)}.txt" not in short_stories_path_contents:
                subject_short_stories_text_txt_to_gen.append({ "index": index, "item": item })
            else:
                short_story_text_txt = open(current_path + "short_stories/" + f"{str(index)}.txt", 'r', encoding='utf-8').read()
                character_count += len(short_story_text_txt)
        
        # Generate /short_stories/0.txt ...
        if len(subject_short_stories_text_txt_to_gen) != 0:
            print("Generating short_stories/texts for " + current_path)
            all_generated_texts_char_count = generate_short_stories_text_txt(subject_short_stories_text_txt_to_gen, parent_item, current_path)
            character_count += all_generated_texts_char_count
            display_character_count(character_count)
        
        # Get subject items
        depth = 2
        subject_items = []
        subject_path_contents = os.listdir("./dataset/subjects/items/" + str(i))
        if "items.txt" in subject_path_contents:
            subject_items_txt = open("./dataset/subjects/items/" + str(i) + "/items.txt", 'r', encoding='utf-8').read()
            subject_items = subject_items_txt.split("\n")
    
        subject_items_short_stories_txt_to_gen = []
        for subject_item_index, subject_item in enumerate(subject_items):
            os.makedirs(current_path + "items/" + str(subject_item_index), exist_ok=True)
            path_contents = os.listdir(current_path + "items/" + str(subject_item_index))
            if "short_stories.txt" not in path_contents:
                subject_items_short_stories_txt_to_gen.append({ "index": subject_item_index, "item": subject_item })
            
        # Generate short_stories.txt
        current_path = "./dataset/short_stories/" + str(i) + "/items/"
        if len(subject_items_short_stories_txt_to_gen) != 0:
            print("Generating short_stories.txt for " + current_path)
            generate_short_stories_txt(subject_items_short_stories_txt_to_gen, subject, depth, current_path)
        
        # For each subject item
        for subject_item_index, subject_item in enumerate(subject_items):
            parent_item = subject_item
            current_path = "./dataset/short_stories/" + str(i) + "/items/" + str(subject_item_index) + "/"
            os.makedirs(current_path + "short_stories", exist_ok=True)
            short_stories_txt = open(current_path + "short_stories.txt", 'r', encoding='utf-8').read()
            
            subject_item_short_stories_text_txt_to_gen = []
            short_stories_path_contents = os.listdir(current_path + "short_stories")
            for index, item in enumerate(short_stories_txt.split("\n")):
                if f"{str(index)}.txt" not in short_stories_path_contents:
                    subject_item_short_stories_text_txt_to_gen.append({ "index": index, "item": item })
                else:
                    short_story_text_txt = open(current_path + "short_stories/" + f"{str(index)}.txt", 'r', encoding='utf-8').read()
                    character_count += len(short_story_text_txt)
            
            # Generate /short_stories/0.txt ...
            if len(subject_item_short_stories_text_txt_to_gen) != 0:
                print("Generating short_stories/texts for " + current_path)
                all_generated_texts_char_count = generate_short_stories_text_txt(subject_item_short_stories_text_txt_to_gen, parent_item, current_path)
                character_count += all_generated_texts_char_count
                display_character_count(character_count)



def generate_short_stories_txt(items, subject, depth, current_path):
    llm = LLM()
    
    batch = []
    
    for item in items:
        prompt = "Please come up with a numbered list of 17 simple activity scenario titles related to <|label|>. Never write anything innapropriate in a work setting. The titles should be line separated and start with a number."
        if depth == 1 or str(subject) == "-1":
            prompt = prompt.replace("<|label|>", "the subject of " + str(item["item"]))
        else:
            prompt = prompt.replace("<|label|>", str(item["item"]) + ", in the field of " + str(subject))

        prompt += " Structure: 1. Full Name\n2. Full Name\n3. Full Name"
        
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    print(f"Generating list of short stories for {subject} |", end=" ")
    
    max_length = 400
    skip_special_tokens = False
    reset_on_low_tok_sec = True
    batch_size = 8
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec, batch_size)
    
    for index, generated_text in enumerate(generated_texts):
        if generated_text.lower().replace(" ", "") == "na<|end|>":
            continue
            
        def check_filter_1(e):
            if len(str(e).lower().strip()) == 0:
                return False
            return True
        
        decoded_list = generated_text.split("\n")
        decoded_list = filter(check_filter_1, decoded_list)
        
        decoded_list = [". ".join(e.split(". ")[1:]).strip().rstrip(".") for e in decoded_list]
        
        if decoded_list[-1][-7:] != "<|end|>":
            decoded_list.pop()
        else:
            decoded_list[-1] = decoded_list[-1][:-7]
            
        decoded_list = list(dict.fromkeys(decoded_list))
        
        decoded_list = decoded_list[:16]
            
        def check_filter_2(e):
            if len(str(e).lower().strip()) == 0:
                return False
            if str(e).lower().strip() == "na":
                return False
            if str(e).lower().strip() == "end":
                return False
            if str(e).lower().strip() == "end.":
                return False
            if str(e).lower().strip() == "end of list.":
                return False
            return True
        
        decoded_list = filter(check_filter_2, decoded_list)
        decoded_list = "\n".join(decoded_list)
        
        print("")
        print("")
        print(decoded_list)
        print("")
        print("")
        
        save_txt(current_path + str(items[index]["index"]) + "/short_stories.txt", decoded_list)



def generate_short_stories_text_txt(items, parent_item, current_path):
    llm = LLM()
    
    batch = []
    
    for item in items:
        prompt = f"Please write a very simple story with the title \"{item['item']}\". Start with the title and then a new line. Do not use cliche phrases in story writing. Never write anything innapropriate in a work setting. The story is short and no longer than two paragraphs. If you start a third paragraph, type <|end|>. The story should involve activities and be matter of fact."
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    print(f"Generating of short stories for {parent_item} |", end=" ")
    
    max_length = 312
    skip_special_tokens = False
    reset_on_low_tok_sec = True
    batch_size = 8
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec, batch_size)
    
    all_generated_texts_char_count = 0
    for index, generated_text in enumerate(generated_texts):
        new_generated_text = "\n".join(generated_text.strip().split("\n")[1:]).strip()
        new_generated_text = new_generated_text.lstrip("-").strip()
        new_generated_text = new_generated_text.split("<|endoftext|>")[0]
        new_generated_text = new_generated_text.split("<|end|>")[0]
        all_generated_texts_char_count += len(new_generated_text)
        save_txt(current_path + "short_stories/" + str(items[index]["index"]) + ".txt", new_generated_text)

    return all_generated_texts_char_count



# --------------------------------------------------
# Textbooks
# --------------------------------------------------



def generate_data_for_textbooks(subjects):
    global character_count
    
    print("")
    print("")
    print("")
    print("generate_data_for_textbooks")
    print("")
    print("")
    print("")
    
    os.makedirs("./dataset/textbooks", exist_ok=True)
    subject_textbooks_txt_to_gen = []
    for i, subject in enumerate(subjects):
        os.makedirs("./dataset/textbooks/" + str(i), exist_ok=True)
        os.makedirs("./dataset/textbooks/" + str(i) + "/items", exist_ok=True)
        
    # For each subject
    for i, subject in enumerate(subjects):
        depth = 0
        
        current_path = "./dataset/textbooks/" + str(i) + "/"
        
        os.makedirs(current_path, exist_ok=True)
        os.makedirs(current_path + "items", exist_ok=True)
        
        # Generate textbook
        current_path_contents = os.listdir(current_path)
        if "textbook.txt" not in current_path_contents:
            print("Generating textbooks for " + current_path)
            textbook_char_count = generate_textbook(subject, subject, depth, current_path)
            character_count += textbook_char_count
        else:
            textbook_txt = open(current_path + "textbook.txt", 'r', encoding='utf-8').read()
            character_count += len(textbook_txt)
        display_character_count(character_count)
        
        # Get subject items
        depth = 1
        item_items = []
        subject_path_contents = os.listdir("./dataset/subjects/items/" + str(i))
        if "items.txt" in subject_path_contents:
            item_items_txt = open("./dataset/subjects/items/" + str(i) + "/items.txt", 'r', encoding='utf-8').read()
            item_items = item_items_txt.split("\n")
            
        for subject_item_index, subject_item in enumerate(item_items):
            current_path = "./dataset/textbooks/" + str(i) + "/items/" + str(subject_item_index) + "/"
            os.makedirs(current_path + "textbooks", exist_ok=True)
            
            # Generate textbook
            current_path_contents = os.listdir(current_path)
            if "textbook.txt" not in current_path_contents:
                print("Generating textbooks for " + current_path)
                textbook_char_count = generate_textbook(subject_item, subject, depth, current_path)
                character_count += textbook_char_count
            else:
                textbook_txt = open(current_path + "textbook.txt", 'r', encoding='utf-8').read()
                character_count += len(textbook_txt)
            display_character_count(character_count)
            
            # Get subject items
            depth = 2
            item_items = []
            subject_path_contents = os.listdir("./dataset/subjects/items/" + str(i) + "/items/" + str(subject_item_index))
            if "items.txt" in subject_path_contents:
                item_items_txt = open("./dataset/subjects/items/" + str(i) + "/items/" + str(subject_item_index) + "/items.txt", 'r', encoding='utf-8').read()
                item_items = item_items_txt.split("\n")
                
            for subject_item_index_2, subject_item_2 in enumerate(item_items):
                current_path = "./dataset/textbooks/" + str(i) + "/items/" + str(subject_item_index) + "/items/" + str(subject_item_index_2) + "/"
                os.makedirs(current_path, exist_ok=True)
                
                # Generate textbook
                current_path_contents = os.listdir(current_path)
                if "textbook.txt" not in current_path_contents:
                    print("Generating textbooks for " + current_path)
                    textbook_char_count = generate_textbook(subject_item_2, subject, depth, current_path)
                    character_count += textbook_char_count
                else:
                    textbook_txt = open(current_path + "textbook.txt", 'r', encoding='utf-8').read()
                    character_count += len(textbook_txt)
                display_character_count(character_count)



def generate_textbook(item, subject, depth, current_path):
    textbook_name = item
    if depth != 0:
        textbook_name = item + f" ({subject})"
    
    print(f"Generating Textbook: {textbook_name}")
    
    textbook = f"# {textbook_name}"
    
    headings = generate_textbook_headings(textbook_name, depth)
    subheadings = generate_textbook_subheadings(textbook_name, headings, depth)
    for i, heading in enumerate(headings):
        paragraphs = generate_textbook_paragraphs(textbook_name, headings, i, subheadings[i], depth)
        textbook += "\n\n" + paragraphs
        
    save_txt(current_path + "/textbook.txt", textbook)
    print("")
    
    return len(textbook)



def generate_textbook_headings(textbook_name, depth):
    llm = LLM()
    
    generate_headings_prompt_template = " ".join([
        "Please write a numbered list of 16 headings for an original textbook called \"<|textbook_name|>\".",
        "If you accidentally write \"17. \" then please write \"<|end|>\" after that.",
        "Thank you!",
        "The structure of your response should be: 1. Heading\n2. Heading\n3. Heading"
    ])
    prompt = generate_headings_prompt_template.replace("<|textbook_name|>", textbook_name)
    
    messages = [ {"role": "user", "content": prompt } ]
    
    max_length = 400
    skip_special_tokens = False
    
    print(f"   Generating Headings |", end=" ")
    generated_text, _ = llm.generate(messages, max_length, skip_special_tokens)
    
    headings = generated_text.split("\n")
        
    def filter_fun(e):
        if len(str(e).strip()) == 0:
            return False
        return True
    
    headings = list(filter(filter_fun, headings))[:16]
    headings = [e.split("<|end|>")[0].split("<|endoftext|>")[0] for e in headings]

    return headings


def generate_textbook_subheadings(textbook_name, headings, depth):
    llm = LLM()
    
    generate_subheadings_prompt_template = " ".join([
        "Please write a numbered list of 8 subheadings which would go under the heading \"<|heading|>\" for an original textbook called \"<|textbook_name|>\".",
        "If you accidentally write \"9. \" then please write \"<|end|>\" after that.",
        "Thank you!",
        "The structure of your response should be: 1. Subheading\n2. Subheading\n3. Subheading"
    ])
    
    batch = []
    for heading in headings:
        prompt = str(generate_subheadings_prompt_template)
        prompt = prompt.replace("<|textbook_name|>", textbook_name)
        prompt = prompt.replace("<|heading|>", heading)
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    max_length = 400
    skip_special_tokens = False
    reset_on_low_tok_sec = True
    batch_size = 8
    print(f"   Generating Subheadings |", end=" ")
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec, batch_size)
    
    subheadings = []
    
    for subheadings_set in generated_texts:
        subheadings_set = subheadings_set.split("\n")
        
        def filter_fun(e):
            if len(str(e).strip()) == 0:
                return False
            return True
        
        subheadings_set = list(filter(filter_fun, subheadings_set))[:8]
        subheadings_set = [e.split("<|end|>")[0].split("<|endoftext|>")[0] for e in subheadings_set]
        
        subheadings.append(subheadings_set)

    return subheadings



def generate_textbook_paragraphs(textbook_name, headings, heading_index, subheadings, depth):
    llm = LLM()
    
    generate_paragraph_prompt_template = " ".join([
        "Please write a comprehensive, concise paragraph section for an original textbook.",
        "The subheading is \"<|subheading|>\", the heading is \"<|heading|>\", and the textbook is called \"<|textbook_name|>\".",
        "Write at least 5 sentences. No new headings. Never mention the textbook or heading in the paragraph.",
        "Please respond only with the paragraph. Thank you!",
    ])
    
    batch = []
    for subheading in subheadings:
        prompt = str(generate_paragraph_prompt_template)
        prompt = prompt.replace("<|textbook_name|>", textbook_name)
        prompt = prompt.replace("<|heading|>", headings[heading_index])
        prompt = prompt.replace("<|subheading|>", subheading)
        messages = [ {"role": "user", "content": prompt } ]
        batch.append(messages)
    
    max_length = 288
    skip_special_tokens = True
    reset_on_low_tok_sec = True
    batch_size = 8
    print(f"   Generating Paragraphs for \"{headings[heading_index]}\" ({heading_index + 1}/{len(headings)}) |", end=" ")
    generated_texts = llm.generate_batch(batch, max_length, skip_special_tokens, reset_on_low_tok_sec, batch_size)
    
    paragraphs = f"## {headings[heading_index]}\n\n"
    
    for i, paragraph in enumerate(generated_texts):
        # paragraph = "\n".join(paragraph.strip().split("\n")[1:]).strip()
        # if paragraph.startswith("Paragraph: "):
        #     paragraph = paragraph[len("Paragraph: "):]
        # if paragraph.startswith("<|paragraph|>"):
        #     paragraph = paragraph[len("<|paragraph|>"):]
        # if paragraph.endswith("<|end|>"):
        #     paragraph = "<|end|>".join(paragraph.split("<|end|>")[:-1])
            
        # paragraph = paragraph.split("<|endoftext|>")[0]
        
        # def filter_fun(e):
        #     if e.startswith("### "):
        #         return False
        #     return True
        
        # paragraph = "\n".join(list(filter(filter_fun, paragraph.split("\n")))).strip()
        
        paragraphs += "### " + subheadings[i] + "\n" + paragraph + "\n\n"

    return paragraphs
