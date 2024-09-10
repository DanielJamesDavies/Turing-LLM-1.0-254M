import torch
from torch.nn import functional as F
from datasets import load_dataset



class TruthfulQA:
    def __init__(self):
        self.dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    
    
    def evaluate(self, model, tokenizer, device):
        num_correct_norm = 0
        num_total = 0  
        
        mc1_num_correct_norm, mc1_num_total = self.eval_mc1(model, tokenizer, device)
        num_correct_norm += mc1_num_correct_norm
        num_total += mc1_num_total
        
        acc_norm = num_correct_norm / num_total
        
        return num_correct_norm, num_total, acc_norm


    def eval_mc1(self, model, tokenizer, device):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(self.dataset["validation"]):
            question_tokens = tokenizer.encode(example["question"])
            answers_tokens = [tokenizer.encode(answer) for answer in example["mc1_targets"]["choices"]]
            tokens = [question_tokens + answer_tokens for answer_tokens in answers_tokens]
            mask = [[0 for _ in question_tokens] + [1 for _ in answer_tokens] for answer_tokens in answers_tokens]
            label = example["mc1_targets"]["labels"].index(1)
            
            avg_losses = [-1 for _ in tokens]
            for i, token_sequence in enumerate(tokens):
                token_sequence = torch.tensor([token_sequence]).to(device)
                curr_mask = torch.tensor([mask[i]]).to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, _ = model(token_sequence)
                avg_losses[i] = self.get_row_avg_loss(token_sequence, curr_mask, logits).item()
            pred_norm = torch.tensor(avg_losses).argmin().item()
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        
        return num_correct_norm, num_total
        
    
    def get_row_avg_loss(self, tokens, mask, logits):
        shift_logits = logits[..., :-1, :]
        shift_tokens = tokens[..., 1:]
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        return avg_loss
