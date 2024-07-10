import torch
import numpy as np
import abc

def tokenize(tokenizer, prompts):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f for f in features if f is not None and isinstance(
            f, torch.Tensor)]  # .float() .detach()
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f for k, f in features.items()}  # .float()
        setattr(module, name, features)
    else:
        setattr(module, name, features)  # .float()


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


def build_normal(u_x, u_y, d_x, d_y, step, device):
    x, y = torch.meshgrid(torch.linspace(0,1,step), torch.linspace(0,1,step))
    x = x.to(device)
    y = y.to(device)
    out_prob = (1/2/torch.pi/d_x/d_y)*torch.exp(-1/2*(torch.square((x-u_x)/d_x)+torch.square((y-u_y)/d_y)))
    return out_prob

def uniq_masks(all_masks, zero_masks=None, scale=1.0):
    uniq_masks = torch.stack(all_masks)
    # num = all_masks.shape[0]
    uniq_mask = torch.argmax(uniq_masks, dim=0)
    if zero_masks is None:
        all_masks = [((uniq_mask==i)*mask*scale).float().clamp(0, 1.0) for i, mask in enumerate(all_masks)]
    else:
        all_masks = [((uniq_mask==i)*mask*scale).float().clamp(0, 1.0) for i, mask in enumerate(zero_masks)]

    return all_masks
def build_masks(bboxes, size, mask_mode="gaussin_zero_one", focus_rate=1.0):
    all_masks = []
    zero_masks = []
    for bbox in bboxes:
        x0,y0,x1,y1 = bbox
        mask = build_normal((y0+y1)/2, (x0+x1)/2, (y1-y0)/4, (x1-x0)/4, size, bbox.device)
        zero_mask = torch.zeros_like(mask)
        zero_mask[int(y0 * size):min(int(y1 * size)+1, size), int(x0 * size):min(int(x1 * size)+1, size)] = 1.0
        zero_masks.append(zero_mask)
        all_masks.append(mask)
    if mask_mode == 'zero_one':
        return zero_masks
    elif mask_mode == 'guassin':
        all_masks = uniq_masks(all_masks, scale=focus_rate)
        return all_masks
    elif mask_mode == 'gaussin_zero_one':
        all_masks = uniq_masks(all_masks, zero_masks, scale=focus_rate)
        return all_masks
    else:
        raise ValueError("Not supported mask_mode.")
    
class BboxCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet, bboxes, entity_indexes, mask_control, mask_self, with_uncond, mask_mode, soft_mask_rate, focus_rate):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.bboxes = bboxes
        self.entity_indexes = entity_indexes
        self.mask_control = mask_control
        self.mask_self = mask_self
        self.with_uncond = with_uncond
        self.mask_mode = mask_mode
        self.soft_mask_rate = soft_mask_rate
        self.focus_rate = focus_rate

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.with_uncond:
            cond_attention_probs = attention_probs[batch_size//2:]
        else:
            cond_attention_probs = attention_probs

        if self.mask_control:
            
            if is_cross:
                size = int(np.sqrt(sequence_length))
                all_masks = build_masks(self.bboxes, size, mask_mode=self.mask_mode, focus_rate=self.focus_rate)
                for pos, mask in zip(self.entity_indexes, all_masks):
                    start = pos[0]
                    end = pos[-1]
                    if mask.sum() <= 0:  # sequence_length *  0.004:
                        continue
                    mask = mask.reshape((sequence_length, -1)).to(hidden_states.device)
                    mask = mask.expand(-1, (end-start+1))
                    cond_attention_probs[:, :, start+1:end+2] = cond_attention_probs[:, :, start+1:end+2] * mask
            elif self.mask_self:
                size = int(np.sqrt(sequence_length))
                # must be 1/0
                all_masks = build_masks(self.bboxes, size, mask_mode=self.mask_mode, focus_rate=self.focus_rate)
                for img_mask in all_masks:
                    if img_mask.sum() <= 0:  # sequence_length *  0.004:
                        continue
                    img_mask = img_mask.reshape(sequence_length)
                    mask_index = img_mask.nonzero().squeeze(-1)
                    mask = torch.ones(sequence_length, sequence_length).to(hidden_states.device)

                    mask[:, mask_index] = mask[:, mask_index] * img_mask.unsqueeze(-1)
                    cond_attention_probs = cond_attention_probs * mask + cond_attention_probs * (1-mask) * self.soft_mask_rate
            if self.with_uncond:
                attention_probs[batch_size//2:] = cond_attention_probs
            else:
                attention_probs = cond_attention_probs

        self.attnstore(cond_attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
def register_attention_control_bbox(model, controller, bboxes, entity_indexes, mask_control=False, mask_self=False, 
                                    with_uncond=False, mask_mode='gaussin_zero_one', soft_mask_rate=0.2, focus_rate=1.0):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = BboxCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet, bboxes=bboxes, entity_indexes=entity_indexes, 
            mask_control=mask_control, mask_self=mask_self, with_uncond=with_uncond, mask_mode=mask_mode,
            soft_mask_rate=soft_mask_rate, focus_rate=focus_rate
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0