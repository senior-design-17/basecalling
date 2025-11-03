"""
Beam Search Decoding Implementation for CPU Basecalling

This implementation matches Dorado's beam search decoding implementation (beam_search.cpp),
adapted to Python for CPU execution.

The decoder performs beam search over CRF scores to generate DNA sequences with quality scores.
"""

import numpy as np
import torch
from typing import Tuple, List
import math


# Constants
NUM_BASE_BITS = 2
NUM_BASES = 1 << NUM_BASE_BITS  # 4
CRC_SEED = 0x12345678
CRC_POLYNOMIAL = 0x82f63b78  # CRC32C polynomial (reversed)


class BeamElement:
    """Beam element data that needs to be retained for the whole beam."""
    __slots__ = ['state', 'prev_element_index', 'stay']
    
    def __init__(self, state: int, prev_element_index: int, stay: bool):
        self.state = state  # uint16
        self.prev_element_index = prev_element_index  # uint8
        self.stay = stay  # bool


class BeamFrontElement:
    """Beam front element data for a single timestep."""
    __slots__ = ['hash', 'state', 'prev_element_index', 'stay']
    
    def __init__(self, hash_val: int, state: int, prev_element_index: int, stay: bool):
        self.hash = hash_val  # uint32
        self.state = state  # uint16
        self.prev_element_index = prev_element_index  # uint8
        self.stay = stay  # bool


def log_sum_exp(x: float, y: float) -> float:
    """Compute log(exp(x) + exp(y)) in a numerically stable way."""
    abs_diff = abs(x - y)
    if abs_diff < 17.0:
        return max(x, y) + math.log1p(math.exp(-abs_diff))
    else:
        return max(x, y)


def crc32c(crc: int, new_bits: int, num_bits: int = 2) -> int:
    """
    Incorporate new bits into a Castagnoli CRC32 (CRC32C).
    
    Args:
        crc: Current CRC value
        new_bits: New bits to incorporate
        num_bits: Number of bits to process (default 2 for base transitions)
    
    Returns:
        Updated CRC value
    """
    for i in range(num_bits):
        b = (new_bits ^ crc) & 1
        crc >>= 1
        if b:
            crc ^= CRC_POLYNOMIAL
        new_bits >>= 1
    return crc & 0xFFFFFFFF


def get_num_states(num_trans_states: int) -> int:
    """Calculate number of states from number of transition states."""
    if num_trans_states % NUM_BASES != 0:
        raise ValueError("Unexpected number of transition states in beam search decode.")
    return num_trans_states // NUM_BASES


def generate_sequence(
    moves: List[int],
    states: List[int],
    qual_data: np.ndarray,
    q_shift: float,
    q_scale: float
) -> Tuple[str, str]:
    """
    Generate sequence and quality string from moves, states, and quality data.
    
    Args:
        moves: List of move indicators (0=stay, 1=emit base)
        states: List of state values for each timestep
        qual_data: Quality data array of shape [num_blocks, NUM_BASES]
        q_shift: Quality score shift parameter
        q_scale: Quality score scale parameter
    
    Returns:
        Tuple of (sequence, quality_string)
    """
    num_blocks = len(moves)
    seq_len = sum(moves)
    
    sequence = ['N'] * seq_len
    qstring = ['!'] * seq_len
    alphabet = ['A', 'C', 'G', 'T']
    
    base_probs = np.zeros(seq_len, dtype=np.float32)
    total_probs = np.zeros(seq_len, dtype=np.float32)
    
    seq_pos = 0
    for blk in range(num_blocks):
        state = states[blk]
        move = moves[blk]
        base = state & 3
        offset = 0 if blk == 0 else move - 1
        prob_pos = seq_pos + offset
        
        # Get probability for the called base
        base_probs[prob_pos] += qual_data[blk * NUM_BASES + base]
        
        # Accumulate total probability for all bases at this position
        for k in range(NUM_BASES):
            total_probs[prob_pos] += qual_data[blk * NUM_BASES + k]
        
        if blk == 0:
            sequence[seq_pos] = base  # Store as integer index
            seq_pos += 1
        else:
            for j in range(move):
                sequence[seq_pos] = base  # Store as integer index
                seq_pos += 1
    
    # Convert to quality scores and base characters
    for i in range(seq_len):
        sequence[i] = alphabet[int(sequence[i])]  # Convert from int to char
        base_probs[i] = 1.0 - (base_probs[i] / total_probs[i])
        base_probs[i] = -10.0 * math.log10(base_probs[i])
        qscore = base_probs[i] * q_scale + q_shift
        qscore = max(1.0, min(50.0, qscore))
        qstring[i] = chr(int(33.5 + qscore))
    
    return ''.join(sequence), ''.join(qstring)


def beam_search(
    scores: np.ndarray,
    back_guide: np.ndarray,
    posts: np.ndarray,
    num_state_bits: int,
    num_blocks: int,
    max_beam_width: int,
    beam_cut: float,
    fixed_stay_score: float,
    score_scale: float = 1.0,
    posts_scale: float = 1.0
) -> Tuple[List[int], List[int], np.ndarray, float]:
    """
    Perform beam search decoding.
    
    Args:
        scores: Transition scores array of shape [num_blocks, num_transitions]
        back_guide: Backward guide scores of shape [num_blocks + 1, num_states]
        posts: Posterior probabilities of shape [num_blocks + 1, num_states]
        num_state_bits: Number of bits for state representation (log2(num_states))
        num_blocks: Number of time steps (blocks)
        max_beam_width: Maximum beam width
        beam_cut: Beam cutoff parameter (log space threshold)
        fixed_stay_score: Fixed score for stay transitions
        score_scale: Scale factor for scores
        posts_scale: Scale factor for posterior probabilities
    
    Returns:
        Tuple of (states, moves, qual_data, final_score)
    """
    num_states = 1 << num_state_bits
    states_mask = num_states - 1
    
    if max_beam_width > 256:
        raise ValueError("Beam search max_beam_width cannot be greater than 256.")
    
    log_beam_cut = math.log(beam_cut) if beam_cut > 0.0 else float('inf')
    
    # Create beam storage
    beam_vector = [BeamElement(0, 0, False) for _ in range(max_beam_width * (num_blocks + 1))]
    
    # Beam front storage
    max_beam_candidates = (NUM_BASES + 1) * max_beam_width
    current_beam_front = [BeamFrontElement(0, 0, 0, False) for _ in range(max_beam_candidates)]
    prev_beam_front = [BeamFrontElement(0, 0, 0, False) for _ in range(max_beam_candidates)]
    
    current_scores = np.full(max_beam_candidates, -np.inf, dtype=np.float32)
    prev_scores = np.full(max_beam_candidates, -np.inf, dtype=np.float32)
    
    # Initialize beam with top states from backward guide
    beam_init_threshold = float('-inf')
    if max_beam_width < num_states:
        sorted_back_guides = np.sort(back_guide[0])[::-1]  # Descending order
        beam_init_threshold = sorted_back_guides[max_beam_width - 1]
    
    # Initialize the beam
    beam_element = 0
    for state in range(num_states):
        if beam_element >= max_beam_width:
            break
        if back_guide[0, state] >= beam_init_threshold:
            hash_val = crc32c(CRC_SEED, state, 32)
            prev_beam_front[beam_element] = BeamFrontElement(
                hash_val, state, 0, False
            )
            prev_scores[beam_element] = 0.0
            beam_element += 1
    
    # Copy initial beam front into persistent state
    current_beam_width = min(max_beam_width, num_states)
    for element_idx in range(current_beam_width):
        beam_vector[element_idx] = BeamElement(
            prev_beam_front[element_idx].state,
            prev_beam_front[element_idx].prev_element_index,
            prev_beam_front[element_idx].stay
        )
    
    # Iterate through blocks
    for block_idx in range(num_blocks):
        block_scores = scores[block_idx]
        block_back_scores = back_guide[block_idx + 1]
        
        max_score = float('-inf')
        
        # Bloom filter for step hashes
        HASH_PRESENT_BITS = 4096
        HASH_PRESENT_MASK = HASH_PRESENT_BITS - 1
        step_hash_present = np.zeros(HASH_PRESENT_BITS, dtype=bool)
        
        # Generate candidate elements
        new_elem_count = 0
        
        # Expand all possible steps
        for prev_elem_idx in range(current_beam_width):
            previous_element = prev_beam_front[prev_elem_idx]
            
            for new_base in range(NUM_BASES):
                new_state = ((previous_element.state << NUM_BASE_BITS) & states_mask) | new_base
                
                # Calculate move index
                move_idx = (new_state << NUM_BASE_BITS) + \
                          ((previous_element.state << NUM_BASE_BITS) >> num_state_bits)
                
                new_score = (prev_scores[prev_elem_idx] +
                           block_scores[move_idx] * score_scale +
                           block_back_scores[new_state])
                
                new_hash = crc32c(previous_element.hash, new_base, NUM_BASE_BITS)
                
                step_hash_present[new_hash & HASH_PRESENT_MASK] = True
                
                current_beam_front[new_elem_count] = BeamFrontElement(
                    new_hash, new_state, prev_elem_idx, False
                )
                current_scores[new_elem_count] = new_score
                max_score = max(max_score, new_score)
                new_elem_count += 1
        
        # Add stay operations
        for prev_elem_idx in range(current_beam_width):
            previous_element = prev_beam_front[prev_elem_idx]
            stay_score = (prev_scores[prev_elem_idx] +
                         fixed_stay_score +
                         block_back_scores[previous_element.state])
            
            current_beam_front[new_elem_count] = BeamFrontElement(
                previous_element.hash,
                previous_element.state,
                prev_elem_idx,
                True
            )
            current_scores[new_elem_count] = stay_score
            max_score = max(max_score, stay_score)
            
            # Check for stay/step merging
            if step_hash_present[previous_element.hash & HASH_PRESENT_MASK]:
                stay_elem_idx = current_beam_width * NUM_BASES + prev_elem_idx
                stay_latest_base = previous_element.state & 3
                
                # Find matching step extensions
                for prev_elem_comp_idx in range(current_beam_width):
                    step_elem_idx = (prev_elem_comp_idx << NUM_BASE_BITS) | stay_latest_base
                    if current_beam_front[stay_elem_idx].hash == current_beam_front[step_elem_idx].hash:
                        if current_scores[stay_elem_idx] > current_scores[step_elem_idx]:
                            # Fold step into stay
                            folded_score = log_sum_exp(
                                current_scores[stay_elem_idx],
                                current_scores[step_elem_idx]
                            )
                            current_scores[stay_elem_idx] = folded_score
                            max_score = max(max_score, folded_score)
                            current_scores[step_elem_idx] = float('-inf')
                        else:
                            # Fold stay into step
                            folded_score = log_sum_exp(
                                current_scores[stay_elem_idx],
                                current_scores[step_elem_idx]
                            )
                            current_scores[step_elem_idx] = folded_score
                            max_score = max(max_score, folded_score)
                            current_scores[stay_elem_idx] = float('-inf')
            
            new_elem_count += 1
        
        # Beam pruning
        beam_cutoff_score = max_score - log_beam_cut
        
        # Count elements meeting cutoff
        def get_elem_count():
            return np.sum(current_scores[:new_elem_count] >= beam_cutoff_score)
        
        elem_count = get_elem_count()
        
        # Binary search for optimal beam cutoff if needed
        if elem_count > max_beam_width:
            min_beam_width = (max_beam_width * 8) // 10  # 80%
            low_score = beam_cutoff_score
            hi_score = max_score
            num_guesses = 1
            MAX_GUESSES = 10
            
            while (elem_count > max_beam_width or elem_count < min_beam_width) and num_guesses < MAX_GUESSES:
                if elem_count > max_beam_width:
                    low_score = beam_cutoff_score
                    beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0
                else:
                    hi_score = beam_cutoff_score
                    beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0
                
                elem_count = get_elem_count()
                num_guesses += 1
            
            if num_guesses == MAX_GUESSES:
                beam_cutoff_score = hi_score
                elem_count = get_elem_count()
            
            elem_count = min(elem_count, max_beam_width)
        
        # Sort candidates by score (descending) to select top elements
        # Create indices array for sorting
        candidate_indices = np.arange(new_elem_count)
        # Sort by score in descending order
        sorted_indices = candidate_indices[np.argsort(-current_scores[:new_elem_count])]
        
        # Select top elements meeting cutoff
        write_idx = 0
        for read_idx in sorted_indices:
            if current_scores[read_idx] >= beam_cutoff_score:
                if write_idx < max_beam_width:
                    prev_beam_front[write_idx] = current_beam_front[read_idx]
                    prev_scores[write_idx] = current_scores[read_idx]
                    write_idx += 1
                else:
                    break
        
        # At last timestep, ensure best path is element 0
        if block_idx == num_blocks - 1:
            best_score = float('-inf')
            best_score_index = 0
            for i in range(elem_count):
                if prev_scores[i] > best_score:
                    best_score = prev_scores[i]
                    best_score_index = i
            # Swap
            prev_beam_front[0], prev_beam_front[best_score_index] = \
                prev_beam_front[best_score_index], prev_beam_front[0]
            prev_scores[0], prev_scores[best_score_index] = \
                prev_scores[best_score_index], prev_scores[0]
        
        # Update beam persistent state
        beam_offset = (block_idx + 1) * max_beam_width
        for i in range(elem_count):
            # Remove backwards contribution from score
            prev_scores[i] -= block_back_scores[prev_beam_front[i].state]
            
            # Copy to persistent state
            beam_vector[beam_offset + i] = BeamElement(
                prev_beam_front[i].state,
                prev_beam_front[i].prev_element_index,
                prev_beam_front[i].stay
            )
        
        current_beam_width = elem_count
    
    # Extract final score
    final_score = prev_scores[0]
    
    # Reconstruct path
    moves = [0] * num_blocks
    states_list = [0] * num_blocks
    
    element_index = 0
    for beam_idx in range(num_blocks, 0, -1):
        beam_addr = beam_idx * max_beam_width + element_index
        states_list[beam_idx - 1] = int(beam_vector[beam_addr].state)
        moves[beam_idx - 1] = 0 if beam_vector[beam_addr].stay else 1
        element_index = beam_vector[beam_addr].prev_element_index
    
    moves[0] = 1  # Always step in first event
    
    # Compute quality data
    num_states_calc = get_num_states(scores.shape[1])
    qual_data = np.zeros((num_blocks, NUM_BASES), dtype=np.float32)
    
    for block_idx in range(num_blocks):
        state = states_list[block_idx]
        states_list[block_idx] = state % NUM_BASES
        base_to_emit = states_list[block_idx]
        
        # Get posterior probability for this state
        timestep_posts = posts[block_idx + 1]
        block_prob = timestep_posts[state] * posts_scale
        
        # Get indices of left- and right-shifted kmers
        l_shift_idx = state >> NUM_BASE_BITS
        r_shift_idx = (state << NUM_BASE_BITS) % num_states_calc
        msb = num_states_calc >> NUM_BASE_BITS
        
        shifted_states = []
        for shift_base in range(NUM_BASES):
            l_shift_state = l_shift_idx + msb * shift_base
            r_shift_state = r_shift_idx + shift_base
            shifted_states.append(l_shift_state)
            shifted_states.append(r_shift_state)
        
        # Add probabilities for unique states
        for state_idx, candidate_state in enumerate(shifted_states):
            count_state = (candidate_state != state)
            if count_state:
                for inner_state in range(state_idx):
                    if shifted_states[inner_state] == candidate_state:
                        count_state = False
                        break
            if count_state:
                block_prob += timestep_posts[candidate_state] * posts_scale
        
        block_prob = max(0.0, min(1.0, block_prob))
        block_prob = block_prob ** 0.4  # Power scaling factor
        
        # Calculate quality scores
        wrong_base_prob = (1.0 - block_prob) / 3.0
        
        for base in range(NUM_BASES):
            qual_data[block_idx, base] = block_prob if base == base_to_emit else wrong_base_prob
    
    return states_list, moves, qual_data.flatten(), final_score


def beam_search_decode(
    scores_t: torch.Tensor,
    back_guides_t: torch.Tensor,
    posts_t: torch.Tensor,
    max_beam_width: int = 32,
    beam_cut: float = 100.0,
    fixed_stay_score: float = 2.0,
    q_shift: float = 0.0,
    q_scale: float = 1.0,
    byte_score_scale: float = 1.0
) -> Tuple[str, str, List[int]]:
    """
    Main beam search decode function.
    
    Args:
        scores_t: Transition scores tensor of shape [num_blocks, num_transitions]
        back_guides_t: Backward guide scores tensor of shape [num_blocks + 1, num_states]
        posts_t: Posterior probabilities tensor of shape [num_blocks + 1, num_states]
        max_beam_width: Maximum beam width (default: 32)
        beam_cut: Beam cutoff parameter (default: 100.0)
        fixed_stay_score: Fixed stay score (default: 2.0)
        q_shift: Quality score shift (default: 0.0)
        q_scale: Quality score scale (default: 1.0)
        byte_score_scale: Scale factor for byte scores (default: 1.0)
    
    Returns:
        Tuple of (sequence, quality_string, moves)
    """
    num_blocks = scores_t.shape[0]
    num_trans_states = scores_t.shape[1]
    num_states = get_num_states(num_trans_states)
    num_state_bits = int(np.log2(num_states))
    
    if (1 << num_state_bits) != num_states:
        raise ValueError("num_states must be an integral power of 2")
    
    # Convert tensors to numpy
    if scores_t.dtype == torch.float32:
        scores_np = scores_t.cpu().numpy()
        posts_np = posts_t.cpu().numpy()
        score_scale = 1.0
        posts_scale = 1.0
    elif scores_t.dtype == torch.int8:
        scores_np = scores_t.cpu().numpy().astype(np.float32)
        posts_np = posts_t.cpu().numpy().astype(np.float32) / 32767.0
        score_scale = byte_score_scale
        posts_scale = 1.0 / 32767.0
    elif scores_t.dtype == torch.float16:
        scores_np = scores_t.cpu().float().numpy()
        posts_np = posts_t.cpu().numpy()
        score_scale = 1.0
        posts_scale = 1.0
    else:
        raise ValueError(f"Unsupported tensor dtype: {scores_t.dtype}")
    
    back_guides_np = back_guides_t.cpu().numpy()
    if back_guides_t.dtype != torch.float32:
        back_guides_np = back_guides_t.cpu().float().numpy()
    
    states, moves, qual_data, _ = beam_search(
        scores_np,
        back_guides_np,
        posts_np,
        num_state_bits,
        num_blocks,
        max_beam_width,
        beam_cut,
        fixed_stay_score,
        score_scale,
        posts_scale
    )
    
    sequence, qstring = generate_sequence(moves, states, qual_data, q_shift, q_scale)
    
    return sequence, qstring, moves

