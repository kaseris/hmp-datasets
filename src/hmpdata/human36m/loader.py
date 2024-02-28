import numpy as np

def prepare_next_batch_impl(batch_size, dataset, prefix_length, target_length, sequences):
        
        buffer_quat = np.zeros((batch_size, prefix_length+target_length, 32*4), dtype='float32')
        buffer_euler = np.zeros((batch_size, target_length, 32*3), dtype='float32')
        
        sequences = np.random.permutation(sequences)

        batch_idx = 0
        for i, (subject, action) in enumerate(sequences):
            # Pick a random chunk from each sequence
            start_idx = np.random.randint(0, dataset[subject][action]['rotations'].shape[0] - prefix_length - target_length + 1)
            mid_idx = start_idx + prefix_length
            end_idx = start_idx + prefix_length + target_length
            
            buffer_quat[batch_idx] = dataset[subject][action]['rotations'][start_idx:end_idx].reshape( \
                                          prefix_length+target_length, -1)
            buffer_euler[batch_idx] = dataset[subject][action]['rotations_euler'][mid_idx:end_idx].reshape( \
                                          target_length, -1)
            
            batch_idx += 1
            if batch_idx == batch_size or i == len(sequences) - 1:
                yield buffer_quat[:batch_idx], buffer_euler[:batch_idx]
                batch_idx = 0