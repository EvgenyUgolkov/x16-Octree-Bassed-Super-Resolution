#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import MinkowskiEngine as ME
import torch.nn.functional as F

##########################################################################################################################################################
# Was not able to find the fast transfer for the ME Sparse tensors between the different devices. This function will do it instead
##########################################################################################################################################################

def next_device(sparse_tensor, new_device):

    new_sparse_tensor = ME.SparseTensor(features = sparse_tensor.F, 
                                        coordinates = sparse_tensor.C, 
                                        tensor_stride = sparse_tensor.tensor_stride, 
                                        device = new_device)
    
    return new_sparse_tensor
    
##########################################################################################################################################################
# This one is to take the ME sparse tensor, and transfer it back to the dense torch tensor of specific size. Was made for the G_0 to create a dense tensor
# to slice it and then comparing with processed high-res images. All of it is made to generate "mixed" in a more stable way
##########################################################################################################################################################

def back_to_dense_0(sparse_tensor, factor, dense_size):

    # Create the dense tensor
    dense_tensor = torch.zeros(dense_size).cuda()
    
    # Take the elements of the sparse tensor
    coordinates = sparse_tensor.C
    features = sparse_tensor.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)
    
    # Fill this dense tensor with your values
    dense_tensor = update_dense_tensor(dense_tensor, coordinates, features, flag=True)

    return dense_tensor

##########################################################################################################################################################
# This one was created for the G_1. This time we have a previous memorized, which we need to upsample x2, and current sftmx, which we need to reduce in 
# coordinates. Then, all of it should be set into the single style of coordinates
##########################################################################################################################################################

def back_to_dense_1(sftmx, memorized_0, factor, step, dense_size):

    # Create the empty dense tensor to fill it in the next steps
    dense_tensor = torch.zeros(dense_size).cuda()

                                                            # TREATING THE MEMORIZED TENSOR   
    # Take the elements of the sparse tensor
    coordinates = memorized_0.C
    features = memorized_0.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times and fill the dense (we have a filling function inside)
    dense_tensor = expand_coordinates_and_properties(coordinates, features, step, dense_tensor)

                                                             # TREATING THE CURRENT TENSOR
    # Take the elements of the current tensor
    coordinates = sftmx.C
    features = sftmx.F

    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor

    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Fill the dense
    dense_tensor = update_dense_tensor(dense_tensor, coordinates, features, flag=True)

    return dense_tensor

##########################################################################################################################################################
# This one was created for the G_2
##########################################################################################################################################################

def back_to_dense_2(sftmx, memorized_1, memorized_0, factor, step, dense_size):

    # Create the empty dense tensor to fill it in the next steps
    dense_tensor = torch.zeros(dense_size).cuda()

                                                            # TREATING THE NEW MEMORIZED TENSOR   
    # Take the elements of the sparse tensor
    coordinates = memorized_1.C
    features = memorized_1.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = expand_coordinates_and_properties(coordinates, features, step, dense_tensor)
    
                                                                # TREATING THE OLD MEMORIZED TENSOR   
    # Take the elements of the sparse tensor
    coordinates = memorized_0.C
    features = memorized_0.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = expand_coordinates_and_properties(coordinates, features, step*2, dense_tensor)

                                                             # TREATING THE CURRENT TENSOR
    # Take the elements of the current tensor
    coordinates = sftmx.C
    features = sftmx.F

    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor

    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = update_dense_tensor(dense_tensor, coordinates, features, flag=True)

    return dense_tensor

##########################################################################################################################################################
# This one was created for the G_3
##########################################################################################################################################################

def back_to_dense_3(sftmx, memorized_2, memorized_1, memorized_0, factor, step, dense_size):

    # Create the empty dense tensor to fill it in the next steps
    dense_tensor = torch.zeros(dense_size).cuda()

                                                            # TREATING THE NEW MEMORIZED TENSOR   
    # Take the elements of the sparse tensor
    coordinates = memorized_2.C
    features = memorized_2.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = expand_coordinates_and_properties(coordinates, features, step, dense_tensor)
    
                                                                # TREATING THE previous MEMORIZED TENSOR   
    # Take the elements of the sparse tensor
    coordinates = memorized_1.C
    features = memorized_1.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = expand_coordinates_and_properties(coordinates, features, step*2, dense_tensor)

                                                                 # TREATING THE first MEMORIZED TENSOR   
    # Take the elements of the sparse tensor
    coordinates = memorized_0.C
    features = memorized_0.F
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = expand_coordinates_and_properties(coordinates, features, step*4, dense_tensor)

                                                             # TREATING THE CURRENT TENSOR
    # Take the elements of the current tensor
    coordinates = sftmx.C
    features = sftmx.F

    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor

    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    # Upsample all the features 2 times
    dense_tensor = update_dense_tensor(dense_tensor, coordinates, features, flag=True)

    return dense_tensor

##########################################################################################################################################################
# This one was created for the G_4
##########################################################################################################################################################

def back_to_dense_4(sftmx, memorized_3, memorized_2, memorized_1, memorized_0, step, dense_size):

    # Create the empty dense tensor to fill it in the next steps
    dense_tensor = torch.zeros(dense_size).cuda()

                                                                # TREATING THE NEW MEMORIZED TENSOR   
    # Upsample all the features 2 times
    dense_tensor = expand_coordinates_and_properties(memorized_3.C, memorized_3.F, step, dense_tensor)
    
                                                                # TREATING THE MID MEMORIZED TENSOR   
    # Upsample all the features 4 times
    dense_tensor = expand_coordinates_and_properties(memorized_2.C, memorized_2.F, step*2, dense_tensor)

                                                                # TREATING THE OLD MEMORIZED TENSOR   
    # Upsample all the features 8 times
    dense_tensor = expand_coordinates_and_properties(memorized_1.C, memorized_1.F, step*4, dense_tensor)

                                                                # TREATING THE OLD MEMORIZED TENSOR   
    # Upsample all the features 16 times
    dense_tensor = expand_coordinates_and_properties(memorized_0.C, memorized_0.F, step*8, dense_tensor)

                                                                 # TREATING THE CURRENT TENSOR
    # Upsample all the features 2 times
    dense_tensor = update_dense_tensor(dense_tensor, sftmx.C, sftmx.F, flag=True)

    return dense_tensor

##########################################################################################################################################################
# This one is for the case when we take the input -> process through the standard Pytorch convolutions -> and now need to transfer it to the ME Sparse Tensor before going to the ME layers. Thus, we create the coordinate grid, and repeat it bs times. We don't repeat the features because they are the tensors by itself: we need just to reshape them. Then, we create two lists: just coordinates (no batch number in the beginning) and corresponding features. We need it because this way ME creates the batched coordinates. Next, create the Sparse Tensor as an output. Notice, that here we "* factor" the coordinates, because in ME they have to be all the time in the final resolution numbers. For different networks on different layers it will be different - check in the "model" codes. The more we apply the upsample, the smaller this factor will be. The more we apply downsample, the larger this factor will be. Again: it is because the ME sparse tensors format - all the time to have the coordinates of the final resolution, but different stride
##########################################################################################################################################################

def separate_coordinates_and_features_dense(tensor, factor):
    
    bs, C, X, Y, Z = tensor.shape
    
    # Create meshgrid of coordinates
    coords_x = torch.arange(X, device=tensor.device)
    coords_y = torch.arange(Y, device=tensor.device)
    coords_z = torch.arange(Z, device=tensor.device)
    
#    grid_x, grid_y, grid_z = torch.meshgrid(coords_x, coords_y, coords_z, indexing='ij')
    grid_x, grid_y, grid_z = torch.meshgrid(coords_x, coords_y, coords_z)
    all_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # Shape (X, Y, Z, 3)
    
    # Reshape the coordinate tensor to (bs, M, 3)
    all_coords = all_coords.reshape(-1, 3).repeat(bs, 1, 1)  # Shape (bs, X*Y*Z, 3)
    
    # Reshape and permute the features tensor to (bs, M, C)
    feats_all = tensor.permute(0, 2, 3, 4, 1).reshape(bs, X*Y*Z, C)  # Shape (bs, X*Y*Z, C)

    all_coords = [all_coords[i] * factor for i in range(len(all_coords))]
    feats_all =  [feats_all[i] for i in range(len(feats_all))]

    all_coords, feats_all = ME.utils.sparse_collate(all_coords, feats_all)

    sparse_tensor = ME.SparseTensor(features=feats_all, coordinates=all_coords, device=tensor.device, tensor_stride=factor)  
    
    return sparse_tensor

##########################################################################################################################################################
# This one is how to add features from the sparse tensor to the dense tensor, if they both have the step 1 btw the coordinates. This way, we use it either after the final upconvolutional layer, either after artificial reorganization "expand_coordinates_and_properties" below
##########################################################################################################################################################
    
def update_dense_tensor(dense_tensor, coordinates, features, flag=False):
    # Assumes dense_tensor is initialized to the correct shape and dtype
    # Coordinates are assumed to be in the format [batch_index, x, y, z]
    # Features are assumed to be in the format [n, ch]

    # dense_tensor = dense_tensor.clone()
    
    # Extract indices for batch, x, y, z from coordinates
    b = coordinates[:, 0].long()  # Convert to long for indexing
    x = coordinates[:, 1].long()
    y = coordinates[:, 2].long()
    z = coordinates[:, 3].long()
    
    # # Use advanced indexing to update the dense tensor
    # for i in range(features.shape[1]):
    #     dense_tensor[b, i, x, y, z] = features[:, i]

    dense_tensor[b, :, x, y, z] = features                       ########### here i removed the cycle to make it faster. Let's chack the memory ###########

    # # This one is to check if we don't have any unfilled values in the dense tensor
    # if (dense_tensor == 0).all(dim=1).any() and flag:
    #     print("ALARM!")

    return dense_tensor

##########################################################################################################################################################

def update_dense_tensor_from_sparse(coordinates, features, dense_size):
    # Assumes dense_tensor is initialized to the correct shape and dtype
    # Coordinates are assumed to be in the format [batch_index, x, y, z]
    # Features are assumed to be in the format [n, ch]

    dense_tensor = torch.zeros(dense_size).cuda()
    
    # Extract indices for batch, x, y, z from coordinates
    b = coordinates[:, 0].long()  # Convert to long for indexing
    x = coordinates[:, 1].long()
    y = coordinates[:, 2].long()
    z = coordinates[:, 3].long()
    
    # Use advanced indexing to update the dense tensor
    for i in range(features.shape[1]):
        dense_tensor[b, i, x, y, z] = features[:, i]

    # This one is to check if we don't have any unfilled values in the dense tensor
    if (dense_tensor == 0).all(dim=1).any():
        print("ALARM!")
        
    return dense_tensor
    
##########################################################################################################################################################
# For the tensors in his format. Here we take the initial coordinates, and add a special grid to them, depends on the level of the tensor. It happens because when we use this format, the nodes are gradually separated by some specific step, so we fill this space with our new grid 

# step_size here is the step between the nodes in this new format
##########################################################################################################################################################

def expand_coordinates_and_properties(coordinates, properties, step_size, dense_tensor): 
    """
    Expands coordinates and properties for a Minkowski Engine sparse tensor.

    Args:
    coordinates: Tensor of shape [num_points, 4] with format [batch_index, x, y, z].
    properties: Tensor of shape [num_points, feature_dim] aligned with coordinates.
    step_size: Integer representing the step size for x, y, and z dimensions.

    Returns:
    new_coords: Tensor with expanded coordinates.
    new_props: Tensor with properties duplicated for the new coordinates.
    """
    
    # Generate offsets within the specified step range for each dimension........
    range_d = torch.arange(0, step_size, device=coordinates.device)
    # grid_x, grid_y, grid_z = torch.meshgrid(range_d, range_d, range_d, indexing='ij')
    grid_x, grid_y, grid_z = torch.meshgrid(range_d, range_d, range_d)
    offsets = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    # Separate the batch indices and spatial coordinates
    batch_indices = coordinates[:, 0].unsqueeze(1)
    spatial_coords = coordinates[:, 1:]

    # Expand the input spatial coordinates to match the number of offsets
    spatial_coords = (spatial_coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    
    # Repeat the batch indices to match the number of generated coordinates
    batch_indices = (batch_indices.repeat(1, offsets.size(0))).reshape(-1, 1)
    
    # Combine the expanded batch indices with the expanded spatial coordinates
    spatial_coords = torch.cat([batch_indices, spatial_coords], dim=1).to(torch.int)
    
    # Repeat properties to match the number of generated coordinates
    properties = properties.repeat_interleave(offsets.size(0), dim=0)

    final_tensor = update_dense_tensor(dense_tensor, spatial_coords, properties)
    
    return final_tensor

##########################################################################################################################################################
# This one below is the copy for the one above, but in the end instead of filling the dense tensor we create a new sparse tensor with expanded coordinates
# and features. Needed not to proceed the dense tensor for memorized for all the GPUs in Pipeline Parallel
##########################################################################################################################################################

def expand_coordinates_and_properties_sparse(coordinates, properties, step_size): 
    """
    Expands coordinates and properties for a Minkowski Engine sparse tensor.

    Args:
    coordinates: Tensor of shape [num_points, 4] with format [batch_index, x, y, z].
    properties: Tensor of shape [num_points, feature_dim] aligned with coordinates.
    step_size: Integer representing the step size for x, y, and z dimensions.

    Returns:
    new_coords: Tensor with expanded coordinates.
    new_props: Tensor with properties duplicated for the new coordinates.
    """
    
    # Generate offsets within the specified step range for each dimension........
    range_d = torch.arange(0, step_size, device=coordinates.device)
    # grid_x, grid_y, grid_z = torch.meshgrid(range_d, range_d, range_d, indexing='ij')
    grid_x, grid_y, grid_z = torch.meshgrid(range_d, range_d, range_d)
    offsets = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    # Separate the batch indices and spatial coordinates
    batch_indices = coordinates[:, 0].unsqueeze(1)
    spatial_coords = coordinates[:, 1:]

    # Expand the input spatial coordinates to match the number of offsets
    spatial_coords = (spatial_coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    
    # Repeat the batch indices to match the number of generated coordinates
    batch_indices = (batch_indices.repeat(1, offsets.size(0))).reshape(-1, 1)
    
    # Combine the expanded batch indices with the expanded spatial coordinates
    spatial_coords = torch.cat([batch_indices, spatial_coords], dim=1).to(torch.int)
    
    # Repeat properties to match the number of generated coordinates
    properties = properties.repeat_interleave(offsets.size(0), dim=0)

    final_tensor = ME.SparseTensor(features=properties, coordinates=spatial_coords, tensor_stride=1)
    
    return final_tensor

##########################################################################################################################################################

def compute_phase_consistency_loss(images, coordinates):
    """
    Compute the phase consistency loss for the given 3D images and pre-defined coordinates.
    
    Args:
    - images (torch.Tensor): Tensor of shape (bs, ch, x, y, z) representing the 3D images.
    - coordinates (torch.Tensor): Tensor of shape (num_coords, 4) representing the pre-defined coordinates.
    
    Returns:
    - loss (torch.Tensor): The computed loss value.
    """

    # Get the phases (channel with the maximum value)
    phases = torch.argmax(images, dim=1)  # Shape: (bs, x, y, z)
    
    # Define neighbor offsets (26 neighbors in 3D)
    neighbor_offsets = torch.tensor([
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], 
        [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], 
        [1, 1, -1], [1, 1, 0], [1, 1, 1]
    ], device=images.device, dtype=torch.long)
    
    # Extract coordinates for easy access
    batch_indices, x_coords, y_coords, z_coords = coordinates[:, 0].long(), coordinates[:, 1].long(), coordinates[:, 2].long(), coordinates[:, 3].long()
    
    # Gather phases of the main voxels
    main_phases = phases[batch_indices, x_coords, y_coords, z_coords]
    
    # Initialize the loss
    loss = torch.tensor(0.0, device=images.device)
    
    # Iterate over each offset to handle neighbors
    for offset in neighbor_offsets:
        # Compute neighbor coordinates
        neighbor_x = x_coords + offset[0]
        neighbor_y = y_coords + offset[1]
        neighbor_z = z_coords + offset[2]
        
        # Check for valid neighbors
        valid_mask = (neighbor_x >= 0) & (neighbor_x < images.shape[2]) & \
                     (neighbor_y >= 0) & (neighbor_y < images.shape[3]) & \
                     (neighbor_z >= 0) & (neighbor_z < images.shape[4])
        
        # This one is just checking if all of them are out of boundaries, then just skip this step
        if valid_mask.sum() == 0:
            continue
        
        # Gather phases of the neighbors
        neighbor_phases = phases[batch_indices[valid_mask], neighbor_x[valid_mask], neighbor_y[valid_mask], neighbor_z[valid_mask]]
        
        # Compare phases with the main phases and phase 5
        main_phases_valid = main_phases[valid_mask]
        phase_mismatch = (neighbor_phases != main_phases_valid) & (neighbor_phases != 5)
        
        # Accumulate the loss
        loss = loss + phase_mismatch.float().sum()
    
    # # Normalize the loss by the number of coordinates
    # loss = loss / coordinates.shape[0]
    
    return loss

##########################################################################################################################################################

def correct_coordinates(coordinates, factor):
    
    # Separate the batch indices and spatial coordinates, and divide the coordinates by the factor because in ME format we have them in the final resolution
    batch_indices = coordinates[:, 0].unsqueeze(1)
    coordinates = coordinates[:, 1:] / factor
    
    # Combine the batches and divided coordinates back into the single structure 
    coordinates = torch.cat([batch_indices, coordinates], dim=1)

    return coordinates
    
##########################################################################################################################################################
# This part below is for the case where we would like to propagate not only the mixed, but also their neighbors for more efficient convolution
##########################################################################################################################################################

def find_max_last_channel(features):
    # Check if the last channel is the maximum channel in each feature set
    max_mask = torch.argmax(features, dim=1) == features.size(1) - 1
    return max_mask

###################################################################################

def find_neighbors(coords, bounds, factor):
    # Define the neighbor offsets in 3D space, excluding the (0,0,0) combination
    neighbor_offsets = torch.tensor([[dx*factor, dy*factor, dz*factor] for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]])

    # Extend the neighbor offsets to include a zero column for the batch number, which does not change
    neighbor_offsets = torch.cat([torch.zeros((27, 1), dtype=torch.int), neighbor_offsets], dim=1).cuda()

    # Calculate potential neighbors for each coordinate, considering the batch number
    potential_neighbors = coords[:, None, :] + neighbor_offsets[None, :, :]
    
    # Clamping spatial coordinates to stay within the bounds without using a loop
    # We apply clamping only to spatial dimensions
    spatial_clamp_min = torch.tensor([0, 0, 0], dtype=torch.int).cuda()
    spatial_clamp_max = (torch.tensor(bounds[1:], dtype=torch.int) - 1).cuda()
    potential_neighbors[..., 1:4] = torch.max(torch.min(potential_neighbors[..., 1:4], spatial_clamp_max), spatial_clamp_min)

    # Remove duplicates and return unique neighbors
    unique_neighbors = torch.unique(potential_neighbors.reshape(-1, 4), dim=0)
    return unique_neighbors

###################################################################################

def create_mask(all_coords, relevant_coords):
    # Combine the batch size and coordinates into a single unique identifier
    all_coords_ids = (all_coords * torch.tensor([1, 2, 3, 4], device=all_coords.device)).sum(dim=1)
    relevant_coords_ids = (relevant_coords * torch.tensor([1, 2, 3, 4], device=all_coords.device)).sum(dim=1)

    # Convert relevant_coords_ids to a set for fast membership checking
    relevant_set = set(relevant_coords_ids.cpu().numpy())

    # Create a mask based on whether each element in all_coords_ids is in relevant_set
    mask = torch.tensor([id in relevant_set for id in all_coords_ids.cpu().numpy()], device=all_coords.device)

    return mask

###################################################################################

def next_level_octree(coords):
    # Offsets for generating the eight children of each octree node in the spatial dimensions
    offsets = torch.tensor([[dx, dy, dz] for dx in [0, 1] for dy in [0, 1] for dz in [0, 1]]).to(coords.device)

    # Prepare the offsets to work with the coordinates
    expanded_offsets = offsets[None, :, :]  # Expand offsets for broadcasting

    # Expand coords to apply offsets
    expanded_coords = coords[:, None, 1:4] * 2 + expanded_offsets  # Apply offsets to spatial dimensions

    # Preserve batch indices
    batch_indices = coords[:, 0].view(-1, 1, 1).expand(-1, 8, 1)  # Expand batch indices for each of the 8 new coords

    # Concatenate batch indices with calculated spatial coordinates
    next_level_coords = torch.cat((batch_indices, expanded_coords), dim=2)

    # Reshape to have a flat list of coordinates
    next_level_coords = next_level_coords.reshape(-1, 4)

    return next_level_coords

###################################################################################

def masks_here_neighs_next(sparse_tensor_cls, sparse_tensor_conv, bounds, factor):
    
    coordinates_cls = sparse_tensor_cls.C
    features_cls = sparse_tensor_cls.F
    
    mask_mixed_current_level = find_max_last_channel(features_cls)
    coords_mixed_current_level = coordinates_cls[mask_mixed_current_level]

    mixed_and_neighbors_current_level = find_neighbors(coords_mixed_current_level, bounds, factor)
    
    coords_mixed_next_level = next_level_octree(coords_mixed_current_level)
    # print('coords_mixed_next_level', coords_mixed_next_level.size())

    coordinates_conv = sparse_tensor_conv.C
    
    mask_mixed_neighbors_current_level = create_mask(coordinates_conv, mixed_and_neighbors_current_level)

    return mask_mixed_current_level, mask_mixed_neighbors_current_level, coords_mixed_next_level