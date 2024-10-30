import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st

def preprocess_image(image):
    """ Preprocess the image to match the input requirements of your model """
    img_resized = cv2.resize(image, (1152, 2048))
    x_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255
    x_tensor = x_tensor.unsqueeze(0).to('cuda')
    return x_tensor

def create_single_masks(pr_masks, pr_classes, image_shape):
        single_channel_mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Iterate through each mask and apply the corresponding class label
        for mask, class_id in zip(pr_masks, pr_classes):
            if mask.shape != single_channel_mask.shape:
                raise ValueError(f"Shape mismatch: mask shape {mask.shape} does not match single_channel_mask shape {single_channel_mask.shape}")
            single_channel_mask[mask > 0] = class_id  # Ensure we apply the class_id to the correct positions

        #resized_mask = cv2.resize(single_channel_mask, (1152, 2048), interpolation=cv2.INTER_NEAREST)
        return single_channel_mask

def predict(model_path, image_tensor):
    """ Make a prediction using the selected model path """
    # Load the model dynamically based on the provided path
    model = torch.load(model_path, map_location=torch.device('cuda'))
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        pr_mask = model(image_tensor)
        pr_probs = nn.Softmax(dim=1)(pr_mask)

        pr_masks = []
        pr_classes = []
        pr_confidences = []

        # Use max along the class dimension
        class_probs, class_indices = torch.max(pr_probs[0], dim=0)
        # Get unique classes excluding the background
        # Convert to CPU for numpy unique if needed
        unique_classes = torch.unique(class_indices[class_indices >= 1])
        for class_idx in unique_classes:
            class_mask = class_indices == class_idx
            
            class_confidence = class_probs[class_mask].mean().item()
            
            # Resize mask and collect results in one go
            mask_resized = cv2.resize(
                class_mask.cpu().numpy().astype(np.float32),
                (1170, 2048),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.bool_)
            pr_masks.append(mask_resized)
            pr_classes.append(class_idx.item())
            pr_confidences.append(class_confidence)
    # Create the single_channel_mask
    single_channel_mask = create_single_masks(pr_masks, pr_classes, (2048, 1170))
    return single_channel_mask

def visualize_prediction(original_img, mask):
    """ Visualize the segmentation output over the original image """
    colors = {
            1: (255, 0, 0),     # Red
            2: (0, 255, 0),     # Green
            3: (0, 0, 255),     # Blue
            4: (255, 255, 0),   # Cyan
            5: (255, 0, 255),   # Magenta
            6: (0, 255, 255),   # Yellow
            7: (128, 0, 128),   # Purple
            8: (0, 128, 128)    # Teal
        }
    class_dict  = {
            0: "background",
            1:"pnt_blister",
            2:"pnt_dent",
            3:"pnt_orange",
            4:"pnt_scratch",
            5:"pnt_stuck",
            6:"pnt_craters",
            7:"pnt_filmleftovers",
            8:"pnt_wrongpowder"
        }
    # Ensure the mask is squeezed to (H, W)
    mask = mask.squeeze()

    # Ensure the image is in the correct format and scaled to 0-255
    image = original_img
    # Initialize an overlay with the same shape as the image (H, W, C)
    overlay = np.zeros_like(image, dtype=np.uint8)
    present_classes = set()

    # Apply each class color to the corresponding mask area
    for class_id, color in colors.items():
        class_mask = (mask == class_id)
        if np.any(class_mask):  # Only process and record classes that are present in the mask
            present_classes.add(class_id)
        # Apply color to each channel
        for channel in range(3):
            overlay[..., channel][class_mask] = color[channel]

    # Detect boundaries and draw a 1-pixel border
    bordered_image = image.copy()
    for class_id, color in colors.items():
        
        class_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours with the same color as the class on the original image without opacity
        cv2.drawContours(bordered_image, contours, -1, color, 1)

    # Blend the image with the overlay on specific regions, excluding borders
    opaque_mask = (mask != 0)
    blended = bordered_image.copy()
    blended[opaque_mask] = cv2.addWeighted(bordered_image, 0.85, overlay, 0.15, 0)[opaque_mask]

    # Create and add a legend
    # Add class names directly onto the bottom part of the blended image
    # Add class names directly onto the bottom part of the blended image for only present classes
    text_y = image.shape[0] - (20 * len(present_classes))  # Adjust starting position based on the number of classes
    for i, class_id in enumerate(present_classes):
        color = colors[class_id]
        class_name = class_dict.get(class_id, f"Class {class_id}")
        # Write the class name in its respective color
        cv2.putText(blended, class_name, (10, text_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), caption="Predictions", use_column_width=True)
    #return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    #plt.axis('off')  # Hide the axis
    #plt.title('Predictions')
    # Use st.pyplot instead of plt.show to render in Streamlit
    #st.pyplot(plt)