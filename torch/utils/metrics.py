import torch  
import matplotlib.pyplot as plt  
import numpy as np  

def confusion_matrix(preds, targets, num_classes, normalize=False, class_names=None, cmap='Blues'):
    with torch.no_grad():  # Disable gradient calculation for efficiency
        
        # Check that num_classes is a positive integer
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        
        # Check that class_names has length equal to num_classes
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("Length of class_names must be equal to num_classes.")
        
        preds = preds.view(-1)  # Flatten the predictions tensor
        targets = targets.view(-1)  # Flatten the targets tensor
        
        # Check that preds and targets have the same number of elements
        if preds.shape[0] != targets.shape[0]:
            raise ValueError("Predictions and targets must have the same number of elements.")
        
        # Check that preds and targets are within valid class range
        if preds.min() < 0 or preds.max() >= num_classes:
            raise ValueError("Predictions contain invalid class indices.")
        
        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError("Targets contain invalid class indices.")
        
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # Initialize confusion matrix
    
        # Now we want to fill the confusion matrix
        # To do this, we need to find every combination of target and prediction and count them in the confusion matrix
        # First, find the combination of target and prediction in a single index
        indices = num_classes * targets + preds  
        
        # Then, count how often each unique index occurs in the data.
        conf_matrix_flat = torch.bincount(indices, minlength=num_classes ** 2)  
        
        # Then reshape the confusion matrix to the final num_classes x num_classes matrix
        conf_matrix = conf_matrix_flat.reshape(num_classes, num_classes)  
    
        # If we want the confusion matrix to add up to 1, we can normalize it
        if normalize:
            conf_matrix = conf_matrix.float()  # Convert for division
            conf_matrix = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)  # Normalize
            conf_matrix = conf_matrix.nan_to_num()  # Replace NaNs with 0
    
        cm_array = conf_matrix.cpu().numpy()  # Convert tensor to np array for plotting
    
        # Display the confusion matrix 
        plt.imshow(cm_array, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix')
        plt.colorbar()
        # Set class names for the x and y axis ticks
        class_names = class_names or [str(i) for i in range(num_classes)]
    
        # Set the labels on the x-axis to the class names with a 45-degree rotation
        plt.xticks(np.arange(num_classes), class_names, rotation=45)
    
        # Set the labels on the y-axis to the class names
        plt.yticks(np.arange(num_classes), class_names)
    
        # Set the format for the text inside the matrix (decimals if normalized, integers if not)
        fmt = '.2f' if normalize else 'd'
    
        # Loop over every cell in the matrix and add the count as text
        for i, j in np.ndindex(cm_array.shape):
            plt.text(j, i, format(cm_array[i, j], fmt), 
                    ha="center", va="center", 
                    color="white" if cm_array[i, j] > cm_array.max() / 2. else "black")
    
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    return conf_matrix  # Return the confusion matrix tensor if the user wants to use it



# Example predicted labels and true labels
preds = torch.tensor([2, 0, 2, 2, 0, 1])
targets = torch.tensor([0, 0, 2, 2, 0, 2])

# Number of classes
num_classes = 3

# Optional: Class names
class_names = ['Class A', 'Class B', 'Class C']

# Compute and plot the confusion matrix
cm = confusion_matrix(preds, targets, num_classes, normalize=True, class_names=class_names, cmap='Blues')

print("Confusion Matrix Tensor:")
print(cm)