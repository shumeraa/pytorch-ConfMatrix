import torch  
import matplotlib.pyplot as plt  
import numpy as np  

def confusion_matrix(preds, targets, num_classes, normalize=False, class_names=None, cmap='Blues'):
    with torch.no_grad():  # Disable gradient calculation for efficiency
        preds = preds.view(-1)  # Flatten the predictions tensor
        targets = targets.view(-1)  # Flatten the targets tensor

        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # Initialize confusion matrix

        # we need to make sure that all given values are valid
        # so, make sure that all values are at least 0 and at most num_classes
        mask = (targets >= 0) & (targets < num_classes)  
        targets = targets[mask]  
        preds = preds[mask]  

        # now we want to fill the confusion matrix
        # to do this, we need to find every combination of target and prediction and count them in the confusion matrix
        # first, find the combination of target and prediction in a single index
        indices = num_classes * targets + preds  
        
        # Then, count how often each unique index occurs in the data.
        conf_matrix_flat = torch.bincount(indices, minlength=num_classes ** 2)  
        
        #then reshape the confusion matrix to the final num_classes x num_classes matrix
        conf_matrix = conf_matrix_flat.reshape(num_classes, num_classes)  

        # if we want the confusion matrix to add up to 1, we cannormalize it
        if normalize:
            conf_matrix = conf_matrix.float()  # Convert for division
            conf_matrix = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)  # normalize
            conf_matrix = conf_matrix.nan_to_num()  # if there are any bad values, replace them with 0

        cm_array = conf_matrix.cpu().numpy()  # Convert tensor to np array for plotting

        # display the confusion matrix as an image using the specified color map 
        plt.imshow(cm_array, interpolation='nearest', cmap=cmap)

        plt.title('Confusion Matrix')

        # display a color bar next to the plot to show the color scale
        plt.colorbar()

        # set class names for the x and y axis ticks, if no class names are provided, use default numbers
        class_names = class_names or [str(i) for i in range(num_classes)]

        # set the labels on the x-axis to the class names with a 45-degree rotation
        plt.xticks(np.arange(num_classes), class_names, rotation=45)

        # set the labels on the y-axis to the class names
        plt.yticks(np.arange(num_classes), class_names)

        # set the format for the text inside the matrix (decimals if normalized, integers if not)
        fmt = '.2f' if normalize else 'd'

        # loop over every cell in the matrix and add the count as text
        for i, j in np.ndindex(cm_array.shape):
            plt.text(j, i, format(cm_array[i, j], fmt), 
                    ha="center", va="center", 
                    color="white" if cm_array[i, j] > cm_array.max() / 2. else "black")

        # label the y-axis as 'true label'
        plt.ylabel('True Label')

        # label the x-axis as 'predicted label'
        plt.xlabel('Predicted Label')

        # adjust the layout to prevent overlap or clipping
        plt.tight_layout()

        # show the plot
        plt.show()


    return conf_matrix  # Return the confusion matrix tensor
#ADD: Make sure it handles wrong number of classes
#ADD: handle non matching preds and targets
#ADD: handle non matching num_classes

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