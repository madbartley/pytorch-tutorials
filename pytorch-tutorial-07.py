

# Save and Load the Model

import torch
import torchvision.models as models

# Saving and Loading Model Weights
# pytorch models store the learned parameters in an internal state dictionary, called state_dic
# we can save these parameters in a .pth file for later use
# NOTE: this will save these parameters to C:\Users\MBartley/.cache\torch\hub\checkpoints\vgg16-397923af.pth
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# to load model weights, you need to create an instance of the SAME MODEL first, and then load the paramters using the load_state_dict() method
# here, we will do the same thing but set weights_only=True to limit the functions executed during unpickling to only those necessary for loading the weights
# this is best practice for loading weights
# NOTE: we do not specify ``weights``, i.e. create untrained model
# then we load our weights into the new, untrained model
# Recall: model.eval() will set the model to evaluation mode, meaning it will not try to update/optimize the parameters when it runs
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

# Saving and Loading Models with Shapes

# When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network
# We might want to save the structure of this class together with the model, in which case we can pass model (and not model.state_dict()) to the saving function
# Note, this will be saved to C:\Users\MBartley/.cache\torch\hub\checkpoints\vgg16-397923af.pth
torch.save(model, 'model.pth')

# now, we can load the model
# note: saving state_dict is considered the best practice. However, below we use weights_only=False because this involves loading the model, which is a legacy use case for torch.save
model = torch.load('model.pth', weights_only=False),

# NOTE: This approach uses Python pickle module when serializing the model, thus it relies on the actual class definition to be available when loading the model.

# The following link covers saving and loading models more in-depth: https://pytorch.org/tutorials/beginner/saving_loading_models.html