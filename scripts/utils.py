import torch
from torch import Tensor

from matplotlib import pyplot as plt

def accuracy(nn_output: Tensor, ground_truth: Tensor, k: int=1):
    '''
    Return accuracy@k for the given model output and ground truth
    nn_output: a tensor of shape (num_datapoints x num_classes) which may 
       or may not be the output of a softmax or logsoftmax layer
    ground_truth: a tensor of longs or ints of shape (num_datapoints)
    k: the 'k' in accuracy@k
    '''
    assert k <= nn_output.shape[1], f"k too big. Found: {k}. Max: {nn_output.shape[1]} inferred from the nn_output"
    # get classes of assignment for the top-k nn_outputs row-wise
    nn_out_classes = nn_output.topk(k).indices
    # make ground_truth a column vector
    ground_truth_vec = ground_truth.unsqueeze(-1)
    # and repeat the column k times (= reproduce nn_out_classes shape)
    ground_truth_vec = ground_truth_vec.expand_as(nn_out_classes)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth_vec)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def random_data_extractor(loader, n):
    data, label = next(iter(loader))
    random_ind = torch.randint(len(data), [n])
    return data[random_ind], label[random_ind]

def show_bw_image(image, title=""):
    im = plt.imshow(image.squeeze(), cmap="gray")
    plt.title(title)
    plt.show()

def plot_nn_output(output):
    fig, axes=plt.subplots(1,1)

    output = output.squeeze()
    axes.bar(list(range(len(output))), output.softmax(dim=-1).detach().numpy())
    axes.xaxis.set_ticks(range(len(output)))
    plt.show()

def evaluate_single_data(model, data, transforms):
    batched_data = transforms(data)
    output = model(batched_data)
    classification = output.argmax(1)
    return output, classification

def get_first_n_error(model, loader, n=5):
    wrong_data = []
    ground_truth = []
    wrong_predictions = []
    accumulator = 0
    for data, labels in loader:
        print("...", accumulator)
        prediction = model(data).argmax(1).squeeze()
        match = prediction.eq(labels)
        wrong_classification = torch.where(match==False)[0]
        print(len(wrong_classification))
        if len(wrong_classification) == 0:
            continue
        m = min(len(wrong_classification), n - accumulator)
        print(m)
        wrong_data.extend([torch.Tensor(img) for img in data[wrong_classification[:m]].tolist()])
        ground_truth.extend(labels[wrong_classification[:m]].tolist())
        wrong_predictions.extend(prediction[wrong_classification[:m]].tolist())
        accumulator += m
        if accumulator >= n:
            break
    return wrong_data, ground_truth, wrong_predictions