import functools
import numpy as np
import torch
import torch.optim as optim

from models.classifier_model import ThreeWayJointYClassifier, ThreeWayConditionalYClassifier
from models.score_model import ScoreNet
from models.compositions import BinaryDiffusionComposition

from samplers.pc_sampler import pc_trajectory_sampler

#
# Diffusion Sampling
#

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  

    Returns:
    The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)
  
sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
trajectory_sampler = functools.partial(pc_trajectory_sampler, marginal_prob_std=marginal_prob_std_fn, diffusion_coeff=diffusion_coeff_fn)

#
# Classifier training
#

def train(model, optimizer, joint_y_classifier, score_model1, score_model2, score_model3, batch_size, num_steps, device):
    loss_term = 0.0

    exclude_from_t = 0.7 # do not train from this timestep until t = 1.0. This is because the last timesteps are too noisy to train on.
    train_fraction = 50.0 / num_steps # train on a fraction of randomly selected steps of this size
    loss_step_weight = 1.0 / ((1-exclude_from_t) * train_fraction)
    for (y_1, y_2) in [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]:
        binary_composition = BinaryDiffusionComposition([score_model1, score_model2, score_model3], joint_y_classifier, y_1, y_2, 10.0)
        batch, time_steps = trajectory_sampler(binary_composition, batch_size = batch_size, num_steps = num_steps, device=device, show_progress=False)

        with torch.no_grad():
            time_term = torch.ones(batch_size, device=device) * time_steps[-1]
            logprobs_term_ema = joint_y_classifier(batch[-1,...], time_term)
            w_mat = torch.sum(logprobs_term_ema.exp(), dim=1)

        for stepIDX in range(num_steps):
            if time_steps[stepIDX] > exclude_from_t:
                continue
            if (np.random.rand() > train_fraction) and (stepIDX != (num_steps-1)):
                continue
            s_1 = batch[stepIDX,...]

            time_term = torch.ones(s_1.shape[0], device=device) * time_steps[stepIDX]
            logprobs_non_term = model(s_1, time_term, [y_1] * batch_size, [y_2] * batch_size)

            loss_term -= torch.sum(w_mat * logprobs_non_term) * loss_step_weight

    loss_term /= (num_steps * batch_size * 6)
    print('Average Loss: {:5f}'.format(loss_term.item()))

    optimizer.zero_grad()
    loss_term.backward()
    optimizer.step()

def test(model, joint_y_classifier, score_model1, score_model2, score_model3, batch_size, num_steps, device):
    for (y_1, y_2) in [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]:
        binary_composition = BinaryDiffusionComposition([score_model1, score_model2, score_model3], joint_y_classifier, y_1, y_2, 10.0)
        batch, time_steps = trajectory_sampler(binary_composition, batch_size = batch_size, num_steps = num_steps, device=device, show_progress=False)
        batch = batch.detach()

        s_1 = batch[-1,...]
        time_term = torch.ones(s_1.shape[0], device=device) * time_steps[-1]
        logprobs_non_term = model(s_1, time_term, [y_1] * batch_size, [y_2] * batch_size).exp().mean(dim=0)
        print(str(y_1) + " " + str(y_2) + ": " + str(logprobs_non_term))


batch_size = 128
test_batch_size = 300
num_steps = 500

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

score_model1 = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=3)
score_model1 = score_model1.to(device)
score_model1.load_state_dict(torch.load('gen_MN1_ckpt_195.pth', map_location=device))
for param in score_model1.parameters():
    param.requires_grad = False

score_model2 = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=3)
score_model2 = score_model2.to(device)
score_model2.load_state_dict(torch.load('gen_MN2_ckpt_195.pth', map_location=device))
for param in score_model2.parameters():
    param.requires_grad = False

score_model3 = ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_channels=3)
score_model3 = score_model3.to(device)
score_model3.load_state_dict(torch.load('gen_MN3_ckpt_195.pth', map_location=device))
for param in score_model3.parameters():
    param.requires_grad = False

joint_y_classifier = ThreeWayJointYClassifier(input_channels=3)
joint_y_classifier = joint_y_classifier.to(device)
cls_ckpt = torch.load('3way_classifier_ckpt_700.pth', map_location=device)
joint_y_classifier.load_state_dict(cls_ckpt)
for param in joint_y_classifier.parameters():
    param.requires_grad = False

model = ThreeWayConditionalYClassifier(input_channels=score_model1.input_channels).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=0.1)

for epoch in range(1, 200 + 1):
    train(model, optimizer, joint_y_classifier, score_model1, score_model2, score_model3, batch_size, num_steps, device)
    if epoch % 10 == 0:
        print("EPOCH " + str(epoch))
        test(model, joint_y_classifier, score_model1, score_model2, score_model3, test_batch_size, num_steps, device)
    if epoch % 20 == 0:
        torch.save(model.state_dict(), '3way_conditional_classifier_ckpt_' + str(epoch) + '.pth')
