import copy
import functools
import numpy as np
import torch
import torch.optim as optim

from models.classifier_model import ThreeWayJointYClassifier
from models.score_model import ScoreNet

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

def test(model, score_model1, score_model2, score_model3, batch_size, num_steps, device):
    with torch.no_grad():
        batch1, time_steps1 = trajectory_sampler(score_model1, batch_size = batch_size, num_steps = num_steps, device=device)
        batch2, time_steps2 = trajectory_sampler(score_model2, batch_size = batch_size, num_steps = num_steps, device=device)
        batch3, time_steps3 = trajectory_sampler(score_model3, batch_size = batch_size, num_steps = num_steps, device=device)

        x_1 = batch1[-1,...]
        x_2 = batch2[-1,...]
        x_3 = batch3[-1,...]

        # compute terminal loss
        x_term = torch.cat([x_1, x_2, x_3], dim=0)
        time_term = torch.cat([torch.ones(x_1.shape[0], device=device) * time_steps1[-1],
                            torch.ones(x_2.shape[0], device=device) * time_steps2[-1],
                            torch.ones(x_3.shape[0], device=device) * time_steps3[-1]], dim=0)
        logprobs_term = model(x_term, time_term)

        ce_eq1 = torch.zeros((x_1.shape[0], 3), device=device)
        ce_eq1[:,0] = 1.0
        ce_eq2 = torch.zeros((x_2.shape[0], 3), device=device)
        ce_eq2[:,1] = 1.0
        ce_eq3 = torch.zeros((x_3.shape[0], 3), device=device)
        ce_eq3[:,2] = 1.0

        print(torch.mean(torch.exp(logprobs_term)[:batch_size,...], dim=0))
        print(torch.mean(torch.exp(logprobs_term)[batch_size:2*batch_size,...], dim=0))
        print(torch.mean(torch.exp(logprobs_term)[2*batch_size:,...], dim=0))

        ce_target_term = torch.cat([ce_eq1, ce_eq2, ce_eq3], dim=0)
        loss_term = -torch.mean(torch.logsumexp(logprobs_term, dim=1) * ce_target_term)
        loss1 = -torch.mean(torch.logsumexp(logprobs_term, dim=1)[:batch_size,0])
        loss2 = -torch.mean(torch.logsumexp(logprobs_term, dim=1)[batch_size:2*batch_size,1])
        loss3 = -torch.mean(torch.logsumexp(logprobs_term, dim=1)[2*batch_size:,2])
        acc = (torch.argmax(torch.sum(logprobs_term.exp(), dim=1), dim=1) == torch.tensor([0] * x_1.shape[0] + [1] * x_2.shape[0] + [2] * x_3.shape[0], dtype=torch.int, device=device)).float().mean()
        acc1 = (torch.argmax(torch.sum(logprobs_term.exp(), dim=1), dim=1)[:batch_size] == torch.tensor([0] * x_1.shape[0], dtype=torch.int, device=device)).float().mean()
        acc2 = (torch.argmax(torch.sum(logprobs_term.exp(), dim=1), dim=1)[batch_size:2*batch_size] == torch.tensor([1] * x_1.shape[0], dtype=torch.int, device=device)).float().mean()
        acc3 = (torch.argmax(torch.sum(logprobs_term.exp(), dim=1), dim=1)[2*batch_size:] == torch.tensor([2] * x_1.shape[0], dtype=torch.int, device=device)).float().mean()
        print('Average terminal loss: {:5f} ({:.5f}, {:.5f}, {:.5f}), accuracy: {:.5f} ({:.5f}, {:.5f}, {:.5f})'.format(loss_term.item(), loss1, loss2, loss3, acc, acc1, acc2, acc3))

        #
        # non-terminal states
        #
        logprobs_term_ema = model(x_term, time_term)
        p_x_y2_eq_1 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 0]
        p_x_y2_eq_2 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 1]
        p_x_y2_eq_3 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 2]

        for stepIDX in range(0, batch1.shape[0]-1, 10):
            s_1 = batch1[stepIDX,...]
            s_2 = batch2[stepIDX,...]
            s_3 = batch3[stepIDX,...]
            s_non_term = torch.cat([s_1, s_2, s_3], dim=0)

            time_term = torch.cat([torch.ones(s_1.shape[0], device=device) * time_steps1[stepIDX],
                                   torch.ones(s_2.shape[0], device=device) * time_steps2[stepIDX],
                                   torch.ones(s_3.shape[0], device=device) * time_steps3[stepIDX]], dim=0)
            logprobs_non_term = model(s_non_term, time_term)

            w_mat = torch.zeros((s_non_term.shape[0], 3, 3), device=device)
            # set y1 = 0
            w_mat[:s_1.shape[0], 0, :] = 1.0
            # set y1 = 1
            w_mat[s_1.shape[0]:s_1.shape[0]+s_2.shape[0], 1, :] = 1.0
            # set y1 = 2
            w_mat[s_1.shape[0]+s_2.shape[0]:, 2, :] = 1.0

            w_mat[:, :, 0] *= p_x_y2_eq_1[:, None]
            w_mat[:, :, 1] *= p_x_y2_eq_2[:, None]
            w_mat[:, :, 2] *= p_x_y2_eq_3[:, None]

            step_loss = -torch.mean(w_mat * logprobs_non_term)
            loss1 = -torch.mean(torch.logsumexp(logprobs_non_term, dim=1)[:batch_size,0])
            loss2 = -torch.mean(torch.logsumexp(logprobs_non_term, dim=1)[batch_size:2*batch_size,1])
            loss3 = -torch.mean(torch.logsumexp(logprobs_non_term, dim=1)[2*batch_size:,2])
            acc = (torch.argmax(torch.sum(logprobs_non_term.exp(), dim=1), dim=1) == torch.tensor([0] * x_1.shape[0] + [1] * x_2.shape[0] + [2] * x_3.shape[0], dtype=torch.int, device=device)).float().mean()
            acc1 = (torch.argmax(torch.sum(logprobs_non_term.exp(), dim=1), dim=1)[:batch_size] == torch.tensor([0] * x_1.shape[0], dtype=torch.int, device=device)).float().mean()
            acc2 = (torch.argmax(torch.sum(logprobs_non_term.exp(), dim=1), dim=1)[batch_size:2*batch_size] == torch.tensor([1] * x_1.shape[0], dtype=torch.int, device=device)).float().mean()
            acc3 = (torch.argmax(torch.sum(logprobs_non_term.exp(), dim=1), dim=1)[2*batch_size:] == torch.tensor([2] * x_1.shape[0], dtype=torch.int, device=device)).float().mean()
            print('Average Loss at step {:2f}: {:5f} ({:.5f}, {:.5f}, {:.5f}), accuracy: {:5f} ({:.5f}, {:.5f}, {:.5f})'.format(time_steps1[stepIDX], step_loss.item(), loss1, loss2, loss3, acc, acc1, acc2, acc3))

def train(model, target_model, optimizer, score_model1, score_model2, score_model3, batch_size, num_steps, device, terminal_only = True):
    batch1, time_steps1 = trajectory_sampler(score_model1, batch_size = batch_size, num_steps = num_steps, device=device)
    batch2, time_steps2 = trajectory_sampler(score_model2, batch_size = batch_size, num_steps = num_steps, device=device)
    batch3, time_steps3 = trajectory_sampler(score_model3, batch_size = batch_size, num_steps = num_steps, device=device)

    x_1 = batch1[-1,...]
    x_2 = batch2[-1,...]
    x_3 = batch3[-1,...]

    # compute terminal loss
    x_term = torch.cat([x_1, x_2, x_3], dim=0)
    time_term = torch.cat([torch.ones(x_1.shape[0], device=device) * time_steps1[-1],
                           torch.ones(x_2.shape[0], device=device) * time_steps2[-1],
                           torch.ones(x_3.shape[0], device=device) * time_steps3[-1]], dim=0)
    logprobs_term = model(x_term, time_term)

    ce_eq1 = torch.zeros((x_1.shape[0], 3), device=device)
    ce_eq1[:,0] = 1.0
    ce_eq2 = torch.zeros((x_2.shape[0], 3), device=device)
    ce_eq2[:,1] = 1.0
    ce_eq3 = torch.zeros((x_3.shape[0], 3), device=device)
    ce_eq3[:,2] = 1.0

    ce_target_term = torch.cat([ce_eq1, ce_eq2, ce_eq3], dim=0)
    loss_term = -torch.mean(torch.logsumexp(logprobs_term, dim=1) * ce_target_term)

    #
    # non-terminal states
    #
    if not terminal_only:
        with torch.no_grad():
            logprobs_term_ema = target_model(x_term, time_term)
            p_x_y2_eq_1 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 0]
            p_x_y2_eq_2 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 1]
            p_x_y2_eq_3 = torch.sum(logprobs_term_ema.exp(), dim=1)[:, 2]

        exclude_from_t = 0.7 # do not train from this timestep until t = 1.0. This is because the last timesteps are too noisy to train on.
        train_fraction = 0.1 # train on a fraction of randomly selected steps of this size
        loss_step_weight = 1.0 / (batch1.shape[0] * (1-exclude_from_t) * train_fraction)
        for stepIDX in range(batch1.shape[0]-1):
            if time_steps1[stepIDX] > exclude_from_t:
               continue
            if np.random.rand() > train_fraction:
               continue
            s_1 = batch1[stepIDX,...]
            s_2 = batch2[stepIDX,...]
            s_3 = batch3[stepIDX,...]
            s_non_term = torch.cat([s_1, s_2, s_3], dim=0)

            time_term = torch.cat([torch.ones(s_1.shape[0], device=device) * time_steps1[stepIDX],
                                   torch.ones(s_2.shape[0], device=device) * time_steps2[stepIDX],
                                   torch.ones(s_3.shape[0], device=device) * time_steps3[stepIDX]], dim=0)
            logprobs_non_term = model(s_non_term, time_term)

            w_mat = torch.zeros((s_non_term.shape[0], 3, 3), device=device)
            # set y1 = 0
            w_mat[:s_1.shape[0], 0, :] = 1.0
            # set y1 = 1
            w_mat[s_1.shape[0]:s_1.shape[0]+s_2.shape[0], 1, :] = 1.0
            # set y1 = 2
            w_mat[s_1.shape[0]+s_2.shape[0]:, 2, :] = 1.0

            w_mat[:, :, 0] *= p_x_y2_eq_1[:, None]
            w_mat[:, :, 1] *= p_x_y2_eq_2[:, None]
            w_mat[:, :, 2] *= p_x_y2_eq_3[:, None]

            loss_term -= torch.mean(w_mat * logprobs_non_term) * loss_step_weight

    print('Average Loss: {:5f}'.format(loss_term.item()))

    optimizer.zero_grad()
    loss_term.backward()
    optimizer.step()

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

model = ThreeWayJointYClassifier(input_channels=score_model1.input_channels).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

test(model, score_model1, score_model2, score_model3, test_batch_size, num_steps, device)

for epoch in range(1, 700 + 1):
    if epoch % 5 == 1:
        target_model = copy.deepcopy(model)
        for p in target_model.parameters():
            p.requires_grad = False

    train(model, target_model, optimizer, score_model1, score_model2, score_model3, batch_size, num_steps, device, (epoch <= 100))
    if epoch % 50 == 0:
       test(model, score_model1, score_model2, score_model3, test_batch_size, num_steps, device)
       torch.save(model.state_dict(), '3way_classifier_ckpt_' + str(epoch) + '.pth')
