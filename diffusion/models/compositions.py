import torch
import torch.nn as nn

class BinaryDiffusionComposition(nn.Module):
    """
    Composition of m diffusion models using 2 y variables
    score_models: list of score_model models
    classifier: classifier for classifier guidance
    y_1, y_2: int defining the composition
    guidance_scale" float representing the guidance scaling factor
    """
    def __init__(self, score_models, classifier, y_1, y_2, guidance_scale = 1.0):
      super().__init__()
      self.score_models = score_models
      self.m = len(self.score_models)
      self.classifier = classifier
      self.y_1 = y_1
      self.y_2 = y_2
      self.guidance_scale = guidance_scale

      self.input_channels = score_models[0].input_channels

    def classifier_grad(self, x, t):
        x_tmp = torch.clone(x).requires_grad_(True).to(x.device)
        t.requires_grad_(False)
        cls_logprobs_x_t = self.classifier(x_tmp,t)

        grd = torch.zeros((x.shape[0],self.m,self.m), device = x.device)   # same shape as cls_logprobs_x_t
        grd[:, self.y_1 - 1, self.y_2 - 1] = 1.0    # column of Jacobian to compute
        cls_logprobs_x_t.backward(gradient = grd, retain_graph = True)
        grad = x_tmp.grad
        grad.requires_grad_(False)

        return grad

    def forward(self, x, t):
        cls_grad = self.classifier_grad(x,t)
        with torch.no_grad():
            scores = []
            for score_model in self.score_models:
                scores.append(score_model(x, t))

            cls_logprobs_x_t = self.classifier(x, t)

            mixture_score = torch.zeros_like(scores[0], device=x.device)
            for i in range(self.m):
                mixture_score += torch.mul(scores[i], torch.sum(torch.exp(cls_logprobs_x_t), dim=2)[:, i].view(-1, 1, 1, 1))

            composition_score = mixture_score + self.guidance_scale * cls_grad
            return composition_score
        
class ConditionalDiffusionComposition(nn.Module):
    """
    Composition of m diffusion models using 2 y variables
    score_models: list of score_model models
    classifier: classifier for classifier guidance
    y_1, y_2: int defining the composition
    guidance_scale" float representing the guidance scaling factor
    """
    def __init__(self, binary_diffusion, conditional_classifier, y_3, guidance_scale = 1.0):
      super().__init__()
      self.binary_diffusion = binary_diffusion
      self.conditional_classifier = conditional_classifier
      self.m = binary_diffusion.m
      self.y_1 = binary_diffusion.y_1
      self.y_2 = binary_diffusion.y_2
      self.y_3 = y_3
      self.guidance_scale = guidance_scale

      self.input_channels = binary_diffusion.input_channels

    def classifier_grad(self, x, t):
        x_tmp = torch.clone(x).requires_grad_(True).to(x.device)
        t.requires_grad_(False)
        cls_logprobs_x_t = self.conditional_classifier(x_tmp,t,[self.y_1] * x.shape[0], [self.y_2] * x.shape[0])

        grd = torch.zeros((x.shape[0],self.m), device = x.device)   # same shape as cls_logprobs_x_t
        grd[:, self.y_3 - 1] = 1.0    # column of Jacobian to compute
        cls_logprobs_x_t.backward(gradient = grd, retain_graph = True)
        grad = x_tmp.grad
        grad.requires_grad_(False)

        return grad

    def forward(self, x, t):
        binary_score = self.binary_diffusion(x,t)
        cls_grad = self.classifier_grad(x,t)
        return binary_score + cls_grad * self.guidance_scale