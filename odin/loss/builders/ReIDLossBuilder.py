from ..SoftmaxLogitsLoss import SoftmaxLogitsLoss
from ..SoftmaxLabelSmooth import SoftmaxLabelSmooth
from ..TripletLoss import TripletLoss
from ..MarginLoss import MarginLoss

class ReIDLossBuilder(object):
  LOSS_PARAMS = {}
  LOSS_PARAMS['SoftmaxLogitsLoss'] = {}
  LOSS_PARAMS['SoftmaxLogitsLoss']['fn'] = SoftmaxLogitsLoss
  LOSS_PARAMS['SoftmaxLogitsLoss']['args'] = ['logits', 'labels']
  LOSS_PARAMS['TripletLoss'] = {}
  LOSS_PARAMS['TripletLoss']['fn'] = TripletLoss
  LOSS_PARAMS['TripletLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['MarginLoss'] = {}
  LOSS_PARAMS['MarginLoss']['fn'] = MarginLoss
  LOSS_PARAMS['MarginLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['SoftmaxLabelSmooth'] = {}
  LOSS_PARAMS['SoftmaxLabelSmooth']['fn'] = SoftmaxLabelSmooth
  LOSS_PARAMS['SoftmaxLabelSmooth']['args'] = ['logits', 'labels']

  def __init__(self, loss_functions, loss_lambda, loss_kwargs, **kwargs):
    self.loss = []
        
    fn_len = len(loss_functions)
    lambda_len = len(loss_lambda)
    kwargs_len = len(loss_kwargs)
    # Set up the logger
    self.logger = kwargs.get("logger")
    # Sanity check
    if fn_len != lambda_len:
      raise ValueError("Loss function list length is %i. Expected %i length loss_lambdas, got %i"%(fn_len, fn_len, lambda_len))
    if fn_len != kwargs_len:
      raise ValueError("Loss function list length is %i. Expected %i length loss_kwargs, got %i"%(fn_len, fn_len, kwargs_len))
    # Add the loss functions with the correct features. Lambda is applied later
    for idx, fn in enumerate(loss_functions):
      self.loss.append(self.LOSS_PARAMS[fn]['fn'](**loss_kwargs[idx]))
      self.logger.info("Added {loss} with lambda = {lamb} and loss arguments {largs}".format(loss=fn, lamb=loss_lambda[idx], largs=str(loss_kwargs[idx])))
    
    lambda_sum = sum(loss_lambda)
    loss_lambda = [float(item)/float(lambda_sum) for item in loss_lambda]
    self.loss_lambda = loss_lambda
    self.loss_fn = loss_functions
    
  def __call__(self,**kwargs):
    """Call operator of the loss builder.

    This returns the sum of each individual loss provided in the initialization, multiplied by their respective loss_lambdas.

    Args (kwargs only):
        logits: Torch tensor of shape (batch_size, softmax_dimensions). The output logits of the softmax layer. None if not available.
        labels: Torch tensor of shape (batch_size, 1). The class labels.
        features: Torch tensor of shape (batch_size, embedding_dimensions). The feature embeddings generated by the ReID model.
    """
    loss = 0.0
    for idx, fn in enumerate(self.loss):
      loss += self.loss_lambda[idx] * fn(kwargs.get(self.LOSS_PARAMS[self.loss_fn[idx]]['args'][0]), kwargs.get(self.LOSS_PARAMS[self.loss_fn[idx]]['args'][1]))
    return loss