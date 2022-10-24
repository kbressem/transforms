class WeightedMSELoss: 
  
  def __init__(self, scale_factor): 
    self.scale_factor = scale_factor
    
  def __call__(self, input, target): 
    weigth = (target + 1) ** self.scale_factor
    
