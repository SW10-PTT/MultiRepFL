class ChallengeTrainingSpecs:
  def __init__(self, model_hash, min_collateral, max_collateral, manager_address, reward, min_rounds, punishfactor, punishfactorContrib, freeriderPenalty, taskType):
      self.modelHash:bytes = model_hash
      self.min_collateral:int = min_collateral
      self.max_collateral:int = max_collateral
      self.manager_address = manager_address
      self.reward:int = reward
      self.min_rounds: int = min_rounds
      self.punishfactor: int = punishfactor
      self.punishfactorContrib: int = punishfactorContrib
      self.freeriderPenalty: int = freeriderPenalty
      self.taskType: int = taskType #Enum in future please