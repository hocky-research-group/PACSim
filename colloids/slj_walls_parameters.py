class ShiftedLennardJonesWallsParameters(object):
    """
    This class stores the parameters to compute the shifted Lennard Jones potential forces for closed-wall simulations .

    :param box_length:
        The dimensions of the simulation box. This is used to determine the location of the SLJ walls. 
        The unit of the box_length must be compatible with nanometer. The value must be greater than 0.
        Defaults to 1000 nanometer.
    :type box_length: unit.Quantity
    :param epsilon:
        The Lennard Jones well depth.
        Defaults to 1.0.
    :type epsilon: float
    :param alpha:
        The cutoff factor in the shifted Lennard Jones potential. 
        This affects the continuity/ differentiability of the SLJ potential functional form.
        Defaults to 0.0 nanometers.
    :type alpha: float
  
    :raises TypeError:
        If the box_length is not a Quantity with a proper unit or if epsilon or alpha is not a float.
    :raises ValueError:
        If the box_length is not greater than zero.
    """

    ENERGY_CONVERSION_FACTOR = 2.477709860209665*unit.kilojoule_per_mole
          
    def __init__(self, box_length, epsilon, alpha):
      box_length = unit.Quantity = 1000.0 * (unit.nano * unit.meter)
      epsilon = 1.0 * ENERGY_CONVERSION_FACTOR
      alpha = 0.0
                 
      """Constructor of the ColloidPotentialsParameters class."""
      if not box_length.unit.is_compatible((unit.nano * unit.meter)):
          raise TypeError("argument box_length must have a unit that is compatible with nanometers")
      if not box_length.value_in_unit(unit.nano * unit.meter) > 0.0:
          raise ValueError("argument box_length must have a value greater than zero")
      if not type(epsilon) == float:
         raise TypeError("argument epsilon must be a float")
      if not type(alpha) == float: 
         raise TypeError("argument alpha must be a float")
           
     
      self._box_length = box_length.in_units_of(unit.nano * unit.meter))
      self._epsilon = epsilon.in_units_of(kilojoule_per_mole)
      self._alpha = alpha


  @property
  def box_length(self) -> unit.Quantity:
      return self._box_length

  @property
  def epsilon(self) -> unit.Quantity:
      return self._epsilon

  @property
  def alpha(self) -> float:
      return self._alpha


if __name__ == '__main__':
    parameters = ShiftedLennardJonesWallsParameters()
    print("Box length for slj walls:", parameters.box_length)
    print("Epsilon for slj walls:", parameters.epsilon)
    print("Alpha for slj walls:", parameters.alpha)
