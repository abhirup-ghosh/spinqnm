import numpy as np
import nr_fits as nr
import sys
sys.path.append('../../testGR_IR/scripts')
import imrtgrutils_final as tgr
import lal, lalsimulation

lmModes_list = [[2,2], [2,1], [3,3], [4,4]]

mass1 = 50.
mass2 = 50.
spin1 = lal.CreateREAL8Vector(3)
spin2 = lal.CreateREAL8Vector(3)
spin1.data = [0.,0.,0.]
spin2.data = [0.,0.,0.]

nmodes = len(lmModes_list)
approximant = 41

for lmModes in lmModes_list:
  l = lmModes[0]
  m = lmModes[1]

  modefreq = lal.CreateCOMPLEX16Vector(1)
  qnmfreq = lalsimulation.SimIMREOBGenerateQNMFreqV2(modefreq, mass1, mass2, spin2.data, spin2.data, l, m, 0, approximant)
  print qnmfreq
