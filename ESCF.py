#!/usr/bin/env python3

"""
@author: fb445
"""
import os
import subprocess as sp
import sys
import shutil
import datetime
import copy
import time
import numpy as np

bohr2angstroem = 0.529177
angstroem2bohr = 1.88973
hartree2cm = 219474.63
cm2hartree = 4.5563352812122295e-06
hartree2ev = 27.210736268101943
ev2hartree = 0.0367502

#How many frozen core orbitals for each atom
FROZENS = {'H': 0, 'He': 0,
           'Li': 0, 'Be': 0, 'B': 1, 'C': 1, 'N': 1, 'O': 1, 'F': 1, 'Ne': 1,
           'Na': 1, 'Mg': 1, 'Al': 5, 'Si': 5, 'P': 5, 'S': 5, 'Cl': 5, 'Ar': 5,
           'K': 5, 'Ca': 5,
           'Sc': 5, 'Ti': 5, 'V': 5, 'Cr': 5, 'Mn': 5, 'Fe': 5, 'Co': 5, 'Ni': 5, 'Cu': 5, 'Zn': 5,
           'Ga': 9, 'Ge': 9, 'As': 9, 'Se': 9, 'Br': 9, 'Kr': 9,
           'Rb': 9, 'Sr': 9,
           'Y': 14, 'Zr': 14, 'Nb': 14, 'Mo': 14, 'Tc': 14, 'Ru': 14, 'Rh': 14, 'Pd': 14, 'Ag': 14, 'Cd': 14,
           'In': 18, 'Sn': 18, 'Sb': 18, 'Te': 18, 'I': 18, 'Xe': 18,
           'Cs': 18, 'Ba': 18,
           'La': 18,
           'Ce': 18, 'Pr': 18, 'Nd': 18, 'Pm': 18, 'Sm': 18, 'Eu': 18, 'Gd': 18, 'Tb': 18, 'Dy': 18, 'Ho': 18, 'Er': 18, 'Tm': 18, 'Yb': 18, 'Lu': 23,
           'Hf': 23, 'Ta': 23, 'W': 23, 'Re': 23, 'Os': 23, 'Ir': 23, 'Pt': 23, 'Au': 23, 'Hg': 23,
           'Tl': 34, 'Pb': 34, 'Bi': 34, 'Po': 34, 'At': 34, 'Rn': 34,
           'Fr': 34, 'Ra': 34,
           'Ac': 34,
           'Th': 34, 'Pa': 34, 'U': 34, 'Np': 34, 'Pu': 34, 'Am': 34, 'Cm': 34, 'Bk': 34, 'Cf': 34, 'Es': 34, 'Fm': 34, 'Md': 34, 'No': 34, 'Lr': 34,
           'Rf': 50, 'Db': 50, 'Sg': 50, 'Bh': 50, 'Hs': 50, 'Mt': 50, 'Ds': 50, 'Rg': 50, 'Cn': 50,
           'Nh': 50, 'Fl': 50, 'Mc': 50, 'Lv': 50, 'Ts': 50, 'Og': 50
           }
    
class TDM(): # Transition Density Matrix composed of transition energy, numpy matrix and dimension of that matrix
    
    def __init__(self):
        self.eigenvalue = 0
        self.tensorspace = 0
        
    def setTDM(self,NBR_of_virt,NBR_of_occ): #constructor
        self.TDM = np.zeros([NBR_of_virt,NBR_of_occ])
        return
    
    def __add__(self,other):
        Sum_of_TDMs = TDM()
        Sum_of_TDMs.eigenvalue = self.eigenvalue
        Sum_of_TDMs.tensorspace = self.tensorspace
        Sum_of_TDMs.TDM = self.TDM + other.TDM
        return Sum_of_TDMs
    
    def copy(self):
        new_tdm = TDM()
        new_tdm.eigenvalue = self.eigenvalue
        new_tdm.tensorspace = self.tensorspace
        new_tdm.TDM = np.copy(self.TDM)
        return new_tdm   
    
class QMoutput():
    
    def __init__(self, QMin):
        self.QMInput = QMin
        return
    
    # Read in Hamiltonian, Dipolmoments and Orbitals that play a role in transition.
    # This function is called once for each escf output file, therefore once for every multiplicity (so far only 1 or 3) and generates the attributes each time
    
    def parseESCF(self,multiplicity=1): 
        
        
        prevdir = os.getcwd()
        path = self.QMInput.scratchdir
        if multiplicity==1:
            unrestriced = False
        else:
            unrestriced = True
        os.chdir(path)       
        print('parsing escf output file from: ' +str(os.getcwd()))
        
        if not unrestriced:
            n = self.QMInput.number_of_singlets+1
            m = self.QMInput.number_of_singlets+1
        else: 
            n = self.QMInput.number_of_triplets+1
            m = self.QMInput.number_of_triplets+1
            
        Hamiltonian = MakeMatrix(n, m)   
        DipoleMatrix = MakeMatrix(n, 3,float)
        
        if self.QMInput.method[0]=='gw':
            if multiplicity == 1:
                file = readfile('bse.out')
            elif multiplicity == 3:
                file = readfile('bse_triplet.out')
        elif self.QMInput.method[0]=='dft':
            if multiplicity == 1:
                file = readfile('escf.out')
            elif multiplicity == 3:
                file = readfile('escf_triplet.out')
        
        current_state = 0
        current_line = 0
        NRG_complete = False
        Dipole_complete = False  
        cartesian_basis_function_not_found = True
        
        # Go through ESCF file line by line and find the corresponding keywords and write the information to variables
        # Once Dipole and transition energies are parsed NRG_complete and Dipole_complete are set to true, so that the state index can be increased by one for the next one
        
        for line in file:
            if NRG_complete and Dipole_complete:
                current_state += 1
                NRG_complete = False
                Dipole_complete = False
            if 'Total energy:' in line:
                Hamiltonian[current_state,current_state] = float(line.split()[-1])+0j
                NRG_complete = True
            if 'Electric dipole moment:' in line: #Ground state
                dipole_starts_at = current_line
                for dipole_part in file[dipole_starts_at:dipole_starts_at+7]:
                    if ' x ' in dipole_part:
                        DipoleMatrix[current_state,0] = float(dipole_part.split()[3])
                    elif ' y ' in dipole_part: 
                        DipoleMatrix[current_state,1] = float(dipole_part.split()[3])
                    elif ' z ' in dipole_part:
                        DipoleMatrix[current_state,2] = float(dipole_part.split()[3])
                        Dipole_complete = True
                            
            if 'Electric transition dipole moment (length rep.)' in line: #each excited state
                dipole_starts_at = current_line
                for dipole_part in file[dipole_starts_at:dipole_starts_at+5]:
                    if ' x ' in dipole_part:
                        DipoleMatrix[current_state,0] = float(dipole_part.split()[1])
                    elif ' y ' in dipole_part: 
                        DipoleMatrix[current_state,1] = float(dipole_part.split()[1])
                    elif ' z ' in dipole_part:
                        DipoleMatrix[current_state,2] = float(dipole_part.split()[1])
                        Dipole_complete = True
            
            # Parse some additional information that is needed later            
            if 'number of basis functions' in line:
                self.nbr_of_scf_basis_functions = int(line.split()[-1])
            if 'number of occupied orbitals' in line:
                self.nbr_of_occupied_orbitals = int(line.split()[-1])
            if 'total number of cartesian basis functions' in line and cartesian_basis_function_not_found:
                self.nbr_of_cartesian_basis_functions = int(line.split()[-1])
                cartesian_basis_function_not_found = False
            current_line += 1
        
        # Check if everything went through without problems    
        if multiplicity==1:
            assert current_state == self.QMInput.number_of_singlets+1, '!!! Warning: Number of parsed states is not equal to number of requested states !!!'
        else:
            assert current_state == self.QMInput.number_of_triplets+1, '!!! Warning: Number of parsed states is not equal to number of requested states !!!'
            
            
        # Store Data
        if multiplicity == 1:
            self.original_dipole_matrix = DipoleMatrix
            self.all_diplote_matrices = build_all_dipole_matrices(DipoleMatrix)
            self.hamiltonian = separate_complex(Hamiltonian)
        else:
            self.original_dipole_matrix_triplet = DipoleMatrix
            self.hamiltonian_triplet = separate_complex(Hamiltonian)
            self.all_diplote_matrices_triplet = build_all_dipole_matrices(DipoleMatrix)
        os.chdir(prevdir)
        
        return
        
    def parseTDM(self,tda=False,multiplicity=1): #Parse all TDMs from sing_a or ciss_a, one could extend this so symmetry, then this functions need to be executed for each irrep
        
        path = self.QMInput.scratchdir
        if multiplicity == 1: # The same function is called once for every multiplicity
            self.All_TDMs = []
        elif multiplicity == 3:
            self.All_TDMs_triplet = []
        prevdir = os.getcwd()
        os.chdir(path)
        
        print('parsing transition density matrix')        
        
        NBR_of_occ = self.nbr_of_occupied_orbitals
        NBR_of_virt = self.nbr_of_scf_basis_functions - self.nbr_of_occupied_orbitals
        
        if tda: #ciss_a only saves X vector in MO basis like this: 5 orbitals and 1,2 are occupied -> ciss_a saves from left to right, row by row x in the following way: (3,1),(4,1),(5,1),(3,2),(4,2) ....
            X = TDM()
            X.setTDM(NBR_of_virt,NBR_of_occ)        
            if multiplicity == 1:
                lines = readfile('ciss_a')#singlet
            elif multiplicity == 3:
                lines = readfile('cist_a')#triplet, one will have to extend this if unrestricted calculations or doublets are needed
            current_state = 1
            TDM_found = False
            Time_to_calculate_TDM = False
            for line in lines:
                if 'tensor space dimension' in line:
                    words = line.split()
                    X.tensorspace = int(words[-1])
                elif 'eigenvalue' in line:
                    words = line.split()
                    X.eigenvalue = float(words[-1].replace('D','E'))
                    TDM_found = True
                    indx_occ  = 0
                    indx_virt = 0
                    continue
                if TDM_found:#This part is very similar to the theodore interface. One goes through the file in steps of 4
                    words = [line[0+20*i:20+20*i] for i in range(4)]
                    for word in words:
                        X.TDM[indx_virt,indx_occ] = float(word.replace('D','E'))
                        indx_virt += 1
                        if indx_virt == NBR_of_virt:
                            indx_occ += 1
                            indx_virt = 0
                            if indx_occ == NBR_of_occ:
                                current_state += 1
                                TDM_found = False
                                if multiplicity == 1:
                                    self.All_TDMs.append(X.copy())
                                else:
                                    self.All_TDMs_triplet.append(X.copy())
                                break                    
        else: #sing_a saves data same way like ciss_a, but after X+Y comes X-Y with no linebreak!
            assert multiplicity == 1, 'For now, only TDA for triplet !' # Does Turbomole save X+Y for alpha followed by X+Y for beta and then X-Y or alpha for both first followed by beta for both? 
            lines = readfile('sing_a')
            X_plus_Y = TDM()
            X_plus_Y.setTDM(NBR_of_virt,NBR_of_occ)
            X_minus_Y = TDM()
            X_minus_Y.setTDM(NBR_of_virt,NBR_of_occ)
            current_state = 1
            TDM_found = False
            X_plus_Y_Done = False
            Time_to_calculate_TDM = False
            for line in lines:
                if 'tensor space dimension' in line:
                    words = line.split()
                    X_plus_Y.tensorspace = int(words[-1])
                elif 'eigenvalue' in line:
                    words = line.split()
                    X_plus_Y.eigenvalue = float(words[-1].replace('D','E'))
                    TDM_found = True
                    indx_occ  = 0
                    indx_virt = 0
                    continue
                if TDM_found:
                    words = [line[0+20*i:20+20*i] for i in range(4)]
                    for word in words:
                        if not X_plus_Y_Done:
                            X_plus_Y.TDM[indx_virt,indx_occ] = float(word.replace('D','E'))
                            indx_virt += 1
                            if indx_virt == NBR_of_virt:
                                indx_occ += 1
                                indx_virt = 0
                                if indx_occ == NBR_of_occ:
                                    X_plus_Y_Done = True
                                    indx_occ  = 0
                                    indx_virt = 0  
                        else:
                            X_minus_Y.TDM[indx_virt,indx_occ] = float(word.replace('D','E'))
                            indx_virt += 1
                            if indx_virt == NBR_of_virt:
                                indx_occ += 1
                                indx_virt = 0
                                if indx_occ == NBR_of_occ:
                                    current_state += 1
                                    TDM_found = False
                                    Time_to_calculate_TDM = True
                                    X_plus_Y_Done = False
                                    break
                if Time_to_calculate_TDM:
                    HelpingTDM = (X_plus_Y + X_minus_Y)
                    HelpingTDM.TDM = HelpingTDM.TDM/2
                    self.All_TDMs.append(HelpingTDM.copy())
                    Time_to_calculate_TDM = False
        
        os.chdir(prevdir)
        #later add function in the case of non hypbrid functionals. Plus since Turbomole 7.8 one can write responsevector in binary format $mosbinary, need to add this maybe ?
        return

    def writeQMout(self): #Write QM.out file
        
        matrix_dimension = self.QMInput.number_of_singlets + 3*self.QMInput.number_of_triplets + 1
        #print hamilton
        string = ''
        string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' %(1, matrix_dimension, matrix_dimension)
        string += '%i %i\n' % (matrix_dimension, matrix_dimension)
        for i in self.hamiltonian:
            for j in i:
                string += '%s ' %(eformat(j, 12, 3))
            string += '\n'
        string += '\n'
        #print dipole matrix
        string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, matrix_dimension, matrix_dimension)
        for i in self.all_diplote_matrices:
            string += '%i %i\n' % (matrix_dimension, matrix_dimension)
            for j in i:
                for k in j:
                    string += '%s ' %(eformat(k, 12, 3))
                string += '\n'
        string += '\n'       
        writefile('QM.out', string)
        os.popen('cp QM.out ' + self.QMInput.savedir+'QM.out')
        
        
        #if self.QMInput.overlap: #start writing ci determinants
            
        nocc = self.nbr_of_occupied_orbitals
        nvirt = self.nbr_of_scf_basis_functions - self.nbr_of_occupied_orbitals
        # Det file header
        string = ''
        string += str(self.QMInput.number_of_singlets+1)  + ' ' + str(nvirt+nocc) + ' ' + str(nocc*nvirt*2+1) + '\n'
            
        # generate a base string, which is modified for each determinant
        base_determinant = []
        for i in range(0,nvirt+nocc):
            if i < nocc:
                base_determinant.append('d')
            else:
                base_determinant.append('e')
        
        string += list2string(base_determinant) + '  '
        string += self.plotCIVector(0,0,Groundstate=True) + ' \n' #Ground sate determinant plus CI vector
        
        for j in range(nocc,0,-1):
            for k in range(0,nvirt):
                #edit string with contributing orbitals and put alpha/beta at the correct position
                helping_string = copy.deepcopy(base_determinant)
                helping_string[j-1] = 'a'
                helping_string[nocc+k] = 'b'
                string += list2string(helping_string) + '  '
                string += self.plotCIVector(j,k)
                string += '\n'
                        
                #and now the reverse determinant wich alpha and beta swapped
                helping_string = copy.deepcopy(base_determinant)
                helping_string[nocc+k] = 'a'
                helping_string[j-1] = 'b'
                string += list2string(helping_string) + '  '
                string += self.plotCIVector(j,k,Groundstate=False,invert=True)
                string += '\n'
        
        # save lines in the correspondig directory in the case of the init calculation        
        if not self.QMInput.inital_calc:
            writefile(self.QMInput.savedir+'det.DSPL', self.truncate_ci(string.splitlines(),1))
        else: 
            writefile(self.QMInput.savedir+'det', self.truncate_ci(string.splitlines(),1))
        # Same procedure again for the triplets    
        if self.QMInput.number_of_triplets != 0:
            
            string = ''
            string += str(self.QMInput.number_of_triplets)  + ' ' + str(nvirt+nocc) + ' ' + str(nocc*nvirt*2) + '\n'
                
            base_determinant = []
            for i in range(0,nvirt+nocc):
                if i < nocc:
                    base_determinant.append('d')
                else:
                    base_determinant.append('e')
            
            for j in range(nocc,0,-1):
                for k in range(0,nvirt):
                    #edit string with contributing orbitals
                    helping_string = copy.deepcopy(base_determinant)
                    helping_string[j-1] = 'a'
                    helping_string[nocc+k] = 'b'
                    string += list2string(helping_string) + '  '
                    string += self.plotCIVector(j,k,Groundstate=False,multiplicity=3)
                    string += '\n'
                            
                    #and now the reverse determinant wich alpha and beta swapped
                    helping_string = copy.deepcopy(base_determinant)
                    helping_string[nocc+k] = 'a'
                    helping_string[j-1] = 'b'
                    string += list2string(helping_string) + '  '
                    string += self.plotCIVector(j,k,Groundstate=False,multiplicity=3)
                    string += '\n'
            
            if not self.QMInput.inital_calc:
                writefile(self.QMInput.savedir+'det.DSPL.trip', self.truncate_ci(string.splitlines(),3))
            else:
                writefile(self.QMInput.savedir+'det.trip', self.truncate_ci(string.splitlines(),3))
        
        return

    def plotCIVector(self,indx_occ,indx_virt,Groundstate=False,invert=False,multiplicity=1): #generate CI vector from transition desnity matrix
        CIVector = []
        # Which information should be used ?
        if multiplicity == 1:
            Matrix = self.All_TDMs
            nbr_of_coefficients = self.QMInput.number_of_singlets
        elif multiplicity == 3:
            Matrix = self.All_TDMs_triplet
            nbr_of_coefficients = self.QMInput.number_of_triplets
        # for the Groundstate
        if Groundstate:
            CIVector.append(eformat(1, 12, 3))
            CIVector.append(' ')
            for i in range(0,nbr_of_coefficients):
                CIVector.append(eformat(0, 12, 3))
                CIVector.append(' ')
            return list2string(CIVector)
        elif invert:
            if multiplicity == 1:
                CIVector.append(eformat(0, 12, 3))
            CIVector.append(' ')
            for i in Matrix:
                value = i.TDM[indx_virt][indx_occ-1]*np.sqrt(0.5)*(-1)
                CIVector.append(eformat(value, 12, 3))
                CIVector.append(' ')
            return list2string(CIVector)
        else:
            if multiplicity == 1:
                CIVector.append(eformat(0, 12, 3))
            CIVector.append(' ')
            for i in Matrix:
                value = i.TDM[indx_virt][indx_occ-1]*np.sqrt(0.5)
                CIVector.append(eformat(value, 12, 3))
                CIVector.append(' ')
            return list2string(CIVector)
    
    def calculate_Overlap(self): #sets up input for wf overlap, calls it and saves it in QMout
        
        prevdir = os.getcwd()
        os.chdir(self.QMInput.savedir)   
        print('Calculating overlap matrix in: ' + os.getcwd())
        
        string = ''
        string += 'a_mo=mos.DSPL_000_eq\n'
        string += 'a_mo_read=2\n'
        string += 'b_mo=mos.DSPL\n'
        string += 'b_mo_read=2\n'
        string += 'a_det=det.DSPL_000_eq\n'
        string += 'b_det=det.DSPL\n'
        string += 'mix_aoovl=AOovl\n'
        if self.QMInput.frozen_core:
            string += 'ncore=' + str(self.QMInput.ncore) + '\n'
        
        
        writefile('wfo.in', string)
        
        run_WFOVLP(self.QMInput)
        
        if self.QMInput.number_of_triplets != 0:
            print('Calculating triplet overlap matrix in: ' + os.getcwd())
        
            string = ''
            string += 'a_mo=mos.DSPL_000_eq\n'
            string += 'a_mo_read=2\n'
            string += 'b_mo=mos.DSPL\n'
            string += 'b_mo_read=2\n'
            string += 'a_det=det.DSPL_000_eq.trip\n'
            string += 'b_det=det.DSPL.trip\n'
            string += 'mix_aoovl=AOovl\n'
            if self.QMInput.frozen_core:
                string += 'ncore=' + str(self.QMInput.ncore) + '\n'
        
            writefile('wfo.in', string)
            
            run_WFOVLP(self.QMInput,multiplicity = 3)
    
    
        self.wf_ovlp = read_WF_OVLP(self.QMInput)
    
        os.chdir(prevdir)  
        return
    
    def truncate_ci(self,det, multiplicity = 1): #truncates det files and save it in QM.out
        
        if multiplicity == 1:
            nbr_of_states = self.QMInput.number_of_singlets + 1
        elif multiplicity == 3:
            nbr_of_states = self.QMInput.number_of_triplets
        # First find wich dets are important by sorting and rearranging them
        which_dets_to_keep = []
        for CI_vector_index in range(0,nbr_of_states):
            CI_vector = []
            for line_index, line in enumerate(det):
                if line_index == 0:
                    continue
                CI_vector.append([line_index,float(line.split()[CI_vector_index+1])])
            CI_vector_squared = [[index, item ** 2] for index, item in CI_vector]
            CI_vector_sorted = sorted(CI_vector_squared, key=lambda x: x[1], reverse=True)
            
            Norm = 0
            for item_index, item in CI_vector_sorted:
                Norm += item
                which_dets_to_keep.append(item_index)
                if multiplicity == 1:
                    if Norm >= self.QMInput.wfthres:
                        if len(which_dets_to_keep)%2 == 1:
                            break
                        else:
                            continue
                elif multiplicity == 3:
                    if Norm >= self.QMInput.wfthres:
                        if len(which_dets_to_keep)%2 == 0:
                            break
                        else:
                            continue
        # Now write new det file with only the relevant lines    
        which_dets_to_keep = sorted(list(set(which_dets_to_keep)))    
        
        for item_index, det_index in enumerate(copy.deepcopy(which_dets_to_keep)):
            det_string = list(det[det_index].split()[0])
            if 'a' in det_string[0:int(self.QMInput.ncore)] or 'b' in det_string[0:int(self.QMInput.ncore)]:
                which_dets_to_keep.remove(det_index)
                
        nbr_of_dets = len(which_dets_to_keep)
        
        det_truncated = ''
        line_indx = 0
        for line in det:
            if line_indx == 0:
                header = line.split()
                header[-1] = str(nbr_of_dets)
                list2string(header,True) + '\n'
                det_truncated +=  list2string(header,True) + '\n'
                line_indx += 1
                continue
            if line_indx in which_dets_to_keep:
                det_truncated += line +'\n'
            line_indx += 1 
        
        return det_truncated
    
    
    def truncate_ci_simple(self,equlibrium_det_filename,displaced_det_filename): #first attempt at a more simple truncation, where all entries with small norm are discarded
        
        equlibrium_det = readfile(equlibrium_det_filename)
        displaced_det = readfile(displaced_det_filename)
        
        equlibrium_det_truncated = ''
        displaced_det_truncated = ''
        
        nbr_of_deleted_dets = 0
        line_indx = 0
        for line in equlibrium_det:
            if line_indx == 0:
                equlibrium_det_truncated += equlibrium_det[line_indx]
                displaced_det_truncated += displaced_det[line_indx]
                line_indx += 1
                continue
            
            eq_ci_vec = equlibrium_det[line_indx].split()[1:]
            dis_ci_vec = displaced_det[line_indx].split()[1:]
            
            keep_entry = False
            for item in eq_ci_vec:
                if float(item) > 1 - self.QMInput.wfthres:
                    keep_entry = True
            for item in dis_ci_vec:
                if float(item) > 1 - self.QMInput.wfthres:
                    keep_entry = True            
            if keep_entry:
                equlibrium_det_truncated += equlibrium_det[line_indx]
                displaced_det_truncated += displaced_det[line_indx]
            else:
                nbr_of_deleted_dets += 1
            
            line_indx += 1
    
        det_part = equlibrium_det_truncated.split('\n')
        header = det_part[0].split()
        header[-1] = str(int(header[-1])-nbr_of_deleted_dets)
        det_part = '\n'.join(det_part[1:])
        equlibrium_det_truncated = list2string(header,True) + '\n' + list2string(det_part)
        
        det_part = displaced_det_truncated.split('\n')
        header = det_part[0].split()
        header[-1] = str(int(header[-1])-nbr_of_deleted_dets)
        det_part = '\n'.join(det_part[1:])
        displaced_det_truncated = list2string(header,True) + '\n' + list2string(det_part)
        
        writefile(equlibrium_det_filename, equlibrium_det_truncated)
        writefile(displaced_det_filename, displaced_det_truncated)
        return
    
    def writeOVLP(self): #Write Overlap matrix to QM.out
        self.wf_ovlp = separate_complex(self.wf_ovlp)
        nmstates = self.QMInput.number_of_singlets + 1 +3*self.QMInput.number_of_triplets
        #print overlap
        string = ''
        string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
        string += '%i %i\n' % (nmstates, nmstates)
        for i in self.wf_ovlp:
            for j in i:
                string += '%s ' %(eformat(j, 12, 3))
            string += '\n'
        string += '\n'
        with open('QM.out','a') as file:
            file.write(string)
        return 
    
    def parsePROPER(self): #parse proper for spherical SOCs
        
        path = self.QMInput.scratchdir
        prevdir = os.getcwd()
        os.chdir(path)
        
        SOCs = []
        lines = readfile('tddft-socme-data.spherical.dat')
        current_SOC = []
        for line_index, line in enumerate(lines):
            
            if line_index%2 == 0:
                current_SOC = listofstring2float(line.split()[5:-1])
            else:
                for item_index, item in enumerate(line.split()[:-1]):
                    replaced_item = item.replace('D', 'E').replace('i', 'j')        
                    current_SOC[item_index] =  current_SOC[item_index] + complex(replaced_item)
                SOCs.append(current_SOC)
                current_SOC = []
        self.SOCs = np.matrix(SOCs) 
        os.chdir(prevdir)
        return

    def parsePROPER_cartesian(self): #parse proper for cartesian SOCs <- leads to instabilities with MCTDH
        
        path = self.QMInput.scratchdir
        prevdir = os.getcwd()
        os.chdir(path)
        
        SOCs = []
        lines = readfile('tddft-socme-data.cartesian.dat')
        for line in lines:
            SOCs.append(listofstring2float(line.split()[5:-1]))
            
        self.SOCs = np.matrix(SOCs) 
        os.chdir(prevdir)
        return

    
    def combine_S_and_T_Hamilton(self): # ESCF output gets parsed 2 times for S and T. Now merge final result before writing it to QM.out
        # For different multiplicities this has to be generalized...
        Singlet = merge_complex(self.hamiltonian)
        old_Triplet = merge_complex(self.hamiltonian_triplet)
        Triplet = MakeMatrix(old_Triplet.shape[0]-1, old_Triplet.shape[1]-1, complex)
        
        for i_indx,i in enumerate(old_Triplet): # remove the Groundstate
            for j_indx, j in enumerate(i):
                if i_indx == 0 or j_indx == 0:
                    continue
                Triplet[i_indx-1][i_indx-1] = old_Triplet[i_indx][i_indx]
        
        matrix_dimension = self.QMInput.number_of_singlets + 3*self.QMInput.number_of_triplets + 1
        Hamilton = MakeMatrix(matrix_dimension, matrix_dimension, complex)
        
        Hamilton[:self.QMInput.number_of_singlets+1, :self.QMInput.number_of_singlets+1] = Singlet #set up correct diagonal and the correspondig blocks
        Hamilton[self.QMInput.number_of_singlets+1:self.QMInput.number_of_singlets+1+self.QMInput.number_of_triplets, self.QMInput.number_of_singlets+1:self.QMInput.number_of_singlets+1+self.QMInput.number_of_triplets] = Triplet
        Hamilton[self.QMInput.number_of_singlets+1+self.QMInput.number_of_triplets:self.QMInput.number_of_singlets+1+2*self.QMInput.number_of_triplets, self.QMInput.number_of_singlets+1+self.QMInput.number_of_triplets:self.QMInput.number_of_singlets+1+2*self.QMInput.number_of_triplets] = Triplet
        Hamilton[self.QMInput.number_of_singlets+1+2*self.QMInput.number_of_triplets:self.QMInput.number_of_singlets+1+3*self.QMInput.number_of_triplets, self.QMInput.number_of_singlets+1+2*self.QMInput.number_of_triplets:self.QMInput.number_of_singlets+1+3*self.QMInput.number_of_triplets] = Triplet
        
        #Add SOCs to the Hamiltonian, by building the two blocks first and inserting them into H later
        if self.QMInput.inital_calc:
            SOC_XYZ = []
            for coupling in self.SOCs.T:
                SOC = MakeMatrix(self.QMInput.number_of_singlets+1, self.QMInput.number_of_triplets, complex)
                for triplet_indx in range(0,self.QMInput.number_of_triplets):
                    singlet_index = 0
                    while singlet_index < self.QMInput.number_of_singlets+1:
                        SOC[singlet_index,triplet_indx] = coupling[0,triplet_indx*(self.QMInput.number_of_singlets+1)+singlet_index]
                        singlet_index += 1
            
                SOC_XYZ.append(copy.deepcopy(SOC))
            
            
            all_SOCs = MakeMatrix(self.QMInput.number_of_singlets+1, 3*self.QMInput.number_of_triplets, complex) # here one assigns <1,1> = x, <1,0> = y and <1,-1> = z. One can change this by changing the index of SOC_XYZ accordingly
            all_SOCs[:self.QMInput.number_of_singlets+1,:self.QMInput.number_of_triplets] = SOC_XYZ[0]
            all_SOCs[:self.QMInput.number_of_singlets+1,self.QMInput.number_of_triplets:2*self.QMInput.number_of_triplets] = SOC_XYZ[2]
            all_SOCs[:self.QMInput.number_of_singlets+1,2*self.QMInput.number_of_triplets:3*self.QMInput.number_of_triplets] = SOC_XYZ[1]
        
        
            Hamilton[:self.QMInput.number_of_singlets+1,self.QMInput.number_of_singlets+1:self.QMInput.number_of_singlets+1+3*self.QMInput.number_of_triplets] = all_SOCs
            Hamilton[self.QMInput.number_of_singlets+1:self.QMInput.number_of_singlets+1+3*self.QMInput.number_of_triplets,:self.QMInput.number_of_singlets+1] = all_SOCs.T.conj()
        
        
        self.hamiltonian = separate_complex(Hamilton)
        
        return
    
    def combine_S_and_T_Dipole(self): # Just like Hamilton above
        
        matrix_dimension = self.QMInput.number_of_singlets + 3*self.QMInput.number_of_triplets + 1
        new_dipole_matrix = MakeMatrix(matrix_dimension, 2*matrix_dimension,float)
        
        all_new_diplote_matrices = []
        
        
        for old_matrix in self.all_diplote_matrices:
            new_dipole_matrix[:self.QMInput.number_of_singlets+1, :2*(self.QMInput.number_of_singlets+1)] = old_matrix
            all_new_diplote_matrices.append(copy.deepcopy(new_dipole_matrix))
        
        
        self.all_diplote_matrices = all_new_diplote_matrices
        
        
        #For now only S-S Dipole as well as GS included. Is is it possible to calculate S1,S2, ... and T1,T2, ... Dipole ? What about Transition dipole from T to T ?
        
        return
        

class QM(): #Q.in Class. -> read all input data and prepare turbomole calculation

    requested_molden_file = False
    inital_calc = False
    overlap = False
    cleanup = False
    tda = False
    grid = None
    scfconv = None
    range_sep_functional = False
    frozen_core = False
    method = ['dft']
    number_of_singlets = 0
    number_of_triplets = 0
    define_input = ''
    
    
    def __str__(self): # Plot all Variables
        list = []
        for attr in vars(self):
            list.append([attr, getattr(self, attr)])
        return str(list)
    
    
    def setGeometry(self,Inputfile): # Read in Geometry file
        line_indx = 0
        self.geometry = []
        for line in Inputfile:
            if line_indx == 0:
                Nbr_of_atoms = int(line)
                self.geometry = [[Nbr_of_atoms]]
            if line_indx > Nbr_of_atoms+1:
                break
            if line_indx > 1:
                self.geometry.append(line.split())
            line_indx += 1
        self.Nbr_of_atoms = Nbr_of_atoms
        return
    
    def setParameters(self,Inputfile): # Read QM.in for parameters, most of them do nothing so far, since the script always performs the full routine
        for line in Inputfile:
            if 'molden' in line:
                self.requested_molden_file = True            
            if 'states' in line:
                self.number_of_singlets = int(line.split()[1])-1
                self.number_of_triplets = int(line.split()[-1])
            if 'init' in line:
                self.inital_calc = True
            if 'overlap' in line:
                self.overlap = True
            if 'cleanup' in line:
                self.cleanup = True
            if 'savedir' in line:
                self.savedir = line.split()[-1]
        return
    
    def readTemplate(self,Inputfile): # Read Template file
        for line in Inputfile:
            if 'basis' in line:# can either be basis all *basis set* or basis '*ATOM*' *basis set* '*ATOM*' *basis set* '*ATOM*' *basis set* ... <--- The basis sets musst be availabe in Turbomole
                self.basis = []
                indx = 0
                for item in line.split():
                    if indx%2 == 0:
                        indx += 1
                        continue
                    if line.split()[indx] == 'all':
                        self.basis.append([line.split()[indx], line.split()[indx+1]])
                        break
                    else:
                        self.basis.append(["\""+line.split()[indx]+"\"", line.split()[indx+1]])
                    indx += 1
            if 'auxbasis' in line: # same as above for cbas and jbas (RI approximation)
                self.auxbasis = []
                indx = 0
                for item in line.split():
                    if indx%2 == 0:
                        indx += 1
                        continue
                    if line.split()[indx] == 'all':
                        self.auxbasis.append([line.split()[indx], line.split()[indx+1]])
                        break
                    else:
                        self.auxbasis.append(["\""+line.split()[indx]+"\"", line.split()[indx+1]])
                    indx += 1
            if 'charge' in line:
                self.charge = line.split()[-1]      
            if 'rpacor' in line:
                self.rpacor = line.split()[-1]  
            if 'functional' in line: # functional *functional* <--- musst be available in Turbomole or functional LC-blyp alpha beta omega
                self.dft_functional = line.split()[1]
                if self.dft_functional == 'LC-blyp':#LC-BLYP as the only range seperated functional, one could generalize this
                    self.range_sep_functional = True
                    self.dft_alpha = line.split()[2]
                    self.dft_beta = line.split()[3]
                    self.dft_omega = line.split()[4]
            if 'grid' in line: # DFT grid 
                self.grid = line.split()[-1]
            if 'gw' in line:
                self.method.pop()
                self.method.append('gw')   
            if 'qpeiter' in line:
                self.qpeiter = line.split()[-1]
            if 'QP_energies' in line:
                self.QP_energies = [line.split()[-2],line.split()[-1]]
            if 'tda' in line:
                self.tda = True
            if 'scfconv' in line:
                self.scfconv = line.split()[-1]
            if 'ncore' in line:
                self.frozen_core = True
                self.ncore = 0
                skip_header = True
                for item in self.geometry:
                    if skip_header:
                        skip_header = False
                        continue
                    self.ncore += FROZENS[item[0]]
                    
        return
    def readResources(self,Inputfile): # Read additional information
        for line in Inputfile:
            if 'turbodir' in line:
                self.turbodir = line.split()[-1]
            if 'scratchdir' in line:
                # debugging for slurm:
                #self.scratchdir = line.split()[-1] + '/' +str(os.environ['USER']) + '-' + str(os.environ['SLURM_JOB_ID']) 
                self.scratchdir = line.split()[-1]
            if 'memory' in line:
                self.memory = line.split()[-1]     
            if 'ncpu' in line:
                self.ncpu = int(line.split()[-1])
            if 'dipolelevel' in line:
                self.dipolelevel = line.split()[-1]
            if 'wfoverlap' in line:
                self.wfoverlap = line.split()[-1]     
            if 'wfthres' in line: 
                self.wfthres = float(line.split()[-1])  
        
        return
    def set_up_initial_define(self): #Prepare Define by generating define.in.
        if self.method[0] == 'gw':
            with open("define.in", "w") as file:
                file.write('\n')
                file.write('\n')
                file.write('a coord\n')
                file.write('*\n')
                file.write('no\n')
                for item in self.basis:
                    file.write('b ' + list2string(item,True) + '\n')
                file.write('*\n')
                file.write('eht\n')
                file.write('\n')
                file.write(self.charge+'\n')
                file.write('\n')
                file.write('cc\n')
                file.write('memory ' + self.memory + '\n')
                file.write('cbas\n')
                for item in self.auxbasis:
                    file.write('b ' + list2string(item,True) + '\n')
                file.write('*\n')
                file.write('*\n')
                file.write('dft\n')
                file.write('on\n')
                if self.dft_functional != 'LC-blyp':
                    file.write('func '+ self.dft_functional + '\n')
                else:
                    file.write('func b3-lyp\n')
                if self.grid is not None:
                    file.write('grid '+ self.grid + '\n')
                file.write('*\n')
                file.write('ri\n')
                file.write('jbas\n')
                for item in self.auxbasis:
                    file.write('b ' + list2string(item,True) + '\n')
                file.write('*\n')
                file.write('on\n')
                file.write('*\n')
                file.write('scf\n')
                file.write('iter\n')
                file.write('500\n')
                if self.scfconv is not None:
                    file.write('conv\n')
                    file.write(self.scfconv+'\n')
                file.write('\n')
                file.write('gw\n')
                file.write('rigw on\n')
                file.write('contour on\n')
                file.write('memory ' + self.rpacor + '\n')
                file.write('mxdiis 16\n')
                if self.qpeiter.isdigit():
                    file.write('qpeiter ' + self.qpeiter + '\n')
                elif self.qpeiter == 'evgw':
                    file.write(self.qpeiter + ' on\n')
                else:
                    print('qpeiter Keyword missing or broken!')
                    sys.exit()
                file.write('*\n')
                file.write('ex\n')
                file.write('bse\n')
                if self.tda:
                    file.write('ciss\n')
                file.write('*\n')
                file.write('a ' + str(self.number_of_singlets) + '\n')
                file.write('*\n')
                file.write('*\n')
                file.write('q\n')
                file.close()
        elif self.method[0] == 'dft':
            with open("define.in", "w") as file:
                file.write('\n')
                file.write('\n')
                file.write('a coord\n')
                file.write('*\n')
                file.write('no\n')
                for item in self.basis:
                    file.write('b ' + list2string(item,True) + '\n')
                file.write('*\n')
                file.write('eht\n')
                file.write('\n')
                file.write(self.charge+'\n')
                file.write('\n')
                file.write('cc\n')
                file.write('memory ' + self.memory + '\n')
                file.write('cbas\n')
                for item in self.auxbasis:
                    file.write('b ' + list2string(item,True) + '\n')
                file.write('*\n')
                file.write('*\n')
                file.write('dft\n')
                file.write('on\n')
                if self.dft_functional != 'LC-blyp':
                    file.write('func '+ self.dft_functional + '\n')
                else:
                    file.write('func b3-lyp\n')
                if self.grid is not None:
                    file.write('grid '+ self.grid + '\n')
                file.write('*\n')
                file.write('ri\n')
                file.write('jbas\n')
                for item in self.auxbasis:
                    file.write('b ' + list2string(item,True) + '\n')
                file.write('*\n')
                file.write('on\n')
                file.write('*\n')
                file.write('scf\n')
                file.write('iter\n')
                file.write('500\n')
                if self.scfconv is not None:
                    file.write('conv\n')
                    file.write(self.scfconv+'\n')
                file.write('\n')
                file.write('ex\n')
                file.write('rpas\n')
                if self.tda:
                    file.write('ciss\n')
                file.write('*\n')
                file.write('a ' + str(self.number_of_singlets) + '\n')
                file.write('*\n')
                file.write('*\n')
                file.write('q\n')
                file.close()
        return
    
    def generateCoord(self): # Write coordinate file in angstroem and use x2t to generate coord. Turbomole needs to be loaded here
            with open("geometry.xyz", "w") as file:
                line_index = 0
                for line in self.geometry:
                    if line_index == 0:
                        for item in line: 
                            file.write(str(item)+' ')
                    if line_index == 1:
                        file.write('\n')
                        for item in line:    
                            if is_float(item):
                                file.write(str(float(item)*bohr2angstroem)+' ')
                            else:
                                file.write(str(item)+' ')
                        file.write('\n')
                        line_index += 1
                    else:
                        for item in line:
                            if is_float(item):
                                file.write(str(float(item)*bohr2angstroem)+' ')
                            else:
                                file.write(str(item)+' ')
                        file.write('\n')
                        line_index += 1
            file.close()
            os.popen('x2t geometry.xyz > coord')
            return
    
    def runDefine(self): #execute Define with define.in input and fix control file in the case of GW
    
        path = self.scratchdir
        # again debugging for slurm:
        mkdir(path)
        loop_indx = 0
        while True: # wait for system to create directory and define.in. Terrible implementation, but does the trick for now
            time.sleep(1)
            try:
                os.listdir(path)
            except FileNotFoundError:
                print("Not yet!")
                loop_indx += 1
                if loop_indx > 30:
                    print("Directory can not be found! Do you have the correct permissions?")
                    sys.exit()
            else:
                print("Directory found!")
                break
        os.popen('mv define.in ' + path + '/define.in')
        os.popen('mv coord ' + path + '/coord')
        prevdir = os.getcwd()
        os.chdir(path)        
        loop_indx = 0
        while True: 
            time.sleep(1)
            try:
                with open("define.in", "r"):
                    pass
            except FileNotFoundError:
                print("Not yet!")
                loop_indx += 1
                if loop_indx > 30:
                    print("Directory can not be found! Do you have the correct permissions?")
                    sys.exit()
            else:
                print("define.in found!")
                break
        runerror = runProgram('define<define.in', path, 'define.output')
        if runerror.returncode != 0:
            print('Seems like Tubomole Define is in severe trouble...')
            sys.exit()        
        os.popen('cp '+ prevdir + '/mos.init ' + path + '/mos')
        #Now fix format for GW by reading the entire file and making changes to ips, gap and contour
        if self.method[0] == 'gw':
            with open("control", "r") as f:
                lines = f.readlines()
            with open("control", "w") as f:
                for line in lines:
                    if 'ips' not in line and 'gap' not in line:
                        if 'contour' not in line:
                            f.write(line)
                        else:
                            f.write('  contour start='+ self.QP_energies[0] +' end='+ self.QP_energies[1] +' irrep=1 ! a \n')# one needs to take care of spin=1 or spin=2 if one wants to extend this to unrestricted calculations

            f.close()
        if self.range_sep_functional:#if fucntional == LC-BLYP then build LC-BLYP from scratch using the libxc library
            remove_section_in_control('control', 'dft')
            add_section_to_control('control', 'dft')
            add_option_to_control_section('control', 'dft','functional libxc XC_GGA_X_ITYH')
            add_option_to_control_section('control', 'dft','functional libxc add 1 XC_GGA_C_LYP')
            add_option_to_control_section('control', 'dft','functional libxc set-rangesep ' + self.dft_alpha + ' ' + self.dft_beta + ' ' + self.dft_omega)
            add_option_to_control_section('control', 'dft','functional libxc factors ' + self.dft_beta + ' 1' )
            add_option_to_control_section('control', 'dft','functional libxc extparams 1 ' + self.dft_omega)
            add_option_to_control_section('control', 'dft','gridsize   ' + self.grid)
        os.chdir(prevdir)
        return            
    
    def save_files(self): #copy mos, coord and molden file to calc directory from scratch directory
        prevdir = os.getcwd()
        os.chdir(self.scratchdir)
        writefile('tm2molden.in','\n\n\n')
        os.chdir(prevdir)
        runerror = runProgram('tm2molden<tm2molden.in', self.scratchdir, 'tm2molden.output')
        if runerror.returncode != 0:
            print('Seems like Tubomole Tm2molden is in severe trouble...')
            sys.exit()
        mkdir(self.savedir+'MOLDEN/')
        os.popen('cp ' + self.scratchdir + '/molden.input ' + self.savedir+'MOLDEN/molden.input')
        if not self.inital_calc:
            os.popen('cp ' + self.scratchdir + '/mos ' + self.savedir + '/mos.DSPL')
        else:
            os.popen('cp ' + self.scratchdir + '/mos ' + self.savedir + '/mos')
        os.popen('cp ' + self.scratchdir + '/coord ' + self.savedir)
        if not self.inital_calc:
            os.popen('cp ' + '../DSPL_000_eq/SAVE/coord '  + self.savedir +'/coord.old')
            os.popen('cp ' + '../DSPL_000_eq/SAVE/det '  + self.savedir +'/det.DSPL_000_eq')
            os.popen('cp ' + '../DSPL_000_eq/SAVE/mos ' + self.savedir +'/mos.DSPL_000_eq')
            if self.number_of_triplets !=0:
                os.popen('cp ' + '../DSPL_000_eq/SAVE/det.trip ' + self.savedir +'/det.DSPL_000_eq.trip')
        return
    

def is_float(element): #Is element a float?
    try:
        float(element)
        return True
    except ValueError:
        return False
    
def list2string(input_list,spacing=False): #converts a list of entries and generates a string
    string = ''
    if not spacing:
        for i in input_list:
            string += str(i)
    else: 
        for i in input_list:
            string += str(i)+" "
    return string     

def listofstring2float(input_list):# Turns a list of strings into a list of floats
    
    float_list = []
    for item in input_list:
        try:
            float_list.append(float(item.replace('D','E')))
        except ValueError:
            print('Error while converting strings into floats')
            sys.exit()
            return None
    return float_list

def find_entry(list_of_lists, target_item):
    
    for inner_list in list_of_lists:
        if inner_list[1] == target_item:
            return inner_list[0]  
    sys.exit('error in findig item in CI vector')

def print_header():
    
    sharcdir = os.environ['SHARC']
    turbodir = os.environ['TURBODIR']
    
    if len(sharcdir) != 0 or len(turbodir) != 0:
        print('Using SHARC location: ' + sharcdir + '\nand Turbomole location: ' + turbodir)
    else:
        print('either Sharc or Turbomole is not loaded!')
        sys.exit()
    print('Starting ESCF calculation at: ' + str(datetime.datetime.now()))
    return

def convert_time(seconds): #what time is it?
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

def eformat(f, prec, exp_digits): #rearange format to plot numbers
    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))
    
def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit()
    return out

def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit()

def mkdir(DIR):
    # mkdir the DIR, or clean it if it exists
   #DIR = DIR + '/Singlet'
    if os.path.exists(DIR):
        if os.path.isfile(DIR):
            print('%s exists and is a file!' % (DIR))
            sys.exit()
        elif os.path.isdir(DIR):
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            os.makedirs(DIR)
        except OSError:
            print('Can not create %s\n' % (DIR))
            sys.exit()


def runProgram(string, workdir, outfile, errfile=''):
    prevdir = os.getcwd()
    os.chdir(workdir)

    try:
        with open(os.path.join(workdir, outfile), 'w') as stdoutfile:
            if errfile:
                with open(os.path.join(workdir, errfile), 'w') as stderrfile:
                    runerror = sp.run(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
            else:
                runerror = sp.run(string, shell=True, stdout=stdoutfile, stderr=sp.STDOUT)
    except OSError as e:
        print('Call had some serious problems:', e)
        sys.exit()
    finally:
        os.chdir(prevdir)
    return runerror

#Modify control file
def add_section_to_control(path, section):
    # adds a section keyword to the control file, before $end
    # if section does not start with $, $ will be prepended
    if not section[0] == '$':
        section = '$' + section
    infile = readfile(path)

    # get iline of $end
    iline = -1
    while True:
        iline += 1
        line = infile[iline]
        if '$end' in line:
            break

    outfile = infile[:iline]
    outfile.append(section + '\n')
    outfile.extend(infile[iline:])
    writefile(path, outfile)
    return

def remove_section_in_control(path, section):
    # removes a keyword and its options from control file
    if not section[0] == '$':
        section = '$' + section
    infile = readfile(path)

    # get iline where section starts
    iline = -1
    while True:
        iline += 1
        if iline == len(infile):
            return
        line = infile[iline]
        if section in line:
            break

    # get jline where section ends
    jline = iline
    while True:
        jline += 1
        line = infile[jline]
        if '$' in line:
            break

    # splice together new file
    outfile = infile[:iline] + infile[jline:]
    writefile(path, outfile)
    return

def add_option_to_control_section(path, section, newline):
    if not section[0] == '$':
        section = '$' + section
    infile = readfile(path)
    newline = '  ' + newline

    # get iline where section starts
    iline = -1
    while True:
        iline += 1
        if iline == len(infile):
            return
        line = infile[iline]
        if section in line:
            break

    # get jline where section ends
    jline = iline
    while True:
        jline += 1
        line = infile[jline]
        if '$' in line:
            break
        # do nothing if line is already there
        if newline + '\n' == line:
            return

    # splice together new file
    outfile = infile[:jline]
    outfile.append(newline + '\n')
    outfile.extend(infile[jline:])
    writefile(path, outfile)
    return

def define_for_ao_ovl(path, QMin): #prepare seperate define.input to calculate AO Overlapp
    # write define input
    string = ''
    string = '\n'
    string += 'title: SHARC run\n'
    string += 'a coord\n'
    string += '*\n'
    string += 'no\n'
    for item in QMin.basis:
        string += 'b ' + list2string(item,True) + '\n'
    string += 'c\n all 0\n'#this works apparently, otherwise turbomole will crash with no output
    string += '*\n'
    string += 'eht\n'
    string += 'y\n'
    string += QMin.charge + '\n'
    string += 'y\n'
    string += '*'

    # string contains the input for define, now call it
    infile = os.path.join(path, 'define.input')
    writefile(infile, string)

    return


def get_AO_OVL(QMin): #same algorithm as ricc2 interface, just new format for ESCF
    # get double geometry

    path = QMin.scratchdir
    loop_indx = 0
    while True:
        time.sleep(1)
        try:
            with open(QMin.savedir + "/coord.old", "r"):
                pass
        except FileNotFoundError:
            print("Searching for coord.old in: " + os.getcwd() + QMin.savedir)
            loop_indx += 1
            if loop_indx > 30:
                print("Coord_old can not be found! Do you have the correct permissions?")
                sys.exit()
        else:
            break
    oldgeom = readfile(os.path.join(QMin.savedir, 'coord.old'))
    while True:
        time.sleep(1)
        try:
            with open(QMin.savedir + "/coord", "r"):
                pass
        except FileNotFoundError:
            print("Searching for coord in: " + os.getcwd() + QMin.savedir)
            loop_indx += 1
            if loop_indx > 30:
                print("Coord can not be found! Do you have the correct permissions?")
                sys.exit()
        else:
            break
    newgeom = readfile(os.path.join(QMin.savedir, 'coord'))
    prevdir = os.getcwd()
    os.chdir(QMin.scratchdir)
    string = '$coord\n'
    wrt = False
    for line in oldgeom:
        if '$coord' in line:
            wrt = True
            continue
        elif '$' in line:
            wrt = False
            continue
        if wrt:
            string += line
    for line in newgeom:
        if '$coord' in line:
            wrt = True
            continue
        elif '$' in line:
            wrt = False
            continue
        if wrt:
            string += line
    string += '$end\n'
    tofile = os.path.join(path, 'coord')
    writefile(tofile, string)
    
    mkdir('AO_OVL')
    os.chdir('./AO_OVL')
    path = path + '/AO_OVL'
    os.popen('cp ../coord .')
    
    #Prepare define and start it this will crash but generate enough input for dscf
    define_for_ao_ovl(path, QMin)
    #os.popen('mv ../define.input .')
    runerror = runProgram('define < define.input', path, 'define.output')
    
    controlfile = os.path.join(path, 'control')
    remove_section_in_control(controlfile, '$scfiterlimit')
    add_section_to_control(controlfile, '$scfiterlimit 0')
    add_section_to_control(controlfile, '$intsdebug 1 sao')
    add_section_to_control(controlfile, '$closed shells')
    add_option_to_control_section(controlfile, '$closed shells', 'a 1-2')
    add_section_to_control(controlfile, '$scfmo none')

    # write geometry again because define tries to be too clever with the double geometry
    tofile = os.path.join(path, 'coord')
    writefile(tofile, string)

    # call dscf, but also works with ridft
    string = 'dscf'
    runerror = runProgram(string, path, 'dscf.out')

    # get AO overlap matrix from dscf.out
    dscf = readfile(os.path.join(path, 'dscf.out'))
    for line in dscf:
        if ' number of basis functions   :' in line:
            nbas = int(line.split()[-1])
            break
    else:
        print('Could not find number of basis functions in dscf.out!')
        sys.exit(98)
    iline = -1
    while True:
        iline += 1
        line = dscf[iline]
        if 'OVERLAP(SAO)' in line:
            break
    iline += 1
    ao_ovl = MakeMatrix(nbas, nbas,float)
    x = 0
    y = 0
    while True:
        iline += 1
        line = dscf[iline].split()
        for el in line:
            ao_ovl[x][y] = float(el)
            ao_ovl[y][x] = float(el)
            x += 1
            if x > y:
                x = 0
                y += 1
        if y >= nbas:
            break
    # the SAO overlap in dscf output is a LOWER triangular matrix
    # hence, the off-diagonal block must be transposed

    # write AO overlap matrix to savedir
    string = '%i %i\n' % (nbas // 2, nbas // 2)
    for irow in range(nbas // 2, nbas):
        for icol in range(0, nbas // 2):
            string += '% .15e ' % (ao_ovl[icol][irow])          # note the exchanged indices => transposition
        string += '\n'
    
    
    os.chdir(prevdir)
    filename = os.path.join(QMin.savedir, 'AOovl')
    writefile(filename, string)
    
    return

#run different programs
def run_dscf(QMin):
    workdir = os.path.join(QMin.scratchdir)
    #if QMin.ncpu > 1:
    #    string = 'dscf_omp'
    #else:
    #    string = 'dscf'
    string = 'dscf'
    runerror = runProgram(string, workdir, 'dscf.out')
    if runerror.returncode != 0:
        print('Seems like Tubomole DSCF is in severe trouble...')
        sys.exit()

    return

def run_ridft(QMin):
    workdir = os.path.join(QMin.scratchdir)
    # if QMin.ncpu > 1:
    #     string = 'ridft_smp'
    # else:
    #     string = 'ridft'
    string = 'ridft'
    runerror = runProgram(string, workdir, 'ridft.out')
    if runerror.returncode != 0:
        print('Seems like Tubomole RIDFT is in severe trouble...')
        sys.exit()

    return

def run_ESCF(QMin,multiplicity=1):
    # if QMin.ncpu > 1:
    #     string = 'escf_smp'
    # else:
    #     string = 'escf'
    string = 'escf'
    if multiplicity == 1:
        workdir = os.path.join(QMin.scratchdir)
        runerror = runProgram(string, workdir, 'escf.out')
    else:
        workdir = os.path.join(QMin.scratchdir+'/Triplet')
        runerror = runProgram(string, workdir, 'escf_triplet.out')
    if runerror.returncode != 0:
        print('Seems like Tubomole ESCF is in severe trouble...')
        sys.exit()

    return

def run_GW(QMin):
    workdir = os.path.join(QMin.scratchdir)
    # if QMin.ncpu > 1:
    #     string = 'escf_smp -gw'
    # else:
    #     string = 'escf -gw'
    string = 'escf -gw'
    runerror = runProgram(string, workdir, 'gw.out')
    if runerror.returncode != 0:
        print('Seems like Tubomole ESCF GW is in severe trouble...')
        sys.exit()

    return

def run_BSE(QMin,multiplicity=1):
    # if QMin.ncpu > 1:
    #     string = 'escf_smp -bse'
    # else:
    #     string = 'escf -bse'
    string = 'escf -bse'
    if multiplicity == 1:
        workdir = os.path.join(QMin.scratchdir)
        runerror = runProgram(string, workdir, 'bse.out')
    else:
        workdir = os.path.join(QMin.scratchdir+'/Triplet')
        runerror = runProgram(string, workdir, 'bse_triplet.out')
    if runerror.returncode != 0:
        print('Seems like Tubomole ESCF BSE is in severe trouble...')
        sys.exit()

    return

def run_Proper(QMin):
    workdir = os.path.join(QMin.scratchdir)
    string = 'proper socme-1e'
    runerror = runProgram(string, workdir, 'proper.out')
    if runerror.returncode != 0:
        print('Seems like Tubomole PROPER is in severe trouble...')
        sys.exit()

    return

def run_WFOVLP(QMin,multiplicity=1):
    workdir = os.getcwd()
    string = '$SHARC/wfoverlap.x -m ' + QMin.memory + ' -f wfo.in'
    if multiplicity == 1:
        runerror = runProgram(string, workdir, 'wfoverlap.out')
    elif multiplicity == 3:
        runerror = runProgram(string, workdir, 'wfoverlap_triplet.out')
    if runerror.returncode != 0:
        print('Seems like Sharc wavefuntion overlap is in severe trouble...')
        sys.exit()

    return

def MakeMatrix(n,m,datatype=complex):
    matrix = np.zeros((n,m),dtype=datatype)
    return matrix

def separate_complex(matrix): #split complex matrix in entries with seperate real and imag part (in Sharc format)
    cols, rows = matrix.shape
    new_matrix = MakeMatrix(cols, 2*rows, float)
    for i in range(rows):
        for j in range(cols):
            real_part = matrix[i][j].real
            imag_part = matrix[i][j].imag
            new_matrix[i][2*j] = real_part
            new_matrix[i][2*j+1] = imag_part
    return new_matrix

def merge_complex(matrix): #merge seperated matrix back into complex form (undoes Sharc format)
    cols, rows = matrix.shape
    new_matrix = MakeMatrix(cols, int(rows/2), complex)
    row_indx = 0
    for i in range(cols):
        for j in range(rows):
            if row_indx % 2 != 0:
                row_indx += 1
                continue
            real_part = matrix[i][j]
            imag_part = matrix[i][j+1]
            new_matrix[i][int(j/2)] = real_part + imag_part*0j
            row_indx += 1
    return new_matrix


def build_all_dipole_matrices (matrix): #takes matrix with each row one dipole element and rearanges it in sharc format

    list_of_matrices = []
    dimension = matrix.shape
    helpingmatrix = MakeMatrix(dimension[0], dimension[0],float)
    
    for column in matrix.T:
        indx = 0
        for item in column:
            helpingmatrix[indx][0] = item
            helpingmatrix[0][indx] = item
            indx += 1
        asign_matrix = separate_complex(helpingmatrix)
        list_of_matrices.append(asign_matrix.copy())
    return list_of_matrices

def readQMin(QMinfilename):
    
    QMin = QM()
    QMin.setGeometry(QMinfilename)
    QMin.setParameters(QMinfilename)
    QMin.readTemplate(readfile('ESCF.template'))
    QMin.readResources(readfile('ESCF.resources'))
    
    return QMin

def runGWBSECalculation(QMin): #run ridft, escf -gw, escf -bse and read in all the data
    
    QMout = QMoutput(QMin)
    
    QMin.set_up_initial_define()
    QMin.generateCoord()
    QMin.runDefine()
    
    ts1 = time.time()
    run_ridft(QMin)
    ts2 = time.time()
    print('RIDFT Done after: ' + convert_time(ts2-ts1))
    run_GW(QMin)
    ts1 = time.time()
    print('GW Done after ' + convert_time(ts1-ts2))
    run_BSE(QMin)
    ts2 = time.time()
    print('BSE Done after ' + convert_time(ts2-ts1))
    
    #print(os.getcwd())
    # Now, prepare Input for Triplets
    if QMin.number_of_triplets != 0:
        print('Time for a Triplet Calculation!')
        if QMin.tda:
            remove_section_in_control(QMin.scratchdir+'/control', 'scfinstab ciss')
            remove_section_in_control(QMin.scratchdir+'/control', 'soes')
            add_section_to_control(QMin.scratchdir+'/control', 'scfinstab cist')
            add_section_to_control(QMin.scratchdir+'/control', 'soes')
            add_option_to_control_section(QMin.scratchdir+'/control', 'soes','  a            ' + str(QMin.number_of_triplets))
        else:
            remove_section_in_control(QMin.scratchdir+'/control', 'scfinstab rpas')
            remove_section_in_control(QMin.scratchdir+'/control', 'soes')
            add_section_to_control(QMin.scratchdir+'/control', 'scfinstab rpat')
            add_section_to_control(QMin.scratchdir+'/control', 'soes')
            add_option_to_control_section(QMin.scratchdir+'/control', 'soes','  a            ' + str(QMin.number_of_triplets))

        
        prevdir = os.getcwd()
        triplet_dir = QMin.scratchdir +'/Triplet' # Move triplet calculation into a separate directory
        mkdir(triplet_dir)
        os.chdir(triplet_dir)
        os.popen('cp ../auxbasis .')
        os.popen('cp ../basis .')
        os.popen('cp ../control .')
        os.popen('cp ../coord .')
        os.popen('cp ../energy .')
        os.popen('cp ../exspectrum .')
        os.popen('cp ../HOMO .')
        os.popen('cp ../LUMO .')
        os.popen('cp ../moments .')
        os.popen('cp ../mos .')
        os.popen('cp ../qpenergies.dat .')
        time.sleep(1)
        ts1 = time.time()
        run_BSE(QMin,multiplicity=3)
        ts2 = time.time()
        print('BSE triplet Done after ' + convert_time(ts2-ts1))
        if QMin.tda:
            os.popen('cp cist_a ../.')
        else:
            os.popen('cp trip_a ../.') 
        os.popen('cp bse_triplet.out ../.')
        os.chdir(prevdir)
        time.sleep(1)
        if QMin.inital_calc:
            run_Proper(QMin) #SOCs
            print('Proper Done!')
    
    QMin.save_files()
    QMout.parseESCF()
    if QMin.number_of_triplets != 0:
        QMout.parseESCF(multiplicity=3)
        if QMin.inital_calc:
            QMout.parsePROPER()
        QMout.combine_S_and_T_Hamilton()
        QMout.combine_S_and_T_Dipole()
        
        
    if QMin.overlap:
        get_AO_OVL(QMin)
    
    #QMout.parseMOs()
    QMout.parseTDM(QMout.QMInput.tda)
    if QMin.number_of_triplets != 0:
        QMout.parseTDM(QMout.QMInput.tda, multiplicity=3)
    
    return QMout

def runDFTCalculation(QMin): # Same as above
    
        QMout = QMoutput(QMin)
        
        QMin.set_up_initial_define()
        QMin.generateCoord()
        QMin.runDefine()
        
        ts1 = time.time()
        run_ridft(QMin)
        ts2 = time.time()
        print('RIDFT Done after: ' + convert_time(ts2-ts1))
        run_ESCF(QMin)
        ts1 = time.time()
        print('ESCF Done after ' + convert_time(ts1-ts2))
        
        #print(os.getcwd())
        # Now, prepare Input for Triplets
        if QMin.number_of_triplets != 0:
            print('Time for a Triplet Calculation!')
            if QMin.tda:
                remove_section_in_control(QMin.scratchdir+'/control', 'scfinstab ciss')
                remove_section_in_control(QMin.scratchdir+'/control', 'soes')
                add_section_to_control(QMin.scratchdir+'/control', 'scfinstab cist')
                add_section_to_control(QMin.scratchdir+'/control', 'soes')
                add_option_to_control_section(QMin.scratchdir+'/control', 'soes','  a            ' + str(QMin.number_of_triplets))
            else:
                remove_section_in_control(QMin.scratchdir+'/control', 'scfinstab rpas')
                remove_section_in_control(QMin.scratchdir+'/control', 'soes')
                add_section_to_control(QMin.scratchdir+'/control', 'scfinstab rpat')
                add_section_to_control(QMin.scratchdir+'/control', 'soes')
                add_option_to_control_section(QMin.scratchdir+'/control', 'soes','  a            ' + str(QMin.number_of_triplets))

            prevdir = os.getcwd()
            triplet_dir = QMin.scratchdir +'/Triplet'
            mkdir(triplet_dir)
            os.chdir(triplet_dir)
            os.popen('cp ../auxbasis .')
            os.popen('cp ../basis .')
            os.popen('cp ../control .')
            os.popen('cp ../coord .')
            os.popen('cp ../energy .')
            os.popen('cp ../moments .')
            os.popen('cp ../mos .')
            time.sleep(1) # Again, not pretty... Needs a fix
            ts1 = time.time()
            run_ESCF(QMin,multiplicity=3)
            ts2 = time.time()
            print('ESCF triplet Done after ' + convert_time(ts2-ts1))
            if QMin.tda:
                os.popen('cp cist_a ../.')
            else:
                os.popen('cp trip_a ../.')
            os.popen('cp escf_triplet.out ../.')
            os.chdir(prevdir)
            time.sleep(1)
            if QMin.inital_calc:
                run_Proper(QMin) #SOCs
                print('Proper Done!')
        
        QMin.save_files()
        QMout.parseESCF()
        if QMin.number_of_triplets != 0:
            QMout.parseESCF(multiplicity=3)
            if QMin.inital_calc:
                QMout.parsePROPER()
            QMout.combine_S_and_T_Hamilton()
            QMout.combine_S_and_T_Dipole()
            
            
        if QMin.overlap:
            get_AO_OVL(QMin)
        
        #QMout.parseMOs()
        QMout.parseTDM(QMout.QMInput.tda)
        if QMin.number_of_triplets != 0:
            QMout.parseTDM(QMout.QMInput.tda, multiplicity=3)
        
        return QMout

def read_WF_OVLP(QMin):#parse wfoverlap, should be similar to the ricc2 routine
    
    lines = readfile('wfoverlap.out')
    matrix = []
    
    for line in lines:
        if 'Overlap matrix <PsiA_i|PsiB_j>' in line:
            continue
        elif line.startswith('<'):
            row = [float(value) for value in line.split()[2:]]
            matrix.append(row)
        if 'Renormalized overlap matrix <PsiA_i|PsiB_j>' in line:
            break
    
    if QMin.number_of_triplets == 0:
        return np.array(matrix)
    else:
        lines = readfile('wfoverlap_triplet.out')
        matrix2 = []
        for line in lines:
            if 'Overlap matrix <PsiA_i|PsiB_j>' in line:
                continue
            elif line.startswith('<'):
                row = [float(value) for value in line.split()[2:]]
                matrix2.append(row)
            if 'Renormalized overlap matrix <PsiA_i|PsiB_j>' in line:
                break
        
        matrix_dimension = QMin.number_of_singlets + 3*QMin.number_of_triplets + 1
        combined_matrix = MakeMatrix(matrix_dimension, matrix_dimension, float)
        
        combined_matrix[:QMin.number_of_singlets+1,:QMin.number_of_singlets+1] = np.array(matrix)
        
        combined_matrix[QMin.number_of_singlets+1:QMin.number_of_singlets+1+QMin.number_of_triplets,QMin.number_of_singlets+1:QMin.number_of_singlets+1+QMin.number_of_triplets] = np.array(matrix2)
        combined_matrix[QMin.number_of_singlets+1+QMin.number_of_triplets:QMin.number_of_singlets+1+2*QMin.number_of_triplets,QMin.number_of_singlets+1+QMin.number_of_triplets:QMin.number_of_singlets+1+2*QMin.number_of_triplets] = np.array(matrix2)
        combined_matrix[QMin.number_of_singlets+1+2*QMin.number_of_triplets:QMin.number_of_singlets+1+3*QMin.number_of_triplets,QMin.number_of_singlets+1+2*QMin.number_of_triplets:QMin.number_of_singlets+1+3*QMin.number_of_triplets] = np.array(matrix2)
        
        return combined_matrix
        
def main():

    if len(sys.argv) != 2:
        print('Usage:\n./ESCF.py <QMin>\n')
        sys.exit()
    QMinfilename = sys.argv[1]

    print_header()
    
    QMinfilename = readfile(QMinfilename)
    
    # Read QMinfile
    QMin = readQMin(QMinfilename)
    
    #What calculation should we do today?
    if QMin.method[0] == 'gw':
        QMout = runGWBSECalculation(QMin)
    elif QMin.method[0] == 'dft':
        QMout = runDFTCalculation(QMin)
    else:
        print('No method?')
        sys.exit()
    QMout.writeQMout()
    
    #calculate overlap matrix and add to QM.out
    if QMin.overlap:
        QMout.calculate_Overlap()
        QMout.writeOVLP()
    print('SHARC_ESCF all done!')

    
if __name__ == '__main__':
         main()
