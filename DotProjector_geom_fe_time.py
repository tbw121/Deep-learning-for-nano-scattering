
import numpy as np
class Fixed_Heights(): #this class is generating DotProjector with atoms of equal height

 def __init__(self, TotalA, full_w):
  
   self.No_row = int(TotalA//2 + TotalA%2) # number of meta-atoms in a row of computational domain including half or quarter atoms
   self.pitch_w = full_w / 2.0
   self.pitch_h = full_w / TotalA
   m_c = full_w / TotalA
   m_c_centres = -self.pitch_w + m_c/2 + m_c*np.arange(0 , TotalA)
   m_c_centr_comp_dom = m_c_centres[np.where(m_c_centres>=-1e-12)]

 
   self.domain = []
   self.keys_fixed = {}
   self.keys_geom = {}   
   self.keys_source = {}
   self.keys_material = {}

   if(full_w == 7150.0):
       fixed_H = 850.0
       self.keys_source['wavelength'] =15.5e-07
       self.keys_material['n_m'] = 3.48
       self.keys_material['n_s'] = 1.444
   elif(full_w == 4340.0):
       fixed_H = 550.0
       self.keys_source['wavelength'] =9.4e-07
       self.keys_material['n_m'] = 3.70
       self.keys_material['n_s'] = 1.45
   else:
       fixed_H = 500.0
       self.keys_source['wavelength'] =8.5e-07
       self.keys_material['n_m'] = 3.80
       self.keys_material['n_s'] = 1.452
   

   low_bound_mesh_CD_trans, high_bound_mesh_CD_trans  = (self.keys_source['wavelength']*1e9)/2.5 , (self.keys_source['wavelength']*1e9)/9.0
   low_bound_mesh_grat, high_bound_mesh_grat = low_bound_mesh_CD_trans * (1.0/self.keys_material['n_m']), high_bound_mesh_CD_trans*(1.0/self.keys_material['n_m'])
   low_bound_mesh_subs_z, high_bound_mesh_subs_z = low_bound_mesh_CD_trans* (1.0/self.keys_material['n_s']), high_bound_mesh_CD_trans* (1.0/self.keys_material['n_s'])
   low_bound_mesh_air_z, high_bound_mesh_air_z  = (self.keys_source['wavelength']*1e9)/1.5 , (self.keys_source['wavelength']*1e9)/9.0

   self.keys_geom['pitch_w'] = self.pitch_w
   self.keys_geom['pitch_h'] = self.pitch_h
   self.keys_geom['h_top'] =  100.0
   self.keys_geom['h_subst'] = 100.0
   
 
   y_co =  0.0
  
   for jj in range (0, self.No_row):
     x_co = m_c_centr_comp_dom[jj]
     
     x_tag = 'x' + str(jj+1)
     y_tag = 'y' + str(jj+1)
     
     ang_tag = 'ang' +  str(jj+1)
     H_tag = 'height' + str(jj+1)
     
     self.keys_fixed[ang_tag] = 0.0
     self.keys_fixed[H_tag] = fixed_H # fixed heights of atoms
     self.keys_fixed[x_tag] = x_co
     self.keys_fixed[y_tag] = y_co

       
   # Definition of the search domain
   self.design_space =[
       {'name': 'fem_degree', 'type': 'discrete', 'domain': [1,2,3]},
       {'name': 'slc_CD_trans', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_CD_trans), np.log(low_bound_mesh_CD_trans))},
       {'name': 'slc_grat1', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_grat), np.log(low_bound_mesh_grat))},
       {'name': 'slc_grat2', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_grat), np.log(low_bound_mesh_grat))},
       {'name': 'slc_grat3', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_grat), np.log(low_bound_mesh_grat))},
       {'name': 'slc_grat4', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_grat), np.log(low_bound_mesh_grat))},
       {'name': 'slc_grat5', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_grat), np.log(low_bound_mesh_grat))},
       {'name': 'slc_grat6', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_grat), np.log(low_bound_mesh_grat))},
       {'name': 'slc_subs_z', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_subs_z), np.log(low_bound_mesh_subs_z))},
       {'name': 'slc_air_z', 'type': 'continuous', 'domain': (np.log(high_bound_mesh_air_z), np.log(low_bound_mesh_air_z))}
    ]

                
            
               



