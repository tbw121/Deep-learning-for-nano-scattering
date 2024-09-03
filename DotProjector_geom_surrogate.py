
import numpy as np
class Fixed_Heights(): #this class is generating DotProjector with atoms of equal height

 def __init__(self):
  
   TotalA = 11
   full_w = 7150.0
   self.No_row = 6 # number of meta-atoms in a row of computational domain including half or quarter atoms
   self.pitch_w = full_w / 2.0
   self.pitch_h = full_w / TotalA
   m_c = full_w / TotalA
   m_c_centres = -self.pitch_w + m_c/2 + m_c*np.arange(0 , TotalA)
   m_c_centr_comp_dom = m_c_centres[np.where(m_c_centres>=-1e-12)]

 
   self.domain = []
   self.keys_fixed = {}
   self.keys_geom = {}   
   self.keys_mesh = {}
   self.keys_source = {}
   self.keys_material = {}

   
   fixed_H = 850.0
   self.keys_source['wavelength'] =15.5e-07
   self.keys_material['n_m'] = 3.48
   self.keys_material['n_s'] = 1.444
   
   
   self.keys_mesh['slc_CD_trans'] =  np.exp(5.561191250626993)-35
   self.keys_mesh['slc_grat1'] = np.exp(4.772506064996035)-35
   self.keys_mesh['slc_grat2'] = np.exp(4.9098877834050825)-35
   self.keys_mesh['slc_grat3'] = np.exp(4.952459683002432)-35
   self.keys_mesh['slc_grat4'] = np.exp(4.706042343431929)-35
   self.keys_mesh['slc_grat5'] = np.exp(4.939225883627035)-35
   self.keys_mesh['slc_grat6'] = np.exp(4.994120302910986)-35
   self.keys_mesh['slc_subs_z'] = np.exp(5.449095891201055)-35
   self.keys_mesh['slc_air_z'] = np.exp(6.3494315119038305)-35

   
   
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
       
                    
   self.domain =  [{'name': 'radius1', 'type': 'continuous', 'domain': (180.0, 240.0)},
               {'name': 'radius2', 'type': 'continuous', 'domain': (180.0, 240.0)},
               {'name': 'radius3', 'type': 'continuous', 'domain': (180.0, 240.0)},
               {'name': 'radius4', 'type': 'continuous', 'domain': (140.0, 200.0)},
               {'name': 'radius5', 'type': 'continuous', 'domain': (200.0, 260.0)},
               {'name': 'radius6', 'type': 'continuous', 'domain': (180.0, 240.0)}
               ]



                
            
               



