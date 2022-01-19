"""
    Inherits form the default environment and uses a different .xml, allowing only sliding of the target_object instead of a freejoint.

    This class requires a different set of set of rewards and done calculations.
        -- These are done through calculating the table's area and checking whether or not the object is within the boundaries
        -- It assumes the table is rectangular
        -- This is necessary as there is no z-axis joint
"""


if __name__ == '__main__':
    from  RobotArm__BaseEnv import RobotArm
else:
    from .RobotArm__BaseEnv import RobotArm


# Slide class, generates the sliding environment
class RobotArmSlide(RobotArm):
    # user input
    def __init__(self, 
                 render             = True,
                 return_real        = True,      
                 return_tact        = True,       
                 return_kine        = True,       
                 return_raw         = False,
                 gel_width          = 224,         
                 gel_height         = 224,         
                 image_grayscale    = False,  
                 rescale            = 224,      
                 image_normalise    = True,
                 envcam_width       = 720,     
                 envcam_height      = 720,
                 frame_stack        = True,      
                 k_frames           = 3,
                 random_object      = False,   
                 random_target      = True,    
                 t_dist_threshold   = 0.07,
                 frame_skip         = 120,
                 seed               = 42):

        # load the base class's init to setup the environment
        super().__init__(
            render          = render,
            return_real     = return_real, 
            return_tact     = return_tact, 
            return_kine     = return_kine, 
            return_raw      = return_raw,
            gel_width       = gel_width, 
            gel_height      = gel_height, 
            envcam_width    = envcam_width,
            envcam_height   = envcam_height,
            image_grayscale = image_grayscale, 
            rescale         = rescale, 
            image_normalise = image_normalise,
            frame_stack     = frame_stack,    
            k_frames        = k_frames,
            has_object      = True,            
            random_object   = random_object,  
            has_target      = True,           
            random_target   = random_target,  
            t_dist_threshold= t_dist_threshold,
            calc_table_lims = True,
            model_name      = 'gripper_testbed_slide.xml',
            frame_skip      = frame_skip,
            seed            = seed
        )             


    # goals for this env - for use with hddqn
    # -- 0: up
    # -- 1: down
    # -- 2: left
    # -- 3: right
    def get_goals(self):
        return [0, 1, 2, 3, 4, 5]


    # intrinsic agent rewards for this env
    def get_intrinsic_reward(self, reward, goal):
        if self._closer_by_UDLR(goal):
            return True, -1
        
        return False, -2
    

    # reset all the metrics associated with the environment
    def _reset_metrics(self):
        # metric related attributes
        self.missing_surface_contact_time = 0  