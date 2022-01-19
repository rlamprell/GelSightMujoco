"""
    Inherits form the base environment and uses a different .xml - specifically this environment contains only the gripper
    and a target position, which the center of the gripper must reach.

    This class requires a different set of rewards and done calculations.
        -- These are done through calculating the table's area and checking whether or not the object is within the boundaries
        -- It assumes the table is rectangular
"""


if __name__ == '__main__':
    from  RobotArm__BaseEnv import RobotArm
else:
    from .RobotArm__BaseEnv import RobotArm


# Reach class, generates the reach environment
class RobotArm__Reach(RobotArm):
    # user input
    def __init__(self, 
                 render             = True,
                 return_real        = False,   
                 return_tact        = False,    
                 return_kine        = False,      
                 return_raw         = False,
                 gel_width          = 224,       
                 gel_height         = 224,       
                 image_grayscale    = False,  
                 rescale            = 128,      
                 image_normalise    = False,
                 envcam_width       = 720,    
                 envcam_height      = 720,
                 frame_stack        = False,   
                 k_frames           = 3,
                 random_target      = False, 
                 dist_threshold     = 0.07,
                 frame_skip         = 600,
                 seed               = 42
                 ):

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
            has_object      = False,            
            random_object   = False,  
            has_target      = True,           
            random_target   = random_target,  
            t_dist_threshold= dist_threshold,
            model_name      = 'gripper_testbed_reach.xml',
            frame_skip      = frame_skip,
            seed            = seed
        )              


    # goals for this env - for use with hddqn
    # -- 0: y+
    # -- 1: y-
    # -- 2: x+
    # -- 3: x-
    # -- 4: z+
    # -- 5: z-
    # -- 6: static
    def get_goals(self):
        return [0, 1, 2, 3, 4, 5]


    # intrinsic agent rewards for this env
    def get_intrinsic_reward(self, reward, goal):
        if self._closer_by_UDLR(goal):
            return True, 0
        
        return False, -1
        

    def _reset_metrics(self):
        # metric related attributes
        self.contact_time   = 0
        self.hard_grip_steps= 0
        self.lifted_steps   = 0 