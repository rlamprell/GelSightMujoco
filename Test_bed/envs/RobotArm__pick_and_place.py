"""
    Inherits form the slide environment and uses a different .xml, uses a freejoint like in the default env.

    This class requires a different set of set of rewards and done calculations.
        -- Not only does it require rewards related to the grasping and lifting the object, but also in moveing the object to the target position
        -- The done method requires similar updates as the reward method
"""


if __name__ == '__main__':
    from  RobotArm__BaseEnv import RobotArm
else:
    from .RobotArm__BaseEnv import RobotArm


# Pick and Place class, generates the pickandplace environment
class RobotArmPickAndPlace(RobotArm):
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
                 random_object      = True,   
                 random_target      = True,    
                 t_dist_threshold   = 0.07,
                 frame_skip         = 120,
                 seed               = 42
                 ):

        # load the base class's init to setup the environment
        super().__init__(
            render              = render,
            return_real         = return_real, 
            return_tact         = return_tact, 
            return_kine         = return_kine, 
            return_raw          = return_raw,
            gel_width           = gel_width, 
            gel_height          = gel_height, 
            envcam_width        = envcam_width,
            envcam_height       = envcam_height,
            image_grayscale     = image_grayscale, 
            rescale             = rescale, 
            image_normalise     = image_normalise,
            frame_stack         = frame_stack,    
            k_frames            = k_frames,
            has_object          = True,            
            random_object       = random_object,  
            has_target          = True,           
            random_target       = random_target,  
            t_dist_threshold    = t_dist_threshold,
            model_name          = 'gripper_testbed_PickandPlace.xml',
            frame_skip          = frame_skip,
            object_is_freejoint = True,
            seed                = seed
        )              


    # goals for this env - for use with hddqn
    # -- 0: up
    # -- 1: down (x or y-axis)
    # -- 2: left
    # -- 3: right
    # -- 4: grab
    # -- 5: pickup
    # -- 6: place (move down in the z-axis)
    def get_goals(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]


    # intrinsic agent rewards for this env
    def get_intrinsic_reward(self, reward, goal):
        # if moving up, down, left, or right
        if goal<=5:
            if self._closer_by_UDLR(goal):
                return True, -1
        # if grabbing
        elif goal==6:
            if reward>=-5 and reward<=-1:
                return True, -1
        # if releasing
        elif goal==7:
            if reward>=-2:
                return True, -1

        # goal not acheived
        return False, -2


    def _reset_metrics(self):
        # metric related attributes
        self.contact_time    = 0
        self.soft_grip_steps = 0
        self.hard_grip_steps = 0
        self.lifted_steps    = 0       