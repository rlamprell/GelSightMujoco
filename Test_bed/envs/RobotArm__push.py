"""
    Inherits form the slide environment and uses a different .xml, allowing only sliding of the target_object instead of a freejoint.

    This class requires a different set of set of rewards and done calculations.
        -- These are done through calculating the table's area and checking whether or not the object is within the boundaries
        -- It assumes the table is rectangular
        -- The z-axis is present but restricted in the xml
"""


if __name__ == '__main__':
    from  RobotArm__BaseEnv import RobotArm
else:
    from .RobotArm__BaseEnv import RobotArm


# Push class, generates the pushing environment
class RobotArmPush(RobotArm):
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
            has_object      = True,            
            random_object   = random_object,  
            has_target      = True,           
            random_target   = random_target,  
            t_dist_threshold= t_dist_threshold,
            model_name      = 'gripper_testbed_push.xml',
            frame_skip      = frame_skip,
            seed            = seed
        )      


    # is our object still in contact with the table?
    # -- no z-axis joint so this checks if the object is above the positon coordinates of the table
    def _get_table_contact(self):
        # take the bot left and top right of the valid area (assumes a rectangle)
        # -- checks both x and y coordinates fall inbetween this range
        inside_x = (self.curr_targ_obj[0] > self.table_bot_left_lim[0] and 
                    self.curr_targ_obj[0] < self.table_top_right_lim[0])
        inside_y = (self.curr_targ_obj[1] > self.table_bot_left_lim[1] and 
                    self.curr_targ_obj[1] < self.table_top_right_lim[1])
        
        if inside_x and inside_y:
            return True
        
        return False

    
    # timing adjustment for contact
    # -- sometimes the model will throw out a false positional reading when the gripper
    #    comes into contact with the object, this is usually corrected within a step or two
    #    hence this method gives the calculations some leeway
    def _get_table_contact_with_leeway(self):
        
        if not self._get_table_contact():
            self.missing_surface_contact_time += 1
        else:
            self.missing_surface_contact_time  = 0
        
        if self.missing_surface_contact_time  >= 2:
            return False

        return True

   
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