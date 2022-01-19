"""
    Format the xml so that it is readable by dm_control's parser.

    This class generates a duplicate of an environment model, which it 
    places in a file named 'temp_xml.xml', substituting 'include' 
    tags for the referenced contents.

    This is to overcome dm_control's limitation of not being able to read
    those 'include' tags.
"""

import os
import lxml.etree as ET


class xml_dmControl_formatting:
    def __init__(self, filename, outputName="temp_xml.xml"):
        # get in file we wish strip 'include' tags from and the output name
        self.filename   = filename
        self.output     = outputName

        # get the relative filepaths for each
        self.source     = self.__get_paths(os.path.dirname(__file__) + "/../models/" + filename)
        self.output     = self.__get_paths(os.path.dirname(__file__) + "/../models/" + outputName, error_=False)

        # remove the output file if it already exists
        # copy the data across from our source file
        self.__remove_if_exists(self.output)
        self.__copy_data_across(self.source, self.output)

        # check if there are any 'include' tags in the output file
        # -- remove them if there are, exchanging for the associate file content
        # -- check again if there are now any more 'include' tags
        # -- loop until there are no more 'include' tags
        isThere = self.__check_include_is_in_file(self.output)
        while (isThere):
            self.__format(self.output)
            isThere = self.__check_include_is_in_file(self.output)


    # get the paths of the files
    def __get_paths(self, file, error_=True):
        # Mujoco env parameters
        fullpath = file
        return fullpath
        

    # delete the temp file if it exists
    def __remove_if_exists(self, filename_and_path):
        if os.path.isfile(filename_and_path):   os.remove(filename_and_path)


    # copy all the data from the source to the new file
    # -- taken from sage88's post,
    #       https://stackoverflow.com/questions/15343743/copying-from-one-text-file-to-another-using-python 
    def __copy_data_across(self, source, output):
        with open(output, 'a') as f_out, open(source, 'r') as f_in:
            for line in f_in:
                f_out.write(line)


    # make sure there is at least one include in the file to substitute  
    def __check_include_is_in_file(self, output):
        # root of the temp_xml.xml tree
        root    = ET.parse(output).getroot()
        isThere = root.find('.//include')

        # return true is there is an include tag
        if isThere==None:
            return False
        else:
            return True

    
    # for every 'include' tag
    # -- open the associate file
    # -- delete the include tag
    # -- copy everything nested within the first 'body'
    # -- paste in-place of the 'include' tag
    # -- do for every 'include' which was in the file at the start of the process
    def __format(self, output):

        # root of the temp_xml.xml tree
        root    = ET.parse(output).getroot()

        # find each 'include' tag
        for i in root.iterfind(".//include"):
            # get the tag's associated file name
            include_filename = i.attrib['file']

            # find the file in the directory
            include_filename = self.__get_paths(os.path.dirname(__file__) + "/../models/" + include_filename)
            
            # get the root of the nested file 
            inner_root      = ET.parse(include_filename).getroot()

            # get the include tag's parent in our output file and remove the include tag
            include_parent  = i.getparent()
            include_parent.remove(i)

            # find the first 'body' tag in the nested file (this excludes the <mujoco/> tag)
            inner_body      = inner_root.find('body')

            # copy across the entire body to our source file
            include_parent.append(inner_body)

        # write the update to our source file (must be in binary 'wb')
        f = open(output, 'wb')
        f.write(ET.tostring(root, pretty_print=True))
        f.close()


    # return the output name of the file 
    def get_output_name(self):
        return self.output