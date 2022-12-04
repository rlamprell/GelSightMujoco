<h1 align="left">Solution to 'exa-data-eng-assessment'.</h1>

<h3 align="left">Project Description</h3>
<p align="left">
    This is a solution to the EMIS-group DE test found here: https://github.com/emisgroup/exa-data-eng-assessment.
    The solution is simple and uses Python to create a pipeline for unpacking json files and transferring them into a tabular format within a MySQL database.   Both of these are held within Docker containers.  
</p>

<h3 align="left">Installation</h3>
<p>
<ol>
  <li>Simply clone this Repo to your machine</li>
  <li>Open a terminal and navigate to its location</li>
  <li>Run:</li>
      
      docker-compose up --build
</ol>
    (this takes a few minutes)
</p>


<h3 align="left">Running The Pipeline</h3>
<p align="left">
    Open a terminal within the python_pipeline container
    and run:
    
    python3 main.py
</p>

<h3 align="left">Exploring the Database</h3>
<ul>
    <li>Open a terminal within the mysql container</li>
    <li>Enter:</li> 
        
        mysql -u root -p
</ul>
<ul>
    <li>Type the password (any MySQL Commands now work):</li>
    
        password
</ul>
<ul>
    <li>Type:
        
        USE emis_test_db;</li>
</ul>
<ul>
    <li>Type: 
        
        SHOW TABLES;</li>
</ul>
<ul>
    <li>Type (only works if you've main.py from above): 
    
        select * from Patient; </li>
</ul>
```lua
this line is very long
```
<h3 align="left">Languages and Tools:</h3>
<p align="left"> 
    <a href="https://www.python.org" target="_blank" rel="noreferrer"> 
        <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> 
    </a> 
    <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> 
        <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> 
    </a> 
</p>
