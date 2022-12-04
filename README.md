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
  <li>Start by running the command below (this takes a few minutes):</li>
      
      docker-compose up --build
</ol>
    
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
    <li>Type in the password:</li>
    
        password
</ul>
<ul>
    <li>Any MySQL Commands now work, for example:
        
        USE emis_test_db;
        SHOW TABLES;
        select * from Patient;</li>
</ul>







<h3 align="left">Languages and Tools:</h3>
<p align="left"> 
    <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> 
        <img src="https://skillicons.dev/icons?i=python,mysql,docker,git" />
    </a> 
</p>
