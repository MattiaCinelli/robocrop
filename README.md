# robocrop
[![LinkedIn](https://img.shields.io/badge/LinkedIn-MattiaCinelli-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mattia-cinelli-b8a06879/)
[![Medium](https://img.shields.io/badge/Medium-MattiaCinelli-green?style=flat-square&logo=medium)](https://mattia-cinelli.medium.com/)
<a class="github-button" href="https://github.com/MattiaCinelli" aria-label="Follow @MattiaCinelli on GitHub">Follow @MattiaCinelli</a>
<a class="github-button" href="https://github.com/MattiaCinelli/robocrop" data-icon="octicon-star" aria-label="Star MattiaCinelli/robocrop on GitHub">Star</a>

A toy project to practice the creation of gym-like environments. 

In this project a robot has to plant and harvest crops in different settings and challenges.

Clone and install the package
```bash
pip install -e .
```


## Version 1
<img src="commons\RobocropV1.svg" width=300 height=256 align='centre'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Harvest

## Under Construction
### Version 1.1
<img src="commons\RobocropV1.1.svg" width=550 height=256 align='centre'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Water -> Harvest


### Version 2
Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step

Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step