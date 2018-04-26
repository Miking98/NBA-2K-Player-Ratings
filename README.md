# Statistical Analysis of NBA 2K Player Rating Trends

Used Python to analyze trends in NBA 2K player ratings. 

Created beautifully color-coded histograms, bell curves, dot plots, boxplots, and line graphs to break down trends in player ratings by game year (2K13, 2K14, 2K15, 2K16) and position (PG, SG, SF, PF, C).

## Technology

Written in Python, relying heavily on the Numpy, Matplotlib, and Pandas libraries for statistical analysis and graphing.

## Images

Look in the `/plots` folder to view all the graphs generated by running `scripts.py`.

## Data

Parsed into a master Excel spreadsheet by `parse.py`. Manual adjusts needed to be made for nicknames, conflicting records, duplicates, etc.

Data taken from [here](http://hoopshype.com/2016/08/29/these-are-the-ratings-of-all-players-in-nba-2k17/) and [here](http://thereal2kinsider.blogspot.com/2012/09/official-nba-2k13-ratings.html). Unfortunately, this information is not very machine-friendly, so I compiled it into the files available in '/playerratingstuples'.

## Inspiration

I'm a huge NBA 2K fan and have played every game since 2K13 on the Xbox. Aside from the game itself, I appreciate the fact that instead of merely updating rosters every year and re-releasing the same game, like most sports video game franchises, 2K makes a concerted effort every year to improve the gameplay and realistically update player ratings based on the past season, which gives each year's release a unique feel. 

For example, in some years, a disproportionate amount of teams feel like they have really deep rosters, while in others there seems to be an oddly large number of really good shooting guards, or a significant number of players clustered around a 77-rating. Unsurprisingly, there's not a lot of peer-reviewed academic studies about trends in NBA 2K player ratings floating around on the Internet.

Luckily, however, I took AP Statistics during my senior year of high school. And during the year, being geniunely fascinated with what we were learning, I wanted to apply my knowledge to some computer science-related project outside of school, and use statistics to shed light on a real-world question. Also, I wanted to sharpen my Python skills, so the logical result was to create this project.
