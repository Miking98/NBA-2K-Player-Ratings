''' Python Library Dependencies 
numpy
pandas
matplotlib
xlrd
openpyxl
scipy
pprint
'''

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import pprint as pp
from scipy.stats import norm
import openpyxl as op
import math
import random
import re

#----------
#----- FUNCS
#----------

#Matplotlib colors
colors = ['darkred', 'darkorange', 'darkgreen', 'darkblue', 'orchid']
colormap = ['cool', 'autumn', 'copper', 'pink', 'spring']
defaultCMap = cm.nipy_spectral

#Parameters
#	filename - String - name of file (no PATH info or extension needed)
def savePlot(fig, filename):
	path = "/Users/Student/Desktop/Dropbox/Fun Stuff/Ode to Odekirk/nba2kplayerratings/plots/"
	extension = ".png"
	fig.savefig(path+filename+extension)


def drawLineGraph(func):
	def drawgraph(pos, games, scatter, line, **kwargs):
		fig = plt.figure(figsize=(16,12))
		
		ax = fig.add_subplot(1, 1, 1)

		#Mins and maxes
		minXBound = 0
		maxXBound = len(games)-1 if line else len(games)+1
		minYBound = int(df[games].min().min())
		maxYBound = 100

			
		#Customize title, set position, allow space on top of plot for title
		ax.set_ylabel("Rating", fontdict = { "fontsize" : 10 })
		ax.set_xlabel("Game", fontdict = { "fontsize" : 10 })
		ax.set_xbound(lower=minXBound, upper=maxXBound)
		ax.set_ybound(lower=minYBound, upper=maxYBound)


		#Map each game year to an x-coord, and set x-tick labels as game year
		gameYearToXCoord = {}
		xtickLabels = []
		for gindex, g in enumerate(games):
			gameYearToXCoord[g] = gindex #gameYearToXCoord["2K13"] = 0, gameYearToXCoord["2K14"] = 1, etc.
			xtickLabels.append(g)
			if not line:
				gameYearToXCoord[g] += 1 #Shift over 1 if scatter plot b/c need extra padding

		#Add padding to left and right if scatter plot
		if not line:
			xtickLabels.insert(0, "")
			xtickLabels.append("")
		ax.set_xticks(range(0, maxXBound+1))
		#plt.locator_params(axis='x',nbins=len(xtickLabels))
		ax.set_xticklabels(xtickLabels)

		#Draw graph with unique function
		if 'a1' in kwargs and 'a2' in kwargs and 'b1' in kwargs and 'b2' in kwargs:
			fileName, figTitle = func(fig, ax, pos, games, gameYearToXCoord, kwargs['a1'], kwargs['a2'], kwargs['b1'], kwargs['b2'], scatter, line)
		elif 'a1' in kwargs and 'a2' in kwargs:
			fileName, figTitle = func(fig, ax, pos, games, gameYearToXCoord, kwargs['a1'], kwargs['a2'], scatter, line)
		else:
			fileName, figTitle = func(fig, ax, pos, games, gameYearToXCoord, scatter, line)

		fig.suptitle(figTitle, fontsize=18, fontweight="bold")

		savePlot(fig, "_".join(games)+"_"+"_".join(pos)+"_"+fileName+("_scatter" if scatter else "")+("_line" if line else ""))
	return drawgraph


def drawBoxPlotBarGraph(func):
	def drawgraph(pos, games):
		fig = plt.figure(figsize=(16,12))
		
		ax = fig.add_subplot(1, 1, 1)

		#Mins and maxes
		minXBound = 0
		maxXBound = len(games)
		minYBound = int(df[games].min().min())
		maxYBound = 100

			
		#Customize title, set position, allow space on top of plot for title
		ax.set_ylabel("Rating", fontdict = { "fontsize" : 10 })
		ax.set_xlabel("Game", fontdict = { "fontsize" : 10 })
		ax.set_xbound(lower=minXBound, upper=maxXBound)
		ax.set_ybound(lower=minYBound, upper=maxYBound)


		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		ax.get_yaxis().tick_right()

		#Draw graph with unique function
		fileName, figTitle = func(fig, ax, pos, games)

		fig.suptitle(figTitle, fontsize=18, fontweight="bold")

		savePlot(fig, "_".join(games)+"_"+"_".join(pos)+"_"+fileName+"_boxplot_bargraph")
	return drawgraph

#Parameters
#	pos - List - position names (e.g. ["PG", "SG"])
#	games - List - game title (e.g. ["2K14", "2K15"])
#Return
#	Plot of ratings for a position for each game year
def compareposwithingame_samegraphs(pos, games, histogram, dotplot, bestfit):

	fig = plt.figure(figsize=(16,12))
	fig.suptitle(", ".join(games)+" Player Ratings by Position", fontsize=18, fontweight="bold")

	#Find max frequency for a rating, to set y-upper of our graphs
	maxFreqForARating = 0
	for p in pos:
		for game in games:
			count = df[df["Position"]==p][game].value_counts().max()
			if count>maxFreqForARating:
				maxFreqForARating = count

	#Mins and maxes
	minXBound = int(df[games].min().min())
	maxXBound = 100
	minYBound = 0
	maxYBound = int(maxFreqForARating)

	i = 1

	for gameindex, game in enumerate(games):

		#Create subplot
		ax = fig.add_subplot(len(games), 1, i)
		i += 1

		histdata = []
		for pindex, p in enumerate(pos):
			#Get data
			df1 = df[np.isfinite(df[game])]
			players = df1[df1["Position"]==p] #Return players who match the requested position
			ratings = players[game].astype(int)

			#
			#Draw graph
			#

			if histogram:
				#Plot actual histogram
				if pindex==0:
					heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
				else:
					heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
				histdata.append([heights, bins])
				width = (histdata[0][1][1]-histdata[0][1][0])/(len(pos)+1)
				ax.bar(bins[:-1]+width*pindex, heights, width=width, facecolor=colors[pindex])	
			elif dotplot:
				data = ratings.values
				#Plot dot plot of histogram
				#Generate (x,y) coords for each dot
				x = []
				y = []
				width = float(1)/((len(pos)+1)) #Width of dotplot column
				data.sort()
				xvalprevious = None
				ycounter = 0
				for xval in np.nditer(data):
					if xvalprevious is not None and xval==xvalprevious:
						ycounter += 1
					else:
						ycounter = 1
					x.append(xval+width*(pindex-int(len(pos)/2))) #Shift xval over to make room for multiple games with xvals at same rating
					y.append(ycounter)
					xvalprevious = xval

				x = np.asarray(x) #Convert list to numpy array
				y = np.asarray(y)
				s = np.full((1, x.size), float(15)/len(pos))
				ax.scatter(x, y, marker='o', s=s, color=colors[pindex], alpha=1)

			if bestfit:
				#Plot best fit of histogram
				mean = ratings.mean()
				std = ratings.std()
				bestfit_x = np.arange(minXBound, maxXBound, .1)
				bestfit_y = norm.pdf(bestfit_x, mean, std)
				ax2 = ax.twinx()
				ax2.plot(bestfit_x, bestfit_y, linewidth=2, color=colors[pindex])
				plt.setp(ax2.get_yticklabels(), visible=False)
				ax2.yaxis.set_tick_params(size=0)

				#Plot mean line
				plt.axvline(mean, linestyle=('solid' if dotplot else 'dashed'), linewidth=3, color=colors[pindex])

			#Customize title, set position, allow space on top of plot for title
			ax.set_title(game)
			ax.set_xlabel("Rating", fontdict = { "fontsize" : 10 }) #Only have Rating label on last graph in column
			ax.set_ylabel("Number of Players", fontdict = { "fontsize" : 10 })
			ax.set_xbound(lower=minXBound, upper=maxXBound)	
			ax.set_ybound(lower=minYBound, upper=maxYBound)
			ax.set_xticks(np.arange(round(minXBound, -1), maxXBound+1, 5)) #Show major ticks every 5    

		#Legend
		#Set legend key for each position in graph
		handles = []
		for p2index, p2 in enumerate(pos):
			handles.append(mpatches.Patch(color=colors[p2index], label=p2))
		# Position legend in upper right corner
		leg = ax.legend(handles=handles,
						loc=1,
						ncol=1,
						fontsize=10, 
						frameon=True,
						fancybox=True,
						title='Legend')

	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	savePlot(fig, "_".join(games)+"_"+"_".join(pos)+"_poswithingame_samegraphs"+("_hist" if histogram else "")+("_dotplot" if dotplot else "")+("_bestfit" if bestfit else ""))


#Parameters
#	pos - List - position names (e.g. ["PG", "SG"])
#	games - List - game title (e.g. ["2K14", "2K15"])
#Return
#	Plot of ratings for a position for each game year
def compareposwithingame_separategraphs(pos, games, histogram, bestfit):

	fig = plt.figure(figsize=(16,12))
	fig.suptitle("Player Ratings by Position ("+", ".join(games)+")", fontsize=18, fontweight="bold")

	i = 1

	#Find max frequency for a rating, to set y-upper of our graphs
	maxFreqForARating = 0
	for p in pos:
		for game in games:
			count = df[df["Position"]==p][game].value_counts().max()
			if count>maxFreqForARating:
				maxFreqForARating = count

	#Mins and maxes
	minXBound = int(df[games].min().min())
	maxXBound = 100
	minYBound = 0
	maxYBound = int(maxFreqForARating)

	for p in pos:
		for game in games:

			#Get data
			df1 = df[np.isfinite(df[game])]
			players = df1[df1["Position"]==p] #Return players who match the requested position
			ratings = players[game].astype(int)

			#
			#Draw graph
			#

			#Create subplot
			ax = fig.add_subplot(len(pos), len(games), i)

			#Add game title to top of subplot column
			if (pos[0]==p):
				ax.set_title(game+"\n"+p, fontdict={ "fontsize" : 16 })
			else:
				ax.set_title(p, fontdict={ "fontsize" : 16 })

			i += 1

			if histogram:
				#Plot actual histogram
				ratings.plot(kind='hist', ax=ax, bins=100, color='blue')
			
			if bestfit:
				#Plot best fit of histogram
				mean = ratings.mean()
				std = ratings.std()
				bestfit_x = np.arange(minXBound, maxXBound, .1)
				bestfit_y = norm.pdf(bestfit_x, mean, std)
				ax2 = ax.twinx()
				ax2.plot(bestfit_x, bestfit_y, linewidth=2, color='darkred')
				plt.setp(ax2.get_yticklabels(), visible=False)
				ax2.yaxis.set_tick_params(size=0)

				#Plot mean line
				plt.axvline(mean, linestyle='dashed', linewidth=3, color='darkred')

			#Customize title, set position, allow space on top of plot for title
			ax.set_xlabel("Rating" if pos[len(pos)-1]==p else "", fontdict = { "fontsize" : 10 }) #Only have Rating label on last graph in column
			ax.set_ylabel("Number of Players", fontdict = { "fontsize" : 10 })
			ax.set_xbound(lower=minXBound, upper=maxXBound)	
			ax.set_ybound(lower=minYBound, upper=maxYBound)
			ax.set_xticks(np.arange(round(minXBound, -1), maxXBound+1, 5)) #Show major ticks every 5    

			# #Set legend keys
			key = mlines.Line2D([], [], 
								linewidth=0, markersize=0)
			# Set legend labels
			labels = [	"Range: "+str(int(ratings.min()))+"-"+str(int(ratings.max())),
						"Median: "+str(int(ratings.median())), 
						"Std "+str(int(ratings.std())), 
						"n = "+str(int(ratings.size))]

			# Position legend in upper right corner
			leg = ax.legend([key, key, key],  labels, 
							loc=1,
							ncol=1,
							fontsize=10, 
							frameon=True,
							fancybox=True,
							title='Stats')
			for t in leg.get_texts():
				t.set_ha("left")
				t.set_position((-20,0))
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	savePlot(fig, "_".join(games)+"_"+"_".join(pos)+"_poswithingame_separategraphs"+("_hist" if histogram else "")+("_dotplot" if dotplot else "")+("_bestfit" if bestfit else ""))

#Parameters
#	pos - List - position names (e.g. ["PG", "SG"])
#	games - List - game title (e.g. ["2K14", "2K15"])
#Return
#	Plot of ratings for a position for each game year
def compareposacrossgames(pos, games, histogram, dotplot, bestfit):

	fig = plt.figure(figsize=(16,12))
	fig.suptitle("Position Ratings by Game ("+", ".join(games)+")", fontsize=18, fontweight="bold")

	#Find max frequency for a rating, to set y-upper of our graphs
	maxFreqForARating = 0
	for p in pos:
		for game in games:
			count = df[df["Position"]==p][game].value_counts().max()
			if count>maxFreqForARating:
				maxFreqForARating = count

	#Mins and maxes
	minXBound = int(df[games].min().min())
	maxXBound = 100
	minYBound = 0
	maxYBound = int(maxFreqForARating)

	i = 1

	for p in pos:
		#Create subplot
		ax = fig.add_subplot(len(pos), 1, i)
		i += 1

		#Legend
		handles = []
		#Set legend key for each position in graph
		for gindex, g in enumerate(games):
			handles.append(mpatches.Patch(color=colors[gindex], label=g))
		# Position legend in upper right corner
		leg = ax.legend(handles=handles,
						loc=1,
						ncol=1,
						fontsize=10, 
						frameon=True,
						fancybox=True,
						title='Legend')

		histdata = []
		for gameindex, game in enumerate(games):
			#Get data
			df1 = df[np.isfinite(df[game])]
			players = df1[df1["Position"]==p] #Return players who match the requested position
			ratings = players[game].astype(int)

			#
			#Draw graph
			#

			if histogram:
				#Plot actual histogram
				if gameindex==0:
					heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
				else:
					heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
				histdata.append([heights, bins])
				width = (histdata[0][1][1]-histdata[0][1][0])/(len(games)+1)
				ax.bar(bins[:-1]+width*gameindex, heights, width=width, facecolor=colors[gameindex])
			elif dotplot:
				data = ratings.values
				#Plot dot plot of histogram
				#Generate (x,y) coords for each dot
				x = []
				y = []
				width = float(1)/((len(games)+1)) #Width of dotplot column
				data.sort()
				xvalprevious = None
				ycounter = 0
				for xval in np.nditer(data):
					if xvalprevious is not None and xval==xvalprevious:
						ycounter += 1
					else:
						ycounter = 1
					x.append(xval+width*(gameindex-int(len(games)/2))) #Shift xval over to make room for multiple games with xvals at same rating
					y.append(ycounter)
					xvalprevious = xval

				x = np.asarray(x) #Convert list to numpy array
				y = np.asarray(y)
				s = np.full((1, x.size), float(15)/len(games))
				ax.scatter(x, y, marker='o', s=s, color=colors[gameindex], alpha=1)

			if bestfit:
				#Plot best fit of histogram
				mean = ratings.mean()
				std = ratings.std()
				bestfit_x = np.arange(minXBound, maxXBound, .1)
				bestfit_y = norm.pdf(bestfit_x, mean, std)
				ax2 = ax.twinx()
				ax2.plot(bestfit_x, bestfit_y, linewidth=2, color=colors[gameindex])
				plt.setp(ax2.get_yticklabels(), visible=False)
				ax2.yaxis.set_tick_params(size=0)
				#Plot mean line
				plt.axvline(mean, linestyle=('solid' if dotplot else 'dashed'), linewidth=3, color=colors[gameindex])
			

		#Customize title, set position, allow space on top of plot for title
		ax.set_title(p, fontdict={ "fontsize" : 16 })
		ax.set_xlabel("Rating" if pos[len(pos)-1]==p else "", fontdict = { "fontsize" : 10 }) #Only have Rating label on last graph in column
		ax.set_ylabel("Number of Players", fontdict = { "fontsize" : 10 })
		ax.set_xbound(lower=minXBound, upper=maxXBound)
		ax.set_ybound(lower=minYBound, upper=maxYBound)
		ax.set_xticks(np.arange(round(minXBound, -1), maxXBound+1, 5)) #Show major ticks every 5                                                       

	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	savePlot(fig, "_".join(games)+"_"+"_".join(pos)+"_acrossgames"+("_hist" if histogram else "")+("_dotplot" if dotplot else "")+("_bestfit" if bestfit else ""))

def compareposacrossgames_boxplot(pos, games):

	fig = plt.figure(figsize=(16,12))
	fig.suptitle("Position Ratings by Game ("+", ".join(games)+")", fontsize=18, fontweight="bold")

	#Find max frequency for a rating, to set y-upper of our graphs
	maxFreqForARating = 0
	for p in pos:
		for game in games:
			count = df[df["Position"]==p][game].value_counts().max()
			if count>maxFreqForARating:
				maxFreqForARating = count

	#Mins and maxes
	minXBound = int(df[games].min().min())
	maxXBound = 100
	minYBound = 0
	maxYBound = int(maxFreqForARating)

	i = 1

	for p in pos:
		#Create subplot
		ax = fig.add_subplot(len(pos), 1, i)
		i += 1

		#Legend
		handles = []
		#Set legend key for each position in graph
		for gindex, g in enumerate(games):
			handles.append(mpatches.Patch(color=colors[gindex], label=g))
		# Position legend in lower lefthand corner
		leg = ax.legend(handles=handles,
						loc=3,
						ncol=1,
						fontsize=10, 
						frameon=True,
						fancybox=True,
						title='Legend')

		data = []
		for gameindex, game in enumerate(games):
			#Get data
			df1 = df[np.isfinite(df[game])]
			players = df1[df1["Position"]==p] #Return players who match the requested position
			data.append(players[game].astype(int).values)
		
		#
		#Draw graph
		#

		data.reverse() #Reverse order of data so that 2K14 is on top of graph, while 2K16 is on bottom

		#Plot boxplot
		bps = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False)

		#Color boxplot
		for box, c in zip(bps['boxes'], reversed(colors[0:len(games)])): #Reverse color order b/c we reversed data order
			plt.setp(box, color=c, linewidth=1.5, alpha=0.8)
		plt.setp(bps['whiskers'], color='black', linewidth=1, linestyle='solid')
		plt.setp(bps['caps'], color='black', linewidth=1.5)
		plt.setp(bps['medians'], color='black', linewidth=1.5)

		'''
		#Overlay scatterplot on top of boxplot
		for dindex, d in enumerate(data):
			#Eliminate duplicates and count how many of each x-value there are
			d.sort()
			dprevious = None
			x = []
			xcounts = []
			for datapoint in np.nditer(d):
				if dprevious is not None and datapoint==dprevious:
					xcounts[len(xcounts)-1] += 1
				else:
					x.append(datapoint)
					xcounts.append(1)
				dprevious = datapoint

			x = np.asarray(x) #Convert list to numpy array
			s = np.multiply(20, np.asarray(xcounts)) #Convert list to numpy array, then multiply to get bigger size
			y = np.full((1, x.size), dindex+1)[0] #Give all datapoints the same y-value so they are inline with boxplot
			ax.scatter(x, y, marker='d', color='darkblue', s=s, alpha=1, zorder=3)
		'''
		#Customize tick marks (rename them after game title, then move to the right)
		ax.yaxis.tick_right()
		ax.yaxis.set_ticks_position('both')
		plt.yticks(np.arange(1,len(games)+1), reversed(games))

		#Customize title, set position, allow space on top of plot for title
		ax.set_title(p, fontdict={ "fontsize" : 16 })
		ax.set_ylabel("")
		ax.set_xlabel("Rating" if pos[len(pos)-1]==p else "", fontdict = { "fontsize" : 10 })
		ax.set_xbound(lower=minXBound, upper=maxXBound)

	plt.tight_layout()
	plt.subplots_adjust(top=0.93)
	savePlot(fig, "_".join(games)+"_"+"_".join(pos)+"_acrossgames_boxplot")

def allratingsingame_separategraphs(games, a1, a2, histogram, dotplot, bestfit):
	fig = plt.figure(figsize=(16,12))
	fig.suptitle("All Ratings in "+", ".join(games)+(" for Ratings "+str(a1)+"-"+str(a2) if a1>0 and a2<100 else ""), fontsize=18, fontweight="bold")

	#Find max frequency for a rating, to set y-upper of our graphs
	maxFreqForARating = 0
	for game in games:
		players = df[(df[game]>=a1) & (df[game]<=a2)]
		count = players[game].value_counts().max()
		if count>maxFreqForARating:
			maxFreqForARating = count

	#Mins and maxes
	minXBound = int(df[games].min().min()) if a1==0 else a1-1
	maxXBound = 100 if a2==100 else a2+1
	minYBound = 0
	maxYBound = int(maxFreqForARating)+1

	for gameindex, game in enumerate(games):
		#Get data
		players = df[(df[game]>=a1) & (df[game]<=a2)]
		ratings = players[game].astype(int)

		#Draw graph
		ax = fig.add_subplot(len(games), 1, gameindex+1)

		if histogram:
			#Plot actual histogram
			heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
			width = (bins[1]-bins[0])/(len(games)+1)
			ax.bar(bins[:-1], heights, width=width, facecolor=colors[gameindex])			
		elif dotplot:
			data = ratings.values
			#Plot dot plot of histogram
			#Generate (x,y) coords for each dot
			x = []
			y = []
			data.sort()
			xvalprevious = None
			ycounter = 0
			for xval in np.nditer(data):
				if xvalprevious is not None and xval==xvalprevious:
					ycounter += 1
				else:
					ycounter = 1
				x.append(xval)
				y.append(ycounter)
				xvalprevious = xval

			x = np.asarray(x) #Convert list to numpy array
			y = np.asarray(y)
			s = np.full((1, x.size), float(500)/(maxXBound-minXBound))
			ax.scatter(x, y, marker='o', s=s, color=colors[gameindex], alpha=1)

		if bestfit:
			#Plot best fit of histogram
			mean = ratings.mean()
			std = ratings.std()
			bestfit_x = np.arange(minXBound, maxXBound, .1)
			bestfit_y = norm.pdf(bestfit_x, mean, std)
			ax2 = ax.twinx()
			ax2.plot(bestfit_x, bestfit_y, linewidth=2, color=colors[gameindex])
			plt.setp(ax2.get_yticklabels(), visible=False)
			ax2.yaxis.set_tick_params(size=0)
			#Plot mean line
			plt.axvline(mean, linestyle='dashed', linewidth=3, color=colors[gameindex])

		#Customize title, set position, allow space on top of plot for title
		ax.set_title(game)
		ax.set_xlabel("Rating" if games[len(games)-1]==game else "", fontdict = { "fontsize" : 10 }) #Only have Rating label on last graph in column
		ax.set_ylabel("Number of Players", fontdict = { "fontsize" : 10 })
		ax.set_xbound(lower=minXBound, upper=maxXBound)
		ax.set_ybound(lower=minYBound, upper=maxYBound)
		ax.set_xticks(np.arange(round(minXBound, -1), maxXBound+1, 5)) #Show major ticks every 5    

		#Legend

		# #Set legend keys
		key = mlines.Line2D([], [], 
							linewidth=0, markersize=0)
		# Set legend labels
		labels = [	"Range: "+str(int(ratings.min()))+"-"+str(int(ratings.max())),
					"Median: "+str(int(ratings.median())), 
					"Std "+str(int(ratings.std())), 
					"n = "+str(int(ratings.size))]

		# Position legend in upper right corner
		leg = ax.legend([key, key, key, key],  labels, 
						loc=1,
						ncol=1,
						fontsize=10, 
						frameon=True,
						fancybox=True,
						title='Stats')
		for t in leg.get_texts():
			t.set_ha("left")
			t.set_position((-20,0))
	savePlot(fig, "_".join(games)+"_allratings_separategraphs_ratingsfrom"+str(a1)+"to"+str(a2)+("_hist" if histogram else "")+("_dotplot" if dotplot else "")+("_bestfit" if bestfit else ""))

def allratingsingame_samegraphs(games, histogram, dotplot, bestfit):
	fig = plt.figure(figsize=(16,12))
	fig.suptitle("All Ratings in "+", ".join(games), fontsize=18, fontweight="bold")
	
	ax = fig.add_subplot(1, 1, 1)

	#Find max frequency for a rating, to set y-upper of our graphs
	maxFreqForARating = 0
	for game in games:
		count = df[game].value_counts().max()
		if count>maxFreqForARating:
			maxFreqForARating = count

	#Mins and maxes
	minXBound = int(df[games].min().min())
	maxXBound = 100
	minYBound = 0
	maxYBound = int(maxFreqForARating)


	#Legend
	handles = []
	#Set legend key for each position in graph
	for gameindex, game in enumerate(games):
		handles.append(mpatches.Patch(color=colors[gameindex], label=game))
	# Position legend in upper right corner
	leg = ax.legend(handles=handles,
					loc=1,
					ncol=1,
					fontsize=10, 
					frameon=True,
					fancybox=True,
					title='Legend')

	histdata = []
	for gameindex, game in enumerate(games):
		#Get data
		df1 = df[np.isfinite(df[game])]
		ratings = df1[game].astype(int)

		#
		#Draw graph
		#
		if histogram:
			#Plot actual histogram
			if gameindex==0:
				heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
			else:
				heights, bins = np.histogram(ratings, ratings.max()-ratings.min())
			histdata.append([heights, bins])
			width = (histdata[0][1][1]-histdata[0][1][0])/(len(games)+1)
			ax.bar(bins[:-1]+width*gameindex, heights, width=width, facecolor=colors[gameindex])			
		elif dotplot:
			data = ratings.values
			#Plot dot plot of histogram
			#Generate (x,y) coords for each dot
			x = []
			y = []
			width = float(1)/((len(games)+1)) #Width of dotplot column
			data.sort()
			xvalprevious = None
			ycounter = 0
			for xval in np.nditer(data):
				if xvalprevious is not None and xval==xvalprevious:
					ycounter += 1
				else:
					ycounter = 1
				x.append(xval+width*(gameindex-int(len(games)/2))) #Shift xval over to make room for multiple games with xvals at same rating
				y.append(ycounter)
				xvalprevious = xval

			x = np.asarray(x) #Convert list to numpy array
			y = np.asarray(y)
			s = np.full((1, x.size), float(15)/len(games))
			ax.scatter(x, y, marker='o', s=s, color=colors[gameindex], alpha=1)

		if bestfit:
			#Plot best fit of histogram
			mean = ratings.mean()
			std = ratings.std()
			bestfit_x = np.arange(minXBound, maxXBound, .1)
			bestfit_y = norm.pdf(bestfit_x, mean, std)
			ax2 = ax.twinx()
			ax2.plot(bestfit_x, bestfit_y, linewidth=2, color=colors[gameindex])
			plt.setp(ax2.get_yticklabels(), visible=False)
			ax2.yaxis.set_tick_params(size=0)
			#Plot mean line
			plt.axvline(mean, linestyle=('solid' if dotplot else 'dashed'), linewidth=3, color=colors[gameindex])

	#Customize title, set position, allow space on top of plot for title
	ax.set_xlabel("Rating", fontdict = { "fontsize" : 10 }) #Only have Rating label on last graph in column
	ax.set_ylabel("Number of Players", fontdict = { "fontsize" : 10 })
	ax.set_xbound(lower=minXBound, upper=maxXBound)
	ax.set_ybound(lower=minYBound, upper=maxYBound)
	ax.set_xticks(np.arange(round(minXBound, -1), maxXBound+1, 5)) #Show major ticks every 5  

	savePlot(fig, "_".join(games)+"_allratings_samegraphs"+("_hist" if histogram else "")+("_dotplot" if dotplot else "")+("_bestfit" if bestfit else ""))


@drawBoxPlotBarGraph
def boxplotallratingsacrossgames(fig, ax, pos, games):
	ratings = [] # Each game gets one element in this list (a Dataframe of ratings)
	for game in games:
		df1 = df[np.isfinite(df[game])]
		ratings.append(df1[game])
	
	#Plot data
	ax.boxplot(ratings)
	ax.set_xticklabels(games)

	return "boxplotallratingsacrossgames", "Player Rating Progression in "+", ".join(games)

@drawLineGraph
def allratingsacrossgames_line_samegraphs(fig, ax, pos, games, gameYearToXCoord, a1, a2, scatter, line):
	df1 = df[np.isfinite(df[games])]
	ratings = df1[games]
	data = ratings.values

	#Plot data
	players = []
	cmap = defaultCMap
	xShiftTracker = {} #Track how much each rating should shift the x-coord
	for playerindex, player in enumerate(data):
		c = cmap(playerindex/float(len(data)))
		x = []
		y = []
		for ratingindex, rating in enumerate(player):
			xval = gameYearToXCoord[games[ratingindex]]
			yval = rating
			xShift = 0
			if not line: #Shift ratings to the left or right so that they don't overlap on a single x-coord (e.g. all 6 ratings that are 77 become one point on the x-coord 2K13 - this spreads them out horizontally)
				if xval not in xShiftTracker:
					xShiftTracker[xval] = {}
				if rating not in xShiftTracker[xval]:
					xShiftTracker[xval][rating] = 0
				xShift = xShiftTracker[xval][rating]*0.02
				xShiftTracker[xval][rating] += 1
			#Only add datapoint if it is in the range of ratings a1<y<a2
			if yval>=a1 and yval<=a2:
				x.append(xval+xShift)
				y.append(yval)
		if scatter:
			ax.scatter(x, y, marker='o', s=5, color='red', alpha=1)
		if line:
			ax.plot(x, y, color=c)
	return "allratings_acrossgames_samegraphs", "Player Rating Progression in "+", ".join(games)+(" for Ratings "+str(a1)+"-"+str(a2) if a1>0 and a2<100 else "")


@drawLineGraph
def shiftfromratingsbrackets(fig, ax, pos, games, gameYearToXCoord, a1, a2, b1, b2, scatter, line):

	#Tick marks
	ax.tick_params(labelright=True)

	#Get data
	xShiftTracker = {} #Track how much each rating should shift the x-coord
	players = df[(df[games[0]]>=a1) & (df[games[0]]<=a2) & (df[games[-1]]>=0) & (df[games[-1]]<=100)] #Return players who had a rating a1<=x<=a2 in the oldest game and had a rating in the most recent game
	ratings = players[games]
	data = ratings.values

	#Plot data
	cmap = defaultCMap
	increasedCount = 0
	decreasedCount = 0
	for playerindex, player in enumerate(data):
		c = cmap(playerindex/float(len(data)))
		x = []
		y = []
		for ratingindex, rating in enumerate(player):
			xval = gameYearToXCoord[games[ratingindex]]
			yval = rating
			xShift = 0
			if not line: #Shift ratings to the left or right so that they don't overlap on a single x-coord (e.g. all 6 ratings that are 77 become one point on the x-coord 2K13 - this spreads them out horizontally)
				if xval not in xShiftTracker:
					xShiftTracker[xval] = {}
				if rating not in xShiftTracker[xval]:
					xShiftTracker[xval][rating] = 0
				xShift = xShiftTracker[xval][rating]*0.02
				xShiftTracker[xval][rating] += 1
			x.append(xval+xShift)
			y.append(yval)
		#If player raised rating to >=b1, color him red
		if (y[-1]>=b1):
			c = 'red'
			increasedCount += 1
		else:
			c = 'blue'
			decreasedCount += 1
		if scatter:
			ax.scatter(x, y, marker='o', s=5, color=c, alpha=1)
		if line:
			ax.plot(x, y, color=c)

	#Draw dotted line from left a2 to right a2 to make the rating shift more obvious
	ax.plot([gameYearToXCoord[games[0]], gameYearToXCoord[games[-1]]], [a2, a2], color='purple', linestyle='dashed')
	
	#Legend
	handles = []
	#Red is for increased rating, blue is for stagnant or decreased rating
	handles.append(mpatches.Patch(color='red', label='Increased to >='+str(b1)+' (n='+str(increasedCount)+')'))
	handles.append(mpatches.Patch(color='blue', label='Decreased / Stagnant'+' (n='+str(decreasedCount)+')'))
	# Position legend in upper right corner
	leg = ax.legend(handles=handles,
					loc=2,
					ncol=1,
					fontsize=10, 
					frameon=True,
					fancybox=True,
					title='Legend')
	return "shiftfrom"+str(a1)+"to"+str(a2)+"to"+str(b1)+"to"+str(b2), "Progression of Players with Ratings between "+str(a1)+"-"+str(a2)+" in "+games[0]


@drawLineGraph
def compareposacrossgames_line_samegraphs(fig, ax, pos, games, gameYearToXCoord, scatter, line):
	#Legend
	handles = []
	#Set legend key for each position in graph
	for pindex, p in enumerate(pos):
		handles.append(mpatches.Patch(color=colors[pindex], label=p))
	# Position legend in upper right corner
	leg = ax.legend(handles=handles,
					loc=1,
					ncol=1,
					fontsize=10, 
					frameon=True,
					fancybox=True,
					title='Legend')
	#Get data

	xShiftTracker = {} #Track how much each rating should shift the x-coord
	for pindex, p in enumerate(pos):
		players = df[df["Position"]==p] #Return players who match the requested position
		ratings = players[games]
		data = ratings.values

		#Plot data
		cmap = colors[pindex]
		for playerindex, player in enumerate(data):
			c = colors[pindex] #cmap(playerindex/float(len(data)))
			x = []
			y = []
			for ratingindex, rating in enumerate(player):
				xval = gameYearToXCoord[games[ratingindex]]
				yval = rating
				xShift = 0
				if not line: #Shift ratings to the left or right so that they don't overlap on a single x-coord (e.g. all 6 ratings that are 77 become one point on the x-coord 2K13 - this spreads them out horizontally)
					if xval not in xShiftTracker:
						xShiftTracker[xval] = {}
					if rating not in xShiftTracker[xval]:
						xShiftTracker[xval][rating] = 0
					xShift = xShiftTracker[xval][rating]*0.02
					xShiftTracker[xval][rating] += 1
				x.append(xval+xShift)
				y.append(yval)
			if scatter:
				ax.scatter(x, y, marker='o', s=5, color=c, alpha=1)
			if line:
				ax.plot(x, y, color=c)

	return "acrossgames_samegraphs", ", ".join(pos)+" Rating Progression in "+", ".join(games)


#----------
#----- INIT
#----------

#Get player data
allpositions = ["PG","SG","SF","PF","C"]
allgames = ["2K13", "2K14", "2K15", "2K16"]
k14on = ["2K14", "2K15", "2K16"]
alltypes = ["MVP", "MVPcandidate", "allstar", "starter", "sixthman", "bench"]
#Excel data fetching
df = pd.read_excel("nba2kplayerratings.xlsx")
cols = list(df) #list of column names
#Convert strings (e.g. hyphens for games when player didn't have a rating) in Ratings columns to floats
for c in cols:
	if re.compile("2K\d\d").match(c): #If column is a game title in the form "2K__"
		df[c] = pd.to_numeric(df[c], errors='coerce')


#
#Generate plots
#

compareposacrossgames_boxplot(allpositions, allgames)
compareposacrossgames_line_samegraphs(allpositions, allgames, True, False)
compareposacrossgames_line_samegraphs(allpositions, allgames, False, True)
allratingsacrossgames_line_samegraphs(allgames, True, False)
allratingsacrossgames_line_samegraphs(allgames, False, True)

# ALL GAMES AND POSITIONS AT ONCE
truthTable_h_d_b =	[ 	[True, False, False], #Histogram only
						[True, False, True], #Histogram and bestfit
						[False, True, False], #Dotplot only
						[False, True, True], #Dotplot and bestfit
						[False, False, True] #Bestfit only
					]
truthTable_h_b =	[	[True, False], #Histogram only
						[True, True], #Histogram and bestfit
						[False, True] #Bestfit only
					]
truthTable_s_l =	[	[True, False], #Scatterplot only
						[False, True] #Linegraph only
					]
compareposacrossgames_boxplot(allpositions, ["2K14", "2K15", "2K16"])

for t in truthTable_h_d_b:
	compareposwithingame_samegraphs(allpositions, ["2K14", "2K15", "2K16"], t[0], t[1], t[2])
	compareposacrossgames(allpositions, ["2K14", "2K15", "2K16"], t[0], t[1], t[2])
	allratingsingame_samegraphs(["2K14", "2K15", "2K16"], t[0], t[1], t[2])
	plt.close('all')
for t in truthTable_h_b:
	allratingsingame_separategraphs(allgames, t[0], t[1])
	compareposwithingame_separategraphs(allpositions, ["2K14", "2K15", "2K16"], t[0], t[1])
	allratingsacrossgames_line_samegraphs(["2K14", "2K15", "2K16"], t[0], t[1])
	compareposacrossgames_line_samegraphs(allpositions, ["2K14", "2K15", "2K16"], t[0], t[1])
	plt.close('all')

# FOR EACH GAME
for g in allgames:
	compareposacrossgames_boxplot(allpositions, [g])
	for t in truthTable_h_d_b:
		compareposwithingame_samegraphs(allpositions, [g], t[0], t[1], t[2])
		compareposacrossgames(allpositions, [g], t[0], t[1], t[2])
		allratingsingame_samegraphs([g], t[0], t[1], t[2])
		plt.close('all')
	for t in truthTable_h_b:
		allratingsingame_separategraphs([g], t[0], t[1])
		compareposwithingame_separategraphs(allpositions, [g], t[0], t[1])
		plt.close('all')

# FOR EACH POSITION
for p in allpositions:
	for t in truthTable_s_l:
		compareposacrossgames_line_samegraphs([p], allgames, t[0], t[1])
