import os
from selenium import webdriver
from PIL import Image
from io import BytesIO
import numpy as np

class Highlighter():

	def __init__(self):
		# self.driver = webdriver.Chrome("./libraries/chromedriver");
		self.driver = webdriver.PhantomJS();

	def getSnapshot(self, query, webpage, img_target):
		self.driver.get(webpage)
		self.prepare()
		self.setHighlights(query)
		# screenshot = self.driver.save_screenshot(img_target)

		self.storeSnapshot(img_target)



	def storeSnapshot(self, img_target):
		data = self.driver.get_screenshot_as_png()
		img = Image.open(BytesIO(data)).convert('LA')
		img.save(img_target)
		numpy_array = np.asarray(img)
		print(numpy_array)


	def prepare(self):
		# TODO configure page size
		self.driver.execute_script(self.injectJquery)
		self.driver.execute_script(self.injectHighlighter)
		self.driver.execute_script(self.injectCSS)

	def setHighlights(self, query):
		for word in query.split(" "):
			self.driver.execute_script(self.search.format(word))

		self.driver.execute_script(self.coverAll)

	injectJquery = """
	// Inject Jquery 
	var script = document.createElement('script');
	script.src = '../../libraries/jquery-3.2.1.min.js';
	script.type = 'text/javascript';
	document.getElementsByTagName('head')[0].appendChild(script);"""
	injectHighlighter = """
	// Inject highlighter
	var script = document.createElement('script');
	script.src = '../../libraries/jquery.highlight-5.js';
	script.type = 'text/javascript';
	document.getElementsByTagName('head')[0].appendChild(script);
	// Inject extra styling
	"""
	injectCSS = """$('head').append('<link rel="stylesheet" href="../../libraries/style.css" type="text/css" />');"""

	search = "$('html').highlight('{}');"
	coverAll = """$("<html>").addClass("cover").appendTo("body").animate({opacity : 1},0).delay(0);"""