from highlighter import Highlighter
import os

# # driver = webdriver.Chrome('./libraries/phantomjs')
# driver = webdriver.PhantomJS()
# injectHighlightLib = """var script = document.createElement('script');
# script.src = '../../libraries/jquery.highlight-5.js';
# script.type = 'text/javascript';
# document.getElementsByTagName('head')[0].appendChild(script);"""

# injectJquery = """var script = document.createElement('script');
# script.src = '../../libraries/jquery-3.2.1.min.js';
# script.type = 'text/javascript';
# document.getElementsByTagName('head')[0].appendChild(script);"""

# injectCSS = """$('head').append('<link rel="stylesheet" href="../../libraries/style.css" type="text/css" />');"""

# setHighlights = "$('body').highlight('hairstyle');"

# coverAll = "$("<html>").addClass("cover").appendTo("body").animate({opacity : 1},0).delay(0);"

# start_time = time.time()
# driver.get('file://' + os.getcwd() + '/test_storage/salon/index.html')
# driver.set_window_position(0, 0)
# driver.set_window_size(1024, 768)
# driver.execute_script(injectJquery)
# driver.execute_script(injectHighlightLib)
# driver.execute_script(injectCSS)
# driver.execute_script(setHighlights)
# driver.execute_script(coverAll)
# screenshot = driver.save_screenshot('my_screenshot2.png')
# print("--- %s seconds ---" % (time.time() - start_time))


def main():
	highlighter = Highlighter()
	query = "free hairstyle"
	webpage = 'file://' + os.getcwd() + '/test_storage/salon/index.html'
	# webpage = 'https://www.impressivewebs.com/fixing-parent-child-opacity/'
	img_target = "test.png"
	highlighter.getSnapshot(query, webpage, img_target)

# $("body").html($("body").html().replace(/chrome/g,'<b>abcde-fghi</b>'));
if __name__ == '__main__':
	main()