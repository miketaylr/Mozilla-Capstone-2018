#Raw dictionaries straight from the tables

WORDS_TO_COMPONENT = {
"Bookmark":	[x.strip() for x in "Book mark, favorite, marked site".split(',')],
"Firefox":  [x.strip() for x in "Firefox Browser, Browser, Like, Update, Touch, mobile, gesture, Use, Thank, icon, Menu, button, fast, app, support, extension, addon, integration, look, interface, logo, Quantum, design,  menu, version, ui, safari, chrome, navigation, happy, hate, suck, pretty,beautical,ugly, elegant,good looking,good-looking,nice,pleasant,awful,great,design, user interface, user experience, ux, slow,lag,laggy,quick,speed,smooth,crash, frozen,abort,abandon,freeze,dead,died,update,release,updated,new version,compatible, windows, mac".split(',')],
"History/cookies/cache":	[x.strip() for x in "History,cookies,cache,cook".split(',')],
"Top Sites":	[x.strip() for x in "top sites, topsite, top websites, frequent visit, frequent site, high visit".split(',')],
"Highlights":	[x.strip() for x in "highlight, recent visit, recent site, recent bookmark, latest visit, latest site, latest bookmark".split(',')],
"Pocket":	[x.strip() for x in "pocket, save video, recommended by pocket".split(',')],
"Slowness":	[x.strip() for x in "slow,fast,lag,laggy,quick,speed,smooth".split(',')],
"Crashes":	[x.strip() for x in "crash, frozen,abort,abandon,freeze,dead,died".split(',')],
"Color":	[x.strip() for x in "color, red, orange, yellow, green, blue, purple, brown, magenta, tan, cyan, olive, maroon, navy, aquamarine, turquoise, silver, lime, teal, indigo, violet, pink, white, black".split(',')],
"Preferences":	[x.strip() for x in "preferences, fonts&color, save files to, default browser, general, language, application download, performance settings, scrollling, default search engine, remember log in, history setting, suggest browing history, suggest bookmarks, suggest open tabs, cached web content, set tracking protection, firefox data collection and use, block dangerous content, block dangerous downloads".split(',')],
"Top Sites":	[x.strip() for x in "top sites, topsite, top websites, frequent visit, frequent site, high visit".split(',')],
"Highlights":	[x.strip() for x in "highlight, recent visit, recent site, recent bookmark, latest visit, latest site, latest bookmark".split(',')],
"Pocket":	[x.strip() for x in "pocket, save video, recommended by pocket".split(',')]
}


WORDS_TO_ISSUE = {
"Performance": [x.strip() for x in "slow,fast,lag,laggy,quick,speed,smooth, memory, scroll, scrolling, load, time".split(',')],
"Crashes": [x.strip() for x in "crash, frozen, abort, abandon, freeze, dead, died, shut down, close, restart, start up".split(',')],
"Layout Bugs": [x.strip() for x in "pretty, beautiful,ugly, elegant,good looking,good-looking,nice,pleasant,awful,great,UI,user interface,design,broken,broke, css, style".split(',')],
"Regressions": [x.strip() for x in "update,release,updated,version,new version, upgrade, used to, anymore".split(',')],
"Not Supported": [x.strip() for x in "compatible,support,supported,not supported,unsupported".split(',')],
"Generic Bug": [x.strip() for x in "click,work,working,run,delete,save,find,open,close,turnon,turnoff,remove,clear,hide,add,edit,change,view,see,erase,move,enable,disable,login,log in,logon, log on, search,exit,leave,bookmark,mark,scan,pin,load,reload,share,sign in, signin,signout, sign out,reset,set, doesn’t,doesnot,does not,didn't,did not,don't, donot, do not, can't, cannot, can not,couldn't, couldnot, could not, will not, won't, wouldn't, would not, wouldnot, isn't, not, are not, were not, was not, nothing".split(',')],
"Media Playback": [x.strip() for x in "play, video, stream, watch, listen, audio, pause, volume, autoplay".split(',')],
"Security":	[x.strip() for x in "certificate, ssl, https, connection, secure, open, enter, security".split(',')],
"Search Hijacking":[x.strip() for x in	"search, hijack, homepage, google, redirect".split(',')],
}
