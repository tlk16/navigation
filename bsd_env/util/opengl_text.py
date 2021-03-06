import sys

# math
import math

# python image library
from PIL import Image as img

# OpenGL
from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

def __drawCharacter__( n ):

	#----------------------------------------------------------------[ special ]    

	if n == '.':
		glVertex2i(2,0)
		glVertex2i(4,4)

	elif n == '-':
		glVertex2i(1,4)
		glVertex2i(4,4)

	elif n == '[':
		glVertex2i(2,1)
		glVertex2i(4,1)
		glVertex2i(2,1)
		glVertex2i(2,7)
		glVertex2i(4,7)
		glVertex2i(2,7)

	elif n == ']':
		glVertex2i(1,1)
		glVertex2i(3,1)
		glVertex2i(3,1)
		glVertex2i(3,7)
		glVertex2i(3,7)
		glVertex2i(1,7)

	elif n == '/':
		glVertex2i(1,1)
		glVertex2i(4,7)

	elif n == '=':
		glVertex2i(1,5)
		glVertex2i(4,5)
		glVertex2i(1,3)
		glVertex2i(4,3)

	#----------------------------------------------------------------[ numbers ]

	elif n == '1':
		glVertex2i(4,7)
		glVertex2i(4,1)

	elif n == '2':
		glVertex2i(1,7)
		glVertex2i(4,7)
		glVertex2i(4,7)
		glVertex2i(4,4)
		glVertex2i(4,4)
		glVertex2i(1,4)
		glVertex2i(1,4)
		glVertex2i(1,1)
		glVertex2i(1,1)
		glVertex2i(4,1)

	elif n == '3':
		glVertex2i(1,7)
		glVertex2i(4,7)
		glVertex2i(1,4)
		glVertex2i(4,4)
		glVertex2i(1,1)
		glVertex2i(4,1)
		glVertex2i(4,7)
		glVertex2i(4,1)

	elif n == '4':
		glVertex2i(1,7)
		glVertex2i(1,4)
		glVertex2i(1,4)
		glVertex2i(4,4)
		glVertex2i(4,7)
		glVertex2i(4,1)

	elif n == '5':
		glVertex2i(4,7)
		glVertex2i(1,7)
		glVertex2i(1,7)
		glVertex2i(1,4)
		glVertex2i(1,4)
		glVertex2i(4,4)
		glVertex2i(4,4)
		glVertex2i(4,1)
		glVertex2i(4,1)
		glVertex2i(1,1)

	elif n == '6':
		glVertex2i(4,7)
		glVertex2i(1,7)
		glVertex2i(1,7)
		glVertex2i(1,1)
		glVertex2i(1,1)
		glVertex2i(4,1)
		glVertex2i(4,1)
		glVertex2i(4,4)
		glVertex2i(4,4)
		glVertex2i(1,4)

	elif n == '7':
		glVertex2i(1,7)
		glVertex2i(4,7)

		glVertex2i(4,7)
		glVertex2i(4,1)

	elif n == '8':
		glVertex2i(1,7)
		glVertex2i(1,1)
		glVertex2i(4,7)
		glVertex2i(4,1)
		glVertex2i(1,7)
		glVertex2i(4,7)
		glVertex2i(1,4)
		glVertex2i(4,4)
		glVertex2i(1,1)
		glVertex2i(4,1)

	elif n == '9':
		glVertex2i(1,1)
		glVertex2i(4,1)
		glVertex2i(4,1)
		glVertex2i(4,7)
		glVertex2i(4,7)
		glVertex2i(1,7)
		glVertex2i(1,7)
		glVertex2i(1,4)
		glVertex2i(1,4)
		glVertex2i(4,4)

	elif n == '0':
		glVertex2i(1,7)
		glVertex2i(1,1)
		glVertex2i(1,1)
		glVertex2i(4,1)
		glVertex2i(4,1)
		glVertex2i(4,7)
		glVertex2i(4,7)
		glVertex2i(1,7)


#==========================================================[ Draw Progress Bar ]

def drawNumber( number, pos, scale=None ):
	# setup
	glLineWidth(1)
	glColor( 0.0, 1.0, 0.0 )
	glPushMatrix()
	glTranslate( pos[0], pos[1], 0 )
	if scale != None:
		glScale( scale, scale, scale )
	# draw number
	for c in str(number):
		glBegin( GL_LINES )
		__drawCharacter__(c)
		glEnd()
		glTranslate(5,0,0)
	# return to normal	
	glPopMatrix()
	glLineWidth(2)
	glColor( 1.0, 1.0, 1.0 )

def drawProgressBar( steps, done, pos, scale=None ):
	# setup
	glLineWidth(1)
	glColor( 0.0, 1.0, 0.0 )
	glPushMatrix()
	glTranslate( pos[0], pos[1], 0 )
	if scale != None: 
		glScale(scale,scale,scale)

	# open bar with '['
	glBegin( GL_LINES )
	__drawCharacter__( '[' )
	glEnd()
	glTranslate(5,0,0)
	# bar
	perc_done = int( float(done)/float(steps)*50.0 )
	for i in range(50):
		if i<perc_done:
			glBegin( GL_LINES )
			__drawCharacter__('=')
			glEnd()
			glTranslate(5,0,0)
		else:
			glBegin( GL_LINES )
			__drawCharacter__('-')
			glEnd()
			glTranslate(5,0,0)
	# close bar with ']'
	glBegin( GL_LINES )
	__drawCharacter__( ']' )
	glEnd()

	# go numbers
	glLoadIdentity()
	glTranslate( pos[0]+26*5-((3+len(str(steps))+len(str(done)))/2)*5, pos[1]-8, 0 )
	glBegin( GL_LINES )
	__drawCharacter__( '[' )
	glEnd()
	glTranslate(5,0,0)
	# done so far
	for c in str(done):
		glBegin( GL_LINES )
		__drawCharacter__(c)
		glEnd()
		glTranslate(5,0,0)
	# slash
	glBegin( GL_LINES )
	__drawCharacter__('/')
	glEnd()
	glTranslate(5,0,0)
	# step limit
	for c in str(steps):
		glBegin( GL_LINES )
		__drawCharacter__(c)
		glEnd()
		glTranslate(5,0,0)
	# close numbers with ']'
	glBegin( GL_LINES )
	__drawCharacter__( ']' )
	glEnd()

	# return to normal	
	glPopMatrix()
	glLineWidth(2)
	glColor( 1.0, 1.0, 1.0 )

