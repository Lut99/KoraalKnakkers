# Progress bar class. Easily copy this class or import it and have a progress
#   bar in your project!

################# CHANGELOG #################
#   v1.0                                    #
#   + Added basic functionality             #
#############################################
#   v2.0                                    #
#   + Added WaitIndicator class             #
#   + Added DotsAnimation animation         #
#   o Updated ProgressBar to only refresh   #
#     once in a while                       #
#############################################
#   v2.1                                    #
#   + Added functionality to add some       #
#     preceding text to the progress bar    #
#   + Added the ability to customize the    #
#     character that is printed on          #
#     completion to the progress bar.       #
#############################################
#   v2.2                                    #
#   o Set WaitIndicator thread as a Daemon  #
#     so interrupts are actually handled    #
#############################################
#   v2.3                                    #
#   + Added 'update_preceding_text' to      #
#     progress_bar, so the user can update  #
#     the preceding text mid-execution      #
#############################################
#   v2.4                                    #
#   o Changed printing to use sys.out, out  #
#     to increase printing speed and add    #
#     flushing                              #
#############################################


# For a documentation, see my GitHub branch.
import time
import threading
import os
import sys

# Progress Bar
class ProgressBar():
    # Init the class
    def __init__(self, width=-1, min_amount=0, max_amount=99, refresh_rate=0.5, preceding_text="", ending_character="\n"):
        if width == -1:
            _, screen_width = os.popen('stty size', 'r').read().split()
            width = int(screen_width)
        self.total_width = width
        self.preceding_text = preceding_text
        self.ending_character = ending_character
        self.width = self.total_width - 10 - len(self.preceding_text)
        self.step = min_amount
        self.max_amount = max_amount
        self.last_refresh = -1
        self.refresh_rate = refresh_rate
        self.ended = False

    # Draw the progressbar
    def draw (self):
        percentage = float(self.step) / float(self.max_amount)
        bar = self.preceding_text
        bar += " {:5.1f}% ".format(percentage * 100)
        bar += "["
        bar += "=" * (int(self.width * percentage))
        bar += " " * ((self.width) - int(self.width * percentage))
        bar += "]"
        sys.stdout.write(bar + "\r")
        sys.stdout.flush()
        if percentage >= 1 and not self.ended:
            # We're done
            self.end(ending_character = self.ending_character)
            self.ended = True

    # Only draw once in refresh_rate
    def draw_timed (self):
        if time.time() - self.last_refresh > self.refresh_rate or self.last_refresh == -1:
            if self.last_refresh == -1:
                self.last_refresh = time.time()
            else:
                self.last_refresh += self.refresh_rate
            self.draw()

    # Only update step with amount
    def update_only (self, amount=1):
        self.step += amount
        if self.step > self.max_amount:
            self.step = self.max_amount
        if self.step >= self.max_amount:
            self.draw()

    # Update step with amount and draw
    def update (self, amount=1):
        self.update_only(amount)
        if self.step < self.max_amount:
            self.draw_timed()

    # Only sets step to amount
    def set_only (self, amount):
        self.step = amount
        if self.step > self.max_amount:
            self.step = self.max_amount
        if self.step >= self.max_amount:
            self.draw()

    # Sets step to amount and draws
    def set (self, amount):
        self.set_only(amount)
        if self.step < self.max_amount:
            self.draw_timed()

    # End the program
    def end (self, clean=False, ending_character = "\n"):
        if clean:
            sys.stdout.write("\033[K\r")
        sys.stdout.write(ending_character)
        sys.stdout.flush()

    # Update the preceding text mid-execution
    def update_preceding_text (self, new_text):
        self.preceding_text = new_text
        self.width = self.total_width - 10 - len(self.preceding_text)
        self.draw()

class WaitIndicator (threading.Thread):
    # Animations
    class DotsAnimation ():
        # SIZE: No. of dots that is displayed at maximum depth
        def __init__(self, size=3):
            self.step = 0
            self.size = size + 1

        # Returns the maximum length
        def __len__(self):
            return self.size

        # Gets the next to be printed step
        def next_step (self):
            to_return = "." * self.step
            self.step = (self.step + 1) % self.size
            return to_return

        # Get the current step value
        def get_step (self):
            return self.step

    class SpinAnimation ():
        # SIZE: No. of spinners that are displayed
        def __init__(self, size=1):
            self.step = 0
            self.size = size
            self.steps = [
                "|",
                "/",
                "-",
                "\\"
            ]

        # Return the next to be printed string
        def next_step (self):
            to_return = (self.steps[self.step]) * self.size
            self.step = (self.step + 1) % len(self.steps)
            return to_return

    # Init WaitIndicator
    def __init__(self, animation, width=-1, preceding_text = "", end_text = "", refresh_rate = 1, automatic = False):
        self.running = False
        self.width = width
        self.refresh_rate = refresh_rate
        self.preceding_text = preceding_text
        self.end_text = end_text

        if self.width == -1:
            self.animation = animation()
        else:
            self.animation = animation(self.width)

        self.automatic = automatic
        if self.automatic:
            threading.Thread.__init__(self)
            self.setDaemon(True)
            # Devise a unique name
            name = "WaitIndicator-"
            i = 0
            while True:
                if name + str(i) not in threading.enumerate():
                    break
                i += 1
            self.name = name + str(i)

    # The poke function for non-threaded version
    def draw (self):
        if not self.automatic:
            self.run()

    # Update the preceding text
    def update_preceding_text (self, new_text):
        self.preceding_text = new_text

    # Start the indicator
    def start (self):
        self.last_refresh = time.time()
        self.running = True
        if self.automatic:
            threading.Thread.start(self)

    # The actual work function
    def run (self):
        while self.running:
            if time.time() - self.last_refresh > self.refresh_rate:
                self.last_refresh += self.refresh_rate
                sys.stdout.write("\033[K\r")
                sys.stdout.write(self.preceding_text + self.animation.next_step() + "\r")
            if not self.automatic:
                break

    # The stop function
    def stop (self):
        self.running = False
        # If automated, wait until dead
        if self.automatic:
            while self.isAlive():
                pass
        # If given, replace current line by end_text
        if self.end_text != "":
            sys.stdout.write("\033[K\r")
            sys.stdout.write(self.end_text)
        # Add a newline for ends
        sys.stdout.write("\n")
        sys.stdout.flush()
        # Done.
