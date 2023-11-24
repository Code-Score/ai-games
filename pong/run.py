'''
    References :
    https://codingchallenges.fyi/challenges/challenge-pong
    https://www.geeksforgeeks.org/create-a-pong-game-in-python-pygame/
'''
import pygame
import time
import random
from player import Player
from simple_player import Player as SimplePlayer

pygame.init()

# Font that is used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)

# RGB values of standard colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Basic parameters of the screen
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

clock = pygame.time.Clock() 
FPS = 100

player_states = []

class PlayerState:
    def __init__(self, posx, posy, width, height, speed, color):
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
        self.playerRect = pygame.Rect(posx, posy, width, height)
        _ = pygame.draw.rect(screen, self.color, self.playerRect)

    def reset(self):
        self.posy = 0

    def display(self):
        _ = pygame.draw.rect(screen, self.color, self.playerRect)

    def update(self, yFac = 0):
        self.posy = self.posy + self.speed*yFac

        if self.posy <= 0:
            self.posy = 0
        elif self.posy + self.height >= HEIGHT:
            self.posy = HEIGHT-self.height

        self.playerRect = (self.posx, self.posy, self.width, self.height)

    def getRect(self):
        return self.playerRect

def displayScore(text, score, x, y, color):
    text = font20.render(text+str(score), True, color)
    textRect = text.get_rect()
    textRect.center = (x, y)
    screen.blit(text, textRect)

# Ball class
class Ball:
    def __init__(self, posx, posy, radius, speedx, speedy, color):
        self.posx = posx
        self.posy = posy
        self.radius = radius
        self.speedx = speedx
        self.speedy = speedy
        self.color = color
        self.xFac = 1
        self.yFac = -1
        self.ball = pygame.draw.circle(
            screen, self.color, (self.posx, self.posy), self.radius)

    def display(self):
        self.ball = pygame.draw.circle(
            screen, self.color, (self.posx, self.posy), self.radius)

    def update(self):
        self.posx += self.speedx*self.xFac
        self.posy += self.speedy*self.yFac

        if self.posy <= 0 or self.posy >= HEIGHT:
            self.yFac *= -1

        if self.posx < 10:
            return 1
        elif self.posx > WIDTH-15:
            return -1
        else:
            return 0

    def reset(self):
        self.posx = WIDTH//2
        self.posy = random.randint((int)(0.2*HEIGHT), (int)(0.8*HEIGHT))
        self.speedy = random.randint(4, 10)
        self.xFac *= -1

    def hit(self):
        self.xFac *= -1
        self.speedy = random.randint(4, 10)

    def getRect(self):
        return self.ball

def main():
    player1 = SimplePlayer(0)
    player2 = Player(1)
    listOfPlayers = [player1, player2]

    player_state1 = PlayerState(10, 0, 10, 100, 10, GREEN)
    player_state2 = PlayerState(WIDTH-15, 0, 10, 100, 10, GREEN)
    player_states = [player_state1, player_state2]

    ball = Ball(WIDTH//2, random.randint((int)(0.2*HEIGHT), (int)(0.8*HEIGHT)), 7, 7, random.randint(4, 10), WHITE)

    running = True
    prev_time = time.time()
    while True:
        for player in listOfPlayers:
            if (not player.can_start()):
                break
            player_states[player.id].reset()
            player_states[player.id].display()
        ball.reset()
        ball.display()
        running = True
        player1Score, player2Score = 0, 0

        while running and max(player1Score, player2Score) < 5:
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for player in listOfPlayers:
                if pygame.Rect.colliderect(ball.getRect(), player_states[player.id].getRect()):
                    ball.hit()

            for player in listOfPlayers:
                yfac = player.update_state(**{
                        'ballx': ball.posx, 
                        'bally': ball.posy, 
                        'speedx': ball.speedx*ball.xFac, 
                        'speedy': ball.speedy*ball.yFac,
                        'playerx': player_states[player.id].posx, 
                        'playery': player_states[player.id].posy,
                        'playerheight': player_states[player.id].height
                    }
                )
                player_states[player.id].update(yfac)

            point = ball.update()

            # -1 -> Player_1 has scored
            # +1 -> Player_2 has scored
            # 0 -> None of them scored
            if point == -1:
                player1Score += 1
            elif point == 1:
                player2Score += 1

            if point: 
                ball.reset()

            for player in listOfPlayers:
                player_states[player.id].display()
            ball.display()

            displayScore("Player_1 : ", player1Score, 100, 20, WHITE)
            displayScore("Player_2 : ", player2Score, WIDTH-100, 20, WHITE)

            pygame.display.update()
            clock.tick(FPS)
             


if __name__ == "__main__":
    main()
    pygame.quit()
