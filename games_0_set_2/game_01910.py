
# Generated: 2025-08-27T18:41:11.147654
# Source Brief: brief_01910.md
# Brief Index: 1910

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Control the paddle to bounce a ball and break all the bricks. Reach 50 points to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 15, 40)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.BRICK_COLORS = {
            1: (200, 50, 50),  # Red
            2: (50, 200, 50),  # Green
            3: (50, 50, 200),  # Blue
        }

        # Fonts
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED_INITIAL = 6
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50
        self.INITIAL_LIVES = 2
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.bricks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.ball_launched = False
        
        # Paddle
        paddle_y = self.height - 40
        self.paddle = pygame.Rect(
            (self.width - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        self.ball_vel = [0, 0]

        # Bricks
        self._create_bricks()

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _create_bricks(self):
        self.bricks = []
        brick_width, brick_height = 60, 20
        gap = 4
        rows, cols = 5, 10
        top_offset = 50
        side_offset = (self.width - (cols * (brick_width + gap) - gap)) / 2

        for r in range(rows):
            for c in range(cols):
                points = 1
                if r < 2: points = 3  # Top two rows are blue
                elif r < 4: points = 2 # Middle two rows are green
                
                x = side_offset + c * (brick_width + gap)
                y = top_offset + r * (brick_height + gap)
                brick = {
                    "rect": pygame.Rect(x, y, brick_width, brick_height),
                    "points": points,
                    "color": self.BRICK_COLORS[points]
                }
                self.bricks.append(brick)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # Calculate rewards and update state
        reward = -0.01  # Small penalty per step to encourage speed
        self._handle_input(movement, space_held)
        event_reward = self._update_game_logic()
        reward += event_reward
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 100 # Lose penalty
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.width - self.PADDLE_WIDTH, self.paddle.x))

        # Launch ball
        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi/4, math.pi/4) # Random launch angle
            self.ball_vel = [
                self.BALL_SPEED_INITIAL * math.sin(angle),
                -self.BALL_SPEED_INITIAL * math.cos(angle)
            ]
            # Sound effect: launch.wav

    def _update_game_logic(self):
        if not self.ball_launched:
            # Ball follows paddle
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
            return 0

        reward = 0
        
        # Move ball
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.width:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.width, self.ball.right)
            # Sound effect: wall_bounce.wav
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # Sound effect: wall_bounce.wav

        # Life loss
        if self.ball.top >= self.height:
            self.lives -= 1
            self.ball_launched = False
            reward -= 0.2
            # Sound effect: life_lost.wav
            if self.lives > 0:
                self.reset_ball_on_paddle()
            return reward

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.BALL_SPEED_INITIAL * offset
            
            # Normalize speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED_INITIAL
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED_INITIAL
            # Sound effect: paddle_hit.wav

        # Brick collisions
        for brick in self.bricks[:]:
            if self.ball.colliderect(brick["rect"]):
                # Determine bounce direction
                prev_ball_center = (self.ball.centerx - self.ball_vel[0], self.ball.centery - self.ball_vel[1])
                
                # Check collision with vertical sides
                if prev_ball_center[0] <= brick["rect"].left or prev_ball_center[0] >= brick["rect"].right:
                    self.ball_vel[0] *= -1
                # Check collision with horizontal sides
                if prev_ball_center[1] <= brick["rect"].top or prev_ball_center[1] >= brick["rect"].bottom:
                    self.ball_vel[1] *= -1
                
                reward += brick["points"]
                self.score += brick["points"]
                self._create_particles(brick["rect"].center, brick["color"])
                self.bricks.remove(brick)
                # Sound effect: brick_break.wav
                break # Only one brick per frame

        self._update_particles()
        return reward
    
    def reset_ball_on_paddle(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
            life = self.np_random.integers(10, 25)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2  # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, brick["rect"], 1) # Border

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        center = (int(self.ball.centerx), int(self.ball.centery))
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 80))
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_medium.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.width - lives_text.get_width() - 10, 10))

        # Pre-launch message
        if not self.ball_launched and not self.game_over:
            launch_text = self.font_small.render("Press SPACE to launch", True, self.COLOR_TEXT)
            pos = (self.width/2 - launch_text.get_width()/2, self.height - 70)
            self.screen.blit(launch_text, pos)

        # Game Over / Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            pos = (self.width/2 - end_text.get_width()/2, self.height/2 - end_text.get_height()/2)
            self.screen.blit(end_text, pos)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To display the game, we need a different pygame setup
    # This part is for human playability and visualization, not part of the env itself
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.width, env.height))
    
    done = False
    total_reward = 0
    
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Human controls
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Check for termination
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        # Control the frame rate
        env.clock.tick(30)

    env.close()