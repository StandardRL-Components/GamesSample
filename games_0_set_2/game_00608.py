
# Generated: 2025-08-27T14:11:05.689350
# Source Brief: brief_00608.md
# Brief Index: 608

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ to move the paddle vertically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist grid-based Pong. Return the pixel to score points. "
        "Hit the pixel with the edges of your paddle for a bonus, but be careful!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 10
    PADDLE_WIDTH = 2 * GRID_SIZE
    PADDLE_HEIGHT = 10 * GRID_SIZE
    PADDLE_SPEED = 2 * GRID_SIZE
    BALL_SIZE = 1 * GRID_SIZE
    MAX_STEPS = 1500  # Increased slightly to allow for longer volleys
    WIN_SCORE = 7
    MAX_MISSES = 3

    COLOR_BG = (15, 15, 15)
    COLOR_GRID = (40, 40, 40)
    COLOR_PADDLE = (0, 150, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (200, 200, 200)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    PADDLE_X = WIDTH - PADDLE_WIDTH - 2 * GRID_SIZE

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
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 60, bold=True)
        
        # Game state variables are initialized in reset()
        self.paddle_y = 0
        self.pixel_pos = [0, 0]
        self.pixel_vel = [0, 0]
        self.initial_pixel_speed = 4.0
        self.pixel_speed_increase = 0.2
        self.particles = []
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def _reset_pixel(self, to_player_side=False):
        """Resets the pixel's position and velocity."""
        self.pixel_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        
        # Calculate speed based on score
        speed = self.initial_pixel_speed + self.score * self.pixel_speed_increase
        
        # Set velocity
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        
        # Determine direction
        if to_player_side:
            self.pixel_vel = [abs(vx), vy]
        else: # Start towards the left wall
            self.pixel_vel = [-abs(vx), vy]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle_y = (self.HEIGHT - self.PADDLE_HEIGHT) / 2
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.win = False
        
        self._reset_pixel(to_player_side=False)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # 1. Move Paddle
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        self.paddle_y = np.clip(self.paddle_y, 0, self.HEIGHT - self.PADDLE_HEIGHT)

        # 2. Move Pixel
        self.pixel_pos[0] += self.pixel_vel[0]
        self.pixel_pos[1] += self.pixel_vel[1]
        
        pixel_rect = pygame.Rect(self.pixel_pos[0], self.pixel_pos[1], self.BALL_SIZE, self.BALL_SIZE)

        # 3. Handle Collisions
        # Top/Bottom Walls
        if pixel_rect.top <= 0 or pixel_rect.bottom >= self.HEIGHT:
            self.pixel_vel[1] *= -1
            pixel_rect.top = np.clip(pixel_rect.top, 0, self.HEIGHT - self.BALL_SIZE)
            self.pixel_pos[1] = pixel_rect.top
            # Sound: WALL_BOUNCE_SOUND

        # Left Wall (AI side)
        if pixel_rect.left <= 0:
            self.pixel_vel[0] *= -1
            pixel_rect.left = 0
            self.pixel_pos[0] = pixel_rect.left
            # Sound: WALL_BOUNCE_SOUND

        # Right Wall (Player miss)
        if pixel_rect.right >= self.WIDTH:
            self.misses += 1
            # Sound: MISS_SOUND
            if self.misses >= self.MAX_MISSES:
                self.game_over = True
                self.win = False
                terminated = True
                reward += -10  # Terminal loss reward
            else:
                self._reset_pixel(to_player_side=False)

        # Paddle Collision
        paddle_rect = pygame.Rect(self.PADDLE_X, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if self.pixel_vel[0] > 0 and paddle_rect.colliderect(pixel_rect):
            self.pixel_vel[0] *= -1
            self.pixel_pos[0] = self.PADDLE_X - self.BALL_SIZE # prevent sticking
            
            # Event-based reward
            self.score += 1
            reward += 1

            # Continuous feedback reward
            hit_pos_on_paddle = (pixel_rect.centery - paddle_rect.top) / self.PADDLE_HEIGHT
            if hit_pos_on_paddle < 0.1 or hit_pos_on_paddle > 0.9:
                reward += 0.1  # Risky hit bonus
            else:
                reward += -0.02 # Safe hit penalty

            # Increase speed
            current_speed = math.hypot(*self.pixel_vel)
            new_speed = current_speed + self.pixel_speed_increase
            speed_ratio = new_speed / current_speed if current_speed > 0 else 0
            self.pixel_vel = [v * speed_ratio for v in self.pixel_vel]

            # Create particles
            self._create_particles(pixel_rect.midleft)
            # Sound: PADDLE_HIT_SOUND

        # 4. Update Particles
        self._update_particles()
        
        # 5. Check Termination Conditions
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 10  # Terminal win reward
        
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True
            # No reward/penalty for timeout, just end episode

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            size = self.np_random.integers(2, 5)
            self.particles.append([list(pos), vel, life, size])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw particles
        for pos, vel, life, size in self.particles:
            alpha = max(0, min(255, int(255 * (life / 20.0))))
            color = (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], alpha)
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, size, size))
            self.screen.blit(temp_surf, (int(pos[0]), int(pos[1])))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, (self.PADDLE_X, int(self.paddle_y), self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        
        # Draw pixel
        pygame.draw.rect(self.screen, self.COLOR_BALL, (int(self.pixel_pos[0]), int(self.pixel_pos[1]), self.BALL_SIZE, self.BALL_SIZE))

    def _render_ui(self):
        # Score and Misses
        score_text = f"Score: {self.score}/{self.WIN_SCORE}"
        misses_text = f"Misses: {self.misses}/{self.MAX_MISSES}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        misses_surf = self.font_ui.render(misses_text, True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(misses_surf, (self.WIDTH - misses_surf.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.font_game_over.render(msg, True, color)
            pos_x = (self.WIDTH - game_over_surf.get_width()) // 2
            pos_y = (self.HEIGHT - game_over_surf.get_height()) // 2
            self.screen.blit(game_over_surf, (pos_x, pos_y))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "misses": self.misses,
            "steps": self.steps,
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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    pygame.display.set_caption("Grid Pong")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_ESCAPE]:
            running = False

        # The action space is MultiDiscrete, so we create a full action array
        action = [movement, 0, 0] # space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Misses: {info['misses']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(30)

    env.close()