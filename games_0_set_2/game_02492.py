
# Generated: 2025-08-27T20:31:29.905120
# Source Brief: brief_02492.md
# Brief Index: 2492

        
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
        "Controls: Use arrow keys (↑↓←→) to move the basket."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch the falling balls in your basket. Catch 20 to win, miss 5 and you lose. Good luck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BASKET_WIDTH = 100
    BASKET_HEIGHT = 20
    BASKET_SPEED = 8
    BALL_RADIUS = 10
    INITIAL_BALL_SPEED = 2.0
    BALL_SPAWN_RATE = 45 # Lower is faster

    WIN_CONDITION_CATCHES = 20
    LOSE_CONDITION_MISSES = 5
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_BASKET = (230, 60, 60)
    COLOR_BASKET_HIGHLIGHT = (255, 100, 100)
    COLOR_BALL = (240, 240, 240)
    COLOR_PARTICLE = (255, 220, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_MISS = (200, 0, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        # State variables are initialized in reset()
        self.basket_pos = None
        self.balls = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.caught_balls = None
        self.missed_balls = None
        self.ball_speed = None
        self.ball_spawn_timer = None
        
        # Initialize state variables
        self.reset()

        # Validate implementation after full initialization
        # self.validate_implementation() # Optional: uncomment for debugging
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.caught_balls = 0
        self.missed_balls = 0
        
        self.basket_pos = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.BASKET_WIDTH // 2, 
            self.SCREEN_HEIGHT - self.BASKET_HEIGHT - 20, 
            self.BASKET_WIDTH, 
            self.BASKET_HEIGHT
        )
        
        self.balls = []
        self.particles = []
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.ball_spawn_timer = 0
        
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- 1. Handle Player Action ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        dist_before = self._get_distance_to_nearest_ball()

        if movement == 1:  # Up
            self.basket_pos.y -= self.BASKET_SPEED
        elif movement == 2:  # Down
            self.basket_pos.y += self.BASKET_SPEED
        elif movement == 3:  # Left
            self.basket_pos.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_pos.x += self.BASKET_SPEED

        self.basket_pos.clamp_ip(self.screen.get_rect())
        
        dist_after = self._get_distance_to_nearest_ball()
        
        # Continuous reward for moving closer
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 1.0
            else:
                reward -= 0.1

        # --- 2. Update Game State ---
        self.steps += 1
        self._update_balls(reward)
        self._update_particles()
        
        # Spawn new balls periodically
        self.ball_spawn_timer -= 1
        if self.ball_spawn_timer <= 0:
            self._spawn_ball()

        # --- 3. Check for Termination ---
        terminated = False
        if self.caught_balls >= self.WIN_CONDITION_CATCHES:
            reward += 100
            terminated = True
            self.game_over = True
            # // SFX: Win fanfare
        if self.missed_balls >= self.LOSE_CONDITION_MISSES:
            reward -= 100
            terminated = True
            self.game_over = True
            # // SFX: Lose sound
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_distance_to_nearest_ball(self):
        if not self.balls:
            return None
        basket_center_x = self.basket_pos.centerx
        closest_dist = float('inf')
        for ball in self.balls:
            dist = abs(basket_center_x - ball['pos'][0])
            if dist < closest_dist:
                closest_dist = dist
        return closest_dist

    def _update_balls(self, reward_ref):
        for ball in self.balls[:]:
            ball['pos'][1] += self.ball_speed
            
            ball_rect = pygame.Rect(ball['pos'][0] - self.BALL_RADIUS, ball['pos'][1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Check for catch
            if self.basket_pos.colliderect(ball_rect):
                self.balls.remove(ball)
                self.score += 10
                self.caught_balls += 1
                reward_ref += 10
                self._create_particles(ball['pos'])
                # // SFX: Catch sound
                
                # Increase difficulty
                if self.caught_balls > 0 and self.caught_balls % 10 == 0:
                    self.ball_speed += 0.05
                    
            # Check for miss
            elif ball['pos'][1] > self.SCREEN_HEIGHT + self.BALL_RADIUS:
                self.balls.remove(ball)
                self.missed_balls += 1
                reward_ref -= 5
                # // SFX: Miss sound

    def _spawn_ball(self):
        x_pos = self.np_random.integers(self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
        y_pos = -self.BALL_RADIUS
        self.balls.append({'pos': [x_pos, y_pos]})
        self.ball_spawn_timer = self.BALL_SPAWN_RATE

    def _create_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime})
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30))
            radius = int(self.BALL_RADIUS * 0.2 * (p['lifetime'] / 30))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*self.COLOR_PARTICLE, alpha))

        # Render balls
        for ball in self.balls:
            x, y = int(ball['pos'][0]), int(ball['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Render basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_pos, border_radius=3)
        highlight_rect = self.basket_pos.copy()
        highlight_rect.height = self.BASKET_HEIGHT // 4
        pygame.draw.rect(self.screen, self.COLOR_BASKET_HIGHLIGHT, highlight_rect, border_radius=3)
        
    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render missed balls
        miss_text = self.font_small.render("MISSES:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (10, 40))
        for i in range(self.missed_balls):
            x_text = self.font_small.render("X", True, self.COLOR_MISS)
            self.screen.blit(x_text, (100 + i * 20, 40))
            
        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.caught_balls >= self.WIN_CONDITION_CATCHES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = self.COLOR_MISS
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "caught_balls": self.caught_balls,
            "missed_balls": self.missed_balls,
            "ball_speed": self.ball_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # This is a simple mapping from keyboard keys to the environment's action space.
    # It does not handle simultaneous key presses, but demonstrates how to control the env.
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_SPACE: (1, 1), # Action index 1, value 1
        pygame.K_LSHIFT: (2, 1) # Action index 2, value 1
    }

    def get_human_action():
        action = np.array([0, 0, 0]) # Default no-op action
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT]:
            action[2] = 1
            
        return action

    # --- Game Loop ---
    obs, info = env.reset()
    done = False
    
    # We need a different screen for human rendering
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Catcher")
    clock = pygame.time.Clock()
    
    print("--- Human Play Instructions ---")
    print(env.game_description)
    print(env.user_guide)
    
    while not done:
        # Get action from human player
        action = get_human_action()
        
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    env.close()