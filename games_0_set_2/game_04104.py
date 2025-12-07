
# Generated: 2025-08-28T01:25:58.501991
# Source Brief: brief_04104.md
# Brief Index: 4104

        
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
        "Controls: ←→ to move the paddle. ↑↓ to cycle through colors (Red, Green, Blue, Yellow)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Match your paddle's color to the incoming ball to score points. Missing balls costs lives. Score 15 points to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_CELLS = 10
    GRID_SIZE = 360  # 360x360 pixels
    CELL_SIZE = GRID_SIZE // GRID_CELLS
    
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_SIZE) // 2
    GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_SIZE) // 2 + 20

    WIN_SCORE = 15
    MAX_LIVES = 5
    MAX_STEPS = 1500 # Increased from 1000 to allow more time for winning

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    
    # R, G, B, Y
    PALETTE = [(255, 80, 80), (80, 255, 80), (80, 120, 255), (255, 255, 80)]
    PALETTE_DARK = [(120, 40, 40), (40, 120, 40), (40, 60, 120), (120, 120, 40)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 72)
        
        self.reset()
        
        # self.validate_implementation() # Call this to check your work

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.paddle_pos_x = self.GRID_CELLS // 2
        self.paddle_color_idx = 0
        
        self.ball_speed = 0.1 # Grid units per frame
        self.ball_steps_since_color_change = 0
        
        self.particles = []
        self.flash_effect_timer = 0
        
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        reward += self._update_game_state()
        
        terminated = self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                # sfx: win_fanfare
            else:
                reward -= 100
                # sfx: lose_sad_trombone
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, _, _ = action
        
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up: Cycle color forward
            self.paddle_color_idx = (self.paddle_color_idx + 1) % len(self.PALETTE)
        elif movement == 2: # Down: Cycle color backward
            self.paddle_color_idx = (self.paddle_color_idx - 1 + len(self.PALETTE)) % len(self.PALETTE)
        elif movement == 3: # Left
            self.paddle_pos_x = max(0, self.paddle_pos_x - 1)
        elif movement == 4: # Right
            self.paddle_pos_x = min(self.GRID_CELLS - 1, self.paddle_pos_x + 1)

    def _update_game_state(self):
        reward = 0.0
        
        # Continuous reward for color matching
        if self.paddle_color_idx == self.ball_color_idx:
            reward += 0.01
        else:
            reward -= 0.01

        # Update ball
        self.ball_pos[1] += self.ball_speed
        self.ball_steps_since_color_change += 1
        
        # Randomly change ball color every few steps
        if self.ball_steps_since_color_change > 150 / (self.ball_speed * 100):
            self.ball_color_idx = self.np_random.integers(0, len(self.PALETTE))
            self.ball_steps_since_color_change = 0

        # Difficulty scaling
        if self.steps > 0 and self.steps % 50 == 0:
            self.ball_speed = min(0.3, self.ball_speed + 0.005)

        # Check for interaction with paddle row
        if self.ball_pos[1] >= self.GRID_CELLS - 1:
            event_reward = self._handle_ball_interaction()
            reward += event_reward
            self._spawn_ball()
            
        # Update particles and effects
        self._update_particles()
        if self.flash_effect_timer > 0:
            self.flash_effect_timer -= 1
            
        return reward

    def _handle_ball_interaction(self):
        # Ball reached the bottom row
        ball_grid_x = int(self.ball_pos[0])
        
        # Check for paddle hit
        if ball_grid_x == self.paddle_pos_x:
            # Color match check
            if self.ball_color_idx == self.paddle_color_idx:
                # sfx: success_chime
                self.score += 3
                reward = 3.0
                self.flash_effect_timer = 5
                self._create_particles(self.paddle_pos_x, self.GRID_CELLS -1, self.PALETTE[self.paddle_color_idx])
                
                # Risky play bonus
                if self.paddle_pos_x == 0 or self.paddle_pos_x == self.GRID_CELLS - 1:
                    self.score += 2
                    reward += 2.0
                return reward
            else: # Mismatch
                # sfx: failure_buzz
                self.lives -= 1
                return -1.0
        else: # Miss
            # sfx: miss_swoosh
            self.lives -= 1
            return -1.0
        
    def _spawn_ball(self):
        self.ball_pos = [self.np_random.uniform(0.5, self.GRID_CELLS - 0.5), 0.0]
        self.ball_color_idx = self.np_random.integers(0, len(self.PALETTE))
        self.ball_steps_since_color_change = 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_CELLS + 1):
            # Vertical lines
            start_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y)
            end_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y + self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_ORIGIN_X + self.GRID_SIZE, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw paddle
        paddle_color = self.PALETTE[self.paddle_color_idx]
        paddle_rect = pygame.Rect(
            self.GRID_ORIGIN_X + self.paddle_pos_x * self.CELL_SIZE,
            self.GRID_ORIGIN_Y + (self.GRID_CELLS - 1) * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, paddle_color, paddle_rect)
        pygame.draw.rect(self.screen, tuple(min(255, c + 60) for c in paddle_color), paddle_rect.inflate(-8, -8))

        # Draw ball
        ball_pixel_x = int(self.GRID_ORIGIN_X + self.ball_pos[0] * self.CELL_SIZE)
        ball_pixel_y = int(self.GRID_ORIGIN_Y + self.ball_pos[1] * self.CELL_SIZE)
        ball_color = self.PALETTE[self.ball_color_idx]
        radius = self.CELL_SIZE // 2 - 2
        pygame.gfxdraw.filled_circle(self.screen, ball_pixel_x, ball_pixel_y, radius, ball_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pixel_x, ball_pixel_y, radius, ball_color)
        
        # Draw particles
        for p in self.particles:
            p_color = p['color']
            alpha = int(255 * (p['life'] / p['max_life']))
            p_color_alpha = (*p_color, alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))
            
        # Draw flash effect
        if self.flash_effect_timer > 0:
            alpha = int(128 * (self.flash_effect_timer / 5))
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", (20, 15), self.font_ui)
        
        # Current color indicator
        pygame.draw.rect(self.screen, self.PALETTE[self.paddle_color_idx], (160, 15, 20, 20))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (160, 15, 20, 20), 2)
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 15))
        for i in range(self.MAX_LIVES):
            heart_color = self.PALETTE[0] if i < self.lives else self.COLOR_GRID
            pos = (self.SCREEN_WIDTH - 100 + i * 20, 25)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, heart_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, heart_color)

        # Game Over / Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                self._draw_text("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40), self.font_title, center=True)
            else:
                self._draw_text("GAME OVER", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40), self.font_title, center=True)
            self._draw_text("Press Reset", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20), self.font_ui, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surface = font.render(text, True, color)

        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)
        
    def _create_particles(self, grid_x, grid_y, color):
        px = self.GRID_ORIGIN_X + (grid_x + 0.5) * self.CELL_SIZE
        py = self.GRID_ORIGIN_Y + (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "ball_speed": self.ball_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # Monkey-patch the render method for human mode
        GameEnv.metadata["render_modes"].append("human")
        def render_human(self):
            if not hasattr(self, 'window'):
                pygame.display.init()
                self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            
            # Get the rendered surface from _get_observation
            # This is inefficient but ensures the human view matches the agent's
            obs = self._get_observation()
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            self.window.blit(surf, (0, 0))
            pygame.display.flip()

        GameEnv.render = render_human

    env = GameEnv(render_mode=render_mode)
    env.validate_implementation()

    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    # --- Manual Play Loop ---
    if render_mode == 'human':
        action = env.action_space.sample()
        action[0] = 0 # Start with no-op
        
        while not done:
            movement = 0
            
            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            elif keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            
            action = np.array([movement, 0, 0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Lives: {info['lives']}")
            
            if terminated or truncated:
                print("Game Over!")
                done = True
        
        # Keep window open for a bit after game over
        pygame.time.wait(2000)

    env.close()