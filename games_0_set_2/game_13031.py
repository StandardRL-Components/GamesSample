import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:49:47.037556
# Source Brief: brief_03031.md
# Brief Index: 3031
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a "Stacker" game.

    The player controls a horizontal catcher platform to catch falling colored
    blocks. The goal is to stack the blocks as high as possible, earning
    points for matching the colors of consecutive blocks. The game's speed
    increases with each successful match. The episode ends if a block is
    placed unstably, if a block is missed, if the win condition (15 matches)
    is met, or if the maximum number of steps is reached.

    Visuals are a key focus, with a neon-on-dark aesthetic, glow effects,
    and particle bursts to create a polished, arcade-like experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Catch falling blocks with your platform and stack them as high as you can. "
        "Earn bonus points for matching the colors of consecutive blocks."
    )
    user_guide = "Controls: Use ← and → arrow keys to move the catcher left and right."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Core Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WHITE = (240, 240, 240)
        self.BLOCK_COLORS = [
            (50, 255, 50),   # 0: Green
            (50, 150, 255),  # 1: Blue
            (255, 50, 50),   # 2: Red
            (255, 255, 50),  # 3: Yellow
        ]
        self.COLOR_VALUES = {0: 1, 1: 2, 2: 3, 3: 4}

        # --- Game Parameters ---
        self.CATCHER_WIDTH = 100
        self.CATCHER_HEIGHT = 15
        self.CATCHER_SPEED = 12
        self.BLOCK_WIDTH = 60
        self.BLOCK_HEIGHT = 20
        self.INITIAL_FALL_SPEED = 2.0
        self.FALL_SPEED_INCREMENT = 0.1
        self.WIN_MATCH_COUNT = 15
        self.MAX_STACK_HEIGHT = 18
        self.MAX_STEPS = 2000

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_end = pygame.font.SysFont('Consolas', 64, bold=True)

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.match_count = 0
        self.game_over = False
        self.win = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.catcher_x = 0
        self.stack = []
        self.falling_block = None
        self.particles = []
        
        # Initialize state for the first time
        # self.reset() # Called by user, not in init

        # --- Validation ---
        # self.validate_implementation() # Uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.match_count = 0
        self.game_over = False
        self.win = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.catcher_x = self.SCREEN_WIDTH / 2 - self.CATCHER_WIDTH / 2
        self.stack = []
        self.particles = []
        
        self._spawn_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, subsequent steps do nothing and return 0 reward
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        self._update_catcher(movement)
        self._update_falling_block()
        reward += self._check_collision_and_stack()
        self._update_particles()
        
        self.steps += 1
        
        # Check for termination conditions
        terminated = self.game_over
        truncated = False
        if not terminated:
            if self.match_count >= self.WIN_MATCH_COUNT:
                self.win = True
                terminated = True
                reward += 100.0
            elif len(self.stack) >= self.MAX_STACK_HEIGHT:
                terminated = True
                reward += -100.0
            elif self.steps >= self.MAX_STEPS:
                truncated = True
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_block(self):
        x = self.np_random.uniform(self.BLOCK_WIDTH, self.SCREEN_WIDTH - self.BLOCK_WIDTH * 2)
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        self.falling_block = {
            'x': x, 'y': -self.BLOCK_HEIGHT, 'color_index': color_index
        }

    def _update_catcher(self, movement):
        if movement == 3:  # Left
            self.catcher_x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_x += self.CATCHER_SPEED
        
        self.catcher_x = np.clip(self.catcher_x, 0, self.SCREEN_WIDTH - self.CATCHER_WIDTH)

    def _update_falling_block(self):
        if self.falling_block:
            self.falling_block['y'] += self.fall_speed

    def _check_collision_and_stack(self):
        if not self.falling_block:
            return 0.0

        support_y = self.SCREEN_HEIGHT
        support_x = self.catcher_x
        support_width = self.CATCHER_WIDTH
        is_on_stack = False

        if self.stack:
            top_block = self.stack[-1]
            support_y = top_block['y']
            support_x = top_block['x']
            support_width = self.BLOCK_WIDTH
            is_on_stack = True

        block_bottom = self.falling_block['y'] + self.BLOCK_HEIGHT
        if block_bottom >= support_y:
            new_block_rect = pygame.Rect(self.falling_block['x'], 0, self.BLOCK_WIDTH, 1)
            support_rect = pygame.Rect(support_x, 0, support_width, 1)

            is_stable = (new_block_rect.left >= support_rect.left and 
                         new_block_rect.right <= support_rect.right)

            if new_block_rect.colliderect(support_rect) and is_stable:
                self.falling_block['y'] = support_y - self.BLOCK_HEIGHT
                self.stack.append(self.falling_block)
                # sfx: block_land
                
                catch_reward = 0.1
                match_reward = 0.0

                if len(self.stack) > 1 and self.stack[-1]['color_index'] == self.stack[-2]['color_index']:
                    self.match_count += 1
                    self.fall_speed += self.FALL_SPEED_INCREMENT
                    color_val = self.COLOR_VALUES[self.stack[-1]['color_index']]
                    match_reward = min(10.0, len(self.stack) * color_val)
                    self.score += len(self.stack) * color_val
                    # sfx: match_success
                    self._create_particles(
                        self.falling_block['x'] + self.BLOCK_WIDTH / 2,
                        self.falling_block['y'],
                        self.BLOCK_COLORS[self.falling_block['color_index']]
                    )
                
                self._spawn_block()
                return catch_reward + match_reward
            else:
                self.game_over = True
                # sfx: stack_collapse
                return -100.0
        
        if self.falling_block['y'] > self.SCREEN_HEIGHT:
            self.game_over = True
            # sfx: block_miss
            return -100.0

        return 0.0

    def _create_particles(self, x, y, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.uniform(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stack from bottom to top
        for block in self.stack:
            self._draw_block_with_glow(self.screen, block, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)

        # Draw falling block
        if self.falling_block and not self.game_over:
            self._draw_block_with_glow(self.screen, self.falling_block, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)

        # Draw catcher
        catcher_rect = pygame.Rect(int(self.catcher_x), self.SCREEN_HEIGHT - self.CATCHER_HEIGHT, self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        glow_rect = catcher_rect.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_WHITE, 60), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, catcher_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40))))
            color = (*p['color'], alpha)
            radius = max(1, int(p['life'] / 8))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(p['x']), int(p['y']), radius, color)


    def _draw_block_with_glow(self, surface, block_dict, width, height):
        color = self.BLOCK_COLORS[block_dict['color_index']]
        rect = pygame.Rect(int(block_dict['x']), int(block_dict['y']), width, height)
        
        glow_rect = rect.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*color, 70), glow_surf.get_rect(), border_radius=5)
        surface.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(surface, color, rect, border_radius=3)
        highlight_rect = rect.inflate(-4, -4)
        highlight_color = (255, 255, 255, 50)
        pygame.draw.rect(surface, highlight_color, highlight_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        match_text = self.font_ui.render(f"MATCHES: {self.match_count}/{self.WIN_MATCH_COUNT}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(match_text, (10, 40))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_end.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "match_count": self.match_count,
            "fall_speed": self.fall_speed
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # Can't call _get_observation before reset, so we reset first
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        
        # Test reset again to ensure it's idempotent
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block will not run in the testing environment, but is useful for local testing.
    # To run it, you'll need to unset the dummy video driver.
    # For example, on Linux/macOS: unset SDL_VIDEODRIVER
    # On Windows: set SDL_VIDEODRIVER=
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        # In a headless environment, we can't create a display.
        # We can still step through the environment and validate it.
        print("Running in headless mode. Manual play is disabled.")
        env = GameEnv()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
                env.reset()
        env.close()
        print("Headless test complete.")
    else:
        # --- Manual Play Example ---
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()

        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Stacker")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Action mapping for human player
            movement = 0 # no-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            action = [movement, 0, 0] # Space and Shift are not used

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Run at 30 FPS

        env.close()