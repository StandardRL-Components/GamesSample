
# Generated: 2025-08-28T05:50:31.419428
# Source Brief: brief_02739.md
# Brief Index: 2739

        
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

    # --- User-facing metadata ---
    user_guide = (
        "Controls: Use ← and → to move the falling block. Press Space to drop it quickly. Stack blocks as high as you can!"
    )
    game_description = (
        "A physics-based puzzle game where you stack falling blocks. Achieve the target height to win, but be careful! An unstable stack will collapse."
    )

    # --- Game configuration ---
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (40, 40, 60)
    COLOR_BG_BOTTOM = (10, 10, 20)
    COLOR_BLOCK_OUTLINE = (0, 0, 0)
    COLOR_TARGET_LINE = (255, 215, 0, 150) # Gold
    COLOR_TEXT = (240, 240, 240)
    COLOR_GAMEOVER = (255, 80, 80)
    COLOR_WIN = (80, 255, 80)
    BLOCK_COLORS = [
        (217, 83, 79), (91, 192, 222), (240, 173, 78), 
        (92, 184, 92), (155, 132, 204), (231, 123, 157)
    ]

    # Physics and Gameplay
    GROUND_Y = SCREEN_HEIGHT - 30
    GRAVITY = 0.05
    FALL_SPEED_INITIAL = 1.0
    DROP_SPEED = 5.0
    BLOCK_MIN_WIDTH = 50
    BLOCK_MAX_WIDTH = 100
    BLOCK_HEIGHT = 20
    MOVE_SPEED = 4.0
    WIN_HEIGHT = 300
    MAX_STEPS = 1500

    # Rewards
    REWARD_PLACE_BLOCK = 1.0
    REWARD_PER_HEIGHT_UNIT = 0.1
    REWARD_CAUTIOUS = -0.01
    REWARD_WOBBLE = -5.0
    REWARD_COLLAPSE = -100.0
    REWARD_WIN = 100.0

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks = []
        self.falling_block = None
        self.particles = []
        self.collapsed_blocks = []
        self.current_height = 0
        self.last_space_held = False

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_height = 0
        self.last_space_held = False
        
        # Create the stable base
        base_block = {
            'x': self.SCREEN_WIDTH / 2,
            'y': self.GROUND_Y - self.BLOCK_HEIGHT / 2,
            'w': self.SCREEN_WIDTH * 2, # Effectively infinite width
            'h': self.BLOCK_HEIGHT,
            'color': (80, 80, 80)
        }
        self.blocks = [base_block]
        self.particles = []
        self.collapsed_blocks = []
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            self._handle_actions(action)
            reward += self._update_physics_and_collisions()
            self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or (self.steps >= self.MAX_STEPS)
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_new_block(self):
        width = self.np_random.integers(self.BLOCK_MIN_WIDTH, self.BLOCK_MAX_WIDTH + 1)
        self.falling_block = {
            'x': self.SCREEN_WIDTH / 2,
            'y': -self.BLOCK_HEIGHT,
            'w': width,
            'h': self.BLOCK_HEIGHT,
            'vy': self.FALL_SPEED_INITIAL,
            'color': random.choice(self.BLOCK_COLORS)
        }

    def _handle_actions(self, action):
        if not self.falling_block:
            return

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self.falling_block['x'] -= self.MOVE_SPEED
        elif movement == 4: # Right
            self.falling_block['x'] += self.MOVE_SPEED
        
        # Clamp to screen
        self.falling_block['x'] = np.clip(
            self.falling_block['x'], 
            self.falling_block['w'] / 2, 
            self.SCREEN_WIDTH - self.falling_block['w'] / 2
        )

        # Drop action
        if space_held and not self.last_space_held:
            self.falling_block['vy'] = self.DROP_SPEED
            # sound: whoosh.wav
        
        self.last_space_held = space_held

    def _update_physics_and_collisions(self):
        reward = 0
        if not self.falling_block:
            return 0

        # Apply gravity
        self.falling_block['y'] += self.falling_block['vy']
        self.falling_block['vy'] = min(self.falling_block['vy'] + self.GRAVITY, self.DROP_SPEED)

        # Find landing spot
        highest_support_y = self.GROUND_Y
        fb_left = self.falling_block['x'] - self.falling_block['w'] / 2
        fb_right = self.falling_block['x'] + self.falling_block['w'] / 2

        for block in self.blocks:
            b_left = block['x'] - block['w'] / 2
            b_right = block['x'] + block['w'] / 2
            # Check for horizontal overlap
            if max(fb_left, b_left) < min(fb_right, b_right):
                highest_support_y = min(highest_support_y, block['y'] - block['h'] / 2)

        # Check for landing
        if self.falling_block['y'] + self.falling_block['h'] / 2 >= highest_support_y:
            self.falling_block['y'] = highest_support_y - self.falling_block['h'] / 2
            
            # Identify all supporting blocks at this y-level
            supporting_blocks = [b for b in self.blocks if abs((b['y'] - b['h']/2) - highest_support_y) < 1]
            
            support_min_x = min(b['x'] - b['w']/2 for b in supporting_blocks)
            support_max_x = max(b['x'] + b['w']/2 for b in supporting_blocks)

            # --- Stability Check ---
            if self.falling_block['x'] < support_min_x or self.falling_block['x'] > support_max_x:
                # COLLAPSE
                self.game_over = True
                reward += self.REWARD_COLLAPSE
                self._trigger_collapse_effect()
                self.falling_block = None
                # sound: collapse.wav
                return reward
            
            # --- SUCCESSFUL PLACEMENT ---
            placed_block = self.falling_block
            self.blocks.append(placed_block)
            self.falling_block = None
            
            reward += self.REWARD_PLACE_BLOCK
            # sound: place_block.wav
            
            new_height = self.GROUND_Y - (placed_block['y'] - placed_block['h'] / 2)
            if new_height > self.current_height:
                reward += (new_height - self.current_height) * self.REWARD_PER_HEIGHT_UNIT
                self.current_height = new_height
            
            support_center = (support_min_x + support_max_x) / 2
            overhang_dist = abs(placed_block['x'] - support_center)

            if overhang_dist < 5:
                reward += self.REWARD_CAUTIOUS
            
            support_width = support_max_x - support_min_x
            if support_width > 0 and overhang_dist / (support_width / 2) > 0.8: # Wobble if overhang is >40% of support half-width
                reward += self.REWARD_WOBBLE
            
            self._trigger_land_effect(placed_block)
            
            if self.current_height >= self.WIN_HEIGHT:
                self.game_over = True
                reward += self.REWARD_WIN
                # sound: win.wav
            else:
                self._spawn_new_block()
        
        return reward

    def _trigger_land_effect(self, block):
        # Spawn particles at the bottom corners of the landed block
        for i in range(10):
            side = 1 if i < 5 else -1
            pos = [block['x'] + side * block['w'] / 2, block['y'] + block['h'] / 2]
            vel = [self.np_random.uniform(-1, 1) * side, self.np_random.uniform(0.5, 1.5)]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'color': block['color']})

    def _trigger_collapse_effect(self):
        # Turn all stacked blocks (except base) into falling physics objects
        for block in self.blocks[1:]: # Skip the base
            self.collapsed_blocks.append({
                'x': block['x'], 'y': block['y'], 'w': block['w'], 'h': block['h'],
                'vx': self.np_random.uniform(-2, 2), 'vy': self.np_random.uniform(-2, 0),
                'angle': 0, 'v_angle': self.np_random.uniform(-5, 5), 'color': block['color']
            })
        self.blocks = [self.blocks[0]] # Keep only the base

    def _update_particles(self):
        # Landing particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity on particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # Collapsed block particles
        for b in self.collapsed_blocks:
            b['x'] += b['vx']
            b['y'] += b['vy']
            b['vy'] += self.GRAVITY * 2
            b['angle'] += b['v_angle']
        self.collapsed_blocks = [b for b in self.collapsed_blocks if b['y'] < self.SCREEN_HEIGHT + 50]
    
    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": self.current_height,
        }

    def _render_game(self):
        # Background gradient
        for y in range(self.SCREEN_HEIGHT):
            color_ratio = y / self.SCREEN_HEIGHT
            color = [
                (1 - color_ratio) * self.COLOR_BG_TOP[i] + color_ratio * self.COLOR_BG_BOTTOM[i]
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Target height line
        target_y = self.GROUND_Y - self.WIN_HEIGHT
        if self.current_height < self.WIN_HEIGHT:
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.SCREEN_WIDTH, target_y), 2)
            self._draw_text(f"{int(self.WIN_HEIGHT)}", 10, target_y - 20, self.font_main, self.COLOR_TARGET_LINE)

        # Draw all placed blocks
        for block in self.blocks:
            self._draw_block(block)

        # Draw collapsed blocks
        for block in self.collapsed_blocks:
            self._draw_block(block, rotated=True)

        # Draw falling block with glow
        if self.falling_block:
            # Glow effect
            glow_rect = pygame.Rect(0, 0, self.falling_block['w'] + 10, self.falling_block['h'] + 10)
            glow_rect.center = (int(self.falling_block['x']), int(self.falling_block['y']))
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            self._draw_rounded_rect(glow_surf, glow_surf.get_rect(), (*self.falling_block['color'], 50), 8)
            self.screen.blit(glow_surf, glow_rect.topleft)
            self._draw_block(self.falling_block)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            try:
                # Use a surface for alpha blending
                s = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (2, 2), 2)
                self.screen.blit(s, (int(p['pos'][0])-2, int(p['pos'][1])-2))
            except (ValueError, TypeError):
                # Handle potential color errors gracefully
                pass

    def _draw_block(self, block, rotated=False):
        rect = pygame.Rect(0, 0, block['w'], block['h'])
        rect.center = (block['x'], block['y'])
        
        if rotated:
            surf = pygame.Surface((block['w'], block['h']), pygame.SRCALPHA)
            self._draw_rounded_rect(surf, surf.get_rect(), block['color'], 6)
            self._draw_rounded_rect(surf, surf.get_rect(), self.COLOR_BLOCK_OUTLINE, 6, 2)
            rotated_surf = pygame.transform.rotate(surf, block['angle'])
            new_rect = rotated_surf.get_rect(center=rect.center)
            self.screen.blit(rotated_surf, new_rect.topleft)
        else:
            self._draw_rounded_rect(self.screen, rect, block['color'], 6)
            self._draw_rounded_rect(self.screen, rect, self.COLOR_BLOCK_OUTLINE, 6, 2)

    def _draw_rounded_rect(self, surface, rect, color, radius, width=0):
        if rect.width < 2 * radius or rect.height < 2 * radius:
            if width == 0:
                pygame.draw.rect(surface, color, rect)
            else:
                 pygame.draw.rect(surface, color, rect, width)
            return

        pygame.draw.rect(surface, color, (rect.left + radius, rect.top, rect.width - 2 * radius, rect.height), width)
        pygame.draw.rect(surface, color, (rect.left, rect.top + radius, rect.width, rect.height - 2 * radius), width)

        if width == 0:
            pygame.draw.circle(surface, color, (rect.left + radius, rect.top + radius), radius)
            pygame.draw.circle(surface, color, (rect.right - radius, rect.top + radius), radius)
            pygame.draw.circle(surface, color, (rect.left + radius, rect.bottom - radius), radius)
            pygame.draw.circle(surface, color, (rect.right - radius, rect.bottom - radius), radius)
        else:
            pygame.draw.circle(surface, color, (rect.left + radius, rect.top + radius), radius, width)
            pygame.draw.circle(surface, color, (rect.right - radius - (width-1), rect.top + radius), radius, width)
            pygame.draw.circle(surface, color, (rect.left + radius, rect.bottom - radius - (width-1)), radius, width)
            pygame.draw.circle(surface, color, (rect.right - radius - (width-1), rect.bottom - radius - (width-1)), radius, width)


    def _render_ui(self):
        # Display current height
        height_text = f"Height: {int(self.current_height)}"
        self._draw_text(height_text, 20, 20, self.font_main, self.COLOR_TEXT)
        
        # Display steps
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(steps_text, self.SCREEN_WIDTH - 150, 20, self.font_main, self.COLOR_TEXT)
        
        # Game over / Win message
        if self.game_over:
            if self.current_height >= self.WIN_HEIGHT:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            self._draw_text(msg, self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 40, self.font_large, color, center=True)
    
    def _draw_text(self, text, x, y, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (int(x), int(y))
        else:
            text_rect.topleft = (int(x), int(y))
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
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
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # Override screen for human rendering
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")

    terminated = False
    total_reward = 0

    # Game loop
    while True:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        quit_game = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and terminated:
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0

        if quit_game:
            break

        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score (cumulative reward): {total_reward:.2f}")
    print(f"Final Info: {info}")
    
    env.close()