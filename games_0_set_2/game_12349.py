import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:02:59.701980
# Source Brief: brief_02349.md
# Brief Index: 2349
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the goal is to build the tallest, strongest tower
    possible before it collapses. The player selects from different block types
    and places them on a growing structure, battling against increasing wind
    and the tower's own instability.

    The environment prioritizes visual quality and "game feel", with smooth
    animations, particle effects, and clear feedback for all actions.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Build the tallest tower by placing blocks of different strengths. "
        "Battle against increasing wind and instability to prevent your tower from collapsing."
    )
    user_guide = (
        "Controls: Use ←→ to move the cursor, and ↑↓ to select a block type. "
        "Press space to place the block."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 380
    MAX_STEPS = 1000
    WIN_SCORE = 200

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 35, 50)
    COLOR_GROUND = (80, 80, 90)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_PARTICLE = (100, 120, 150, 150) # RGBA

    # --- Block Type Definitions ---
    # (Color, Size (W, H), Strength, Weight)
    BLOCK_TYPES = [
        {'color': (50, 205, 50), 'size': (80, 20), 'strength': 5, 'weight': 1.0},  # Green (Strong)
        {'color': (65, 105, 225), 'size': (60, 20), 'strength': 3, 'weight': 0.7},  # Blue (Medium)
        {'color': (220, 20, 60),  'size': (40, 20), 'strength': 1, 'weight': 0.4},  # Red (Weak)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.blocks = []
        self.cursor_x = self.SCREEN_WIDTH // 2
        self.selected_block_idx = 0
        self.wind_strength = 0.0
        self.tower_height_blocks = 0
        self.sway = 0.0
        self.sway_velocity = 0.0
        self.particles = []
        self.last_action_state = {'movement': 0, 'space': False}
        
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.blocks = []
        self.cursor_x = self.SCREEN_WIDTH // 2
        self.selected_block_idx = 0
        self.wind_strength = 0.0
        self.tower_height_blocks = 0
        self.sway = 0.0
        self.sway_velocity = 0.0
        self.particles = [self._create_particle() for _ in range(50)]
        self.last_action_state = {'movement': 0, 'space': False}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 3: # Left
            self.cursor_x = max(0, self.cursor_x - 10)
        elif movement == 4: # Right
            self.cursor_x = min(self.SCREEN_WIDTH, self.cursor_x + 10)
        
        # Change block type (on press only)
        is_up_press = movement == 1 and self.last_action_state['movement'] != 1
        is_down_press = movement == 2 and self.last_action_state['movement'] != 2
        if is_up_press:
            self.selected_block_idx = (self.selected_block_idx - 1 + len(self.BLOCK_TYPES)) % len(self.BLOCK_TYPES)
        if is_down_press:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.BLOCK_TYPES)
        
        # Place block (on press only)
        is_space_press = space_held and not self.last_action_state['space']
        if is_space_press:
            place_reward, placed_successfully = self._place_block()
            reward += place_reward
            # SFX Placeholder: Play 'place_block.wav' or 'error.wav'
        
        self.last_action_state = {'movement': movement, 'space': space_held}

        # --- Update Game Logic ---
        self._update_physics()
        self._update_particles()
        
        # --- Check for Collapse ---
        if self._check_collapse():
            self.game_over = True
            reward = -100 # Large penalty for collapse
            # SFX Placeholder: Play 'tower_collapse.wav'
        
        # --- Check for Win Condition ---
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            reward = 100 # Large reward for winning
            # SFX Placeholder: Play 'victory.wav'
        
        # --- Check for Max Steps ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS and not self.game_over
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _place_block(self):
        block_info = self.BLOCK_TYPES[self.selected_block_idx]
        block_w, block_h = block_info['size']
        
        new_block_x = self.cursor_x - block_w / 2
        
        # Find highest supporting block
        support_y = self.GROUND_Y
        supporting_block = None
        for b in self.blocks:
            b_x, b_y, b_w, _, _, _ = b
            # Check for horizontal overlap
            if not (new_block_x + block_w < b_x or new_block_x > b_x + b_w):
                if b_y < support_y:
                    support_y = b_y
                    supporting_block = b

        new_block_y = support_y - block_h

        # Prevent placing below screen
        if new_block_y + block_h < 0:
            return -1, False # Invalid placement penalty
        
        # Add the new block
        new_block = [
            new_block_x, new_block_y, block_w, block_h, 
            self.selected_block_idx, 0.0 # x, y, w, h, type_idx, instability
        ]
        self.blocks.append(new_block)
        
        # Update score and height
        self.score += block_info['strength']
        new_height_blocks = (self.GROUND_Y - new_block_y) // block_h
        
        reward = block_info['strength'] # Reward for placing a block
        if new_height_blocks > self.tower_height_blocks:
            self.tower_height_blocks = new_height_blocks
            reward += 10 # Bonus for new height level
        
        return reward, True

    def _update_physics(self):
        # Update wind based on height
        self.wind_strength = self.tower_height_blocks * 0.005 # Tuned for gameplay
        
        # Update tower sway (visual effect only)
        restoring_force = -self.sway * 0.01
        damping = -self.sway_velocity * 0.1
        wind_force = self.wind_strength * (random.random() * 0.5 + 0.5)
        self.sway_velocity += restoring_force + damping + wind_force
        self.sway += self.sway_velocity
        self.sway = max(-15, min(15, self.sway)) # Clamp max sway

        # Recalculate instability for the entire tower
        self.blocks.sort(key=lambda b: b[1]) # Sort by Y-pos, lowest first
        
        for i, block in enumerate(self.blocks):
            # Find its support
            support_center_x = self.SCREEN_WIDTH / 2
            support_width = self.SCREEN_WIDTH
            support_instability = 0.0
            
            # Find the block directly below it that provides the most support
            best_support = None
            max_overlap = 0
            
            for potential_support in self.blocks[:i]:
                if abs(potential_support[1] - (block[1] + block[3])) < 5: # Is it directly below?
                    overlap_min = max(block[0], potential_support[0])
                    overlap_max = min(block[0] + block[2], potential_support[0] + potential_support[2])
                    overlap = max(0, overlap_max - overlap_min)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_support = potential_support
            
            if best_support:
                support_center_x = best_support[0] + best_support[2] / 2
                support_width = best_support[2]
                support_instability = best_support[5] # instability is at index 5

            # Instability model: offset from support + inherited instability
            offset = (block[0] + block[2] / 2) - support_center_x
            normalized_offset = offset / support_width
            block[5] = normalized_offset + support_instability

    def _check_collapse(self):
        # Check for collapse based on instability
        for b in self.blocks:
            # Add wind effect to instability check
            height_factor = (self.GROUND_Y - b[1]) / self.SCREEN_HEIGHT
            total_instability = b[5] + self.wind_strength * height_factor * 5.0

            if abs(total_instability) > 0.7: # Collapse if effective CoM is >70% off center
                return True
        return False
    
    def _create_particle(self):
        return [
            random.randint(0, self.SCREEN_WIDTH),
            random.randint(0, self.SCREEN_HEIGHT),
            random.uniform(0.5, 2.0) # size
        ]

    def _update_particles(self):
        wind_speed = 1 + self.wind_strength * 200
        for p in self.particles:
            p[0] += wind_speed * (p[2] / 2.0) # Move right based on wind and size
            if p[0] > self.SCREEN_WIDTH:
                p[0] = -5
                p[1] = random.randint(0, self.SCREEN_HEIGHT)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Draw wind particles
        for x, y, size in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(x), int(y)), int(size))

        # Draw placed blocks with sway
        for x, y, w, h, type_idx, instability in self.blocks:
            height_factor = (self.GROUND_Y - y) / self.SCREEN_HEIGHT
            sway_offset = self.sway * height_factor * height_factor
            
            block_color = self.BLOCK_TYPES[type_idx]['color']
            rect = (int(x + sway_offset), int(y), int(w), int(h))
            
            pygame.gfxdraw.box(self.screen, rect, block_color)
            pygame.gfxdraw.rectangle(self.screen, rect, (0,0,0,50)) # Subtle border

        # Draw cursor and block preview
        if not self.game_over:
            block_info = self.BLOCK_TYPES[self.selected_block_idx]
            b_w, b_h = block_info['size']
            preview_x = self.cursor_x - b_w / 2
            
            # Find placement Y
            support_y = self.GROUND_Y
            for b in self.blocks:
                b_x, b_y, b_w_placed, _, _, _ = b
                if not (preview_x + b_w < b_x or preview_x > b_x + b_w_placed):
                    if b_y < support_y:
                        support_y = b_y
            preview_y = support_y - b_h
            
            preview_color = (*block_info['color'], 128) # RGBA with alpha
            rect = (int(preview_x), int(preview_y), int(b_w), int(b_h))
            
            # Use a separate surface for transparency
            shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
            pygame.gfxdraw.box(shape_surf, (0, 0, b_w, b_h), preview_color)
            self.screen.blit(shape_surf, rect)

            # Cursor line
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_x, 0), (self.cursor_x, self.SCREEN_HEIGHT), 1)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Wind
        wind_str = f"WIND: {self.wind_strength:.3f}"
        wind_surf = self.font_ui.render(wind_str, True, self.COLOR_TEXT)
        self.screen.blit(wind_surf, (self.SCREEN_WIDTH - wind_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "VICTORY!" if self.win else "TOWER COLLAPSED"
            color = (50, 255, 50) if self.win else (255, 50, 50)
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Add a background to the text for readability
            bg_rect = over_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, (0,0,0,180), bg_rect, border_radius=10)
            self.screen.blit(over_surf, over_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wind": self.wind_strength,
            "height": self.tower_height_blocks,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example usage:
if __name__ == '__main__':
    # This block will not run in the testing environment, but is useful for local development.
    # It requires pygame to be installed with display support.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    pygame.display.set_caption("Tower Builder")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # --- Action mapping for human play ---
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0 # 0=released, 1=held
        shift_held = 0 # 0=released, 1=held

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()