import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:08:17.482632
# Source Brief: brief_00219.md
# Brief Index: 219
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import defaultdict

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks to build a tower. Merge three identical, adjacent blocks to create a larger one and score points, but be careful not to let your tower become unstable!"
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the falling block. Stack blocks and merge three of the same color to build your tower."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_WIDTH_UNITS = 10
    BLOCK_SIZE = 40
    PLAY_AREA_WIDTH_PX = PLAY_AREA_WIDTH_UNITS * BLOCK_SIZE
    PLAY_AREA_X_OFFSET = (SCREEN_WIDTH - PLAY_AREA_WIDTH_PX) // 2
    WIN_HEIGHT_UNITS = 20
    MAX_STEPS = 6000  # 60 seconds * 100 steps/sec

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (35, 40, 60)
    COLOR_TARGET_LINE = (0, 255, 128, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_event = pygame.font.SysFont("Impact", 28)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_blocks = []
        self.falling_block = None
        self.time_remaining = 0
        self.tower_height_units = 0
        self.block_spawn_timer = 0
        self.particles = []
        self.event_texts = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_blocks = []
        self.falling_block = None
        self.time_remaining = self.MAX_STEPS
        self.tower_height_units = 0
        self.block_spawn_timer = 0
        self.particles = []
        self.event_texts = []

        self._spawn_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        self._handle_input(movement)

        # --- Game Logic Update ---
        self.steps += 1
        self.time_remaining -= 1
        step_reward = 0

        # Update falling block
        if self.falling_block:
            self.falling_block['y'] += 1 # Descends 1 pixel per step
            
            landing_y = self._get_landing_y(self.falling_block['grid_x'])
            if self.falling_block['y'] >= landing_y - self.falling_block['height_px']:
                step_reward += self._place_block()
                step_reward += self._check_and_perform_merges()
                
                if not self.game_over: # Merges might not cause collapse
                    if self._check_collapse():
                        step_reward += -100
                        self.game_over = True
                        self._add_event_text("TOWER COLLAPSED!", (255, 50, 50))
                    else:
                        self._update_tower_height()
                        if self.tower_height_units >= self.WIN_HEIGHT_UNITS:
                            step_reward += 100
                            self.game_over = True
                            self._add_event_text("TOWER COMPLETE!", (50, 255, 50))
        else:
            self.block_spawn_timer -= 1
            if self.block_spawn_timer <= 0:
                self._spawn_block()

        # Timeout Check
        if self.time_remaining <= 0 and not self.game_over:
            step_reward += -10
            self.game_over = True
            self._add_event_text("TIME'S UP!", (255, 150, 0))

        # Update visual effects
        self._update_effects()

        self.score += step_reward
        terminated = self.game_over

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Core Logic Helpers ---

    def _handle_input(self, movement):
        if not self.falling_block:
            return
        
        # Per brief, action[0] is movement. 3=left, 4=right.
        if movement == 3: # Left
            self.falling_block['grid_x'] = max(0, self.falling_block['grid_x'] - 1)
        elif movement == 4: # Right
            self.falling_block['grid_x'] = min(self.PLAY_AREA_WIDTH_UNITS - 1, self.falling_block['grid_x'] + 1)

    def _spawn_block(self):
        block_type = self.np_random.integers(0, len(self.BLOCK_COLORS))
        self.falling_block = {
            'grid_x': self.np_random.integers(0, self.PLAY_AREA_WIDTH_UNITS),
            'y': 0,
            'type': block_type,
            'color': self.BLOCK_COLORS[block_type],
            'height_px': self.BLOCK_SIZE,
            'height_units': 1,
        }
        self.block_spawn_timer = 100 # 1 second spawn delay

    def _get_landing_y(self, grid_x):
        max_y = self.SCREEN_HEIGHT
        for block in self.tower_blocks:
            if block['grid_x'] == grid_x:
                max_y = min(max_y, block['y'])
        return max_y

    def _place_block(self):
        # Snap to grid
        landing_y = self._get_landing_y(self.falling_block['grid_x'])
        self.falling_block['y'] = landing_y - self.falling_block['height_px']
        
        self.tower_blocks.append(self.falling_block)
        self.falling_block = None
        # sfx: block_place.wav
        return 0.1 # Placement reward

    def _check_and_perform_merges(self):
        total_merge_reward = 0
        while True:
            merges_found_this_pass = False
            
            # Group blocks by y-level
            levels = defaultdict(list)
            for block in self.tower_blocks:
                levels[block['y']].append(block)

            blocks_to_remove = []
            blocks_to_add = []

            for y in sorted(levels.keys()):
                level_blocks = sorted(levels[y], key=lambda b: b['grid_x'])
                if len(level_blocks) < 3:
                    continue

                for i in range(len(level_blocks) - 2):
                    b1, b2, b3 = level_blocks[i], level_blocks[i+1], level_blocks[i+2]
                    
                    # Check for 3 adjacent blocks of the same type and height
                    is_match = (b1['type'] == b2['type'] == b3['type'])
                    is_adjacent = (b1['grid_x'] + 1 == b2['grid_x'] and b2['grid_x'] + 1 == b3['grid_x'])
                    is_same_height = (b1['height_px'] == b2['height_px'] == b3['height_px'])

                    if is_match and is_adjacent and is_same_height:
                        merges_found_this_pass = True
                        total_merge_reward += 1.0
                        self._add_event_text("+1 MERGE!", self.BLOCK_COLORS[b1['type']])
                        # sfx: merge.wav

                        for block in [b1, b2, b3]:
                            if block not in blocks_to_remove:
                                blocks_to_remove.append(block)
                        
                        new_height_px = b2['height_px'] * 2
                        new_height_units = b2['height_units'] * 2
                        
                        blocks_to_add.append({
                            'grid_x': b2['grid_x'],
                            'y': b2['y'] + b2['height_px'] - new_height_px,
                            'type': b2['type'],
                            'color': b2['color'],
                            'height_px': new_height_px,
                            'height_units': new_height_units,
                        })
                        
                        # Create particles
                        px = self.PLAY_AREA_X_OFFSET + b2['grid_x'] * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                        py = b2['y'] + b2['height_px'] // 2
                        for _ in range(30):
                            self._create_particle(px, py, b2['color'])
                        
                        # Only one merge per pass on a level to keep it simple
                        break 
            
            if merges_found_this_pass:
                self.tower_blocks = [b for b in self.tower_blocks if b not in blocks_to_remove]
                self.tower_blocks.extend(blocks_to_add)
            else:
                break # No more merges to be found

        return total_merge_reward

    def _check_collapse(self):
        sorted_blocks = sorted(self.tower_blocks, key=lambda b: b['y']) # Top to bottom
        
        for block in sorted_blocks:
            base_y = block['y'] + block['height_px']
            if base_y >= self.SCREEN_HEIGHT: # Block is on the ground
                continue

            is_supported = False
            for supporter in self.tower_blocks:
                # A supporter is directly below this block
                if supporter['y'] == base_y and supporter['grid_x'] == block['grid_x']:
                    is_supported = True
                    break
            
            if not is_supported:
                # sfx: collapse.wav
                return True # Unstable!
        return False
        
    def _update_tower_height(self):
        if not self.tower_blocks:
            self.tower_height_units = 0
            return
            
        max_h = 0
        for i in range(self.PLAY_AREA_WIDTH_UNITS):
            col_blocks = sorted([b for b in self.tower_blocks if b['grid_x'] == i], key=lambda b: b['y'])
            col_h = sum(b['height_units'] for b in col_blocks)
            if col_h > max_h:
                max_h = col_h
        self.tower_height_units = max_h


    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
        
        # Update event texts
        self.event_texts = [e for e in self.event_texts if e['life'] > 0]
        for e in self.event_texts:
            e['y'] -= 0.5
            e['life'] -= 1

    def _create_particle(self, x, y, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
            'life': self.np_random.integers(30, 60),
            'color': color
        })
    
    def _add_event_text(self, text, color):
        self.event_texts.append({
            'text': text, 'color': color,
            'x': self.SCREEN_WIDTH // 2, 'y': self.SCREEN_HEIGHT // 2,
            'life': 90 # 1.5 seconds at 60fps (adjust if needed)
        })

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_tower()
        if self.falling_block:
            self._render_falling_block()
        self._render_effects()

    def _render_background(self):
        # Vertical grid lines
        for i in range(self.PLAY_AREA_WIDTH_UNITS + 1):
            x = self.PLAY_AREA_X_OFFSET + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        # Horizontal grid lines
        for i in range(self.SCREEN_HEIGHT // self.BLOCK_SIZE + 1):
            y = i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X_OFFSET, y), (self.PLAY_AREA_X_OFFSET + self.PLAY_AREA_WIDTH_PX, y), 1)

    def _render_tower(self):
        for block in self.tower_blocks:
            self._draw_block(block)

    def _render_falling_block(self):
        # Draw projection
        landing_y = self._get_landing_y(self.falling_block['grid_x'])
        proj_rect = pygame.Rect(
            self.PLAY_AREA_X_OFFSET + self.falling_block['grid_x'] * self.BLOCK_SIZE,
            landing_y - self.falling_block['height_px'],
            self.BLOCK_SIZE,
            self.falling_block['height_px']
        )
        proj_color = self.falling_block['color'][:3] + (50,) # Add alpha
        shape_surf = pygame.Surface(proj_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, proj_color, (0, 0, *proj_rect.size), border_radius=4)
        self.screen.blit(shape_surf, proj_rect.topleft)
        
        # Draw actual block
        self._draw_block(self.falling_block)

    def _draw_block(self, block):
        rect = pygame.Rect(
            self.PLAY_AREA_X_OFFSET + block['grid_x'] * self.BLOCK_SIZE,
            block['y'],
            self.BLOCK_SIZE,
            block['height_px']
        )
        
        # Glow effect
        glow_rect = rect.inflate(8, 8)
        glow_color = block['color'][:3] + (100,)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, glow_color, (0, 0, *glow_rect.size), border_radius=8)
        self.screen.blit(shape_surf, glow_rect.topleft)

        # Main block
        pygame.draw.rect(self.screen, block['color'], rect, border_radius=4)
        # Border for definition
        pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in block['color']), rect, 2, border_radius=4)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 60.0))))
            color = p['color'][:3] + (alpha,)
            size = max(1, int(5 * (p['life'] / 60.0)))
            # Using gfxdraw for anti-aliased circles
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, size, size, size, color)
            self.screen.blit(temp_surf, (int(p['x']) - size, int(p['y']) - size))

        # Event texts
        for e in self.event_texts:
            alpha = max(0, min(255, int(510 * (e['life'] / 90.0)))) # Fade in and out
            text_surf = self.font_event.render(e['text'], True, e['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(e['x'], e['y']))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Timer
        time_sec = self.time_remaining / (self.MAX_STEPS / 60)
        time_text = f"TIME: {time_sec:.1f}"
        time_ratio = self.time_remaining / self.MAX_STEPS
        
        if time_ratio > 0.5:
            r, g = int(255 * (1 - time_ratio) * 2), 255
        else:
            r, g = 255, int(255 * time_ratio * 2)
        timer_color = (r, g, 0)
        
        time_surf = self.font_main.render(time_text, True, timer_color)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 40))
        
        # Tower Height
        height_text = f"HEIGHT: {self.tower_height_units} / {self.WIN_HEIGHT_UNITS}"
        height_color = (50, 255, 200) if self.tower_height_units < self.WIN_HEIGHT_UNITS else (255, 255, 50)
        height_surf = self.font_main.render(height_text, True, height_color)
        height_rect = height_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(height_surf, height_rect)
        
        # Target Line
        target_y = self.SCREEN_HEIGHT - (self.WIN_HEIGHT_UNITS / 2) * self.BLOCK_SIZE
        if target_y > 0:
            start_x = self.PLAY_AREA_X_OFFSET
            end_x = self.PLAY_AREA_X_OFFSET + self.PLAY_AREA_WIDTH_PX
            for x in range(start_x, end_x, 10):
                pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 5, target_y), 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "tower_height": self.tower_height_units,
        }

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The main loop is for human play and visualization, not for training.
    # It will not work in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Tower Merge Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("--- RESET ---")
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}. Info: {info}")
            # The env will not auto-reset, wait for 'r' key
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(100) # Run at game's native speed

    env.close()