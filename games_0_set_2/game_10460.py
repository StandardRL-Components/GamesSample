import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:26:21.546450
# Source Brief: brief_00460.md
# Brief Index: 460
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Define data structures for clarity
BlockType = namedtuple('BlockType', ['id', 'width', 'height', 'instability', 'color', 'unlock_level', 'unlock_time'])
PlacedBlock = namedtuple('PlacedBlock', ['rect', 'block_type', 'wobble_offset'])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack blocks to build a tower against the clock. Manage instability and use different block "
        "shapes to reach the goal height for each level."
    )
    user_guide = (
        "Use ←→ arrow keys to move the cursor, and ↑↓ to select a block. Press space to place the selected block."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = SCREEN_HEIGHT - 40
    INSTABILITY_LIMIT = 20.0
    LEVEL_TIME = 60.0

    # --- Colors ---
    COLOR_BG = (44, 62, 80) # Dark blue-grey
    COLOR_GRID = (52, 73, 94) # Slightly lighter
    COLOR_GROUND = (127, 140, 141) # Grey
    COLOR_TEXT = (236, 240, 241) # Light grey/white
    COLOR_TEXT_SHADOW = (44, 62, 80)
    COLOR_TIMER_DANGER = (231, 76, 60) # Red
    COLOR_GHOST = (255, 255, 255, 100)

    # --- Block Definitions ---
    ALL_BLOCK_TYPES = [
        BlockType(id=0, width=100, height=20, instability=1.0, color=(46, 204, 113), unlock_level=1, unlock_time=0), # Green
        BlockType(id=1, width=80, height=20, instability=1.5, color=(241, 196, 15), unlock_level=1, unlock_time=0),   # Yellow
        BlockType(id=2, width=60, height=20, instability=2.0, color=(230, 126, 34), unlock_level=1, unlock_time=0),  # Orange
        BlockType(id=3, width=40, height=20, instability=3.0, color=(211, 84, 0), unlock_level=1, unlock_time=0),   # Dark Orange
        BlockType(id=4, width=30, height=20, instability=4.0, color=(192, 57, 43), unlock_level=1, unlock_time=0),   # Red
        BlockType(id=5, width=70, height=15, instability=2.5, color=(142, 68, 173), unlock_level=2, unlock_time=20), # Purple
        BlockType(id=6, width=50, height=25, instability=3.5, color=(155, 89, 182), unlock_level=2, unlock_time=40), # Dark Purple
        BlockType(id=7, width=90, height=10, instability=3.0, color=(52, 152, 219), unlock_level=3, unlock_time=20), # Blue
        BlockType(id=8, width=20, height=40, instability=5.0, color=(41, 128, 185), unlock_level=3, unlock_time=40), # Dark Blue
    ]
    
    LEVEL_GOALS = {1: 150, 2: 250, 3: 350}

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.time_remaining = self.LEVEL_TIME
        self.total_instability = 0.0
        self.placed_blocks = []
        self.particles = []
        self.available_block_types = []
        self.current_selection_idx = 0
        self.cursor_x = self.SCREEN_WIDTH / 2
        self.last_space_held = False
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for dev, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        
        self._start_level(1)
        
        return self._get_observation(), self._get_info()

    def _start_level(self, level_num):
        self.level = level_num
        self.time_remaining = self.LEVEL_TIME
        self.total_instability = 0.0
        self.placed_blocks = []
        self.particles = []
        self.cursor_x = self.SCREEN_WIDTH / 2
        self.current_selection_idx = 0
        
        self.available_block_types = [bt for bt in self.ALL_BLOCK_TYPES if bt.unlock_level <= self.level and bt.unlock_time == 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        placement_reward = [0] # Use a list to pass by reference
        self._handle_input(movement, space_held, placement_reward)
        reward += placement_reward[0]

        # --- Update Game State ---
        self.time_remaining = max(0, self.time_remaining - 1/30) # Assuming 30 FPS
        self._update_available_blocks()
        self._update_particles()
        
        # --- Check Win/Loss Conditions ---
        tower_height = self._get_tower_height()
        
        if self.total_instability > self.INSTABILITY_LIMIT:
            reward = -100
            terminated = True
            self.game_over = True
            # sfx: tower_collapse
        
        if self.time_remaining <= 0:
            reward = -100
            terminated = True
            self.game_over = True
            # sfx: time_up_fail

        if not terminated and tower_height >= self.LEVEL_GOALS[self.level]:
            reward += 1 # Brief specifies +1 for level completion
            if self.level < 3:
                self._start_level(self.level + 1)
                # sfx: level_complete
            else:
                reward += 100 # Brief specifies +100 for winning
                terminated = True
                self.game_over = True
                # sfx: game_win

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, reward_ref):
        # Move cursor
        if movement == 3: # Left
            self.cursor_x -= 5
        elif movement == 4: # Right
            self.cursor_x += 5
        self.cursor_x = np.clip(self.cursor_x, 0, self.SCREEN_WIDTH)

        # Change block selection
        if movement == 1: # Up
            if self.available_block_types:
                self.current_selection_idx = (self.current_selection_idx - 1) % len(self.available_block_types)
        elif movement == 2: # Down
            if self.available_block_types:
                self.current_selection_idx = (self.current_selection_idx + 1) % len(self.available_block_types)

        # Place block on space press (not hold)
        if space_held and not self.last_space_held:
            self._place_block(reward_ref)
        self.last_space_held = space_held

    def _place_block(self, reward_ref):
        if not self.available_block_types: return

        selected_type = self.available_block_types[self.current_selection_idx]
        block_width = selected_type.width
        block_height = selected_type.height
        
        placement_x = self.cursor_x - block_width / 2
        
        # Find placement Y and supporting blocks
        support_y = self.GROUND_Y
        supporting_blocks = []
        for block in self.placed_blocks:
            if (placement_x < block.rect.right and placement_x + block_width > block.rect.left):
                if block.rect.top < support_y:
                    support_y = block.rect.top
                    supporting_blocks = [block]
                elif block.rect.top == support_y:
                    supporting_blocks.append(block)

        placement_y = support_y - block_height
        
        if placement_y < 0: # Cannot place above screen
            return

        # Calculate instability
        instability_added = selected_type.instability
        if supporting_blocks:
            support_center_x = sum(b.rect.centerx for b in supporting_blocks) / len(supporting_blocks)
            offset = abs((placement_x + block_width/2) - support_center_x)
            # More instability for being off-center, scaled by height
            height_factor = 1 + (self.GROUND_Y - placement_y) / self.SCREEN_HEIGHT
            instability_added *= (1 + 0.05 * offset) * height_factor

        self.total_instability += instability_added
        
        # Add block to tower
        new_rect = pygame.Rect(placement_x, placement_y, block_width, block_height)
        new_block = PlacedBlock(rect=new_rect, block_type=selected_type, wobble_offset=self.np_random.uniform(0, 2 * math.pi))
        self.placed_blocks.append(new_block)
        
        # Update score and reward
        self.score += 1
        if selected_type.instability <= 2.0:
            reward_ref[0] = 0.1 # Stable block
        else:
            reward_ref[0] = -0.1 # Unstable block
            
        # sfx: block_place
        self._create_particles(new_rect.midbottom, selected_type.color)

    def _update_available_blocks(self):
        level_time_elapsed = self.LEVEL_TIME - self.time_remaining
        newly_available = [
            bt for bt in self.ALL_BLOCK_TYPES 
            if bt.unlock_level == self.level 
            and 0 < bt.unlock_time <= level_time_elapsed 
            and bt not in self.available_block_types
        ]
        if newly_available:
            self.available_block_types.extend(newly_available)
            # sfx: new_block_unlocked
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "instability": self.total_instability}

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))
        
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, (149, 165, 166), (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)

    def _render_game(self):
        # Render placed blocks with wobble
        wobble_factor = min(1.0, self.total_instability / self.INSTABILITY_LIMIT)
        wobble_angle = math.sin(self.steps * 0.1) * 0.05 * wobble_factor
        
        for block in self.placed_blocks:
            self._render_wobbly_block(block, wobble_angle)

        # Render particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            size = p['size'] * (p['life'] / p['lifespan'])
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (p['pos'][0] - size/2, p['pos'][1] - size/2, size, size))

        # Render ghost block
        if self.available_block_types and not self.game_over:
            selected_type = self.available_block_types[self.current_selection_idx]
            ghost_w, ghost_h = selected_type.width, selected_type.height
            ghost_x = self.cursor_x - ghost_w / 2
            
            support_y = self.GROUND_Y
            for block in self.placed_blocks:
                if (ghost_x < block.rect.right and ghost_x + ghost_w > block.rect.left and block.rect.top < support_y):
                    support_y = block.rect.top
            ghost_y = support_y - ghost_h
            
            ghost_surf = pygame.Surface((ghost_w, ghost_h), pygame.SRCALPHA)
            ghost_surf.fill(selected_type.color[:3] + (100,))
            self.screen.blit(ghost_surf, (ghost_x, ghost_y))

    def _render_wobbly_block(self, block, global_wobble_angle):
        rect = block.rect
        color = block.block_type.color
        
        # Individual block wobble + global wobble
        local_wobble = math.sin(self.steps * 0.2 + block.wobble_offset) * 0.02
        angle = global_wobble_angle + local_wobble

        points = [rect.topleft, rect.topright, rect.bottomright, rect.bottomleft]
        center = rect.center
        
        rotated_points = []
        for x, y in points:
            dx, dy = x - center[0], y - center[1]
            new_x = dx * math.cos(angle) - dy * math.sin(angle) + center[0]
            new_y = dx * math.sin(angle) + dy * math.cos(angle) + center[1]
            rotated_points.append((int(new_x), int(new_y)))

        pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)
        # Add a highlight for 3D effect
        highlight_color = tuple(min(255, c + 30) for c in color)
        pygame.draw.line(self.screen, highlight_color, rotated_points[0], rotated_points[1], 2)


    def _render_ui(self):
        # --- Helper to draw text with shadow ---
        def draw_text(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # --- Instability Meter ---
        bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 20
        fill_ratio = self.total_instability / self.INSTABILITY_LIMIT
        fill_w = min(bar_w, int(bar_w * fill_ratio))
        
        # Interpolate color from green to red
        bar_color = (
            int(46 + (231 - 46) * fill_ratio),
            int(204 + (76 - 204) * fill_ratio),
            int(113 + (60 - 113) * fill_ratio)
        )
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_w, bar_h))
        if fill_w > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 2)
        draw_text("INSTABILITY", (bar_x + 5, bar_y - 2), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # --- Timer and Level ---
        timer_color = self.COLOR_TEXT if self.time_remaining > 10 else self.COLOR_TIMER_DANGER
        draw_text(f"TIME: {int(self.time_remaining):02}", (self.SCREEN_WIDTH - 140, 10), self.font_main, timer_color, self.COLOR_TEXT_SHADOW)
        draw_text(f"LEVEL: {self.level}/3", (self.SCREEN_WIDTH - 140, 40), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # --- Score ---
        draw_text(f"SCORE: {self.score}", (230, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # --- Block Selector UI ---
        if self.available_block_types:
            ui_y = self.SCREEN_HEIGHT - 25
            for i, block_type in enumerate(self.available_block_types):
                is_selected = (i == self.current_selection_idx)
                ui_x = self.SCREEN_WIDTH / 2 - (len(self.available_block_types) * 40 / 2) + i * 40
                
                size = 30 if is_selected else 20
                rect = pygame.Rect(ui_x - size/2, ui_y - size/2, size, size)
                
                pygame.draw.rect(self.screen, block_type.color, rect, border_radius=4)
                if is_selected:
                    pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=4)
        
        # --- Game Over/Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._get_tower_height() >= self.LEVEL_GOALS.get(3, float('inf')):
                msg = "TOWER COMPLETE!"
            else:
                msg = "GAME OVER"
            
            draw_text(msg, (self.SCREEN_WIDTH/2 - self.font_main.size(msg)[0]/2, self.SCREEN_HEIGHT/2 - 50), self.font_main, self.COLOR_TEXT, (0,0,0))
            final_score_msg = f"Final Score: {self.score}"
            draw_text(final_score_msg, (self.SCREEN_WIDTH/2 - self.font_small.size(final_score_msg)[0]/2, self.SCREEN_HEIGHT/2), self.font_small, self.COLOR_TEXT, (0,0,0))

    def _get_tower_height(self):
        if not self.placed_blocks:
            return 0
        min_y = min(b.rect.top for b in self.placed_blocks)
        return self.GROUND_Y - min_y

    def _create_particles(self, pos, color):
        for _ in range(10):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(-2, 0)],
                'life': 20,
                'lifespan': 20,
                'color': color,
                'size': self.np_random.uniform(2, 6)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the evaluation system.
    # Set the video driver to a real one if you want to see the game.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Tower Builder")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    game_is_done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_is_done = False
        
        action = [0, 0, 0] # Default to no-op
        if not game_is_done:
            keys = pygame.key.get_pressed()
            
            movement = 0
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                game_is_done = True
        else:
            # Still get observation to show final screen
            obs = env._get_observation()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()