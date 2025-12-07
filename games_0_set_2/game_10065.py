import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:46:26.788246
# Source Brief: brief_00065.md
# Brief Index: 65
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build a tower to reach a target height. Place colored blocks, but be careful—matching "
        "adjacent blocks of the same color will cause a chain reaction and remove them."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor, space to place a block, and shift to "
        "cycle through available block types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 16
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = 120  # Virtual grid height for tall towers
    GROUND_LEVEL_GRID = 100
    TARGET_HEIGHT = 100
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (26, 26, 46)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (241, 250, 238)
    COLOR_TARGET_LINE = (255, 200, 0)
    
    BLOCK_PALETTE = {
        'red': {'base': (230, 57, 70), 'light': (240, 100, 110), 'glow': (230, 57, 70, 50)},
        'blue': {'base': (29, 172, 214), 'light': (80, 200, 230), 'glow': (29, 172, 214, 50)},
        'green': {'base': (100, 200, 120), 'light': (150, 220, 170), 'glow': (100, 200, 120, 50)}
    }
    BLOCK_TYPES = ['red', 'blue', 'green']

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.blocks = {}
        self.particles = []
        self.cursor_pos = [0, 0]
        self.inventory = {}
        self.selected_block_idx = 0
        self.tower_height = 0
        self.camera_offset_y = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.blocks = {}
        # Create a solid ground base
        for x in range(self.GRID_WIDTH):
            self.blocks[(x, self.GROUND_LEVEL_GRID)] = {'type': 'ground', 'color': (80, 80, 90)}

        self.particles = []
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GROUND_LEVEL_GRID - 1]
        
        self.inventory = {'red': 200, 'blue': 200, 'green': 100}
        self.selected_block_idx = 0
        
        self.tower_height = 0
        self.camera_offset_y = 0.0
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        self._handle_movement(movement)
        
        # Rising edge detection for placing blocks
        if space_held and not self.prev_space_held:
            reward += self._place_block()
        
        # Rising edge detection for cycling block type
        if shift_held and not self.prev_shift_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.BLOCK_TYPES)
            # Sfx: UI_cycle

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self._update_particles()
        self._update_camera()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if self.tower_height >= self.TARGET_HEIGHT and not truncated:
                reward += 100.0 # Victory bonus
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Clamp cursor to be within bounds and not inside the ground
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GROUND_LEVEL_GRID - 1)
    
    def _place_block(self):
        place_pos = tuple(self.cursor_pos)
        block_type = self.BLOCK_TYPES[self.selected_block_idx]

        # Check if placement is valid
        if place_pos in self.blocks or self.inventory[block_type] <= 0:
            # Sfx: Error_buzz
            return 0.0

        # Check for support
        support_pos = (place_pos[0], place_pos[1] + 1)
        if support_pos not in self.blocks:
            # Sfx: Error_buzz
            return 0.0

        # Sfx: Block_place
        self.inventory[block_type] -= 1
        self.blocks[place_pos] = {
            'type': block_type,
            'color': self.BLOCK_PALETTE[block_type]['base']
        }
        
        reward = 0.1 # Base reward for placing a block
        
        # --- Physics and Chain Reactions ---
        reaction_count = self._handle_chain_reaction(place_pos)
        if reaction_count > 0:
            reward += 1.0
            if reaction_count > 20:
                reward += 5.0
            self._apply_gravity()
        
        self._update_tower_height()
        return reward

    def _handle_chain_reaction(self, start_pos):
        start_block = self.blocks.get(start_pos)
        if not start_block or start_block['type'] == 'green' or start_block['type'] == 'ground':
            return 0

        q = deque([start_pos])
        visited = {start_pos}
        chain_group = []
        
        while q:
            pos = q.popleft()
            chain_group.append(pos)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_pos = (pos[0] + dx, pos[1] + dy)
                neighbor_block = self.blocks.get(neighbor_pos)
                
                if neighbor_block and neighbor_pos not in visited:
                    if neighbor_block['type'] == start_block['type']:
                        visited.add(neighbor_pos)
                        q.append(neighbor_pos)

        if len(chain_group) > 1:
            # Sfx: Chain_reaction_start
            for pos in chain_group:
                block_color = self.blocks[pos]['color']
                self._create_particles(pos, block_color, 20)
                del self.blocks[pos]
            return len(chain_group)
        return 0

    def _apply_gravity(self):
        moved = True
        while moved:
            moved = False
            # Iterate from bottom-up to correctly handle falling stacks
            sorted_blocks = sorted([pos for pos, block in self.blocks.items() if block['type'] != 'ground'], key=lambda p: p[1], reverse=True)
            
            for pos in sorted_blocks:
                if pos not in self.blocks: continue # Might have been moved already
                
                block = self.blocks[pos]
                below_pos = (pos[0], pos[1] + 1)
                
                if below_pos not in self.blocks and below_pos[1] <= self.GROUND_LEVEL_GRID:
                    del self.blocks[pos]
                    self.blocks[below_pos] = block
                    moved = True

    def _update_tower_height(self):
        placed_blocks = [pos for pos, block in self.blocks.items() if block['type'] != 'ground']
        if not placed_blocks:
            self.tower_height = 0
            return
        
        lowest_y = min(p[1] for p in placed_blocks)
        self.tower_height = self.GROUND_LEVEL_GRID - lowest_y
    
    def _update_camera(self):
        # Smoothly pan camera up as tower grows
        target_offset = max(0, self.tower_height * self.GRID_SIZE - self.SCREEN_HEIGHT * 0.4)
        self.camera_offset_y += (target_offset - self.camera_offset_y) * 0.1

    def _check_termination(self):
        if self.tower_height >= self.TARGET_HEIGHT:
            return True
        if sum(self.inventory.values()) <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tower_height": self.tower_height,
            "inventory": self.inventory,
            "cursor_pos": self.cursor_pos,
        }
    
    def _world_to_screen(self, x, y):
        return int(x), int(y - self.camera_offset_y)

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_background()
        self._render_blocks()
        self._render_particles()
        self._render_cursor()

    def _render_background(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start_pos = self._world_to_screen(i * self.GRID_SIZE, 0)
            end_pos = self._world_to_screen(i * self.GRID_SIZE, self.GRID_HEIGHT * self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        
        for i in range(self.GRID_HEIGHT + 1):
            start_pos = self._world_to_screen(0, i * self.GRID_SIZE)
            end_pos = self._world_to_screen(self.SCREEN_WIDTH, i * self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        
        # Draw target height line
        target_y_world = (self.GROUND_LEVEL_GRID - self.TARGET_HEIGHT) * self.GRID_SIZE
        start_pos = self._world_to_screen(0, target_y_world)
        end_pos = self._world_to_screen(self.SCREEN_WIDTH, target_y_world)
        if start_pos[1] > 0 and start_pos[1] < self.SCREEN_HEIGHT:
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, start_pos, end_pos, 2)

    def _render_blocks(self):
        for pos, block in self.blocks.items():
            screen_x, screen_y = self._world_to_screen(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE)
            
            if screen_y > self.SCREEN_HEIGHT or screen_y < -self.GRID_SIZE:
                continue

            rect = pygame.Rect(screen_x, screen_y, self.GRID_SIZE, self.GRID_SIZE)
            
            if block['type'] == 'ground':
                pygame.draw.rect(self.screen, block['color'], rect)
                pygame.draw.rect(self.screen, (100,100,110), rect.inflate(-4,-4))
            else:
                palette = self.BLOCK_PALETTE[block['type']]
                # Glow effect
                glow_center = (rect.centerx, rect.centery)
                pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], int(self.GRID_SIZE*0.8), palette['glow'])
                # Main block
                pygame.draw.rect(self.screen, palette['base'], rect)
                # Hilight
                pygame.draw.rect(self.screen, palette['light'], rect.inflate(-self.GRID_SIZE*0.3, -self.GRID_SIZE*0.3))


    def _render_cursor(self):
        if self.game_over: return

        block_type = self.BLOCK_TYPES[self.selected_block_idx]
        color = self.BLOCK_PALETTE[block_type]['base']
        
        screen_x, screen_y = self._world_to_screen(self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE)
        rect = pygame.Rect(screen_x, screen_y, self.GRID_SIZE, self.GRID_SIZE)
        
        # Draw a semi-transparent preview
        preview_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        preview_surface.fill((*color, 100))
        self.screen.blit(preview_surface, rect.topleft)
        
        # Draw a border
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # --- Height Display ---
        height_text = self.font_main.render(f"Height: {self.tower_height} / {self.TARGET_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        # --- Score Display ---
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

        # --- Inventory Display ---
        start_x = self.SCREEN_WIDTH - 200
        for i, b_type in enumerate(self.BLOCK_TYPES):
            palette = self.BLOCK_PALETTE[b_type]
            count = self.inventory[b_type]
            
            # Draw block icon
            icon_rect = pygame.Rect(start_x + i * 65, 10, 20, 20)
            pygame.draw.rect(self.screen, palette['base'], icon_rect)
            pygame.draw.rect(self.screen, palette['light'], icon_rect.inflate(-6, -6))
            
            # Draw count
            count_text = self.font_small.render(f"{count}", True, self.COLOR_TEXT)
            self.screen.blit(count_text, (icon_rect.right + 5, icon_rect.centery - count_text.get_height() // 2))

            # Highlight selected
            if i == self.selected_block_idx:
                pygame.draw.rect(self.screen, palette['light'], (start_x + i * 65 - 4, 6, 28, 28), 2)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.tower_height >= self.TARGET_HEIGHT and not (self.steps >= self.MAX_STEPS) else "GAME OVER"
            end_text = self.font_main.render(msg, True, self.COLOR_TARGET_LINE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Particle System ---
    def _create_particles(self, grid_pos, color, count):
        # Sfx: Explosion
        world_x = (grid_pos[0] + 0.5) * self.GRID_SIZE
        world_y = (grid_pos[1] + 0.5) * self.GRID_SIZE
        
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 41)
            self.particles.append({'pos': [world_x, world_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'][0], p['pos'][1])
            if 0 < screen_pos[0] < self.SCREEN_WIDTH and 0 < screen_pos[1] < self.SCREEN_HEIGHT:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                radius = int(self.GRID_SIZE * 0.2 * (p['life'] / p['max_life']))
                if radius > 0:
                    # Using a rect for particles for a blocky explosion feel
                    rect = pygame.Rect(screen_pos[0]-radius, screen_pos[1]-radius, radius*2, radius*2)
                    temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                    temp_surf.fill(color)
                    self.screen.blit(temp_surf, rect.topleft)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # This block is not run by the tests, but is useful for local debugging.
    # It will open a window and let you play the game.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnetic Block Tower")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        # Continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Event-based for single presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                # Since auto_advance is False, we step on every key press
                obs, reward, terminated, truncated, info = env.step(action)
                action = [0, space_held, shift_held] # Reset movement after step
        
        # If no key is pressed, we can still step with a no-op
        # This part is commented out as it makes manual play difficult.
        # An agent would decide when to send no-ops.
        # if not any(pygame.key.get_pressed()):
        #     obs, reward, terminated, truncated, info = env.step([0,0,0])

        # Convert observation back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Height: {info['tower_height']}")
            # Wait for reset (press R)
            
        clock.tick(30) # Limit FPS for manual play
        
    env.close()