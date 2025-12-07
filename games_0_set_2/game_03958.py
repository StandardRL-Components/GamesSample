import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place the selected block. Press shift to cycle block types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic tower defense game. Build a fortress to withstand waves of enemy attacks. Survive 20 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Visuals & Game Constants
        self.GRID_SIZE = 10
        self.CELL_SIZE = 36
        self.GRID_AREA_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_AREA_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (640 - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (400 - self.GRID_AREA_HEIGHT) // 2
        self.MAX_WAVES = 20
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CURSOR = (0, 150, 255)
        self.COLOR_CURSOR_INVALID = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 80, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (40, 200, 120)
        self.COLOR_HEALTH_BAR_BG = (90, 90, 90)

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Block Definitions
        self.BLOCK_DEFS = {
            1: {"name": "Wall", "max_health": 3, "color": (60, 180, 75)},
            2: {"name": "Bracer", "max_health": 8, "color": (70, 130, 180)},
        }
        self.NUM_BLOCK_TYPES = len(self.BLOCK_DEFS)

        # Initialize state variables
        self.grid = None
        self.block_health = None
        self.fortress_health = None
        self.max_fortress_health = None
        self.wave = None
        self.score = None
        self.cursor_pos = None
        self.selected_block_type = None
        self.prev_shift_state = None
        self.game_over = None
        self.win = None
        self.effects = None
        self.steps = None
        
        # This will be initialized in reset()
        self.np_random = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self.block_health = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        
        self.max_fortress_health = 100.0
        self.fortress_health = self.max_fortress_health
        self.wave = 1
        self.score = 0
        self.steps = 0
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_block_type = 1
        self.prev_shift_state = 0
        
        self.game_over = False
        self.win = False
        self.effects = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        terminated = False
        
        # 1. Handle player input
        self._handle_movement(movement)
        self._handle_block_cycling(shift_held)
        
        # 2. Handle primary action (placing a block)
        if space_held:
            placed, placement_reward = self._place_block()
            if placed:
                reward += placement_reward
                # Placing a block triggers the enemy wave
                wave_reward, terminated = self._simulate_enemy_attack()
                reward += wave_reward
                
                if not terminated:
                    self.wave += 1
                    if self.wave > self.MAX_WAVES:
                        self.win = True
                        terminated = True
                        reward += 100.0  # Victory bonus
                        self.score += 10000

        self.prev_shift_state = shift_held
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated envs are also terminated
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

    def _handle_block_cycling(self, shift_held):
        if shift_held and not self.prev_shift_state:
            self.selected_block_type += 1
            if self.selected_block_type > self.NUM_BLOCK_TYPES:
                self.selected_block_type = 1

    def _place_block(self):
        cx, cy = self.cursor_pos
        if self.grid[cy, cx] == 0:
            self.grid[cy, cx] = self.selected_block_type
            self.block_health[cy, cx] = self.BLOCK_DEFS[self.selected_block_type]["max_health"]
            self.score += 10
            self._add_effect('placement', self.cursor_pos, 10)
            return True, 0.1
        return False, 0.0

    def _simulate_enemy_attack(self):
        num_projectiles = self.wave
        reward = 0.0
        terminated = False
        
        for _ in range(num_projectiles):
            # Choose a random edge and a random target
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                start_pos = (self.np_random.uniform(0, self.GRID_SIZE), -0.5)
            elif edge == 1: # Right
                start_pos = (self.GRID_SIZE - 0.5, self.np_random.uniform(0, self.GRID_SIZE))
            elif edge == 2: # Bottom
                start_pos = (self.np_random.uniform(0, self.GRID_SIZE), self.GRID_SIZE - 0.5)
            else: # Left
                start_pos = (-0.5, self.np_random.uniform(0, self.GRID_SIZE))
            
            target_pos = (self.np_random.uniform(0, self.GRID_SIZE), self.np_random.uniform(0, self.GRID_SIZE))
            
            # Trace path
            path_cells = self._get_line_cells(start_pos, target_pos)
            self._add_effect('projectile_path', (start_pos, target_pos), 15)
            
            hit = False
            for cell_x, cell_y in path_cells:
                if self.grid[cell_y, cell_x] > 0:
                    self.block_health[cell_y, cell_x] -= 1
                    self._add_effect('impact', (cell_x, cell_y), 10)
                    if self.block_health[cell_y, cell_x] <= 0:
                        self.grid[cell_y, cell_x] = 0
                        self.score += 50
                        self._add_effect('destroy', (cell_x, cell_y), 20)
                    hit = True
                    break
            
            if not hit:
                self.fortress_health -= 10
                self._add_effect('fortress_hit', (0,0), 20)

        if self.fortress_health <= 0:
            self.fortress_health = 0
            self.game_over = True
            terminated = True
            reward = -100.0  # Defeat penalty
        else:
            reward = 1.0  # Wave survival reward
            self.score += self.wave * 20
            
        return reward, terminated

    def _get_line_cells(self, start, end):
        x1, y1 = start
        x2, y2 = end
        
        dx, dy = x2 - x1, y2 - y1
        steps = int(max(abs(dx), abs(dy)) * 4) # Sample 4 points per grid cell
        if steps == 0: return []
        
        x_inc, y_inc = dx / steps, dy / steps
        
        cells = set()
        x, y = x1, y1
        for _ in range(steps + 1):
            gx, gy = int(x), int(y)
            if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
                cells.add((gx, gy))
            x += x_inc
            y += y_inc
        return sorted(list(cells), key=lambda p: (p[0]-x1)**2 + (p[1]-y1)**2)

    def _add_effect(self, type, data, lifetime):
        self.effects.append({'type': type, 'data': data, 'life': lifetime, 'max_life': lifetime})

    def _update_and_draw_effects(self):
        new_effects = []
        for effect in self.effects:
            effect['life'] -= 1
            if effect['life'] > 0:
                self._draw_effect(effect)
                new_effects.append(effect)
        self.effects = new_effects

    def _draw_effect(self, effect):
        life_ratio = effect['life'] / effect['max_life']
        if effect['type'] == 'placement':
            gx, gy = effect['data']
            cx, cy = self._grid_to_pixel(gx, gy)
            radius = int((1 - life_ratio) * self.CELL_SIZE * 0.7)
            alpha = int(life_ratio * 255)
            color = self.BLOCK_DEFS[self.selected_block_type]['color']
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, (*color, alpha))
        elif effect['type'] == 'impact':
            gx, gy = effect['data']
            cx, cy = self._grid_to_pixel(gx, gy)
            radius = int(life_ratio * self.CELL_SIZE * 0.5)
            alpha = int(life_ratio * 255)
            color = (255, 255, 150, alpha)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
        elif effect['type'] == 'destroy':
            gx, gy = effect['data']
            cx, cy = self._grid_to_pixel(gx, gy)
            alpha = int(life_ratio * 200)
            for _ in range(5):
                angle = self.np_random.uniform(0, 2 * math.pi)
                dist = (1 - life_ratio) * self.CELL_SIZE * 0.8
                px = cx + int(math.cos(angle) * dist)
                py = cy + int(math.sin(angle) * dist)
                pygame.draw.circle(self.screen, (200, 200, 200, alpha), (px, py), 2)
        elif effect['type'] == 'projectile_path':
            start_grid, end_grid = effect['data']
            start_px = self._grid_to_pixel(start_grid[0], start_grid[1], center=True)
            end_px = self._grid_to_pixel(end_grid[0], end_grid[1], center=True)
            alpha = int(life_ratio * 150)
            pygame.draw.aaline(self.screen, (*self.COLOR_PROJECTILE, alpha), start_px, end_px)
        elif effect['type'] == 'fortress_hit':
            # Flash the screen red
            s = pygame.Surface((640, 400), pygame.SRCALPHA)
            alpha = int(life_ratio * 80)
            s.fill((255, 0, 0, alpha))
            self.screen.blit(s, (0, 0))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_blocks()
        self._update_and_draw_effects() # Effects are drawn over blocks
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_v = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v)
            # Horizontal
            start_h = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_h = (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h)

    def _render_blocks(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                block_type = self.grid[y, x]
                if block_type > 0:
                    block_def = self.BLOCK_DEFS[block_type]
                    health_ratio = self.block_health[y, x] / block_def["max_health"]
                    
                    color = block_def["color"]
                    # Desaturate and darken based on health
                    h, s, v, a = pygame.Color(*color).hsva
                    final_color = pygame.Color(0)
                    final_color.hsva = (h, s * (0.6 + 0.4 * health_ratio), v * (0.7 + 0.3 * health_ratio), 100)
                    
                    px, py = self._grid_to_pixel(x, y, center=False)
                    block_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                    
                    border_color = tuple(c*0.6 for c in final_color[:3])
                    pygame.draw.rect(self.screen, final_color, block_rect)
                    pygame.draw.rect(self.screen, border_color, block_rect, 2)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px, py = self._grid_to_pixel(cx, cy, center=False)
        cursor_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        is_valid_pos = self.grid[cy, cx] == 0
        color = self.COLOR_CURSOR if is_valid_pos else self.COLOR_CURSOR_INVALID
        
        pygame.draw.rect(self.screen, color, cursor_rect, 3) # Draw a thick border

    def _render_ui(self):
        # Wave and Score
        wave_text = self.font_main.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (15, 10))
        self.screen.blit(score_text, (640 - score_text.get_width() - 15, 10))

        # Fortress Health
        health_bar_width = 300
        health_bar_height = 20
        health_ratio = self.fortress_health / self.max_fortress_health
        current_health_width = int(health_bar_width * health_ratio)
        
        bg_rect = pygame.Rect((640 - health_bar_width) // 2, 400 - health_bar_height - 5, health_bar_width, health_bar_height)
        fg_rect = pygame.Rect((640 - health_bar_width) // 2, 400 - health_bar_height - 5, current_health_width, health_bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=5)
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fg_rect, border_radius=5)
        
        # Selected Block Info
        block_def = self.BLOCK_DEFS[self.selected_block_type]
        name_text = self.font_main.render(f"Build: {block_def['name']}", True, self.COLOR_TEXT)
        health_text = self.font_main.render(f"HP: {block_def['max_health']}", True, self.COLOR_TEXT)
        
        preview_rect = pygame.Rect(520, 160, 100, 100)
        pygame.draw.rect(self.screen, (40, 50, 60), preview_rect, border_radius=8)
        self.screen.blit(name_text, (525, 170))
        self.screen.blit(health_text, (525, 190))
        
        # Draw block preview
        block_preview_rect = pygame.Rect(545, 215, 50, 50)
        pygame.draw.rect(self.screen, block_def['color'], block_preview_rect)
        pygame.draw.rect(self.screen, tuple(c*0.6 for c in block_def['color']), block_preview_rect, 2)
        
        # Game Over / Victory
        if self.game_over or self.win:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 150) if self.win else (255, 100, 100)
            end_text = self.font_title.render(msg, True, color)
            text_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, text_rect)

    def _grid_to_pixel(self, gx, gy, center=True):
        px = self.GRID_OFFSET_X + gx * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + gy * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "fortress_health": self.fortress_health,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Mapping from Pygame keys to action components
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame window for human play
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.display.set_caption("Fortress Defense")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()

    print(env.user_guide)
    
    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] # move, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        move_found = False
        for key, move_action in key_map.items():
            if keys[key]:
                if not move_found:
                    action[0] = move_action
                    move_found = True
        
        # Other actions
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Wave: {info['wave']}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")

        if terminated:
            print("Game Over!")
            pygame.time.wait(3000) # Pause for 3 seconds before closing

        clock.tick(10) # Limit FPS for human play

    env.close()