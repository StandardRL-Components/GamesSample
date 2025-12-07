
# Generated: 2025-08-27T21:26:59.777457
# Source Brief: brief_02786.md
# Brief Index: 2786

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a number, "
        "then move the cursor to an adjacent spot and press Space again to move or merge."
    )

    game_description = (
        "A strategic puzzle game. Combine identical numbers on the grid to double their value. "
        "Your goal is to create the number 100 before the time runs out."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 5, 4
    CELL_SIZE = 80
    GRID_AREA_W, GRID_AREA_H = GRID_W * CELL_SIZE, GRID_H * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_W - GRID_AREA_W) // 2
    GRID_Y_OFFSET = (SCREEN_H - GRID_AREA_H) // 2 + 30
    TARGET_NUMBER = 100
    TIME_LIMIT = 200
    MAX_EPISODE_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (45, 55, 65)
    COLOR_GRID_LINES = (65, 75, 85)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_SELECTION = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_DARK = (10, 10, 10)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    NUMBER_COLORS = [
        (60, 120, 220),   # 2
        (60, 220, 120),   # 4
        (60, 200, 200),   # 8
        (220, 220, 60),   # 16
        (220, 160, 60),   # 32
        (220, 80, 60),    # 64
        (220, 60, 160),   # 128
        (255, 255, 255),  # 256+
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_grid = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 60, bold=True)
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tile_pos = None
        self.prev_space_held = False
        self.animations = []
        self.time_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.time_left = self.TIME_LIMIT
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tile_pos = None
        self.prev_space_held = False
        self.animations = []

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        self.steps += 1
        self.time_left -= 1
        reward = self._handle_action(movement, space_pressed)
        
        self._update_animations()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.win_state:
                reward = 100.0  # Goal-oriented win reward
                # sfx: win_sound
            else:
                reward = -50.0 # Goal-oriented lose reward
                # sfx: lose_sound
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_action(self, movement, space_pressed):
        reward = 0.0

        # 1. Handle cursor movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_H - 1)

        # 2. Handle selection/action
        if space_pressed:
            cx, cy = self.cursor_pos
            
            if self.selected_tile_pos is None:
                if self.grid[cy][cx] > 0:
                    self.selected_tile_pos = [cx, cy]
                    # sfx: select_tile
            else:
                sx, sy = self.selected_tile_pos
                
                if (cx, cy) == (sx, sy): # Deselect
                    self.selected_tile_pos = None
                    # sfx: deselect_tile
                elif abs(cx - sx) + abs(cy - sy) != 1: # Invalid move: not adjacent
                    reward = -0.1
                    self.selected_tile_pos = None
                    # sfx: invalid_move
                else:
                    source_val = self.grid[sy][sx]
                    target_val = self.grid[cy][cx]
                    
                    if target_val == 0: # Move to empty space
                        self.grid[cy][cx] = source_val
                        self.grid[sy][sx] = 0
                        self.selected_tile_pos = None
                        # sfx: move_tile
                    elif target_val == source_val: # Merge
                        new_val = source_val * 2
                        self.grid[cy][cx] = new_val
                        self.grid[sy][sx] = 0
                        self.selected_tile_pos = None
                        self.score += new_val
                        
                        # Calculate merge reward
                        reward += 1.0
                        if new_val >= 64: # Bonus for high-value merge
                            reward += 10.0
                        if source_val < 32 and self._is_larger_merge_possible(32):
                            reward -= 1.0 # Penalty for small merge when big one is possible
                        
                        self._add_animation('pop', (cx, cy), duration=15)
                        self._add_animation('particles', (cx, cy), duration=20, data={'value': new_val})
                        # sfx: merge_success
                        
                        if new_val >= self.TARGET_NUMBER:
                            self.win_state = True
                    else: # Invalid merge: different numbers
                        reward = -0.1
                        self.selected_tile_pos = None
                        # sfx: invalid_merge

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left}

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_AREA_W, self.GRID_AREA_H)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw tiles and grid lines
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                cell_rect = self._get_cell_rect(x, y)
                
                # Draw grid lines
                if x > 0:
                    pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (cell_rect.left, cell_rect.top), (cell_rect.left, cell_rect.bottom), 2)
                if y > 0:
                    pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (cell_rect.left, cell_rect.top), (cell_rect.right, cell_rect.top), 2)

                # Draw number tile
                value = self.grid[y][x]
                if value > 0:
                    color = self._get_color_for_value(value)
                    
                    # Pop animation
                    scale = 1.0
                    for anim in self.animations:
                        if anim['type'] == 'pop' and anim['pos'] == (x, y):
                            progress = anim['progress'] / anim['duration']
                            scale = 1.0 + 0.5 * math.sin(progress * math.pi) # Simple pop effect
                    
                    scaled_size = self.CELL_SIZE * 0.85 * scale
                    tile_rect = pygame.Rect(0, 0, scaled_size, scaled_size)
                    tile_rect.center = cell_rect.center
                    
                    pygame.draw.rect(self.screen, color, tile_rect, border_radius=8)
                    
                    # Draw number text
                    text_surf = self.font_grid.render(str(value), True, self.COLOR_TEXT_DARK)
                    text_rect = text_surf.get_rect(center=tile_rect.center)
                    self.screen.blit(text_surf, text_rect)

        # Draw selection highlight
        if self.selected_tile_pos:
            sx, sy = self.selected_tile_pos
            rect = self._get_cell_rect(sx, sy).inflate(6, 6)
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 4, border_radius=12)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = self._get_cell_rect(cx, cy)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=10)
        
        self._render_animations()

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        time_text = f"Time: {self.time_left}"
        target_text = f"Target: {self.TARGET_NUMBER}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        target_surf = self.font_ui.render(target_text, True, self.COLOR_TEXT)

        self.screen.blit(score_surf, (20, 15))
        self.screen.blit(time_surf, (self.SCREEN_W // 2 - time_surf.get_width() // 2, 15))
        self.screen.blit(target_surf, (self.SCREEN_W - target_surf.get_width() - 20, 15))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = "YOU WIN!" if self.win_state else "TIME UP"
        color = self.COLOR_WIN if self.win_state else self.COLOR_LOSE
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_W // 2, self.SCREEN_H // 2))
        
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_cell_rect(self, x, y):
        return pygame.Rect(
            self.GRID_X_OFFSET + x * self.CELL_SIZE,
            self.GRID_Y_OFFSET + y * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )

    def _get_color_for_value(self, value):
        if value == 0: return self.COLOR_GRID_BG
        log_val = int(math.log2(value)) - 1
        return self.NUMBER_COLORS[min(log_val, len(self.NUMBER_COLORS) - 1)]

    def _generate_grid(self):
        while True:
            self.grid = [[0] * self.GRID_W for _ in range(self.GRID_H)]
            possible_values = [2, 2, 2, 4, 4, 8]
            for y in range(self.GRID_H):
                for x in range(self.GRID_W):
                    self.grid[y][x] = random.choice(possible_values)
            if self._has_any_possible_merges():
                break

    def _has_any_possible_merges(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                val = self.grid[y][x]
                if val > 0:
                    if x + 1 < self.GRID_W and self.grid[y][x+1] == val: return True
                    if y + 1 < self.GRID_H and self.grid[y+1][x] == val: return True
        return False
        
    def _is_larger_merge_possible(self, threshold):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                val = self.grid[y][x]
                if val >= threshold:
                    if x + 1 < self.GRID_W and self.grid[y][x+1] == val: return True
                    if y + 1 < self.GRID_H and self.grid[y+1][x] == val: return True
        return False

    def _check_termination(self):
        return self.win_state or self.time_left <= 0 or self.steps >= self.MAX_EPISODE_STEPS

    def _add_animation(self, anim_type, pos, duration, data=None):
        if anim_type == 'particles':
            cell_rect = self._get_cell_rect(pos[0], pos[1])
            center = cell_rect.center
            num_particles = 15 + int(math.log2(data.get('value', 2))) * 2
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = random.randint(15, 30)
                color = self._get_color_for_value(data.get('value', 2))
                self.animations.append({
                    'type': 'particle', 'pos': list(center), 'vel': vel,
                    'life': life, 'max_life': life, 'color': color
                })
        else:
            self.animations.append({'type': anim_type, 'pos': pos, 'progress': 0, 'duration': duration})
    
    def _update_animations(self):
        for anim in self.animations:
            if anim['type'] == 'particle':
                anim['life'] -= 1
                anim['pos'][0] += anim['vel'][0]
                anim['pos'][1] += anim['vel'][1]
                anim['vel'][0] *= 0.95 # friction
                anim['vel'][1] *= 0.95
            else:
                anim['progress'] += 1
        self.animations = [anim for anim in self.animations if (anim.get('life', 1) > 0 and anim.get('progress', 0) < anim.get('duration', 1))]

    def _render_animations(self):
        for anim in self.animations:
            if anim['type'] == 'particle':
                life_ratio = anim['life'] / anim['max_life']
                size = int(8 * life_ratio)
                if size > 0:
                    pos = (int(anim['pos'][0]), int(anim['pos'][1]))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, (*anim['color'], int(255 * life_ratio)))
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (*anim['color'], int(255 * life_ratio)))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Setup a display window
    pygame.display.set_caption("Number Merger")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    
    while not done:
        movement = 0
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Time: {info['time_left']}")

        # Blit the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for manual play

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}")