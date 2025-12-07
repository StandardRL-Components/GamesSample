
# Generated: 2025-08-28T00:09:10.297473
# Source Brief: brief_03700.md
# Brief Index: 3700

        
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
        "Controls: Press Space to cycle selection. Use ←↑→↓ to move the selected number. "
        "Merge identical numbers to double their value."
    )

    game_description = (
        "A fast-paced puzzle game where you merge numbers to reach a target value. "
        "Select a tile and move it onto a matching neighbor to combine them. "
        "Clear three stages before time runs out!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Visuals ---
        self.FONT_LARGE = pygame.font.Font(None, 60)
        self.FONT_MEDIUM = pygame.font.Font(None, 36)
        self.FONT_SMALL = pygame.font.Font(None, 24)
        self.FONT_TINY = pygame.font.Font(None, 18)

        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID_BG = (52, 73, 94)
        self.COLOR_GRID_LINE = (42, 58, 75)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_TEXT_SHADOW = (41, 52, 63)
        self.COLOR_SELECTOR = (241, 196, 15)
        
        self.VALUE_COLORS = {
            1: (52, 152, 219), 2: (46, 204, 113), 4: (26, 188, 156),
            8: (241, 196, 15), 16: (243, 156, 18), 32: (230, 126, 34),
            64: (211, 84, 0), 128: (231, 76, 60), 256: (192, 57, 43),
            512: (155, 89, 182), 1024: (142, 68, 173), 2048: (255, 255, 255)
        }

        # --- Game Config ---
        self.GRID_ROWS, self.GRID_COLS = 4, 5
        self.TILE_SIZE = 70
        self.TILE_MARGIN = 8
        self.GRID_WIDTH = self.GRID_COLS * (self.TILE_SIZE + self.TILE_MARGIN) - self.TILE_MARGIN
        self.GRID_HEIGHT = self.GRID_ROWS * (self.TILE_SIZE + self.TILE_MARGIN) - self.TILE_MARGIN
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 30
        
        self.TIME_PER_ACTION = 0.25 # Seconds deducted per action

        # --- State ---
        self.tiles = []
        self.particles = []
        self.grid = {}
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.targets = [256, 512, 1024]
        self.target_number = 256
        self.time_remaining = 60.0
        self.game_over = False
        self.win_state = False
        self.last_space_held = False
        self.selected_tile_idx = -1
        self.action_reward = 0.0

        self.reset()
        self.validate_implementation()
    
    def _get_color_for_value(self, value):
        if value > 1024: return self.VALUE_COLORS[2048]
        return self.VALUE_COLORS.get(value, self.COLOR_TEXT)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False
        self.win_state = False
        self.last_space_held = False
        
        self.particles.clear()
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.target_number = self.targets[self.stage - 1]
        self.time_remaining = 60.0
        self.score = 0
        self.tiles.clear()
        self.grid.clear()

        initial_tile_count = self.GRID_ROWS * self.GRID_COLS // 2
        for _ in range(initial_tile_count):
            self._add_random_tile()
        
        self.tiles.sort(key=lambda t: (t.row, t.col))
        self.selected_tile_idx = 0 if self.tiles else -1

    def _add_random_tile(self, value=None):
        empty_cells = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in self.grid:
                    empty_cells.append((r, c))
        
        if not empty_cells:
            return False

        r, c = self.np_random.choice(empty_cells, 1)[0]
        if value is None:
            value = self.np_random.choice([1, 2], p=[0.8, 0.2])
        
        new_tile = Tile(value, r, c)
        self.tiles.append(new_tile)
        self.grid[(r, c)] = new_tile
        return True

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0
        
        self.time_remaining = max(0, self.time_remaining - self.TIME_PER_ACTION)

        # 1. Handle selection
        if space_held and not self.last_space_held and self.tiles:
            self.selected_tile_idx = (self.selected_tile_idx + 1) % len(self.tiles)
            # SFX: select_beep.wav
        self.last_space_held = space_held

        # 2. Handle movement
        move_occurred = False
        if movement != 0 and self.tiles and self.selected_tile_idx != -1:
            if self.selected_tile_idx >= len(self.tiles): # Safety check
                self.selected_tile_idx = 0

            selected_tile = self.tiles[self.selected_tile_idx]
            sr, sc = selected_tile.row, selected_tile.col
            
            dr, dc = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            dr, dc = dc, dr # Flip for row/col vs x/y
            
            tr, tc = sr + dr, sc + dc # Target row/col

            if not (0 <= tr < self.GRID_ROWS and 0 <= tc < self.GRID_COLS):
                reward -= 0.1 * math.log2(max(2, selected_tile.value)) # Penalty for hitting wall
                selected_tile.start_bump_anim()
                # SFX: bump_wall.wav
            else:
                target_contents = self.grid.get((tr, tc))
                if target_contents is None: # Move to empty
                    del self.grid[(sr, sc)]
                    self.grid[(tr, tc)] = selected_tile
                    selected_tile.move_to(tr, tc)
                    move_occurred = True
                    # SFX: slide.wav
                elif target_contents.value == selected_tile.value: # Merge
                    new_value = target_contents.value * 2
                    target_contents.set_value(new_value)
                    
                    self.tiles.remove(selected_tile)
                    del self.grid[(sr, sc)]

                    reward += 1.0 + math.log2(new_value) * 0.5
                    self.score += new_value
                    
                    self._create_particles(target_contents)
                    move_occurred = True
                    self.selected_tile_idx = self.tiles.index(target_contents) # Select the new tile
                    # SFX: merge_success.wav

                    if new_value >= self.target_number:
                        reward += 10.0
                        self.stage += 1
                        if self.stage > len(self.targets):
                            self.win_state = True
                            self.game_over = True
                            reward += 100.0
                            # SFX: game_win.wav
                        else:
                            self._setup_stage()
                            # SFX: stage_clear.wav
                else: # Bump into different tile
                    reward -= 0.1 * math.log2(max(2, selected_tile.value))
                    selected_tile.start_bump_anim()
                    # SFX: bump_tile.wav
        
        # 3. Post-move logic
        if move_occurred:
            self._add_random_tile()
            self.tiles.sort(key=lambda t: (t.row, t.col))
            # Re-find the index of the selected tile after sorting
            if self.tiles and self.selected_tile_idx != -1:
                try:
                    current_selection = self.tiles[self.selected_tile_idx]
                    self.selected_tile_idx = self.tiles.index(current_selection)
                except (ValueError, IndexError):
                    self.selected_tile_idx = 0 if self.tiles else -1

        # 4. Update animations
        for tile in self.tiles: tile.update()
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0: self.particles.remove(p)

        # 5. Check for termination
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True
            reward -= 50.0
            # SFX: game_lose.wav
            
        terminated = self.game_over
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, tile):
        px, py = self._get_pixel_pos(tile.row, tile.col)
        px += self.TILE_SIZE / 2
        py += self.TILE_SIZE / 2
        color = self._get_color_for_value(tile.value)
        for _ in range(20):
            self.particles.append(Particle(px, py, color, self.np_random))

    def _get_pixel_pos(self, r, c):
        x = self.GRID_X + c * (self.TILE_SIZE + self.TILE_MARGIN)
        y = self.GRID_Y + r * (self.TILE_SIZE + self.TILE_MARGIN)
        return x, y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_background()
        self._render_tiles()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_background(self):
        grid_rect = pygame.Rect(self.GRID_X - self.TILE_MARGIN, self.GRID_Y - self.TILE_MARGIN,
                                self.GRID_WIDTH + 2 * self.TILE_MARGIN, self.GRID_HEIGHT + 2 * self.TILE_MARGIN)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                x, y = self._get_pixel_pos(r, c)
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, border_radius=5)

    def _render_tiles(self):
        if not self.tiles: return
        
        for i, tile in enumerate(self.tiles):
            is_selected = (i == self.selected_tile_idx) and not self.game_over
            tile.draw(self.screen, self, is_selected)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Helper to draw shadowed text
        def draw_text(text, font, color, pos, shadow_offset=(2,2)):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            self.screen.blit(text_surf, pos)

        # Score
        draw_text(f"SCORE: {self.score}", self.FONT_MEDIUM, self.COLOR_TEXT, (20, 15))
        # Time
        time_str = f"TIME: {self.time_remaining:.1f}"
        time_color = (231, 76, 60) if self.time_remaining < 10 else self.COLOR_TEXT
        draw_text(time_str, self.FONT_MEDIUM, time_color, (self.WIDTH - 200, 15))
        # Stage & Target
        draw_text(f"STAGE {self.stage}", self.FONT_SMALL, self.COLOR_TEXT, (20, 55))
        target_color = self._get_color_for_value(self.target_number)
        draw_text(f"TARGET: {self.target_number}", self.FONT_SMALL, target_color, (110, 55))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((20, 20, 20, 180))
        
        message = "YOU WIN!" if self.win_state else "TIME'S UP!"
        color = (46, 204, 113) if self.win_state else (231, 76, 60)
        
        text_surf = self.FONT_LARGE.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)
        
        score_surf = self.FONT_MEDIUM.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage, "time_remaining": self.time_remaining}
    
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Tile:
    def __init__(self, value, row, col):
        self.value = value
        self.row, self.col = row, col
        self.anim_scale = 0.0 # For spawn
        self.anim_merge_scale = 1.0 # For merge
        self.anim_bump_offset = 0.0 # For bump
        self.anim_state = "spawning"
        self.anim_duration = 10 # frames

    def set_value(self, value):
        self.value = value
        self.anim_merge_scale = 1.3
        self.anim_state = "merging"

    def move_to(self, r, c):
        self.row, self.col = r, c
        
    def start_bump_anim(self):
        self.anim_bump_offset = 10
        self.anim_state = "bumping"
        
    def update(self):
        if self.anim_state == "spawning":
            self.anim_scale += 1.0 / self.anim_duration
            if self.anim_scale >= 1.0:
                self.anim_scale = 1.0
                self.anim_state = "idle"
        elif self.anim_state == "merging":
            self.anim_merge_scale -= 0.05
            if self.anim_merge_scale <= 1.0:
                self.anim_merge_scale = 1.0
                self.anim_state = "idle"
        elif self.anim_state == "bumping":
            self.anim_bump_offset *= 0.6
            if self.anim_bump_offset < 1:
                self.anim_bump_offset = 0
                self.anim_state = "idle"

    def draw(self, surface, env, is_selected):
        x, y = env._get_pixel_pos(self.row, self.col)
        size = env.TILE_SIZE * self.anim_scale * self.anim_merge_scale
        
        # Bump animation
        if self.anim_state == "bumping":
            dx, dy = 0, 0
            # This needs the last move direction, which is complex. Simple shake:
            x += (random.random() - 0.5) * self.anim_bump_offset
            y += (random.random() - 0.5) * self.anim_bump_offset

        rect = pygame.Rect(0, 0, size, size)
        rect.center = (x + env.TILE_SIZE / 2, y + env.TILE_SIZE / 2)
        
        color = env._get_color_for_value(self.value)
        
        # Draw tile
        pygame.draw.rect(surface, color, rect, border_radius=5)
        
        # Draw number
        if self.anim_scale > 0.5:
            font_size = int(36 - len(str(self.value)) * 3)
            font = pygame.font.Font(None, font_size)
            text_surf = font.render(str(self.value), True, env.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=rect.center)
            surface.blit(text_surf, text_rect)
        
        # Draw selector
        if is_selected:
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            alpha = 150 + pulse * 105
            
            # Glow effect
            sel_rect = rect.inflate(10, 10)
            sel_surf = pygame.Surface(sel_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(sel_surf, (*env.COLOR_SELECTOR, alpha/3), (0,0, *sel_rect.size), border_radius=8)
            surface.blit(sel_surf, sel_rect.topleft)

            pygame.draw.rect(surface, env.COLOR_SELECTOR, rect, 3, border_radius=7)

class Particle:
    def __init__(self, x, y, color, np_random):
        self.x, self.y = x, y
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = np_random.uniform(15, 30)
        self.max_lifespan = self.lifespan
        self.color = color
        self.size = np_random.uniform(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)