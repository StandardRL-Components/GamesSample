
# Generated: 2025-08-28T04:06:32.642892
# Source Brief: brief_02191.md
# Brief Index: 2191

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a "
        "monster, then move to an adjacent one and press space again to swap. "
        "Press shift to cancel your selection."
    )

    game_description = (
        "Match 3 or more monsters to clear them from the board. Clear all monsters "
        "within the move limit to advance through three challenging stages."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    CELL_SIZE = 40
    GRID_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_TOP_LEFT_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2 - 80
    GRID_TOP_LEFT_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_BG = (25, 30, 50)
    COLOR_GRID_LINE = (45, 50, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED_GLOW = (255, 255, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    MONSTER_COLORS = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
        (255, 80, 255), (80, 255, 255), (255, 150, 50), (150, 50, 255),
        (50, 255, 150), (200, 200, 200), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 0, 128), (128, 255, 0)
    ]

    # Game settings
    MAX_STAGES = 3
    MOVES_PER_STAGE = 15
    ANIMATION_SPEED = 4 # frames per animation step

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.stage = 1
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.steps = 0
        
        self.animations = []
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False
        self.game_won = False
        self.animations = []
        self.particles = []

        self._start_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        if not self.animations:
            # Process player input only if no animations are running
            reward = self._process_input(action)
        
        self._update_animations()

        # If animations just finished, check for cascades and board state
        if not self.animations:
            matches = self._find_and_process_matches()
            if matches:
                reward += sum(len(m) for m in matches)
                # Chain reaction bonus
                if self.last_action_was_match:
                    reward += 1 
                self.last_action_was_match = True
                self._handle_gravity_and_refill()
                # This will trigger fall animations in the next step
            else:
                self.last_action_was_match = False
                # If board is settled, check for possible moves
                if not self._find_possible_moves():
                    self._shuffle_board()

        # Check for stage clear
        if not self.animations and self._is_board_cleared():
            reward += 5 # Stage clear reward
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_won = True
                terminated = True
                reward += 50 # Game won reward
            else:
                self._start_stage()
        
        # Check for termination
        if self.moves_left <= 0 and not self.animations:
            if not self._is_board_cleared():
                self.game_over = True
            terminated = True

        if self.game_won:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _process_input(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Cancel Selection ---
        if shift_press and self.selected_pos:
            self.selected_pos = None
            # sfx: cancel
            return 0

        # --- Select / Swap ---
        if space_press:
            cx, cy = self.cursor_pos
            if self.grid[cy][cx] == -1: # Cannot select empty space
                # sfx: error
                return 0

            if not self.selected_pos:
                self.selected_pos = [cx, cy]
                # sfx: select
            else:
                sx, sy = self.selected_pos
                is_adjacent = abs(cx - sx) + abs(cy - sy) == 1
                if is_adjacent:
                    # --- Perform Swap ---
                    self.moves_left -= 1
                    self.grid[cy][cx], self.grid[sy][sx] = self.grid[sy][sx], self.grid[cy][cx]
                    
                    matches = self._find_matches()
                    if not matches:
                        # Invalid swap, swap back
                        self.grid[cy][cx], self.grid[sy][sx] = self.grid[sy][sx], self.grid[cy][cx]
                        reward = -0.1
                        self.animations.append({
                            'type': 'invalid_swap', 'pos1': (sx, sy), 'pos2': (cx, cy),
                            'progress': 0, 'duration': self.ANIMATION_SPEED * 2
                        })
                        # sfx: invalid_swap
                    else:
                        # Valid swap, animation will be handled by match processing
                        self.animations.append({
                            'type': 'swap', 'pos1': (sx, sy), 'pos2': (cx, cy),
                            'progress': 0, 'duration': self.ANIMATION_SPEED
                        })
                        # sfx: swap_success
                    
                    self.selected_pos = None
                    self.last_action_was_match = False

                elif (cx, cy) == (sx, sy): # Deselect
                    self.selected_pos = None
                    # sfx: cancel
                else: # Select new monster
                    self.selected_pos = [cx, cy]
                    # sfx: select
        return reward

    def _start_stage(self):
        self.moves_left = self.MOVES_PER_STAGE
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self._generate_grid()
        while not self._find_possible_moves():
            self._generate_grid()
        self.last_action_was_match = False

    def _get_num_monster_types(self):
        return 5 * self.stage

    def _generate_grid(self):
        num_types = self._get_num_monster_types()
        self.grid = [[-1 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                possible_types = list(range(num_types))
                # Avoid creating initial matches
                if x > 1 and self.grid[y][x-1] == self.grid[y][x-2]:
                    if self.grid[y][x-1] in possible_types:
                        possible_types.remove(self.grid[y][x-1])
                if y > 1 and self.grid[y-1][x] == self.grid[y-2][x]:
                    if self.grid[y-1][x] in possible_types:
                        possible_types.remove(self.grid[y-1][x])
                
                if not possible_types: # Fallback if removal leaves no options
                    possible_types = list(range(num_types))

                self.grid[y][x] = self.np_random.choice(possible_types)
    
    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == -1: continue
                # Horizontal
                if x < self.GRID_WIDTH - 2 and self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2]:
                    matches.add(((x, y), (x+1, y), (x+2, y)))
                # Vertical
                if y < self.GRID_HEIGHT - 2 and self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x]:
                    matches.add(((x, y), (x, y+1), (x, y+2)))
        
        # Consolidate overlapping matches
        if not matches: return []
        
        groups = []
        while matches:
            first, *rest = matches
            first = set(first)
            matches = rest
            
            while True:
                merged = False
                for other in matches:
                    if first.intersection(other):
                        first.update(other)
                        matches.remove(other)
                        merged = True
                        break
                if not merged:
                    break
            groups.append(list(first))
        return groups

    def _find_and_process_matches(self):
        matches = self._find_matches()
        if matches:
            # sfx: match_found
            all_matched_coords = set()
            for group in matches:
                for x, y in group:
                    all_matched_coords.add((x, y))

            self.animations.append({
                'type': 'disappear', 'coords': list(all_matched_coords),
                'progress': 0, 'duration': self.ANIMATION_SPEED
            })
            for x, y in all_matched_coords:
                self._create_particles(x, y, self.grid[y][x])
                self.grid[y][x] = -1
        return matches

    def _handle_gravity_and_refill(self):
        # sfx: gravity_fall
        fall_animations = []
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    monster_type = self.grid[y][x]
                    self.grid[y + empty_count][x] = monster_type
                    self.grid[y][x] = -1
                    fall_animations.append({
                        'type': 'fall', 'from_pos': (x, y), 'to_pos': (x, y + empty_count),
                        'monster_type': monster_type, 'progress': 0, 'duration': self.ANIMATION_SPEED * empty_count
                    })
        
        # Refill top rows
        num_types = self._get_num_monster_types()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[y][x] == -1:
                    monster_type = self.np_random.integers(0, num_types)
                    self.grid[y][x] = monster_type
                    fall_animations.append({
                        'type': 'fall', 'from_pos': (x, y - self.GRID_HEIGHT), 'to_pos': (x, y),
                        'monster_type': monster_type, 'progress': 0, 'duration': self.ANIMATION_SPEED * self.GRID_HEIGHT
                    })
        if fall_animations:
            self.animations.extend(fall_animations)

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                    if self._find_matches():
                        self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                    if self._find_matches():
                        self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
        return False
    
    def _shuffle_board(self):
        # sfx: shuffle
        all_monsters = [self.grid[y][x] for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) if self.grid[y][x] != -1]
        self.np_random.shuffle(all_monsters)
        
        idx = 0
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                 if self.grid[y][x] != -1:
                    self.grid[y][x] = all_monsters[idx]
                    idx += 1
        
        # Ensure no matches and possible moves after shuffle
        while self._find_matches() or not self._find_possible_moves():
            self._generate_grid()

    def _is_board_cleared(self):
        return all(self.grid[y][x] == -1 for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH))

    def _update_animations(self):
        if not self.animations: return
        
        # We only process one "type" of animation at a time to keep logic simple
        # e.g., all swaps finish, then all disappears, then all falls
        current_anim_type = self.animations[0]['type']
        
        finished_anims = []
        for anim in self.animations:
            if anim['type'] == current_anim_type:
                anim['progress'] += 1
                if anim['progress'] >= anim['duration']:
                    finished_anims.append(anim)
        
        for anim in finished_anims:
            self.animations.remove(anim)
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # gravity
            p['life'] -= 1

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
            "stage": self.stage,
            "moves_left": self.moves_left,
        }

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        ui_x = self.GRID_TOP_LEFT_X + self.GRID_AREA_WIDTH + 40
        
        self._render_text("Stage", self.font_medium, self.COLOR_TEXT, (ui_x, 50))
        self._render_text(f"{self.stage} / {self.MAX_STAGES}", self.font_large, (255, 255, 255), (ui_x, 80))
        
        self._render_text("Moves Left", self.font_medium, self.COLOR_TEXT, (ui_x, 150))
        self._render_text(str(self.moves_left), self.font_large, (255, 255, 255), (ui_x, 180))
        
        self._render_text("Score", self.font_medium, self.COLOR_TEXT, (ui_x, 250))
        self._render_text(str(self.score), self.font_large, (255, 255, 255), (ui_x, 280))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._render_text("GAME OVER", self.font_large, (255, 50, 50), (self.SCREEN_WIDTH/2 - 120, self.SCREEN_HEIGHT/2 - 50))
        elif self.game_won:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._render_text("YOU WIN!", self.font_large, (50, 255, 50), (self.SCREEN_WIDTH/2 - 100, self.SCREEN_HEIGHT/2 - 50))

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)

        # Draw monsters
        rendered_this_frame = set()
        
        # Draw animated monsters first
        for anim in self.animations:
            p = anim['progress'] / anim['duration']
            if anim['type'] == 'swap' or anim['type'] == 'invalid_swap':
                p1 = anim['pos1']
                p2 = anim['pos2']
                
                if anim['type'] == 'invalid_swap' and p > 0.5:
                    p1, p2 = p2, p1 # swap back
                    p = (p - 0.5) * 2
                
                pos1_x = self.GRID_TOP_LEFT_X + (p1[0] + (p2[0] - p1[0]) * p) * self.CELL_SIZE
                pos1_y = self.GRID_TOP_LEFT_Y + (p1[1] + (p2[1] - p1[1]) * p) * self.CELL_SIZE
                
                pos2_x = self.GRID_TOP_LEFT_X + (p2[0] + (p1[0] - p2[0]) * p) * self.CELL_SIZE
                pos2_y = self.GRID_TOP_LEFT_Y + (p2[1] + (p1[1] - p2[1]) * p) * self.CELL_SIZE
                
                self._draw_monster_at_pixel(self.grid[p2[1]][p2[0]], pos1_x, pos1_y)
                self._draw_monster_at_pixel(self.grid[p1[1]][p1[0]], pos2_x, pos2_y)
                
                rendered_this_frame.add(p1)
                rendered_this_frame.add(p2)
            
            elif anim['type'] == 'disappear':
                scale = 1.0 - p
                for x,y in anim['coords']:
                    px = self.GRID_TOP_LEFT_X + x * self.CELL_SIZE
                    py = self.GRID_TOP_LEFT_Y + y * self.CELL_SIZE
                    self._draw_monster_at_pixel(anim.get('monster_type', 0), px, py, scale)
                    rendered_this_frame.add((x,y))

            elif anim['type'] == 'fall':
                fx, fy = anim['from_pos']
                tx, ty = anim['to_pos']
                pos_x = self.GRID_TOP_LEFT_X + fx * self.CELL_SIZE
                pos_y = self.GRID_TOP_LEFT_Y + (fy + (ty - fy) * p) * self.CELL_SIZE
                self._draw_monster_at_pixel(anim['monster_type'], pos_x, pos_y)
                rendered_this_frame.add(anim['to_pos'])

        # Draw static monsters
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) in rendered_this_frame: continue
                monster_type = self.grid[y][x]
                if monster_type != -1:
                    px = self.GRID_TOP_LEFT_X + x * self.CELL_SIZE
                    py = self.GRID_TOP_LEFT_Y + y * self.CELL_SIZE
                    self._draw_monster_at_pixel(monster_type, px, py)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['life']/10))
            
        # Draw selection and cursor
        if self.selected_pos:
            sx, sy = self.selected_pos
            rect = pygame.Rect(self.GRID_TOP_LEFT_X + sx * self.CELL_SIZE, self.GRID_TOP_LEFT_Y + sy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            
            # Glow effect
            glow_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            glow_surf.fill(self.COLOR_SELECTED_GLOW)
            self.screen.blit(glow_surf, rect.topleft)
            pygame.draw.rect(self.screen, (255,255,255), rect, 2, border_radius=4)
        
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_TOP_LEFT_X + cx * self.CELL_SIZE, self.GRID_TOP_LEFT_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

    def _draw_monster_at_pixel(self, monster_type, px, py, scale=1.0):
        if monster_type == -1: return
        
        size = int(self.CELL_SIZE * 0.7 * scale)
        center_x = int(px + self.CELL_SIZE / 2)
        center_y = int(py + self.CELL_SIZE / 2)
        color = self.MONSTER_COLORS[monster_type % len(self.MONSTER_COLORS)]
        
        shape_type = monster_type % 7
        
        if shape_type == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, size // 2, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size // 2, color)
        elif shape_type == 1: # Square
            rect = pygame.Rect(center_x - size // 2, center_y - size // 2, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif shape_type == 2: # Triangle Up
            points = [(center_x, center_y - size//2), (center_x - size//2, center_y + size//2), (center_x + size//2, center_y + size//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif shape_type == 3: # Diamond
            points = [(center_x, center_y - size//2), (center_x + size//2, center_y), (center_x, center_y + size//2), (center_x - size//2, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif shape_type == 4: # Triangle Down
            points = [(center_x, center_y + size//2), (center_x - size//2, center_y - size//2), (center_x + size//2, center_y - size//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif shape_type == 5: # Plus
            pygame.draw.rect(self.screen, color, (center_x - size//2, center_y - size//6, size, size//3), border_radius=2)
            pygame.draw.rect(self.screen, color, (center_x - size//6, center_y - size//2, size//3, size), border_radius=2)
        elif shape_type == 6: # Hexagon
            radius = size // 2
            points = [(center_x + int(radius * math.cos(math.radians(angle))), center_y + int(radius * math.sin(math.radians(angle)))) for angle in range(30, 361, 60)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _create_particles(self, grid_x, grid_y, monster_type):
        px = self.GRID_TOP_LEFT_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_TOP_LEFT_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.MONSTER_COLORS[monster_type % len(self.MONSTER_COLORS)]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(10, 20),
                'color': color
            })
            
    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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