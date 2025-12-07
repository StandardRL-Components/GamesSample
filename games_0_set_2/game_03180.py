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

    user_guide = (
        "Controls: Arrow keys to move cursor. Hold space and use an arrow key to swap gems."
    )

    game_description = (
        "Strategically maneuver gems on a grid to create matches of 3 or more, aiming to clear the entire board within a limited number of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    CELL_SIZE = 40
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    NUM_GEM_TYPES = 6
    STARTING_MOVES = 30
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 100)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (100, 150, 255), # Blue
        (255, 255, 100), # Yellow
        (200, 100, 255), # Purple
        (255, 150, 50),  # Orange
    ]
    GEM_HIGHLIGHTS = [
        (255, 150, 150),
        (150, 255, 150),
        (170, 200, 255),
        (255, 255, 180),
        (230, 170, 255),
        (255, 200, 130),
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.animations = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.last_action_was_move = False
        self.last_move_dir = None
        self.swap_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_remaining = self.STARTING_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        self.animations = []
        self.particles = []
        self.last_move_dir = None
        self.swap_info = {}

        self._generate_board()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        self.steps += 1
        move_attempted = False

        # --- Handle player input ---
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.last_move_dir = (dx, dy) # Store last direction for swap
            
            if self.selected_gem_pos:
                # If a gem is selected, movement triggers a swap
                target_pos = [self.selected_gem_pos[0] + dx, self.selected_gem_pos[1] + dy]
                if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                    self._attempt_swap(self.selected_gem_pos, target_pos)
                    move_attempted = True
                self.selected_gem_pos = None # Deselect after attempt
            else:
                # Otherwise, just move the cursor
                self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
                self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
        
        if space_pressed:
            if self.selected_gem_pos:
                # Pressing space again deselects
                self.selected_gem_pos = None
            else:
                # Select the gem under the cursor
                self.selected_gem_pos = list(self.cursor_pos)

        # --- Process game logic if a move was made ---
        if move_attempted:
            self.moves_remaining -= 1
            
            # Swap gems
            pos1, pos2 = self.swap_info['from'], self.swap_info['to']
            self.grid[pos1[1]][pos1[0]], self.grid[pos2[1]][pos2[0]] = self.grid[pos2[1]][pos2[0]], self.grid[pos1[1]][pos1[0]]
            
            # Check for matches
            all_matches = self._find_all_matches()
            
            if not all_matches:
                # Invalid move, swap back
                self.grid[pos1[1]][pos1[0]], self.grid[pos2[1]][pos2[0]] = self.grid[pos2[1]][pos2[0]], self.grid[pos1[1]][pos1[0]]
                self.swap_info['is_revert'] = True # Animate the revert
                reward = -0.1
            else:
                # Valid move, process chain reaction
                chain_level = 1
                while all_matches:
                    # Calculate reward
                    num_matched = len(all_matches)
                    reward += num_matched * 1.0
                    if chain_level > 1: reward += 10.0 # Chain reaction bonus
                    self.score += num_matched
                    
                    # Handle matches (clear gems, add particles)
                    self._handle_matches(all_matches)
                    
                    # Apply gravity and refill
                    self._apply_gravity_and_refill()
                    
                    # Check for new matches
                    all_matches = self._find_all_matches()
                    chain_level += 1
        
        # --- Check for termination ---
        is_board_clear = all(gem == 0 for row in self.grid for gem in row)
        if is_board_clear:
            self.game_over = True
            reward += 100.0 # Win bonus
        
        if self.moves_remaining <= 0 and not is_board_clear:
            self.game_over = True
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _attempt_swap(self, pos1, pos2):
        self.swap_info = {
            'type': 'swap',
            'from': list(pos1), 'to': list(pos2), 'progress': 0, 'is_revert': False
        }
        self.animations.append(self.swap_info)

    def _generate_board(self):
        self.grid = [[self.np_random.integers(1, self.NUM_GEM_TYPES + 1) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        # Ensure no initial matches and at least one possible move
        while self._find_all_matches() or not self._find_possible_moves():
            matches = self._find_all_matches()
            if matches:
                for x, y in matches:
                    self.grid[y][x] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
            else: # No matches but no moves
                # Shuffle the board to create moves
                flat_grid = [gem for row in self.grid for gem in row]
                self.np_random.shuffle(flat_grid)
                self.grid = [flat_grid[i*self.GRID_WIDTH:(i+1)*self.GRID_WIDTH] for i in range(self.GRID_HEIGHT)]

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == 0: continue
                # Horizontal
                if x < self.GRID_WIDTH - 2 and self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical
                if y < self.GRID_HEIGHT - 2 and self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                    if self._find_all_matches():
                        self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                    if self._find_all_matches():
                        self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
        return False

    def _handle_matches(self, matches):
        # sound: match.wav
        for x, y in matches:
            gem_type = self.grid[y][x]
            if gem_type > 0:
                self.animations.append({'type': 'match', 'pos': (x, y), 'gem_type': gem_type, 'progress': 0})
                self._create_particles(x, y, gem_type)
                self.grid[y][x] = 0

    def _apply_gravity_and_refill(self):
        # sound: fall.wav
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    gem_type = self.grid[y][x]
                    self.grid[y + empty_slots][x] = gem_type
                    self.grid[y][x] = 0
                    self.animations.append({'type': 'fall', 'from': (x, y), 'to': (x, y + empty_slots), 'gem_type': gem_type, 'progress': 0})
            
            for i in range(empty_slots):
                gem_type = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                self.grid[i][x] = gem_type
                self.animations.append({'type': 'fall', 'from': (x, i - empty_slots), 'to': (x, i), 'gem_type': gem_type, 'progress': 0})

    def _create_particles(self, grid_x, grid_y, gem_type):
        px, py = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2, self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': 20, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _render_game(self):
        self._update_and_draw_animations()
        self._draw_grid()
        
        # Draw static gems not involved in animations
        animated_gems = set()
        for anim in self.animations:
            if anim['type'] == 'swap':
                animated_gems.add(tuple(anim['from']))
                animated_gems.add(tuple(anim['to']))
            elif anim['type'] == 'fall':
                animated_gems.add(tuple(anim['to']))
            elif anim['type'] == 'match':
                 animated_gems.add(tuple(anim['pos']))

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in animated_gems:
                    gem_type = self.grid[y][x]
                    if gem_type > 0:
                        is_selected = self.selected_gem_pos and x == self.selected_gem_pos[0] and y == self.selected_gem_pos[1]
                        self._draw_gem(x, y, gem_type, selected=is_selected)

        self._draw_cursor()

    def _draw_grid(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, y))

    def _draw_gem(self, grid_x, grid_y, gem_type, size_mult=1.0, alpha=255, selected=False, offset_x=0, offset_y=0):
        if gem_type == 0: return
        
        radius = int(self.CELL_SIZE * 0.38 * size_mult)
        center_x = int(self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2 + offset_x)
        center_y = int(self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2 + offset_y)
        
        base_color = self.GEM_COLORS[gem_type - 1]
        highlight_color = self.GEM_HIGHLIGHTS[gem_type - 1]
        
        if selected:
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            s_radius = int(self.CELL_SIZE * 0.45 * (1 + pulse * 0.1))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, s_radius, (*self.COLOR_SELECTED, 100))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, s_radius, (*self.COLOR_SELECTED, 150))

        # Create a temporary surface for alpha blending if needed
        temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        temp_surface.fill((0,0,0,0))
        local_center_x, local_center_y = radius, radius
        
        if alpha < 255:
            # All drawing functions need to handle alpha correctly. Pygame.gfxdraw is tricky.
            # A common way is to draw on a separate surface and blit with alpha.
            # However, for simplicity here, we'll try to pass alpha directly.
            # Note: Pygame.gfxdraw does not support alpha in all functions as expected.
            # The tuple `(*color, alpha)` is a good attempt.
            pass

        if gem_type == 1: # Circle
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*base_color, alpha))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*base_color, alpha))
            pygame.gfxdraw.filled_circle(self.screen, center_x-radius//3, center_y-radius//3, radius//3, (*highlight_color, alpha))
        elif gem_type == 2: # Square
            rect = pygame.Rect(center_x - radius, center_y - radius, radius*2, radius*2)
            pygame.draw.rect(self.screen, (*base_color, alpha), rect, border_radius=4)
            pygame.draw.rect(self.screen, (*highlight_color, alpha), (rect.x+2, rect.y+2, rect.width*0.4, rect.height*0.4), border_radius=3)
        elif gem_type == 3: # Triangle
            points = [(center_x, center_y - radius), (center_x - radius, center_y + radius//2), (center_x + radius, center_y + radius//2)]
            pygame.gfxdraw.filled_polygon(self.screen, points, (*base_color, alpha))
            pygame.gfxdraw.aapolygon(self.screen, points, (*base_color, alpha))
        elif gem_type == 4: # Diamond
            points = [(center_x, center_y - radius), (center_x + radius, center_y), (center_x, center_y + radius), (center_x - radius, center_y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, (*base_color, alpha))
            pygame.gfxdraw.aapolygon(self.screen, points, (*base_color, alpha))
        elif gem_type == 5: # Hexagon
            points = [(center_x + math.cos(math.pi/3 * i) * radius, center_y + math.sin(math.pi/3 * i) * radius) for i in range(6)]
            pygame.gfxdraw.filled_polygon(self.screen, points, (*base_color, alpha))
            pygame.gfxdraw.aapolygon(self.screen, points, (*base_color, alpha))
        elif gem_type == 6: # Star
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius * 0.5
                angle = i * math.pi / 5
                points.append((center_x + r * math.sin(angle), center_y - r * math.cos(angle)))
            pygame.gfxdraw.filled_polygon(self.screen, points, (*base_color, alpha))
            pygame.gfxdraw.aapolygon(self.screen, points, (*base_color, alpha))


    def _draw_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pulse = (math.sin(pygame.time.get_ticks() * 0.02) + 1) / 2
        alpha = 150 + 105 * pulse
        pygame.draw.rect(self.screen, (*self.COLOR_CURSOR, int(alpha)), rect, 3, border_radius=5)

    def _update_and_draw_animations(self):
        # Update and draw particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = int(255 * (p['life'] / 20))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha))

        # Update and draw main animations
        new_animations = []
        for anim in self.animations:
            anim['progress'] += 0.15 # Animation speed
            if anim['progress'] >= 1.0:
                continue

            new_animations.append(anim)
            p = anim['progress']

            if anim['type'] == 'swap':
                p_ease = 1 - (1 - p) ** 3 # Ease out cubic
                from_pos, to_pos = anim['from'], anim['to']
                
                # Draw first gem moving
                off_x1 = (to_pos[0] - from_pos[0]) * self.CELL_SIZE * p_ease
                off_y1 = (to_pos[1] - from_pos[1]) * self.CELL_SIZE * p_ease
                gem_type1 = self.grid[to_pos[1]][to_pos[0]] if not anim['is_revert'] else self.grid[from_pos[1]][from_pos[0]]
                self._draw_gem(from_pos[0], from_pos[1], gem_type1, offset_x=off_x1, offset_y=off_y1)

                # Draw second gem moving
                off_x2 = (from_pos[0] - to_pos[0]) * self.CELL_SIZE * p_ease
                off_y2 = (from_pos[1] - to_pos[1]) * self.CELL_SIZE * p_ease
                gem_type2 = self.grid[from_pos[1]][from_pos[0]] if not anim['is_revert'] else self.grid[to_pos[1]][to_pos[0]]
                self._draw_gem(to_pos[0], to_pos[1], gem_type2, offset_x=off_x2, offset_y=off_y2)

            elif anim['type'] == 'match':
                p_ease = p * 2 if p < 0.5 else 1 - (p-0.5)*2
                size = 1.0 + 0.5 * math.sin(p * math.pi) # Pulse
                alpha = 255 * (1 - p)
                self._draw_gem(anim['pos'][0], anim['pos'][1], anim['gem_type'], size_mult=size, alpha=alpha)

            elif anim['type'] == 'fall':
                p_ease = p * p # Ease in quad
                start_y = anim['from'][1]
                end_y = anim['to'][1]
                current_y = start_y + (end_y - start_y) * p_ease
                offset_y = (current_y - anim['to'][1]) * self.CELL_SIZE
                self._draw_gem(anim['to'][0], anim['to'][1], anim['gem_type'], offset_y=offset_y)
        
        self.animations = new_animations

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_small.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))
        
        if self.game_over:
            is_win = all(gem == 0 for row in self.grid for gem in row)
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "BOARD CLEARED!" if is_win else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)