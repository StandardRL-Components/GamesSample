
# Generated: 2025-08-28T00:48:46.684889
# Source Brief: brief_03900.md
# Brief Index: 3900

        
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


# Helper class for particles
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.randint(15, 30)  # lifespan in frames

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            # Fade out effect
            alpha = max(0, min(255, int(255 * (self.life / 30))))
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (2, 2), 2)
            surface.blit(temp_surf, (int(self.x) - 2, int(self.y) - 2))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a gem, "
        "then move to an adjacent gem and press space again to swap. Press shift to cancel a selection."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Plan your moves to create chain reactions and "
        "reach the target score of 1000 within 30 moves!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 6, 8
    GEM_TYPES = 6
    TARGET_SCORE = 1000
    MAX_MOVES = 30
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    
    GEM_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 100, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width, self.screen_height = 640, 400
        self.grid_offset_x = (self.screen_width - self.GRID_WIDTH * 40) // 2
        self.grid_offset_y = (self.screen_height - self.GRID_HEIGHT * 40) // 2
        self.gem_size = 40

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.game_state = "AWAITING_INPUT"
        self.animation_timer = 0
        self.swap_coords = None
        self.fall_map = {}
        self.match_list = set()
        self.particles = []
        self.current_step_reward = 0
        
        self.reset()

        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        
        self._create_board_and_ensure_moves()
        
        self.game_state = "AWAITING_INPUT"
        self.animation_timer = 0
        self.swap_coords = None
        self.fall_map = {}
        self.match_list = set()
        self.particles = []
        self.current_step_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        self._update_animations_and_state()
        
        if self.game_state == "AWAITING_INPUT":
            self.current_step_reward = 0
            self._handle_input(action)
            # If a swap was invalid, the reward is set in _handle_input
            reward = self.current_step_reward
        
        if self.game_state == "FINISHED_TURN":
            reward = self.current_step_reward
            self.current_step_reward = 0
            
            if not self._find_valid_moves():
                self.game_state = "SHUFFLING"
                self.animation_timer = 30 # Shuffle animation time
            else:
                self.game_state = "AWAITING_INPUT"

        if self.score >= self.TARGET_SCORE and not self.win:
            self.win = True
            reward += 100
            terminated = True
        elif self.moves_left <= 0 and self.game_state in ["AWAITING_INPUT", "FINISHED_TURN"]:
            terminated = True
        
        if terminated:
            self.game_over = True
            self.game_state = "GAME_OVER"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        if movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        if movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        if movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1

        # --- Cancel Selection ---
        if shift_pressed and self.selected_pos:
            self.selected_pos = None
            # sfx: cancel_select

        # --- Select / Swap ---
        if space_pressed:
            if not self.selected_pos:
                self.selected_pos = list(self.cursor_pos)
                # sfx: select_gem
            else:
                # Attempting a swap
                p1 = self.selected_pos
                p2 = self.cursor_pos
                
                # Check for adjacency
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1:
                    self.moves_left -= 1
                    self._swap_gems(p1, p2)
                    
                    matches1 = self._find_matches_at_point(p2)
                    matches2 = self._find_matches_at_point(p1)
                    
                    self.swap_coords = (p1, p2)
                    if not matches1 and not matches2:
                        # Invalid move, swap back
                        self.game_state = "REVERSING_SWAP"
                        self.animation_timer = 10 # frames
                        self.current_step_reward = -0.1
                        # sfx: invalid_swap
                    else:
                        # Valid move
                        self.game_state = "SWAPPING"
                        self.animation_timer = 10 # frames
                        # sfx: valid_swap
                else:
                    # Not adjacent, just change selection
                    self.selected_pos = list(self.cursor_pos)
                    # sfx: select_gem
                
                # If a swap was attempted, clear selection
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1:
                    self.selected_pos = None

    def _update_animations_and_state(self):
        if self.animation_timer > 0:
            self.animation_timer -= 1
            return

        if self.game_state == "SWAPPING":
            self._start_match_finding()
        elif self.game_state == "REVERSING_SWAP":
            p1, p2 = self.swap_coords
            self._swap_gems(p1, p2) # Swap back
            self.swap_coords = None
            self.game_state = "FINISHED_TURN"
        elif self.game_state == "MATCHING":
            self._handle_gravity_and_refill()
        elif self.game_state == "FALLING":
            self._start_match_finding() # Check for cascades
        elif self.game_state == "SHUFFLING":
            self._shuffle_board()
            self.game_state = "AWAITING_INPUT"

    def _start_match_finding(self):
        self.swap_coords = None
        matches = self._find_all_matches()
        if matches:
            self.match_list = matches
            
            # Calculate reward
            base_reward = len(matches)
            bonus = self._calculate_combo_bonus()
            self.current_step_reward += base_reward + bonus
            self.score += base_reward + bonus
            
            # Create particles
            for x, y in matches:
                gem_type = self.grid[y, x]
                if gem_type > 0:
                    px, py = self._grid_to_pixel(x, y)
                    for _ in range(10):
                        self.particles.append(Particle(px + self.gem_size/2, py + self.gem_size/2, self.GEM_COLORS[gem_type-1]))

            self.game_state = "MATCHING"
            self.animation_timer = 15 # frames for match effect
            # sfx: match_found
        else:
            self.game_state = "FINISHED_TURN"

    def _calculate_combo_bonus(self):
        bonus = 0
        # This is a simplification; a real combo system would track chain lengths.
        # Here we just check for long matches.
        # This logic is complex, so we'll just check match list size for a simple bonus.
        if len(self.match_list) == 4:
            bonus += 5
        elif len(self.match_list) >= 5:
            bonus += 10
        return bonus

    def _handle_gravity_and_refill(self):
        # Remove matched gems
        for x, y in self.match_list:
            self.grid[y, x] = 0
        self.match_list = set()

        # Calculate falling
        self.fall_map = {}
        for x in range(self.GRID_WIDTH):
            fall_dist = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] == 0:
                    fall_dist += 1
                elif fall_dist > 0:
                    # Move gem in data grid
                    gem_type = self.grid[y, x]
                    self.grid[y + fall_dist, x] = gem_type
                    self.grid[y, x] = 0
                    # Store animation data
                    self.fall_map[(x, y + fall_dist)] = (gem_type, fall_dist)
        
        # Fill top rows
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[y, x] == 0:
                    gem_type = self.np_random.integers(1, self.GEM_TYPES + 1)
                    self.grid[y, x] = gem_type
                    self.fall_map[(x, y)] = (gem_type, self.GRID_HEIGHT) # Fall from top
        
        self.game_state = "FALLING"
        self.animation_timer = 15 # frames for falling animation
        # sfx: gems_fall

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "game_state": self.game_state,
        }

    # --- Board Logic ---
    def _create_board_and_ensure_moves(self):
        self.grid = self.np_random.integers(1, self.GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches():
            self.grid = self.np_random.integers(1, self.GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        
        if not self._find_valid_moves():
            self._shuffle_board()

    def _swap_gems(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

    def _find_matches_at_point(self, p):
        x, y = p
        gem_type = self.grid[y, x]
        if gem_type == 0: return set()

        # Horizontal
        h_matches = {(x, y)}
        for i in range(x - 1, -1, -1):
            if self.grid[y, i] == gem_type: h_matches.add((i, y))
            else: break
        for i in range(x + 1, self.GRID_WIDTH):
            if self.grid[y, i] == gem_type: h_matches.add((i, y))
            else: break
        
        # Vertical
        v_matches = {(x, y)}
        for i in range(y - 1, -1, -1):
            if self.grid[i, x] == gem_type: v_matches.add((x, i))
            else: break
        for i in range(y + 1, self.GRID_HEIGHT):
            if self.grid[i, x] == gem_type: v_matches.add((x, i))
            else: break

        results = set()
        if len(h_matches) >= 3: results.update(h_matches)
        if len(v_matches) >= 3: results.update(v_matches)
        return results

    def _find_all_matches(self):
        all_matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                all_matches.update(self._find_matches_at_point((x, y)))
        return all_matches

    def _find_valid_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_matches_at_point((x, y)) or self._find_matches_at_point((x + 1, y)):
                        self._swap_gems((x, y), (x + 1, y)) # Swap back
                        return True
                    self._swap_gems((x, y), (x + 1, y)) # Swap back
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_matches_at_point((x, y)) or self._find_matches_at_point((x, y + 1)):
                        self._swap_gems((x, y), (x, y + 1)) # Swap back
                        return True
                    self._swap_gems((x, y), (x, y + 1)) # Swap back
        return False
    
    def _shuffle_board(self):
        flat_list = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_list)
        self.grid = np.array(flat_list).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        
        while self._find_all_matches() or not self._find_valid_moves():
            self.np_random.shuffle(flat_list)
            self.grid = np.array(flat_list).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        # sfx: board_shuffle

    # --- Rendering ---
    def _render_game(self):
        self._render_grid_bg()
        self._render_gems()
        self._render_cursor_and_selection()
        self._render_particles()

    def _grid_to_pixel(self, x, y):
        return self.grid_offset_x + x * self.gem_size, self.grid_offset_y + y * self.gem_size

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _render_grid_bg(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.grid_offset_x, self.grid_offset_y, 
            self.GRID_WIDTH * self.gem_size, self.GRID_HEIGHT * self.gem_size))
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.grid_offset_x, self.grid_offset_y + y * self.gem_size)
            end = (self.grid_offset_x + self.GRID_WIDTH * self.gem_size, self.grid_offset_y + y * self.gem_size)
            pygame.draw.line(self.screen, self.COLOR_BG, start, end, 2)
        for x in range(self.GRID_WIDTH + 1):
            start = (self.grid_offset_x + x * self.gem_size, self.grid_offset_y)
            end = (self.grid_offset_x + x * self.gem_size, self.grid_offset_y + self.GRID_HEIGHT * self.gem_size)
            pygame.draw.line(self.screen, self.COLOR_BG, start, end, 2)

    def _render_gems(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type == 0: continue
                
                px, py = self._grid_to_pixel(x, y)
                size = self.gem_size
                is_swapping = False
                
                # Swap animation
                if self.game_state in ["SWAPPING", "REVERSING_SWAP"] and self.swap_coords:
                    p1, p2 = self.swap_coords
                    t = 1.0 - (self.animation_timer / 10)
                    if (x, y) == tuple(p1):
                        px, py = self._lerp(px, self._grid_to_pixel(*p2)[0], t), self._lerp(py, self._grid_to_pixel(*p2)[1], t)
                        is_swapping = True
                    elif (x, y) == tuple(p2):
                        px, py = self._lerp(px, self._grid_to_pixel(*p1)[0], t), self._lerp(py, self._grid_to_pixel(*p1)[1], t)
                        is_swapping = True
                
                # Match animation
                if self.game_state == "MATCHING" and (x, y) in self.match_list:
                    t = self.animation_timer / 15
                    size = self.gem_size * (1 + (1 - t) * 0.5) # Grow and shrink
                    px -= (size - self.gem_size) / 2
                    py -= (size - self.gem_size) / 2
                
                # Fall animation
                if self.game_state == "FALLING" and (x, y) in self.fall_map:
                    original_gem_type, fall_dist = self.fall_map[(x, y)]
                    gem_type = original_gem_type
                    t = 1.0 - (self.animation_timer / 15)
                    start_y = self._grid_to_pixel(x, y)[1]
                    if fall_dist == self.GRID_HEIGHT: # New gem
                        start_y = self.grid_offset_y - self.gem_size
                    else: # Existing gem
                        start_y = self._grid_to_pixel(x, y - fall_dist)[1]
                    py = self._lerp(start_y, self._grid_to_pixel(x, y)[1], t)
                
                # Don't draw gems that are part of a completed match
                if self.game_state == "FALLING" and (x, y) in self.match_list:
                    continue

                self._draw_gem(px, py, int(size), gem_type, is_swapping)
    
    def _draw_gem(self, x, y, size, gem_type, is_swapping):
        color = self.GEM_COLORS[gem_type - 1]
        rect = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
        
        # Simple shapes for different gems
        center = (int(x + size // 2), int(y + size // 2))
        radius = int(size // 2 - 4)
        
        if gem_type == 1: # Circle
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
        elif gem_type == 2: # Square
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif gem_type == 3: # Diamond
            points = [(center[0], y + 4), (x + size - 4, center[1]), (center[0], y + size - 4), (x + 4, center[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Triangle Up
            points = [(center[0], y + 4), (x + size - 4, y + size - 4), (x + 4, y + size - 4)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Hexagon
            points = [(center[0] + radius, center[1]), 
                      (center[0] + radius//2, center[1] - int(radius * 0.866)),
                      (center[0] - radius//2, center[1] - int(radius * 0.866)),
                      (center[0] - radius, center[1]),
                      (center[0] - radius//2, center[1] + int(radius * 0.866)),
                      (center[0] + radius//2, center[1] + int(radius * 0.866))]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else: # Star
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius / 2
                angle = i * math.pi / 5
                points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        if is_swapping: # Add a highlight for clarity during swaps
             pygame.draw.rect(self.screen, (255,255,255,100), (x,y,size,size), 3)

    def _render_cursor_and_selection(self):
        # Draw selected gem highlight
        if self.selected_pos:
            px, py = self._grid_to_pixel(*self.selected_pos)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, (px, py, self.gem_size, self.gem_size), 4)

        # Draw cursor
        cx, cy = self.cursor_pos
        px, py = self.grid_offset_x + cx * self.gem_size, self.grid_offset_y + cy * self.gem_size
        
        # Pulsating effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 
        color = (int(self._lerp(180, 255, pulse)), int(self._lerp(180, 255, pulse)), 0)
        pygame.draw.rect(self.screen, color, (px, py, self.gem_size, self.gem_size), 4)

    def _render_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
            else:
                p.draw(self.screen)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, x, y, align="left"):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            if align == "left": text_rect.topleft = (x, y)
            elif align == "right": text_rect.topright = (x, y)
            elif align == "center": text_rect.center = (x, y)
            
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Score
        draw_text(f"Score: {self.score}", self.font_large, self.COLOR_TEXT, 20, 10, "left")
        
        # Moves
        draw_text(f"Moves: {self.moves_left}", self.font_large, self.COLOR_TEXT, self.screen_width - 20, 10, "right")

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            draw_text(message, self.font_large, self.COLOR_CURSOR, self.screen_width // 2, self.screen_height // 2 - 20, "center")
            draw_text(f"Final Score: {self.score}", self.font_small, self.COLOR_TEXT, self.screen_width // 2, self.screen_height // 2 + 20, "center")

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()