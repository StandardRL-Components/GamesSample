import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a tile, "
        "then move to an adjacent tile and press space again to swap."
    )
    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Clear the entire board before the time runs out to win!"
    )

    # Game settings
    auto_advance = True
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    NUM_TILE_TYPES = 6
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 215, 0)
    COLOR_TIMER_BAR = (46, 204, 113)
    COLOR_TIMER_BAR_BG = (60, 70, 80)
    TILE_COLORS = [
        (231, 76, 60),   # Red
        (52, 152, 219),  # Blue
        (46, 204, 113),  # Green
        (155, 89, 182),  # Purple
        (241, 196, 15),  # Yellow
        (230, 126, 34),  # Orange
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 20)
        self.font_large = pygame.font.SysFont("sans-serif", 32)
        
        self.screen_width, self.screen_height = self.screen.get_size()
        self.grid_rect = pygame.Rect(120, 40, 400, 320)
        self.tile_size = self.grid_rect.width // self.GRID_WIDTH
        
        # Game state variables are initialized in reset()
        self.board = None
        self.cursor_pos = None
        self.selected_tile = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.prev_space_held = None
        self.particles = None
        self.animations = None
        self.game_phase = None # e.g., 'input', 'animating'
        self.last_move_time = 0
        self.last_swap_info = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tile = None
        self.prev_space_held = False
        self.particles = []
        self.animations = []
        self.game_phase = 'input'
        
        self._initialize_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        if self.game_phase == 'input':
            reward += self._handle_input(action)

        # --- Update Game Logic ---
        self._update_animations()
        
        # If animations just finished, proceed to next phase
        if not self.animations and self.game_phase != 'input':
            if self.game_phase == 'swapping':
                matches = self._find_matches()
                if not matches:
                    # Invalid swap, swap back
                    reward -= 0.01 # Small penalty for invalid swap
                    pos1, pos2 = self.last_swap_info
                    self._swap_tiles_back(pos1, pos2)
                    self.game_phase = 'swapping_back'
                else:
                    self.game_phase = 'clearing'
                    self._process_matches(matches)
                    reward += sum(len(m) for m in matches) # +1 per tile
            
            elif self.game_phase == 'swapping_back':
                self.game_phase = 'input' # Return to player control
            
            elif self.game_phase == 'clearing':
                self._apply_gravity()
                # If gravity caused tiles to fall, new animations were created
                if self.animations:
                    self.game_phase = 'falling'
                else: # Board was full, no tiles fell, just refill
                    new_tiles = self._refill_board()
                    if new_tiles:
                        self.game_phase = 'refilling'
                    else: # No new tiles, check for cascades
                        self.game_phase = 'checking_cascades'

            elif self.game_phase == 'falling':
                new_tiles = self._refill_board()
                if new_tiles:
                    self.game_phase = 'refilling'
                else: # No new tiles needed, check for cascades
                    self.game_phase = 'checking_cascades'

            elif self.game_phase in ('refilling', 'checking_cascades'):
                matches = self._find_matches()
                if matches: # Cascade happened
                    self.game_phase = 'clearing'
                    self._process_matches(matches)
                    reward += sum(len(m) for m in matches)
                else: # No more cascades, return to input
                    if not self._find_possible_moves():
                        self._reshuffle_board()
                    self.game_phase = 'input'

        # --- Update Step and Termination ---
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if np.all(self.board == 0): # Win condition
                reward += 100
            else: # Time out
                reward -= 100
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # Debounce cursor movement
        if movement != 0 and (pygame.time.get_ticks() - self.last_move_time > 100):
            self.last_move_time = pygame.time.get_ticks()
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # Handle space press (on rising edge)
        if space_held and not self.prev_space_held:
            if self.selected_tile is None:
                self.selected_tile = tuple(self.cursor_pos)
            else:
                x1, y1 = self.selected_tile
                x2, y2 = self.cursor_pos
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    self._swap_tiles(self.selected_tile, tuple(self.cursor_pos))
                    reward += 0.1 # Provisional reward for valid swap attempt
                    self.selected_tile = None
                elif (x1, y1) == (x2, y2): # Deselect if same tile
                    self.selected_tile = None
                else: # Invalid non-adjacent swap, deselect
                    reward -= 0.01
                    self.selected_tile = None

        self.prev_space_held = space_held
        return reward

    def _initialize_board(self):
        # Loop until a valid board (no initial matches, at least one move) is generated.
        while True:
            self.board = self.np_random.integers(
                1, self.NUM_TILE_TYPES + 1,
                size=(self.GRID_WIDTH, self.GRID_HEIGHT),
                dtype=int
            )
            
            # Resolve any matches that were generated
            matches = self._find_matches()
            while matches:
                for match in matches:
                    for x, y in match:
                        self.board[x, y] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                matches = self._find_matches()
            
            # If there's at least one possible move, the board is valid
            if self._find_possible_moves():
                break

    def _reshuffle_board(self):
        # Called on stalemate. Shuffles existing tiles to create new moves.
        flat_tiles = list(self.board.flatten())
        
        while True:
            self.np_random.shuffle(flat_tiles)
            self.board = np.array(flat_tiles).reshape((self.GRID_WIDTH, self.GRID_HEIGHT))

            # Resolve any matches created by the shuffle
            matches = self._find_matches()
            while matches:
                for match in matches:
                    for x, y in match:
                        self.board[x, y] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                matches = self._find_matches()

            # Ensure the new board has moves
            if self._find_possible_moves():
                break

    def _swap_tiles(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[x1, y1], self.board[x2, y2] = self.board[x2, y2], self.board[x1, y1]
        self.game_phase = 'swapping'
        self.last_swap_info = (pos1, pos2)
        self._add_swap_animation(pos1, pos2)

    def _swap_tiles_back(self, pos1, pos2):
        # This is just a logical swap, the animation handles the visual
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[x1, y1], self.board[x2, y2] = self.board[x2, y2], self.board[x1, y1]
        self._add_swap_animation(pos1, pos2)

    def _add_swap_animation(self, pos1, pos2):
        self.animations.append({
            'type': 'swap', 'pos1': pos1, 'pos2': pos2, 'progress': 0.0
        })

    def _find_matches(self):
        matches = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                tile_type = self.board[x, y]
                if tile_type != 0 and tile_type == self.board[x+1, y] == self.board[x+2, y]:
                    matches.add(((x, y), (x+1, y), (x+2, y)))
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                tile_type = self.board[x, y]
                if tile_type != 0 and tile_type == self.board[x, y+1] == self.board[x, y+2]:
                    matches.add(((x, y), (x, y+1), (x, y+2)))
        
        if not matches: return []

        # Consolidate overlapping matches (e.g., L or T shapes)
        consolidated = []
        all_matched_coords = {coord for match in matches for coord in match}
        
        while all_matched_coords:
            group = set()
            queue = {all_matched_coords.pop()}
            while queue:
                current = queue.pop()
                group.add(current)
                # Find neighbors in all_matched_coords
                neighbors = {
                    (current[0] + dx, current[1] + dy)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                }
                found_neighbors = all_matched_coords.intersection(neighbors)
                queue.update(found_neighbors)
                all_matched_coords.difference_update(found_neighbors)
            consolidated.append(list(group))
            
        return [match for match in consolidated if len(match) >= 3]


    def _process_matches(self, matches):
        tiles_to_clear = set()
        for match in matches:
            for x, y in match:
                tiles_to_clear.add((x, y))

        for x, y in tiles_to_clear:
            self.score += 10
            tile_type = self.board[x, y]
            if tile_type > 0:
                self._create_particles(x, y, tile_type)
            self.board[x, y] = 0

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_y = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[x, y] != 0:
                    if y != empty_y:
                        tile_type = self.board[x, y]
                        self.board[x, empty_y] = tile_type
                        self.board[x, y] = 0
                        self.animations.append({
                            'type': 'fall', 'from': (x, y), 'to': (x, empty_y), 'progress': 0.0,
                            'tile_type': tile_type
                        })
                    empty_y -= 1

    def _refill_board(self):
        new_tiles_created = False
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[x, y] == 0:
                    new_tile_type = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                    self.board[x, y] = new_tile_type
                    self.animations.append({
                        'type': 'fall', 'from': (x, y - self.GRID_HEIGHT - empty_count), 'to': (x, y), 'progress': 0.0,
                        'tile_type': new_tile_type
                    })
                    new_tiles_created = True
                    empty_count += 1
        return new_tiles_created

    def _find_possible_moves(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self.board[x, y], self.board[x+1, y] = self.board[x+1, y], self.board[x, y]
                    if self._find_matches():
                        self.board[x, y], self.board[x+1, y] = self.board[x+1, y], self.board[x, y]
                        return True
                    self.board[x, y], self.board[x+1, y] = self.board[x+1, y], self.board[x, y]
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self.board[x, y], self.board[x, y+1] = self.board[x, y+1], self.board[x, y]
                    if self._find_matches():
                        self.board[x, y], self.board[x, y+1] = self.board[x, y+1], self.board[x, y]
                        return True
                    self.board[x, y], self.board[x, y+1] = self.board[x, y+1], self.board[x, y]
        return False

    def _check_termination(self):
        if self.game_over:
            return True
        if np.all(self.board == 0): # All tiles cleared
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_bg()
        
        animating_tiles = set()
        for anim in self.animations:
            if anim['type'] == 'swap':
                animating_tiles.add(anim['pos1'])
                animating_tiles.add(anim['pos2'])
            elif anim['type'] == 'fall':
                # Don't draw the tile at its final destination yet
                animating_tiles.add(anim['to'])
        
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.board[x, y] != 0 and (x, y) not in animating_tiles:
                    self._render_tile(x, y, self.board[x, y])

        for anim in self.animations:
            if anim['type'] == 'swap':
                p = anim['progress']
                x1, y1 = anim['pos1']
                x2, y2 = anim['pos2']
                
                # Determine which tile is which after the logical swap
                tile_type_at_pos1_start = self.board[x2, y2]
                tile_type_at_pos2_start = self.board[x1, y1]

                ix1 = x1 + (x2 - x1) * p
                iy1 = y1 + (y2 - y1) * p
                ix2 = x2 + (x1 - x2) * p
                iy2 = y2 + (y1 - y2) * p
                
                self._render_tile(ix1, iy1, tile_type_at_pos1_start, is_grid_coords=False)
                self._render_tile(ix2, iy2, tile_type_at_pos2_start, is_grid_coords=False)
            
            elif anim['type'] == 'fall':
                p = anim['progress']
                x1, y1 = anim['from']
                x2, y2 = anim['to']
                ix = x1 + (x2-x1) * p
                iy = y1 + (y2-y1) * p
                self._render_tile(ix, iy, anim['tile_type'], is_grid_coords=False)

        self._render_particles()
        self._render_cursor()

    def _grid_to_pixel(self, x, y):
        px = self.grid_rect.left + x * self.tile_size
        py = self.grid_rect.top + y * self.tile_size
        return int(px), int(py)

    def _render_tile(self, x, y, tile_type, is_grid_coords=True):
        if tile_type == 0: return
        
        if is_grid_coords:
            px, py = self._grid_to_pixel(x, y)
        else: # Interpolated floating point coords
            px = self.grid_rect.left + x * self.tile_size
            py = self.grid_rect.top + y * self.tile_size
        
        color = self.TILE_COLORS[tile_type - 1]
        shadow_color = tuple(c * 0.6 for c in color)
        highlight_color = tuple(min(255, c * 1.4) for c in color)
        
        inset = 4
        rect = pygame.Rect(px + inset, py + inset, self.tile_size - inset*2, self.tile_size - inset*2)
        
        pygame.draw.rect(self.screen, shadow_color, rect, border_radius=8)
        
        inner_rect = rect.inflate(-4, -4)
        inner_rect.y -= 2
        pygame.draw.rect(self.screen, color, inner_rect, border_radius=7)
        
        highlight_rect = pygame.Rect(inner_rect.x, inner_rect.y, inner_rect.width, inner_rect.height // 2)
        pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=7)


    def _render_grid_bg(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_rect.left + x * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_rect.top), (px, self.grid_rect.bottom))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_rect.top + y * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_rect.left, py), (self.grid_rect.right, py))

    def _render_cursor(self):
        if self.game_phase != 'input': return
        
        cx, cy = self.cursor_pos
        px, py = self._grid_to_pixel(cx, cy)
        
        if self.selected_tile:
            sx, sy = self.selected_tile
            spx, spy = self._grid_to_pixel(sx, sy)
            s_rect = pygame.Rect(spx, spy, self.tile_size, self.tile_size)
            pygame.draw.rect(self.screen, (255, 255, 255), s_rect, 4, border_radius=10)

        cursor_rect = pygame.Rect(px, py, self.tile_size, self.tile_size)
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        color = (200 + 55 * pulse, 200 + 55 * pulse, 200 + 55 * pulse)
        pygame.draw.rect(self.screen, color, cursor_rect, 3, border_radius=10)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 5))
        
        timer_width = 200
        timer_height = 20
        timer_x = self.screen_width - timer_width - 20
        timer_y = 10
        
        progress = max(0, 1 - self.steps / self.MAX_STEPS)
        
        bg_rect = pygame.Rect(timer_x, timer_y, timer_width, timer_height)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, bg_rect, border_radius=5)
        
        bar_rect = pygame.Rect(timer_x, timer_y, int(timer_width * progress), timer_height)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, bar_rect, border_radius=5)
        
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.all(self.board == 0)
            msg = "BOARD CLEARED!" if win_condition else "TIME'S UP!"
            color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_x, grid_y, tile_type):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        center_x, center_y = px + self.tile_size // 2, py + self.tile_size // 2
        color = self.TILE_COLORS[tile_type - 1]
        
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'color': color})

    def _render_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            
            radius = int(p['life'] / 5)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
        
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_animations(self):
        if not self.animations:
            return
            
        done_indices = []
        for i, anim in enumerate(self.animations):
            anim['progress'] += 0.15 # Animation speed
            if anim['progress'] >= 1.0:
                done_indices.append(i)
        
        for i in sorted(done_indices, reverse=True):
            del self.animations[i]
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # The environment can be run headlessly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    assert obs.shape == (400, 640, 3)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (400, 640, 3)
    print("✓ Headless execution test passed.")
    
    # The following code is for manual play and requires a display.
    try:
        # Re-initialize pygame without the dummy driver for display
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        pygame.display.init()
        pygame.font.init()

        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Match-3 Gym Environment")
        clock = pygame.time.Clock()
        
        game_env = GameEnv(render_mode="rgb_array")
        obs, info = game_env.reset()
        done = False
        
        print("\n--- Manual Control ---")
        print(game_env.user_guide)
        
        while not done:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = game_env.step(action)
            done = terminated or truncated
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- Resetting ---")
                    obs, info = game_env.reset()

            clock.tick(30)
            
    except pygame.error as e:
        print(f"\nPygame display error: {e}")
        print("Manual play requires a display. The environment itself runs headlessly.")
    
    pygame.quit()
    print("Game finished.")