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
        "Controls: Arrow keys to move cursor. Space to select a tile. "
        "Move to an adjacent tile and press Space again to swap. Shift to deselect."
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Clear the board to advance through 3 stages, all against the clock!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.TILE_SIZE = 50
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.TILE_SIZE) // 2 + 20
        self.FPS = 30
        self.MAX_STEPS = 5400 # 3 stages * 60s * 30fps

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.TILE_COLORS = {
            1: (255, 80, 80),   # Red
            2: (80, 255, 80),   # Green
            3: (80, 120, 255),  # Blue
            4: (255, 255, 80),  # Yellow
            5: (200, 80, 255)   # Purple
        }
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TIMER_BAR = (80, 200, 255)
        self.COLOR_TIMER_BG = (40, 60, 80)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans", 20)
        self.font_medium = pygame.font.SysFont("sans", 28)
        self.font_large = pygame.font.SysFont("sans", 48)
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_tile_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.stage = None
        self.time_per_stage = 60 * self.FPS
        self.time_remaining = None
        
        self.game_state = 'IDLE' # IDLE, SWAP, MATCH, FALL, REFILL, SHUFFLE
        self.animations = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        self.rng = None

        # self.reset() is not called in __init__ to follow Gymnasium standard practice
        # self.validate_implementation() is also removed from __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # self.rng should be initialized if it's None or if a new seed is provided.
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.time_remaining = self.time_per_stage

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile_pos = None
        
        self.game_state = 'IDLE'
        self.animations = []
        self.particles = []
        
        self._generate_board(self.stage)

        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action
        reward = 0
        self.steps += 1
        self.time_remaining -= 1

        # --- Handle Input ---
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.game_state == 'IDLE':
            reward += self._handle_input(movement, space_press, shift_press)

        # --- Update Game State Machine ---
        finished_animations = self._update_animations()

        for anim in finished_animations:
            if anim['type'] == 'SWAP':
                reward += self._process_swap(anim)

        if not self.animations:
            if self.game_state == 'MATCH_CHECK':
                reward += self._process_matches()
            elif self.game_state == 'FALL_CHECK':
                self._process_fall()
            elif self.game_state == 'REFILL_CHECK':
                self._process_refill()
            elif self.game_state == 'BOARD_CHECK':
                reward += self._process_board_check()
            elif self.game_state == 'SHUFFLE_EXECUTE':
                self._execute_shuffle()

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.time_remaining <= 0 and not self.win:
            self.game_over = True
            terminated = True
            reward -= 100 # Loss penalty
        
        if self.win:
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time/step limits
        
        if self.game_over and self.game_state not in ['GAME_OVER', 'WIN']:
            self.game_state = 'WIN' if self.win else 'GAME_OVER'

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_press, shift_press):
        # Cursor Movement
        if self.move_cooldown <= 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            if movement > 0: self.move_cooldown = 5
        else:
            self.move_cooldown -= 1
        
        # Deselect
        if shift_press and self.selected_tile_pos is not None:
            self.selected_tile_pos = None
            # sfx: deselect
        
        # Select / Swap
        if space_press:
            cx, cy = self.cursor_pos
            if self.grid[cy][cx] == 0: return 0 # Cannot select empty space
            
            if self.selected_tile_pos is None:
                self.selected_tile_pos = list(self.cursor_pos)
                # sfx: select
            else:
                sx, sy = self.selected_tile_pos
                dist = abs(cx - sx) + abs(cy - sy)
                if dist == 1: # Is adjacent
                    self._initiate_swap(self.selected_tile_pos, self.cursor_pos)
                    self.selected_tile_pos = None
                elif (cx, cy) == (sx, sy): # Clicked same tile
                    self.selected_tile_pos = None
                else: # Clicked non-adjacent tile
                    self.selected_tile_pos = list(self.cursor_pos)
                    # sfx: select
        return 0

    def _initiate_swap(self, pos1, pos2, is_swap_back=False):
        self.game_state = 'SWAP_CHECK'
        duration = 8 if not is_swap_back else 6
        self.animations.append({
            'type': 'SWAP', 'pos1': pos1, 'pos2': pos2, 
            'progress': 0, 'duration': duration, 'is_swap_back': is_swap_back
        })
        # sfx: swap

    def _process_swap(self, swap_anim):
        x1, y1 = swap_anim['pos1']
        x2, y2 = swap_anim['pos2']
        
        # Perform swap in the grid data. This happens for the initial swap and the swap-back.
        self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]

        # After swapping, check for matches.
        matches = self._find_matches()

        if swap_anim['is_swap_back']:
            # This was a swap-back animation. The board is now restored. Go idle.
            self.game_state = 'IDLE'
            return 0
        
        if not matches:
            # This was an initial swap, but it was invalid (no matches).
            # The grid is currently in the swapped state. We initiate an animation to swap it back.
            self._initiate_swap(swap_anim['pos1'], swap_anim['pos2'], is_swap_back=True)
            return -0.2 # Penalty for invalid move
        else:
            # This was a valid initial swap. Proceed to clear matches.
            self.game_state = 'MATCH_CHECK'
            return 0

    def _process_matches(self):
        matches = self._find_matches()
        if not matches:
            self.game_state = 'BOARD_CHECK'
            return 0

        reward = 0
        tiles_to_clear = set()
        for match in matches:
            for pos in match:
                tiles_to_clear.add(pos)
        
        for x, y in tiles_to_clear:
            if self.grid[y][x] != 0:
                reward += 1 # Reward per tile cleared
                self._create_particles((x,y))
                self.grid[y][x] = 0
        
        self.score += int(reward)
        self.animations.append({'type': 'MATCH', 'tiles': list(tiles_to_clear), 'progress': 0, 'duration': 10})
        self.game_state = 'FALL_CHECK'
        # sfx: match_clear
        return reward
    
    def _process_fall(self):
        moved = self._apply_gravity()
        if moved:
            self.game_state = 'REFILL_CHECK'
        else:
            self.game_state = 'BOARD_CHECK'

    def _process_refill(self):
        self._refill_board()
        self.game_state = 'MATCH_CHECK' # Check for cascade matches

    def _process_board_check(self):
        # Check for win condition
        if all(self.grid[y][x] == 0 for y in range(self.GRID_SIZE) for x in range(self.GRID_SIZE)):
            self.stage += 1
            if self.stage > 3:
                self.win = True
                self.game_state = 'WIN'
            else:
                self.time_remaining = self.time_per_stage
                self._generate_board(self.stage)
                self.game_state = 'IDLE'
                # sfx: stage_clear
                return 5 # Stage clear reward
        # Check for no possible moves
        elif not self._check_for_possible_moves():
            self.game_state = 'SHUFFLE_EXECUTE'
            self.animations.append({'type': 'SHUFFLE', 'progress': 0, 'duration': 20})
            # sfx: shuffle
        else:
            self.game_state = 'IDLE'
        return 0
    
    def _execute_shuffle(self):
        self._shuffle_board()
        self.game_state = 'IDLE'

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.TILE_SIZE
            y = self.GRID_OFFSET_Y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.TILE_SIZE), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.TILE_SIZE, y), 1)
        
        # Get animated positions
        animated_tiles = {}
        for anim in self.animations:
            if anim['type'] == 'SWAP':
                p = anim['progress'] / anim['duration']
                x1, y1 = anim['pos1']
                x2, y2 = anim['pos2']
                ix = x1 + (x2 - x1) * p
                iy = y1 + (y2 - y1) * p
                
                # During swap, the grid data is already swapped, so we need to get the color from the opposite position
                if not anim['is_swap_back']:
                    animated_tiles[(x1, y1)] = (ix, iy, self.grid[y2][x2])
                    animated_tiles[(x2, y2)] = (x2 + (x1 - x2) * p, y2 + (y1 - y2) * p, self.grid[y1][x1])
                else: # On swap back, grid is also swapped, logic is the same
                    animated_tiles[(x1, y1)] = (ix, iy, self.grid[y2][x2])
                    animated_tiles[(x2, y2)] = (x2 + (x1 - x2) * p, y2 + (y1 - y2) * p, self.grid[y1][x1])

            elif anim['type'] == 'FALL':
                p = anim['progress'] / anim['duration']
                x, y_start, y_end = anim['pos']
                iy = y_start + (y_end - y_start) * p
                animated_tiles[(x, y_end)] = (x, iy, anim['color'])
            elif anim['type'] == 'REFILL':
                p = anim['progress'] / anim['duration']
                x, y = anim['pos']
                iy = y - (1-p)
                animated_tiles[(x, y)] = (x, iy, anim['color'])

        # Draw tiles
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) in animated_tiles:
                    ax, ay, color_idx = animated_tiles[(x, y)]
                    if color_idx > 0:
                        self._draw_tile(ax, ay, color_idx)
                elif self.grid[y][x] > 0:
                    self._draw_tile(x, y, self.grid[y][x])
        
        # Draw match animation
        for anim in self.animations:
            if anim['type'] == 'MATCH':
                p = anim['progress'] / anim['duration']
                for x, y in anim['tiles']:
                    cx = self.GRID_OFFSET_X + x * self.TILE_SIZE + self.TILE_SIZE // 2
                    cy = self.GRID_OFFSET_Y + y * self.TILE_SIZE + self.TILE_SIZE // 2
                    radius = int((self.TILE_SIZE // 2) * p)
                    alpha = int(255 * (1 - p))
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, (255, 255, 255, alpha))

        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), (*p['color'], alpha))

        # Draw cursor and selection
        cx, cy = self.cursor_pos
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (self.GRID_OFFSET_X + cx * self.TILE_SIZE, self.GRID_OFFSET_Y + cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), 3, border_radius=8)

        if self.selected_tile_pos:
            sx, sy = self.selected_tile_pos
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (self.GRID_OFFSET_X + sx * self.TILE_SIZE + 3, self.GRID_OFFSET_Y + sy * self.TILE_SIZE + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6), 2, border_radius=6)
    
    def _draw_tile(self, x, y, color_idx):
        px = self.GRID_OFFSET_X + x * self.TILE_SIZE
        py = self.GRID_OFFSET_Y + y * self.TILE_SIZE
        color = self.TILE_COLORS[color_idx]
        shadow_color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
        
        padding = 4
        pygame.draw.rect(self.screen, shadow_color, (px + padding, py + padding + 2, self.TILE_SIZE - padding*2, self.TILE_SIZE - padding*2), border_radius=8)
        pygame.draw.rect(self.screen, color, (px + padding, py + padding, self.TILE_SIZE - padding*2, self.TILE_SIZE - padding*2), border_radius=8)
        
        # Highlight
        highlight_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
        pygame.gfxdraw.arc(self.screen, int(px + self.TILE_SIZE/2), int(py + self.TILE_SIZE/2), int(self.TILE_SIZE/3), 110, 250, highlight_color)
        
    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        timer_width = 150
        timer_height = 20
        time_ratio = max(0, self.time_remaining / self.time_per_stage)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, (self.WIDTH - timer_width - 15, 15, timer_width, timer_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (self.WIDTH - timer_width - 15, 15, int(timer_width * time_ratio), timer_height), border_radius=5)
        
        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage} / 3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - timer_width - 15, 40))

        # Game Over / Win message
        if self.game_state in ['GAME_OVER', 'WIN']:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "TIME UP!"
            msg_render = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(msg_render, msg_rect)

            final_score_render = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            self.screen.blit(final_score_render, final_score_rect)
    
    def _update_animations(self):
        finished_animations = []
        for anim in self.animations[:]:
            anim['progress'] += 1
            if anim['progress'] >= anim['duration']:
                self.animations.remove(anim)
                finished_animations.append(anim)
        return finished_animations

    def _generate_board(self, stage):
        initial_tiles = 25 + (stage-1) * 5
        self.grid = [[0] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        
        empty_cells = list(range(self.GRID_SIZE * self.GRID_SIZE))
        self.rng.shuffle(empty_cells)
        
        for i in range(initial_tiles):
            idx = empty_cells[i]
            y, x = divmod(idx, self.GRID_SIZE)
            self.grid[y][x] = self.rng.integers(1, len(self.TILE_COLORS) + 1)
        
        # Ensure no initial matches and at least one move
        while self._find_matches() or not self._check_for_possible_moves():
            self._shuffle_board()

    def _find_matches(self):
        matches = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == 0: continue
                # Horizontal
                if x < self.GRID_SIZE - 2 and self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2]:
                    match = [(x, y), (x+1, y), (x+2, y)]
                    i = 3
                    while x + i < self.GRID_SIZE and self.grid[y][x] == self.grid[y][x+i]:
                        match.append((x+i, y))
                        i += 1
                    matches.append(match)
                # Vertical
                if y < self.GRID_SIZE - 2 and self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x]:
                    match = [(x, y), (x, y+1), (x, y+2)]
                    i = 3
                    while y + i < self.GRID_SIZE and self.grid[y][x] == self.grid[y+i][x]:
                        match.append((x, y+i))
                        i += 1
                    matches.append(match)
        # Filter out sub-matches
        unique_matches = []
        seen_tiles = set()
        for match in sorted(matches, key=len, reverse=True):
            is_unique = True
            for pos in match:
                if pos in seen_tiles:
                    is_unique = False
                    break
            if is_unique:
                unique_matches.append(match)
                for pos in match: seen_tiles.add(pos)
        return unique_matches

    def _apply_gravity(self):
        moved = False
        for x in range(self.GRID_SIZE):
            empty_y = -1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y][x] == 0 and empty_y == -1:
                    empty_y = y
                elif self.grid[y][x] != 0 and empty_y != -1:
                    self.grid[empty_y][x] = self.grid[y][x]
                    self.grid[y][x] = 0
                    self.animations.append({'type': 'FALL', 'pos': (x, y, empty_y), 'color': self.grid[empty_y][x], 'progress': 0, 'duration': (empty_y - y) * 2})
                    empty_y -= 1
                    moved = True
        return moved
    
    def _refill_board(self):
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.grid[y][x] == 0:
                    self.grid[y][x] = self.rng.integers(1, len(self.TILE_COLORS) + 1)
                    self.animations.append({'type': 'REFILL', 'pos': (x,y), 'color': self.grid[y][x], 'progress': 0, 'duration': 5})
    
    def _check_for_possible_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == 0: continue
                # Check swap right
                if x < self.GRID_SIZE - 1 and self.grid[y][x+1] != 0:
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                    if self._find_matches():
                        self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                # Check swap down
                if y < self.GRID_SIZE - 1 and self.grid[y+1][x] != 0:
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                    if self._find_matches():
                        self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
        return False

    def _shuffle_board(self):
        tiles = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] > 0:
                    tiles.append(self.grid[y][x])
        self.rng.shuffle(tiles)
        
        tile_idx = 0
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] > 0:
                    self.grid[y][x] = tiles[tile_idx]
                    tile_idx += 1
        
        if self._find_matches() or not self._check_for_possible_moves():
            self._shuffle_board() # Recurse until a valid state is found

    def _create_particles(self, pos):
        x, y = pos
        px = self.GRID_OFFSET_X + x * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_OFFSET_Y + y * self.TILE_SIZE + self.TILE_SIZE // 2
        # This function is called right before the grid tile is set to 0
        color = self.TILE_COLORS[self.grid[y][x]]
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.integers(10, 20),
                'max_life': 20,
                'size': self.rng.uniform(2, 5),
                'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not part of the required Gymnasium interface
    # but is useful for human testing and visualization.
    
    # Un-set the dummy video driver to allow for a display window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # Set human control mappings
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }
    
    # Create a window to display the game
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print("Game Over!")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

    env.close()