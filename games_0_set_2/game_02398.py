
# Generated: 2025-08-27T20:16:16.160939
# Source Brief: brief_02398.md
# Brief Index: 2398

        
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


# Animation helper classes to manage visual effects over time
class TileAnimation:
    """Base class for animations, handling timing and progress."""
    def __init__(self, duration):
        self.start_time = pygame.time.get_ticks()
        self.duration = duration
        self.finished = False

    def get_progress(self):
        """Returns animation progress from 0.0 to 1.0."""
        progress = (pygame.time.get_ticks() - self.start_time) / self.duration
        return min(progress, 1.0)

    def update(self):
        """Marks the animation as finished if its duration has passed."""
        if self.get_progress() >= 1.0:
            self.finished = True

class SwapAnimation(TileAnimation):
    """Animation for two tiles swapping positions."""
    def __init__(self, pos1, pos2, duration=150):
        super().__init__(duration)
        self.pos1 = pos1
        self.pos2 = pos2

class FallAnimation(TileAnimation):
    """Animation for a tile falling down the grid."""
    def __init__(self, from_row, to_row, col, duration_per_tile=50):
        super().__init__(duration_per_tile * (to_row - from_row))
        self.from_row = from_row
        self.to_row = to_row
        self.col = col

class ClearAnimation(TileAnimation):
    """Animation for a tile shrinking and disappearing."""
    def __init__(self, pos, duration=200):
        super().__init__(duration)
        self.pos = pos

class ReshuffleAnimation(TileAnimation):
    """Animation for tiles during a board reshuffle."""
    def __init__(self, pos, target_pos, delay, duration=400):
        super().__init__(duration)
        self.start_time += delay
        self.pos = pos
        self.target_pos = target_pos

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the selector. Press Space to select a tile, then move to an adjacent tile and press Space again to swap. Press Shift to deselect."
    game_description = "A fast-paced tile-matching puzzle game. Swap adjacent tiles to create matches of three or more. Clear the board before you run out of moves!"

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 10, 10
    TILE_SIZE = 32
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * TILE_SIZE) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - GRID_ROWS * TILE_SIZE) // 2 + 20

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (45, 55, 65)
    COLOR_UI_BG = (15, 25, 35, 200)
    COLOR_TEXT = (230, 230, 230)
    TILE_COLORS = [
        (0, 0, 0),       # 0: Empty
        (255, 65, 54),   # 1: Red
        (0, 116, 217),   # 2: Blue
        (46, 204, 64),   # 3: Green
        (255, 220, 0),   # 4: Yellow
        (177, 13, 201),  # 5: Purple
        (255, 133, 27),  # 6: Orange
    ]
    NUM_COLORS = len(TILE_COLORS) - 1

    # Rewards
    REWARD_MATCH_PER_TILE = 1
    REWARD_BIG_MATCH = 5
    REWARD_INVALID_SWAP = -0.1
    REWARD_WIN = 100
    REWARD_LOSS = -50

    MAX_MOVES = 50
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.selector_pos = np.array([self.GRID_ROWS // 2, self.GRID_COLS // 2])
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.game_logic_state = 'IDLE' # 'IDLE', 'PROCESSING'

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.selector_pos = np.array([self.GRID_ROWS // 2, self.GRID_COLS // 2])
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.game_logic_state = 'IDLE'

        self._initialize_board()
        
        return self._get_observation(), self._get_info()

    def _initialize_board(self):
        """Fills the board with tiles, ensuring no initial matches and at least one possible move."""
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS), dtype=int)
            if self._find_matches(self.grid)[0] or not self._check_for_possible_moves(self.grid):
                continue # Regenerate if board starts with matches or has no moves
            break

    def step(self, action):
        self.clock.tick(30)
        self.steps += 1
        reward = 0

        self._update_animations()
        self._update_particles()

        # Game logic is paused while animations are running
        if not self.animations:
            if self.game_logic_state == 'PROCESSING':
                # Animations just finished, process resulting cascades
                cascade_reward = self._process_cascades()
                if cascade_reward > 0:
                    reward += cascade_reward
                else:
                    # No more cascades, check for no-moves-left scenario
                    self.game_logic_state = 'IDLE'
                    if not self._is_board_cleared() and not self._check_for_possible_moves(self.grid):
                        self._reshuffle_board()
            
            # Process player input only when idle and not game over
            if self.game_logic_state == 'IDLE' and not self.game_over:
                input_reward = self._handle_input(action)
                reward += input_reward

        self.last_space_held = (action[1] == 1)
        self.last_shift_held = (action[2] == 1)

        terminated = self.steps >= self.MAX_STEPS
        if not self.game_over:
            board_cleared = self._is_board_cleared()
            if self.moves_left <= 0 and not board_cleared:
                self.game_over = True
                terminated = True
                reward += self.REWARD_LOSS
            elif board_cleared:
                self.game_over = True
                terminated = True
                reward += self.REWARD_WIN
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        reward = 0

        if movement == 1: self.selector_pos[0] -= 1
        elif movement == 2: self.selector_pos[0] += 1
        elif movement == 3: self.selector_pos[1] -= 1
        elif movement == 4: self.selector_pos[1] += 1
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_ROWS - 1)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_COLS - 1)

        if shift_press and self.selected_tile is not None:
            self.selected_tile = None
            # sfx: deselect_sound

        if space_press:
            if self.selected_tile is None:
                self.selected_tile = tuple(self.selector_pos)
                # sfx: select_sound
            else:
                r1, c1 = self.selected_tile
                r2, c2 = self.selector_pos
                
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    self.moves_left -= 1
                    self.animations.append(SwapAnimation((r1, c1), (r2, c2)))
                    # sfx: swap_sound

                    temp_grid = self.grid.copy()
                    temp_grid[r1, c1], temp_grid[r2, c2] = temp_grid[r2, c2], temp_grid[r1, c1]
                    
                    if not self._find_matches(temp_grid)[0]:
                        self.animations.append(SwapAnimation((r1, c1), (r2, c2), duration=100))
                        self.animations[-1].start_time += 150 # Delay the swap back
                        reward += self.REWARD_INVALID_SWAP
                        # sfx: invalid_swap_sound
                    else:
                        self.grid = temp_grid
                        self.game_logic_state = 'PROCESSING'
                    
                    self.selected_tile = None
                else:
                    self.selected_tile = tuple(self.selector_pos)
                    # sfx: select_sound
        return reward

    def _process_cascades(self):
        """Finds and processes all matches on the board, returning the reward."""
        reward = 0
        matches, big_matches = self._find_matches(self.grid)
        
        if not matches:
            return 0
        
        # sfx: match_sound
        reward += len(matches) * self.REWARD_MATCH_PER_TILE
        reward += len(big_matches) * self.REWARD_BIG_MATCH
        self.score += len(matches)

        for r, c in matches:
            self.animations.append(ClearAnimation((r, c)))
            self._spawn_particles(r, c, self.grid[r, c])
            self.grid[r, c] = 0

        self._apply_gravity()
        return reward

    def _apply_gravity(self):
        """Makes tiles fall down to fill empty spaces."""
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.animations.append(FallAnimation(r, empty_row, c))
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
            
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
                self.animations.append(FallAnimation(r - (empty_row + 1), r, c))
                # sfx: tile_fall_sound

    def _find_matches(self, grid):
        """Finds all horizontal and vertical matches of 3 or more tiles."""
        matches = set()
        big_matches = set()
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    match_len = 3
                    while c + match_len < self.GRID_COLS and grid[r, c] == grid[r, c+match_len]:
                        match_len += 1
                    
                    is_big = match_len > 3
                    for i in range(match_len):
                        matches.add((r, c+i))
                        if is_big: big_matches.add((r, c+i))
        
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    match_len = 3
                    while r + match_len < self.GRID_ROWS and grid[r, c] == grid[r+match_len, c]:
                        match_len += 1

                    is_big = match_len > 3
                    for i in range(match_len):
                        matches.add((r+i, c))
                        if is_big: big_matches.add((r+i, c))

        return matches, big_matches

    def _check_for_possible_moves(self, grid):
        """Checks the entire grid for any valid move."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if c < self.GRID_COLS - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_matches(temp_grid)[0]: return True
                if r < self.GRID_ROWS - 1:
                    temp_grid = grid.copy()
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_matches(temp_grid)[0]: return True
        return False

    def _reshuffle_board(self):
        """Shuffles the board when no moves are possible."""
        # sfx: reshuffle_sound
        flat_tiles = self.grid[self.grid > 0].tolist()
        self.np_random.shuffle(flat_tiles)

        new_grid = np.zeros_like(self.grid)
        i = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] > 0:
                    new_grid[r, c] = flat_tiles[i]
                    i += 1
        
        self.grid = new_grid
        if not self._check_for_possible_moves(self.grid) and not self._is_board_cleared():
            self._initialize_board()
        else:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r,c] > 0:
                        delay = (r + c) * 20
                        self.animations.append(ReshuffleAnimation((r,c), (r,c), delay))
            self.game_logic_state = 'PROCESSING'

    def _is_board_cleared(self):
        return np.all(self.grid == 0)

    def _update_animations(self):
        finished_anims = [anim for anim in self.animations if (anim.update(), anim.finished)[1]]
        self.animations = [anim for anim in self.animations if anim not in finished_anims]

    def _spawn_particles(self, r, c, color_idx):
        px, py = self._grid_to_pixel(r, c)
        px += self.TILE_SIZE // 2; py += self.TILE_SIZE // 2
        color = self.TILE_COLORS[color_idx]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, r, c):
        return (self.GRID_MARGIN_X + c * self.TILE_SIZE, self.GRID_MARGIN_Y + r * self.TILE_SIZE)

    def _render_game(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                px, py = self._grid_to_pixel(r, c)
                pygame.draw.rect(self.screen, self.COLOR_GRID, (px, py, self.TILE_SIZE, self.TILE_SIZE), 1)

        anim_overrides = {}
        for anim in self.animations:
            progress = anim.get_progress()
            if isinstance(anim, SwapAnimation):
                r1, c1 = anim.pos1; r2, c2 = anim.pos2
                p1x, p1y = self._grid_to_pixel(r1, c1); p2x, p2y = self._grid_to_pixel(r2, c2)
                anim_overrides[anim.pos1] = (p1x + (p2x-p1x)*progress, p1y + (p2y-p1y)*progress)
                anim_overrides[anim.pos2] = (p2x + (p1x-p2x)*progress, p2y + (p1y-p2y)*progress)
            elif isinstance(anim, FallAnimation):
                start_y = self.GRID_MARGIN_Y + anim.from_row * self.TILE_SIZE
                end_y = self.GRID_MARGIN_Y + anim.to_row * self.TILE_SIZE
                curr_y = start_y + (end_y - start_y) * (1 - (1-progress)**3) # Ease out
                anim_overrides[(anim.to_row, anim.col)] = (self._grid_to_pixel(0, anim.col)[0], curr_y)
            elif isinstance(anim, ClearAnimation): anim_overrides[anim.pos] = 'clearing'
            elif isinstance(anim, ReshuffleAnimation): anim_overrides[anim.pos] = 'reshuffling'

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0: continue
                px, py = self._grid_to_pixel(r, c)
                size = self.TILE_SIZE
                
                if (r, c) in anim_overrides:
                    override = anim_overrides[(r, c)]
                    if override == 'clearing':
                        anim = next(a for a in self.animations if isinstance(a, ClearAnimation) and a.pos == (r,c))
                        size = self.TILE_SIZE * (1 - anim.get_progress())
                        px += (self.TILE_SIZE - size) / 2; py += (self.TILE_SIZE - size) / 2
                    elif override == 'reshuffling':
                        anim = next(a for a in self.animations if isinstance(a, ReshuffleAnimation) and a.pos == (r,c))
                        scale = 1.0 + math.sin(anim.get_progress() * math.pi) * 0.5
                        size = self.TILE_SIZE * scale
                        px -= (size - self.TILE_SIZE) / 2; py -= (size - self.TILE_SIZE) / 2
                    else: px, py = override
                
                pygame.draw.rect(self.screen, self.TILE_COLORS[self.grid[r,c]], (int(px+2), int(py+2), max(0, int(size-4)), max(0, int(size-4))), border_radius=4)
        
        for p in self.particles:
            radius = int(3 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], int(255 * (p['life'] / p['max_life']))))

        if not self.game_over:
            sel_r, sel_c = self.selector_pos
            sel_px, sel_py = self._grid_to_pixel(sel_r, sel_c)
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
            pygame.draw.rect(self.screen, (255,255,255), (sel_px, sel_py, self.TILE_SIZE, self.TILE_SIZE), width=2 + int(pulse*2), border_radius=6)

            if self.selected_tile is not None:
                sel_r, sel_c = self.selected_tile
                sel_px, sel_py = self._grid_to_pixel(sel_r, sel_c)
                pygame.draw.rect(self.screen, (255,255,0), (sel_px, sel_py, self.TILE_SIZE, self.TILE_SIZE), width=3, border_radius=6)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA); ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "BOARD CLEARED!" if self._is_board_cleared() else "OUT OF MOVES"
            text = self.font_gameover.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Matcher")
    
    running = True
    while running:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

    env.close()