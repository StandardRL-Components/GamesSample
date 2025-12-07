
# Generated: 2025-08-28T01:22:13.388782
# Source Brief: brief_04072.md
# Brief Index: 4072

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a tile. "
        "Press space again to swap it with the adjacent tile in the direction you last moved. "
        "Hold shift to deselect."
    )

    game_description = (
        "A vibrant match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Clear the entire board before you run out of moves to win. Plan your swaps carefully to create cascading combos!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 6, 6
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_OFFSET_X, GRID_OFFSET_Y = 160, 20
    TILE_SIZE = 60
    
    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID_BG = (40, 45, 55)
    COLOR_GRID_LINES = (55, 60, 70)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 240, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 150, 80),   # Orange
    ]
    
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

    # --- Game Settings ---
    MAX_MOVES = 15
    MAX_STEPS = 1000
    
    Particle = namedtuple("Particle", ["pos", "vel", "color", "radius", "life"])

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
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.last_move_dir = None
        self.moves_left = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_state = None
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.rng = np.random.default_rng()

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = None
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.last_move_dir = [1, 0] # Default to right
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()

    def _generate_initial_board(self):
        while True:
            self.grid = self.rng.integers(0, len(self.GEM_COLORS), size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            if self._find_all_matches():
                continue # Regenerate if matches exist on start
            if self._check_for_possible_moves():
                break # Board is valid

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        
        # --- Handle Input ---
        if self.selected_pos is None: # IDLE state
            if movement == 1: self.cursor_pos[1] -= 1; self.last_move_dir = [0, -1]
            elif movement == 2: self.cursor_pos[1] += 1; self.last_move_dir = [0, 1]
            elif movement == 3: self.cursor_pos[0] -= 1; self.last_move_dir = [-1, 0]
            elif movement == 4: self.cursor_pos[0] += 1; self.last_move_dir = [1, 0]
            
            # Cursor wrap-around
            self.cursor_pos[0] %= self.GRID_WIDTH
            self.cursor_pos[1] %= self.GRID_HEIGHT

            if space_pressed:
                self.selected_pos = list(self.cursor_pos)
                # sfx: select_gem
        else: # SELECTED state
            if shift_held:
                self.selected_pos = None
                # sfx: deselect_gem
            elif space_pressed:
                reward = self._attempt_swap()
                self.selected_pos = None
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Update Game State ---
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if np.all(self.grid == -1):
            self.game_over = True
            self.win_state = "WIN"
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.win_state = "LOSE"
            terminated = True
            reward += -50
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_state = "TIMEOUT"
            terminated = True
            reward += -50

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _attempt_swap(self):
        p1 = self.selected_pos
        target_x = p1[0] + self.last_move_dir[0]
        target_y = p1[1] + self.last_move_dir[1]

        if not (0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT):
            # sfx: invalid_swap
            return 0 # Invalid swap location

        p2 = [target_x, target_y]
        self.moves_left -= 1
        
        # Perform swap
        self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]
        
        # Check for matches
        chain_reward = self._resolve_matches()

        if chain_reward == 0:
            # No match, swap back
            self.grid[p1[0], p1[1]], self.grid[p2[0], p2[1]] = self.grid[p2[0], p2[1]], self.grid[p1[0], p1[1]]
            # sfx: no_match
            return -0.1
        else:
            # sfx: match_success
            # After a successful move, check if any more moves are possible
            if not self._check_for_possible_moves() and np.any(self.grid != -1):
                self._reshuffle_board()
            return chain_reward
            
    def _resolve_matches(self):
        total_reward = 0
        chain_level = 1
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            num_cleared = len(matches)
            total_reward += num_cleared * 1 # +1 per tile
            if num_cleared >= 4:
                total_reward += 5 # Bonus for 4+
            if chain_level > 1:
                total_reward += num_cleared * (chain_level - 1) # Chain bonus
                # sfx: chain_reaction

            self._clear_tiles(matches)
            self._apply_gravity_and_refill()
            chain_level += 1
        return total_reward

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x, y] != -1 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x, y] != -1 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _clear_tiles(self, tiles_to_clear):
        for x, y in tiles_to_clear:
            if self.grid[x, y] != -1:
                self._spawn_particles(x, y, self.GEM_COLORS[self.grid[x, y]])
                self.grid[x, y] = -1
        # sfx: gems_clear

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == -1:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = -1
            # Refill top
            for y in range(empty_slots):
                self.grid[x, y] = self.rng.integers(0, len(self.GEM_COLORS))
        # sfx: gems_fall

    def _check_for_possible_moves(self):
        temp_grid = self.grid.copy()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    temp_grid[x, y], temp_grid[x+1, y] = temp_grid[x+1, y], temp_grid[x, y]
                    if self._grid_has_match(temp_grid): return True
                    temp_grid[x, y], temp_grid[x+1, y] = temp_grid[x+1, y], temp_grid[x, y] # Swap back
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    temp_grid[x, y], temp_grid[x, y+1] = temp_grid[x, y+1], temp_grid[x, y]
                    if self._grid_has_match(temp_grid): return True
                    temp_grid[x, y], temp_grid[x, y+1] = temp_grid[x, y+1], temp_grid[x, y] # Swap back
        return False
        
    def _grid_has_match(self, grid):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if grid[x,y]!=-1 and grid[x,y]==grid[x+1,y]==grid[x+2,y]: return True
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if grid[x,y]!=-1 and grid[x,y]==grid[x,y+1]==grid[x,y+2]: return True
        return False

    def _reshuffle_board(self):
        # sfx: board_shuffle
        flat_gems = self.grid[self.grid != -1].tolist()
        while True:
            self.rng.shuffle(flat_gems)
            new_grid = np.full_like(self.grid, -1)
            i = 0
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if self.grid[x, y] != -1:
                        new_grid[x, y] = flat_gems[i]
                        i += 1
            if self._grid_has_match(new_grid): continue
            
            temp_check_grid = new_grid.copy()
            if self._check_for_possible_moves():
                self.grid = new_grid
                break
    
    def _spawn_particles(self, grid_x, grid_y, color):
        center_x = self.GRID_OFFSET_X + grid_x * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(20):
            angle = self.rng.random() * 2 * math.pi
            speed = 2 + self.rng.random() * 4
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = 3 + self.rng.random() * 3
            life = 20 + self.rng.integers(0, 20)
            self.particles.append(self.Particle([center_x, center_y], vel, color, radius, life))

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            new_pos = [p.pos[0] + p.vel[0], p.pos[1] + p.vel[1]]
            new_vel = [p.vel[0] * 0.95, p.vel[1] * 0.95 + 0.1] # Damping and gravity
            new_radius = p.radius * 0.95
            new_life = p.life - 1
            if new_life > 0 and new_radius > 0.5:
                new_particles.append(self.Particle(new_pos, new_vel, p.color, new_radius, new_life))
        self.particles = new_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type != -1:
                    color = self.GEM_COLORS[gem_type]
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + x * self.TILE_SIZE + 4,
                        self.GRID_OFFSET_Y + y * self.TILE_SIZE + 4,
                        self.TILE_SIZE - 8,
                        self.TILE_SIZE - 8,
                    )
                    pygame.gfxdraw.box(self.screen, rect, (*color, 200))
                    pygame.gfxdraw.rectangle(self.screen, rect, (*[c+30 if c<225 else 255 for c in color], 255))

        # Grid lines
        for i in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start, end, 1)
        for i in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.TILE_SIZE, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start, end, 1)
            
        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), int(p.radius), (*p.color, int(255 * (p.life / 60))))

        # Cursor and Selection
        if self.selected_pos:
            x, y = self.selected_pos
            rect = pygame.Rect(self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 4)

        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.TILE_SIZE, self.GRID_OFFSET_Y + cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        
    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (15, 20))
        
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 50))
        
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state == "WIN":
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_LOSE)
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
            "is_selected": self.selected_pos is not None,
        }
        
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

# Example usage:
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        movement = 0 # no-op
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Control the speed of manual play

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()