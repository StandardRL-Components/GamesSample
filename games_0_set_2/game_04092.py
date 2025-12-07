
# Generated: 2025-08-28T01:24:17.870522
# Source Brief: brief_04092.md
# Brief Index: 4092

        
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
        "Controls: Arrows to move cursor. Space to select a tile. "
        "Move to an adjacent tile and press Space again to swap. Shift to deselect."
    )

    game_description = (
        "Swap adjacent tiles to create matches of 3 or more. Clear the board before time runs out!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 6, 6
    TILE_SIZE = 50
    GRID_LINE_WIDTH = 2
    ANIMATION_STEPS = 6 # Number of steps for an animation to complete

    MAX_STEPS = 1000
    MAX_TIME = 60.0  # seconds

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_TIMER_BAR = (40, 160, 220)
    COLOR_TIMER_BG = (60, 70, 80)

    TILE_COLORS = {
        0: (40, 50, 60), # Empty
        1: (220, 50, 50), # Red
        2: (50, 220, 50), # Green
        3: (50, 100, 220), # Blue
    }
    TILE_SHINE_COLORS = {
        1: (255, 120, 120),
        2: (120, 255, 120),
        3: (120, 170, 255),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width, self.screen_height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 16)

        self.grid_pixel_width = self.GRID_WIDTH * self.TILE_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.TILE_SIZE
        self.grid_top_left = (
            (self.screen_width - self.grid_pixel_width) // 2,
            (self.screen_height - self.grid_pixel_height) // 2,
        )

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = (0, 0)
        self.selected_tile = None
        
        self.animation_state = "idle" # idle, swapping, reverting, clearing, falling
        self.animation_progress = 0
        self.animated_tiles = {} # { (x, y): { "from": (px, py), "to": (px, py), "color": int } }
        self.particles = []
        self.post_swap_logic_pending = False
        self.swap_pair = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def _get_grid_pos(self, grid_x, grid_y):
        """Converts grid coordinates to pixel coordinates."""
        return (
            self.grid_top_left[0] + grid_x * self.TILE_SIZE,
            self.grid_top_left[1] + grid_y * self.TILE_SIZE,
        )

    def _create_board(self):
        """Generates a valid board with no initial matches and at least one possible move."""
        while True:
            for r in range(self.GRID_HEIGHT):
                for c in range(self.GRID_WIDTH):
                    self.grid[r, c] = self.np_random.integers(1, 4)
            
            if self._find_matches():
                continue # Regenerate if matches exist

            if self._find_possible_moves():
                break # Board is valid
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._create_board()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_tile = None
        
        self.animation_state = "idle"
        self.animation_progress = 0
        self.animated_tiles = {}
        self.particles = []
        self.post_swap_logic_pending = False
        self.swap_pair = None
        self.chain_reaction_level = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty for taking a step to encourage speed

        self._update_animation()

        if self.animation_state == "idle":
            self._handle_action(action)

        self._update_particles()
        
        # Check for potential match reward
        if self._count_potential_matches() > 0:
            reward += 0.05

        self.steps += 1
        self.time_remaining -= 1.0 / 30.0 # Assuming 30fps stepping
        
        # Add rewards from game logic, which are stored in self.score
        # The logic is complex, so we'll calculate reward based on score change
        current_score = self.score
        reward += (current_score - getattr(self, '_last_score', current_score))
        self._last_score = current_score

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if np.all(self.grid == 0):
                reward += 100 # Win bonus
            else:
                reward += -50 # Loss penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if shift_held:
            self.selected_tile = None
            return

        # --- Cursor Movement ---
        cx, cy = self.cursor_pos
        if movement == 1: cy = max(0, cy - 1)
        elif movement == 2: cy = min(self.GRID_HEIGHT - 1, cy + 1)
        elif movement == 3: cx = max(0, cx - 1)
        elif movement == 4: cx = min(self.GRID_WIDTH - 1, cx + 1)
        self.cursor_pos = (cx, cy)

        # --- Selection/Swap Logic ---
        if space_held:
            if self.grid[cy, cx] == 0: return # Can't select empty tiles

            if self.selected_tile is None:
                self.selected_tile = self.cursor_pos
                # sfx: select_tile
            else:
                # Attempt to swap with selected tile
                if self._is_adjacent(self.selected_tile, self.cursor_pos):
                    self._initiate_swap(self.selected_tile, self.cursor_pos)
                else: # Invalid swap target, so select new tile instead
                    self.selected_tile = self.cursor_pos
                    # sfx: select_tile_fail
        
    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _initiate_swap(self, pos1, pos2, revert=False):
        self.animation_state = "reverting" if revert else "swapping"
        self.animation_progress = 0
        self.swap_pair = (pos1, pos2)
        
        c1, r1 = pos1
        c2, r2 = pos2
        
        self.animated_tiles = {
            (r1, c1): {"from": self._get_grid_pos(c1, r1), "to": self._get_grid_pos(c2, r2), "color": self.grid[r1, c1]},
            (r2, c2): {"from": self._get_grid_pos(c2, r2), "to": self._get_grid_pos(c1, r1), "color": self.grid[r2, c2]},
        }
        self.selected_tile = None
        # sfx: swap_start

    def _update_animation(self):
        if self.animation_state == "idle":
            return
            
        self.animation_progress += 1
        if self.animation_progress >= self.ANIMATION_STEPS:
            self.animation_progress = 0
            
            # --- Finish Swapping ---
            if self.animation_state in ["swapping", "reverting"]:
                (c1, r1), (c2, r2) = self.swap_pair
                self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                self.animated_tiles = {}
                
                if self.animation_state == "swapping":
                    self._process_matches()
                else: # Reverting
                    self.animation_state = "idle"
                    self.swap_pair = None
            
            # --- Finish Clearing ---
            elif self.animation_state == "clearing":
                self._apply_gravity_and_refill()

            # --- Finish Falling ---
            elif self.animation_state == "falling":
                self.animated_tiles = {}
                self._process_matches() # Check for chain reactions

    def _process_matches(self):
        matches = self._find_matches()
        if matches:
            if self.animation_state != "falling": # First match in a chain
                self.chain_reaction_level = 1
            else: # Chain reaction
                self.chain_reaction_level += 1
                self.score += 2 * self.chain_reaction_level # Chain reaction bonus
                # sfx: chain_reaction

            self.score += len(matches) # Base score for each tile
            
            for r, c in matches:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0
            
            self.animation_state = "clearing"
            self.animation_progress = 0 # Give a brief pause
            # sfx: match_success
        else:
            # If initial swap had no match, revert it
            if self.animation_state == "swapping":
                self._initiate_swap(self.swap_pair[0], self.swap_pair[1], revert=True)
                # sfx: swap_fail
            else: # No more matches, end of chain
                self.animation_state = "idle"
                self.chain_reaction_level = 0

    def _find_matches(self):
        to_match = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r, c+1] and color == self.grid[r, c+2]:
                    for i in range(3): to_match.add((r, c+i))
                    # Check for longer matches
                    for i in range(c + 3, self.GRID_WIDTH):
                        if self.grid[r, i] == color:
                            to_match.add((r, i))
                        else: break
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                color = self.grid[r, c]
                if color != 0 and color == self.grid[r+1, c] and color == self.grid[r+2, c]:
                    for i in range(3): to_match.add((r+i, c))
                    # Check for longer matches
                    for i in range(r + 3, self.GRID_HEIGHT):
                        if self.grid[i, c] == color:
                            to_match.add((i, c))
                        else: break
        return to_match

    def _apply_gravity_and_refill(self):
        self.animated_tiles = {}
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        # Animate fall
                        start_pos = self._get_grid_pos(c, r)
                        end_pos = self._get_grid_pos(c, empty_row)
                        self.animated_tiles[(empty_row, c)] = {"from": start_pos, "to": end_pos, "color": self.grid[r,c]}
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

            # Refill from top
            for r in range(empty_row, -1, -1):
                new_color = self.np_random.integers(1, 4)
                self.grid[r,c] = new_color
                start_pos = self._get_grid_pos(c, r - (empty_row + 1)) # Start off-screen
                end_pos = self._get_grid_pos(c, r)
                self.animated_tiles[(r, c)] = {"from": start_pos, "to": end_pos, "color": new_color}

        if self.animated_tiles:
            self.animation_state = "falling"
            self.animation_progress = 0
            # sfx: tiles_fall
        else: # No tiles moved, so no new matches possible
            self.animation_state = "idle"


    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches(): moves.append(((r,c), (r,c+1)))
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches(): moves.append(((r,c), (r+1,c)))
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
        return moves

    def _count_potential_matches(self):
        count = 0
        # Horizontal pairs
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 1):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c + 1]:
                    count += 1
        # Vertical pairs
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 1):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r + 1, c]:
                    count += 1
        return count

    def _check_termination(self):
        if self.game_over: return True
        if self.time_remaining <= 0: return True
        if self.steps >= self.MAX_STEPS: return True
        if np.all(self.grid == 0): return True # All tiles cleared
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_remaining}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_top_left[0] + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_top_left[1]), (x, self.grid_top_left[1] + self.grid_pixel_height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_top_left[1] + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_top_left[0], y), (self.grid_top_left[0] + self.grid_pixel_width, y), self.GRID_LINE_WIDTH)

        # Draw tiles
        drawn_animated = set()
        for (r,c), data in self.animated_tiles.items():
            progress = self.animation_progress / self.ANIMATION_STEPS
            fx, fy = data["from"]
            tx, ty = data["to"]
            px = fx + (tx - fx) * progress
            py = fy + (ty - fy) * progress
            self._draw_tile(px, py, data["color"])
            drawn_animated.add((r,c))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r,c) in drawn_animated: continue
                color_idx = self.grid[r, c]
                if color_idx != 0:
                    px, py = self._get_grid_pos(c, r)
                    self._draw_tile(px, py, color_idx)
        
        # Draw cursor and selection
        self._draw_cursor()

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_tile(self, px, py, color_idx):
        rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        
        # Main color
        color = self.TILE_COLORS[color_idx]
        pygame.gfxdraw.box(self.screen, rect.inflate(-6, -6), color)
        
        # Shine/Highlight
        shine_color = self.TILE_SHINE_COLORS.get(color_idx)
        if shine_color:
            shine_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width - 10, 8)
            pygame.draw.rect(self.screen, shine_color, shine_rect, border_top_left_radius=5, border_top_right_radius=5)

    def _draw_cursor(self):
        # Cursor
        cx, cy = self.cursor_pos
        px, py = self._get_grid_pos(cx, cy)
        cursor_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 0), cursor_rect, 3, border_radius=4)
        
        # Selected tile
        if self.selected_tile:
            scx, scy = self.selected_tile
            spx, spy = self._get_grid_pos(scx, scy)
            selected_rect = pygame.Rect(spx, spy, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), selected_rect, 4, border_radius=6)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Timer bar
        timer_bar_width = 200
        timer_bar_height = 20
        timer_x = self.screen_width - timer_bar_width - 20
        timer_y = 25
        
        time_ratio = max(0, self.time_remaining / self.MAX_TIME)
        
        bg_rect = (timer_x, timer_y, timer_bar_width, timer_bar_height)
        fill_rect = (timer_x, timer_y, int(timer_bar_width * time_ratio), timer_bar_height)

        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, bg_rect, border_radius=5)
        if time_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, fill_rect, border_radius=5)

    def _create_particles(self, c, r, color_idx):
        px, py = self._get_grid_pos(c, r)
        center_x, center_y = px + self.TILE_SIZE // 2, py + self.TILE_SIZE // 2
        color = self.TILE_COLORS[color_idx]
        for _ in range(15): # Create 15 particles per match
            self.particles.append(Particle(center_x, center_y, color, self.np_random))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, x, y, color, rng):
        self.x = x
        self.y = y
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = rng.integers(15, 30)
        self.color = color
        self.size = rng.integers(4, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.size = max(0, self.size - 0.2)

    def is_alive(self):
        return self.lifespan > 0

    def draw(self, surface):
        if self.is_alive():
            alpha = int(255 * (self.lifespan / 30))
            r, g, b = self.color
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), (r, g, b, alpha))

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    running = True
    clock = pygame.time.Clock()
    
    # Map pygame keys to gymnasium actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Check for movement keys (only one at a time for simplicity)
        for key, move in key_map.items():
            if keys[key]:
                movement_action = move
                break # Prioritize up/down/left/right in that order
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = (movement_action, space_action, shift_action)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
        
        clock.tick(10) # Control the speed of human play
        
    env.close()