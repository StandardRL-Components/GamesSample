
# Generated: 2025-08-28T05:03:40.446067
# Source Brief: brief_05450.md
# Brief Index: 5450

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. "
        "Hold Space and press an arrow key to swap fruits. Hold Shift to reshuffle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading fruits in a grid-based frenzy to reach a target score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 8
    CELL_SIZE = 40
    GRID_ORIGIN = (160, 40)
    TARGET_SCORE = 1000
    TOTAL_TIME = 180  # seconds
    MAX_STEPS = TOTAL_TIME * 30 # 30 FPS

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_BG = (40, 45, 50)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    COLOR_TIMER_WARN = (255, 150, 0)
    COLOR_TIMER_CRITICAL = (255, 50, 50)
    
    FRUIT_COLORS = [
        (255, 80, 80),   # Red (Cherry)
        (80, 255, 80),   # Green (Apple)
        (80, 80, 255),   # Blue (Blueberry)
        (255, 255, 80),  # Yellow (Lemon)
        (255, 80, 255),  # Magenta (Grape)
        (80, 255, 255),  # Cyan (Plum)
        (255, 165, 0),   # Orange (Orange)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("tahoma", 20, bold=True)
        self.font_large = pygame.font.SysFont("tahoma", 32, bold=True)
        self.font_small = pygame.font.SysFont("tahoma", 14)

        self.np_random = None
        self.game_state = "IDLE"
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [0, 0]
        self.animation_progress = 0
        self.swapping_pair = None
        self.matched_cells = set()
        self.falling_info = {}
        self.particles = []
        self.chain_reaction_level = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.time_remaining = self.TOTAL_TIME
        self.stage = 1
        self.num_fruit_types = 4
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TOTAL_TIME
        self.stage = 1
        self.num_fruit_types = 4
        
        self.game_state = "IDLE"
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.animation_progress = 0
        self.swapping_pair = None
        self.matched_cells = set()
        self.falling_info = {}
        self.particles = []
        self.chain_reaction_level = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._initialize_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = self.game_over

        # Process input only when the board is settled
        if self.game_state == "IDLE":
            reward += self._handle_input(action)
        
        self._update_game_state()

        # Update timer and stage
        self.time_remaining -= 1.0 / 30.0 # Assuming 30 FPS
        self._update_stage()

        self.steps += 1
        
        if not terminated:
            if self.time_remaining <= 0:
                terminated = True
                reward -= 50
                self.game_over = True
                self.game_state = "GAME_OVER"
            elif self.score >= self.TARGET_SCORE:
                terminated = True
                reward += 100
                self.game_over = True
                self.game_state = "VICTORY"
            elif self.steps >= self.MAX_STEPS:
                terminated = True

        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action
        reward = 0

        # Action interpretation: Hold space + direction to swap. Direction only to move.
        if space_held and movement != 0:
            # sfx: swap_attempt
            directions = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dr, dc = directions[movement]
            r1, c1 = self.cursor_pos
            r2, c2 = r1 + dr, c1 + dc

            if 0 <= r2 < self.GRID_SIZE and 0 <= c2 < self.GRID_SIZE:
                # Perform swap
                self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                
                # Check if this swap is valid (creates a match)
                matches1 = self._find_matches_at_cell(r1, c1)
                matches2 = self._find_matches_at_cell(r2, c2)
                
                if not matches1 and not matches2:
                    # Invalid swap, swap back
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    reward -= 0.1
                else:
                    self.swapping_pair = ((r1, c1), (r2, c2))
                    self.animation_progress = 0
                    self.game_state = "SWAPPING"
                    self.chain_reaction_level = 0
        
        elif shift_held and not self.prev_shift_held:
            # sfx: reshuffle
            self._reshuffle_grid()
            reward -= 1.0

        elif not space_held and movement != 0:
            # Move cursor
            directions = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dr, dc = directions[movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dr) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dc) % self.GRID_SIZE

        self.prev_space_held = bool(space_held)
        self.prev_shift_held = bool(shift_held)
        return reward

    def _update_game_state(self):
        """State machine for animations and game logic flow."""
        if self.game_state == "SWAPPING":
            self.animation_progress += 0.15
            if self.animation_progress >= 1.0:
                self.swapping_pair = None
                self.game_state = "MATCH_CHECK"
        
        elif self.game_state == "MATCH_CHECK":
            all_matches = self._find_all_matches()
            if all_matches:
                # sfx: match_found
                self.matched_cells = all_matches
                self.animation_progress = 0
                self.game_state = "CLEARING"
                
                reward = len(all_matches)
                if len(all_matches) >= 5: reward += 5 # Cluster bonus
                if self.chain_reaction_level > 0: reward += 10 # Combo bonus
                self.score += int(reward * (1 + self.chain_reaction_level * 0.5))

                self.chain_reaction_level += 1
                self._spawn_particles(all_matches)
            else:
                self.chain_reaction_level = 0
                if not self._find_possible_moves():
                    self._reshuffle_grid() # Anti-softlock
                self.game_state = "IDLE"
        
        elif self.game_state == "CLEARING":
            self.animation_progress += 0.1
            if self.animation_progress >= 1.0:
                for r, c in self.matched_cells:
                    self.grid[r, c] = -1 # Mark as empty
                self.matched_cells = set()
                self.game_state = "FALLING"
                self._prepare_falling_fruits()
        
        elif self.game_state == "FALLING":
            self.animation_progress += 0.2
            if self.animation_progress >= 1.0:
                self._apply_fall()
                self.falling_info = {}
                self.game_state = "REFILLING"
            
        elif self.game_state == "REFILLING":
            self._refill_grid()
            self.game_state = "MATCH_CHECK" # Check for chain reactions

    def _update_stage(self):
        if self.stage == 1 and self.TOTAL_TIME - self.time_remaining >= 60:
            self.stage = 2
            self.num_fruit_types = 5
        elif self.stage == 2 and self.TOTAL_TIME - self.time_remaining >= 120:
            self.stage = 3
            self.num_fruit_types = 6

    def _initialize_grid(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.grid[r, c] = self._get_new_fruit(r, c)
        
        while self._find_all_matches() or not self._find_possible_moves():
            self._reshuffle_grid(False)

    def _get_new_fruit(self, r, c):
        avoid = []
        # Avoid creating horizontal matches
        if c >= 2 and self.grid[r, c-1] == self.grid[r, c-2]:
            avoid.append(self.grid[r, c-1])
        # Avoid creating vertical matches
        if r >= 2 and self.grid[r-1, c] == self.grid[r-2, c]:
            avoid.append(self.grid[r-1, c])
        
        possible_fruits = [i for i in range(self.num_fruit_types) if i not in avoid]
        if not possible_fruits:
            possible_fruits = list(range(self.num_fruit_types))
            
        return self.np_random.choice(possible_fruits)

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1: continue
                # Horizontal
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_matches_at_cell(self, r, c):
        fruit_id = self.grid[r, c]
        if fruit_id == -1: return set()
        
        # Horizontal check
        h_matches = {(r, c)}
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == fruit_id: h_matches.add((r, i))
            else: break
        for i in range(c + 1, self.GRID_SIZE):
            if self.grid[r, i] == fruit_id: h_matches.add((r, i))
            else: break
        
        # Vertical check
        v_matches = {(r, c)}
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == fruit_id: v_matches.add((i, c))
            else: break
        for i in range(r + 1, self.GRID_SIZE):
            if self.grid[i, c] == fruit_id: v_matches.add((i, c))
            else: break
            
        final_matches = set()
        if len(h_matches) >= 3: final_matches.update(h_matches)
        if len(v_matches) >= 3: final_matches.update(v_matches)
        return final_matches

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_at_cell(r, c) or self._find_matches_at_cell(r, c+1):
                        moves.append(((r, c), (r, c+1)))
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c] # Swap back
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_at_cell(r, c) or self._find_matches_at_cell(r+1, c):
                        moves.append(((r, c), (r+1, c)))
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c] # Swap back
        return moves

    def _reshuffle_grid(self, check_possible_moves=True):
        flat_grid = list(self.grid.flatten())
        self.np_random.shuffle(flat_grid)
        self.grid = np.array(flat_grid).reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        # Ensure no matches and at least one move is possible
        while self._find_all_matches() or (check_possible_moves and not self._find_possible_moves()):
            self.np_random.shuffle(flat_grid)
            self.grid = np.array(flat_grid).reshape((self.GRID_SIZE, self.GRID_SIZE))

    def _prepare_falling_fruits(self):
        self.falling_info = {}
        for c in range(self.GRID_SIZE):
            empty_count = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.falling_info[(r, c)] = empty_count
        self.animation_progress = 0

    def _apply_fall(self):
        for c in range(self.GRID_SIZE):
            write_idx = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_idx:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    write_idx -= 1

    def _refill_grid(self):
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.num_fruit_types)

    def _spawn_particles(self, cells):
        for r, c in cells:
            fruit_id = self.grid[r,c]
            if fruit_id == -1: continue
            color = self.FRUIT_COLORS[fruit_id]
            px, py = self._get_cell_center(r, c)
            for _ in range(10):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifetime = self.np_random.integers(15, 30)
                self.particles.append([px, py, vel[0], vel[1], lifetime, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime--

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_fruits()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "stage": self.stage,
            "game_state": self.game_state
        }

    def _get_cell_center(self, r, c):
        x = self.GRID_ORIGIN[0] + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_ORIGIN[1] + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_grid_bg(self):
        grid_width = self.GRID_SIZE * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GRID, (*self.GRID_ORIGIN, grid_width, grid_width))
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_ORIGIN[0] + i * self.CELL_SIZE, self.GRID_ORIGIN[1])
            end_pos = (self.GRID_ORIGIN[0] + i * self.CELL_SIZE, self.GRID_ORIGIN[1] + grid_width)
            pygame.draw.line(self.screen, self.COLOR_BG, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_ORIGIN[0], self.GRID_ORIGIN[1] + i * self.CELL_SIZE)
            end_pos = (self.GRID_ORIGIN[0] + grid_width, self.GRID_ORIGIN[1] + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start_pos, end_pos, 1)

    def _render_fruits(self):
        radius = self.CELL_SIZE // 2 - 4
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                fruit_id = self.grid[r, c]
                if fruit_id == -1:
                    continue

                px, py = self._get_cell_center(r, c)
                scale = 1.0
                
                # Animations
                if self.swapping_pair:
                    (r1, c1), (r2, c2) = self.swapping_pair
                    if (r, c) == (r1, c1):
                        px, py = self._interpolate_pos((r1, c1), (r2, c2), self.animation_progress)
                    elif (r, c) == (r2, c2):
                        px, py = self._interpolate_pos((r2, c2), (r1, c1), self.animation_progress)
                elif (r, c) in self.falling_info:
                    drop_dist = self.falling_info[(r,c)]
                    start_y = py - drop_dist * self.CELL_SIZE
                    py = start_y + drop_dist * self.CELL_SIZE * self.animation_progress
                elif (r, c) in self.matched_cells:
                    scale = 1.0 - self.animation_progress
                
                current_radius = int(radius * scale)
                if current_radius <= 0: continue

                color = self.FRUIT_COLORS[fruit_id]
                highlight_color = tuple(min(255, val + 60) for val in color)
                
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), current_radius, color)
                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), current_radius, color)
                pygame.gfxdraw.filled_circle(self.screen, int(px - current_radius*0.3), int(py - current_radius*0.3), int(current_radius*0.3), highlight_color)

    def _interpolate_pos(self, pos1, pos2, t):
        x1, y1 = self._get_cell_center(*pos1)
        x2, y2 = self._get_cell_center(*pos2)
        ix = x1 + (x2 - x1) * t
        iy = y1 + (y2 - y1) * t
        return ix, iy

    def _render_cursor(self):
        if self.game_state not in ["IDLE", "GAME_OVER", "VICTORY"]: return
        
        r, c = self.cursor_pos
        x = self.GRID_ORIGIN[0] + c * self.CELL_SIZE
        y = self.GRID_ORIGIN[1] + r * self.CELL_SIZE
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        color = (255, 255, 100 + 155 * pulse)
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect, 3, border_radius=4)
        
    def _render_particles(self):
        for p in self.particles:
            x, y, _, _, lifetime, color = p
            alpha = min(255, int(255 * (lifetime / 15.0)))
            size = int(5 * (lifetime / 15.0))
            if size <= 0: continue
            
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (size, size), size)
            self.screen.blit(s, (int(x-size), int(y-size)))

    def _render_ui(self):
        # Score
        self._render_text("SCORE", (20, 50), self.font_main)
        self._render_text(f"{self.score}", (20, 80), self.font_large)

        # Timer
        timer_color = self.COLOR_TEXT
        if self.time_remaining < 10: timer_color = self.COLOR_TIMER_CRITICAL
        elif self.time_remaining < 30: timer_color = self.COLOR_TIMER_WARN
        
        self._render_text("TIME", (20, 150), self.font_main)
        minutes = int(self.time_remaining) // 60
        seconds = int(self.time_remaining) % 60
        self._render_text(f"{minutes:02}:{seconds:02}", (20, 180), self.font_large, color=timer_color)
        
        # Stage
        self._render_text(f"STAGE {self.stage}/3", (20, 250), self.font_main)

        # Game Over / Victory
        if self.game_state in ["GAME_OVER", "VICTORY"]:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.game_state == "VICTORY" else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Fruit Frenzy")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # In a real game loop, you might wait for a key press to reset
            # For this demo, we'll just let it sit on the final screen
            # To auto-reset: obs, info = env.reset()
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match env's assumed FPS

    pygame.quit()