
# Generated: 2025-08-27T21:41:11.466470
# Source Brief: brief_02872.md
# Brief Index: 2872

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a tile. "
        "Move the cursor to an adjacent tile and press space again to swap. "
        "Match 3 or more to score."
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Create combos and cascades to maximize your score before the 60-second timer runs out!"
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_HEIGHT = 360
    TILE_SIZE = GRID_AREA_HEIGHT // GRID_HEIGHT
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * TILE_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_AREA_HEIGHT)

    NUM_TILE_TYPES = 5
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_SCORE = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_TIMER_BAR = (0, 200, 200)
    COLOR_TIMER_BG = (50, 80, 80)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    LOCKED_COLOR = (100, 110, 120)

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.game_phase = "IDLE"
        self.animation_progress = 0
        self.swap_info = {}
        self.match_info = []
        self.fall_info = {}
        self.particles = []
        self.cascade_level = 0
        
        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_STEPS

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tile = None
        self.prev_space_held = False
        
        self._generate_initial_grid()
        
        self.game_phase = "IDLE"
        self.animation_progress = 0
        self.swap_info = {}
        self.match_info = []
        self.fall_info = {}
        self.particles = []
        self.cascade_level = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.time_remaining -= 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_phase == "IDLE":
            reward += self._handle_input(action)
        elif self.game_phase in ["SWAPPING", "REJECTING"]:
            self._update_swap_animation()
        elif self.game_phase == "MATCHING":
            self._update_match_animation()
        elif self.game_phase == "FALLING":
            self._update_fall_animation()

        self._update_particles()
        
        terminated = self.time_remaining <= 0 or self.score >= self.WIN_SCORE
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                self.win = True
            else:
                reward += -10
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Move Cursor ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Handle Selection/Swap on Space Press ---
        reward = 0
        if space_held and not self.prev_space_held:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy] == -1: # Cannot select locked tile
                # Sound: error.wav
                pass
            elif self.selected_tile is None:
                self.selected_tile = (cx, cy)
                # Sound: select.wav
            else:
                sx, sy = self.selected_tile
                is_adjacent = abs(sx - cx) + abs(sy - cy) == 1
                if is_adjacent:
                    # Attempt swap
                    self.grid[sx, sy], self.grid[cx, cy] = self.grid[cx, cy], self.grid[sx, sy]
                    matches = self._find_matches()
                    if matches:
                        self.grid[sx, sy], self.grid[cx, cy] = self.grid[cx, cy], self.grid[sx, sy] # Swap back for animation
                        self._start_swap((sx, sy), (cx, cy), "SWAPPING")
                        self.cascade_level = 0
                    else:
                        self.grid[sx, sy], self.grid[cx, cy] = self.grid[cx, cy], self.grid[sx, sy] # Swap back
                        self._start_swap((sx, sy), (cx, cy), "REJECTING")
                        reward = -0.1
                        # Sound: invalid_swap.wav
                else: # Not adjacent, just deselect
                    # Sound: deselect.wav
                    pass
                self.selected_tile = None
        
        self.prev_space_held = space_held
        return reward
    
    def _start_swap(self, pos1, pos2, phase):
        self.game_phase = phase
        self.animation_progress = 0
        self.swap_info = {"pos1": pos1, "pos2": pos2}

    def _update_swap_animation(self):
        self.animation_progress += 0.2 # 5 frames for swap
        if self.animation_progress >= 1:
            self.animation_progress = 0
            x1, y1 = self.swap_info["pos1"]
            x2, y2 = self.swap_info["pos2"]

            if self.game_phase == "SWAPPING":
                self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
                self._process_matches()
            else: # REJECTING
                self.game_phase = "IDLE"
            self.swap_info = {}

    def _update_match_animation(self):
        self.animation_progress += 0.25 # 4 frames for match
        if self.animation_progress >= 1:
            self.animation_progress = 0
            for x, y, _ in self.match_info:
                self.grid[x, y] = 0 # Mark as empty
            self.match_info = []
            self._start_fall()

    def _update_fall_animation(self):
        self.animation_progress += 0.25 # 4 frames for fall
        if self.animation_progress >= 1:
            self.animation_progress = 0
            # Apply final positions
            new_grid = np.zeros_like(self.grid)
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if (x, y) in self.fall_info:
                        ox, oy = self.fall_info[(x, y)]["from"]
                        if ox is not None:
                            new_grid[x, y] = self.grid[ox, oy]
                        else: # New tile
                            new_grid[x, y] = self.fall_info[(x, y)]["type"]
                    else:
                        new_grid[x, y] = self.grid[x, y]
            self.grid = new_grid
            self.fall_info = {}
            
            # Check for cascade matches
            if not self._process_matches():
                self.game_phase = "IDLE"
                self._ensure_possible_moves()


    def _process_matches(self):
        matches = self._find_matches()
        if not matches:
            return False

        # Sound: match.wav
        self.cascade_level += 1
        reward = 0
        if self.cascade_level > 1:
            reward += 5 # Cascade bonus
        
        match_coords = set()
        for match in matches:
            length = len(match)
            if length == 3: reward += 1
            elif length == 4: reward += 2
            else: reward += 3
            for x, y in match:
                match_coords.add((x, y))
        
        self.score += reward

        for x, y in match_coords:
            self.match_info.append((x, y, self.grid[x, y]))
            self._spawn_particles(x, y, self.TILE_COLORS[self.grid[x, y] - 1])
        
        self.game_phase = "MATCHING"
        self.animation_progress = 0
        return True

    def _start_fall(self):
        self.fall_info = {}
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.fall_info[(x, y + empty_count)] = {"from": (x, y)}
            # New tiles
            for i in range(empty_count):
                self.fall_info[(x, i)] = {"from": (x, i - empty_count), "type": self.np_random.integers(1, self.NUM_TILE_TYPES + 1)}

        if self.fall_info:
            self.game_phase = "FALLING"
            self.animation_progress = 0
            # Sound: fall.wav
        else:
            self.game_phase = "IDLE"

    def _find_matches(self):
        all_matches = []
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x, y] > 0 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    match = [(x, y), (x+1, y), (x+2, y)]
                    i = x + 3
                    while i < self.GRID_WIDTH and self.grid[i, y] == self.grid[x, y]:
                        match.append((i, y))
                        i += 1
                    all_matches.append(match)
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x, y] > 0 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    match = [(x, y), (x, y+1), (x, y+2)]
                    i = y + 3
                    while i < self.GRID_HEIGHT and self.grid[x, i] == self.grid[x, y]:
                        match.append((x, i))
                        i += 1
                    all_matches.append(match)
        # Filter out duplicates from overlapping matches
        unique_matches = []
        seen_coords = set()
        for match in sorted(all_matches, key=len, reverse=True):
            if not any(coord in seen_coords for coord in match):
                unique_matches.append(match)
                for coord in match:
                    seen_coords.add(coord)
        return unique_matches

    def _generate_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            if self.np_random.random() < 0.9: # 10% chance to have locked tiles
                for _ in range(self.np_random.integers(1, 4)):
                    lx, ly = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
                    self.grid[lx, ly] = -1

            if not self._find_matches() and self._has_possible_moves():
                break

    def _ensure_possible_moves(self):
        if not self._has_possible_moves():
            # Reshuffle board
            flat_grid = self.grid.flatten()
            self.np_random.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
            self._ensure_possible_moves() # Recurse until a valid board is made
            if self._find_matches(): # If shuffle creates matches, clear them
                self._process_matches()

    def _has_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == -1: continue
                # Check swap right
                if x < self.GRID_WIDTH - 1 and self.grid[x+1, y] != -1:
                    self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                    if self._find_matches(): self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]; return True
                    self.grid[x, y], self.grid[x+1, y] = self.grid[x+1, y], self.grid[x, y]
                # Check swap down
                if y < self.GRID_HEIGHT - 1 and self.grid[x, y+1] != -1:
                    self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
                    if self._find_matches(): self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]; return True
                    self.grid[x, y], self.grid[x, y+1] = self.grid[x, y+1], self.grid[x, y]
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE))

        # Draw tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_type = self.grid[x, y]
                if tile_type == 0: continue

                px, py = self.GRID_X_OFFSET + x * self.TILE_SIZE, self.GRID_Y_OFFSET + y * self.TILE_SIZE
                
                # Handle animations
                scale = 1.0
                if self.game_phase == "MATCHING" and any(m[0]==x and m[1]==y for m in self.match_info):
                    scale = 1.0 - self.animation_progress
                elif self.game_phase == "SWAPPING" or self.game_phase == "REJECTING":
                    if (x, y) == self.swap_info["pos1"]:
                        x2, y2 = self.swap_info["pos2"]
                        px = int(self._lerp(px, self.GRID_X_OFFSET + x2 * self.TILE_SIZE, self.animation_progress))
                        py = int(self._lerp(py, self.GRID_Y_OFFSET + y2 * self.TILE_SIZE, self.animation_progress))
                    elif (x, y) == self.swap_info["pos2"]:
                        x1, y1 = self.swap_info["pos1"]
                        px = int(self._lerp(px, self.GRID_X_OFFSET + x1 * self.TILE_SIZE, self.animation_progress))
                        py = int(self._lerp(py, self.GRID_Y_OFFSET + y1 * self.TILE_SIZE, self.animation_progress))
                elif self.game_phase == "FALLING" and (x, y) in self.fall_info:
                    fx, fy = self.fall_info[(x, y)]["from"]
                    if fx is not None:
                        tile_type = self.grid[fx, fy]
                    else: # New tile
                        tile_type = self.fall_info[(x, y)]["type"]
                    
                    start_y = self.GRID_Y_OFFSET + fy * self.TILE_SIZE
                    py = int(self._lerp(start_y, py, self.animation_progress))

                self._draw_tile(px, py, tile_type, scale)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw cursor and selection
        cx, cy = self.cursor_pos
        cursor_rect = (self.GRID_X_OFFSET + cx * self.TILE_SIZE, self.GRID_Y_OFFSET + cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 3)

        if self.selected_tile:
            sx, sy = self.selected_tile
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
            sel_color = self._lerp_color((255,255,255), (255,255,0), pulse)
            sel_rect = (self.GRID_X_OFFSET + sx * self.TILE_SIZE, self.GRID_Y_OFFSET + sy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, sel_color, sel_rect, 4)

    def _draw_tile(self, px, py, tile_type, scale):
        if tile_type == 0: return
        size = int(self.TILE_SIZE * scale)
        offset = (self.TILE_SIZE - size) // 2
        rect = (px + offset, py + offset, size, size)
        
        if tile_type == -1: # Locked tile
            color = self.LOCKED_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, (0,0,0), (rect[0]+5, rect[1]+5, rect[2]-10, rect[3]-10), 3, border_radius=3)
        else:
            color = self.TILE_COLORS[tile_type - 1]
            light_color = self._lerp_color(color, (255, 255, 255), 0.3)
            dark_color = self._lerp_color(color, (0, 0, 0), 0.3)
            
            pygame.draw.rect(self.screen, dark_color, rect, border_radius=8)
            inner_rect = (rect[0] + 2, rect[1] + 2, rect[2] - 4, rect[3] - 4)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=6)
            
            # Add a small shine effect
            shine_rect = (inner_rect[0] + 4, inner_rect[1] + 4, int(inner_rect[2] * 0.5), int(inner_rect[3] * 0.5))
            pygame.gfxdraw.box(self.screen, shine_rect, (*light_color, 50))


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_width = 200
        timer_height = 20
        timer_x = self.SCREEN_WIDTH - timer_width - 10
        timer_y = 10
        
        time_ratio = max(0, self.time_remaining / self.MAX_STEPS)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, (timer_x, timer_y, timer_width, timer_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (timer_x, timer_y, int(timer_width * time_ratio), timer_height), border_radius=5)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "TIME UP!"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_x, grid_y, color):
        px = self.GRID_X_OFFSET + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_Y_OFFSET + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'radius': random.uniform(2, 5),
                'life': random.randint(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['radius'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    @staticmethod
    def _lerp_color(c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )
        
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
    pygame.display.set_caption("Match-3 Gymnasium Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = np.array([movement, space_held, 0]) # Shift is unused

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

        if terminated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # In a real scenario, you'd wait for reset input, but here we just show the final frame.
            # To play again, press 'r'.

        # --- Render ---
        # The observation is already a rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()