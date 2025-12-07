import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import time
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A match-3 puzzle game where the player swaps adjacent gems on a 10x10 grid
    to create matches of three or more. The goal is to reach a target score
    before time runs out or no more moves are possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent square and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A fast-paced match-3 puzzle game. Swap gems to create lines of 3 or more. "
        "Plan your moves to create cascading combos and beat the clock!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.GRID_DIM = 10
        self.NUM_GEM_TYPES = 3  # Not including empty
        self.TARGET_SCORE = 1000
        self.TIME_LIMIT_SECS = 180
        self.MAX_STEPS = 5400 # 30fps * 180s

        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_AREA_WIDTH = 400
        self.CELL_SIZE = self.GRID_AREA_WIDTH // self.GRID_DIM
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_WIDTH) // 2

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Colors ---
        self.COLORS = {
            "bg": (20, 30, 40),
            "grid": (50, 60, 70),
            "cursor": (255, 255, 0),
            "cursor_selected": (0, 255, 0),
            "text": (220, 220, 230),
            "win": (100, 255, 100),
            "lose": (255, 100, 100),
            0: (0, 0, 0), # Empty
            1: (255, 50, 50), # Red
            2: (50, 200, 50), # Green
            3: (80, 150, 255), # Blue
        }
        
        # --- State Variables (initialized in reset) ---
        self.rng = None
        self.grid = None
        self.visual_grid = None
        self.score = 0
        self.steps = 0
        self.time_left = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_gem = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.animations = []
        self.particles = []
        self.is_board_stable = True
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rng is None or seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid = self._create_initial_grid()
        self.visual_grid = np.copy(self.grid) # For rendering during animations
        self.score = 0
        self.steps = 0
        self.time_left = self.TIME_LIMIT_SECS
        self.game_over = False
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.selected_gem = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.animations = []
        self.particles = []
        self.is_board_stable = True
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        # --- Update Timers and Animations ---
        self.time_left -= 1.0 / 30.0 # Assuming 30fps
        self._update_animations()
        self._update_particles()
        
        self.is_board_stable = not self.animations

        # --- Handle Input ---
        if self.is_board_stable and not self.game_over:
            reward += self._handle_input(action)

        # --- Handle Game Logic (Cascades) ---
        if self.is_board_stable and not self.game_over:
            match_reward = self._process_board_state()
            reward += match_reward

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 50 # Lose penalty (time out / no moves)
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLORS["bg"])
        self._render_grid_lines()
        self._render_gems()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "is_board_stable": self.is_board_stable,
        }

    # --- Game Logic Sub-functions ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        reward = 0

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_DIM - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_DIM - 1, self.cursor_pos[0] + 1) # Right

        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Deselect ---
        if shift_pressed and self.selected_gem:
            self.selected_gem = None

        # --- Select / Swap ---
        if space_pressed:
            c, r = self.cursor_pos # Note: cursor is (c, r)
            if not self.selected_gem:
                self.selected_gem = (r, c)
            else:
                # Try to swap
                sr, sc = self.selected_gem
                dr, dc = r - sr, c - sc
                if abs(dr) + abs(dc) == 1: # Is adjacent
                    reward += self._attempt_swap(self.selected_gem, (r, c))
                self.selected_gem = None
        return reward

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Hypothetical swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches1 = self._find_matches_at((r1, c1))
        matches2 = self._find_matches_at((r2, c2))
        
        if not matches1 and not matches2:
            # Invalid swap, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self._add_animation("swap", 0.2, {"pos1": pos1, "pos2": pos2, "reverse": True})
            return -0.1 # Penalty for invalid move
        else:
            # Valid swap
            self.visual_grid = np.copy(self.grid)
            self._add_animation("swap", 0.2, {"pos1": pos1, "pos2": pos2, "reverse": False})
            self.is_board_stable = False
            return 0 # Reward is given when matches are processed

    def _process_board_state(self):
        total_reward = 0
        matches = self._find_all_matches()
        if not matches:
            if not self._find_possible_moves():
                self.game_over = True
            return 0

        combo_multiplier = 1.0
        while matches:
            for length, gem_type in self._get_match_details(matches):
                base_score = (length - 2) * 10
                self.score += int(base_score * combo_multiplier)
                total_reward += (length - 2) * combo_multiplier

            for r, c in matches:
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._create_particles( (r,c), self.COLORS[gem_type] )

            for r, c in matches:
                self.grid[r, c] = 0

            self._apply_gravity_and_refill()
            self.is_board_stable = False
            
            matches = self._find_all_matches()
            if matches:
                combo_multiplier += 0.5
        
        return total_reward

    def _check_termination(self):
        return (
            self.score >= self.TARGET_SCORE
            or self.time_left <= 0
            or self.game_over # Set by no-moves check
        )

    # --- Grid & Match Logic ---
    
    def _create_initial_grid(self):
        while True:
            grid = self.rng.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_DIM, self.GRID_DIM))
            while self._find_all_matches(grid):
                matches = self._find_all_matches(grid)
                for r, c in matches:
                    grid[r, c] = self.rng.integers(1, self.NUM_GEM_TYPES + 1)
            if self._find_possible_moves(grid):
                return grid

    def _find_all_matches(self, grid=None):
        if grid is None: grid = self.grid
        matches = set()
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if grid[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_DIM - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    for i in range(c, self.GRID_DIM):
                        if grid[r, i] == grid[r, c]: matches.add((r, i))
                        else: break
                # Vertical
                if r < self.GRID_DIM - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    for i in range(r, self.GRID_DIM):
                        if grid[i, c] == grid[r, c]: matches.add((i, c))
                        else: break
        return list(matches)

    def _find_matches_at(self, pos, grid=None):
        if grid is None: grid = self.grid
        r, c = pos
        gem_type = grid[r, c]
        if gem_type == 0: return []
        
        h_matches = {pos}
        for i in range(c - 1, -1, -1):
            if grid[r, i] == gem_type: h_matches.add((r, i))
            else: break
        for i in range(c + 1, self.GRID_DIM):
            if grid[r, i] == gem_type: h_matches.add((r, i))
            else: break

        v_matches = {pos}
        for i in range(r - 1, -1, -1):
            if grid[i, c] == gem_type: v_matches.add((i, c))
            else: break
        for i in range(r + 1, self.GRID_DIM):
            if grid[i, c] == gem_type: v_matches.add((i, c))
            else: break

        found = set()
        if len(h_matches) >= 3: found.update(h_matches)
        if len(v_matches) >= 3: found.update(v_matches)
        return list(found)

    def _find_possible_moves(self, grid=None):
        if grid is None: temp_grid = np.copy(self.grid)
        else: temp_grid = np.copy(grid)
        
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                # Try swapping right
                if c < self.GRID_DIM - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_matches_at((r, c), temp_grid) or self._find_matches_at((r, c+1), temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c] # Swap back
                # Try swapping down
                if r < self.GRID_DIM - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_matches_at((r, c), temp_grid) or self._find_matches_at((r+1, c), temp_grid):
                        return True
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c] # Swap back
        return False

    def _get_match_details(self, matches):
        details = []
        checked = set()
        sorted_matches = sorted(list(matches))

        for r_start, c_start in sorted_matches:
            if (r_start, c_start) in checked: continue
            
            h_len = 1
            for c in range(c_start + 1, self.GRID_DIM):
                if (r_start, c) in matches: h_len += 1
                else: break
            if h_len >= 3:
                details.append((h_len, self.grid[r_start,c_start]))
                for c in range(c_start, c_start + h_len): checked.add((r_start, c))

            v_len = 1
            for r in range(r_start + 1, self.GRID_DIM):
                if (r, c_start) in matches: v_len += 1
                else: break
            if v_len >= 3 and (r_start, c_start) not in checked:
                details.append((v_len, self.grid[r_start,c_start]))
                for r in range(r_start, r_start + v_len): checked.add((r, c_start))
        return details

    def _apply_gravity_and_refill(self):
        fall_duration = 0.3
        for c in range(self.GRID_DIM):
            write_idx = self.GRID_DIM - 1
            for r in range(self.GRID_DIM - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if write_idx != r:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                        self._add_animation("fall", fall_duration, {"from": (r, c), "to": (write_idx, c), "gem_type": self.grid[write_idx,c]})
                    write_idx -= 1
            
            for r in range(write_idx, -1, -1):
                new_gem = self.rng.integers(1, self.NUM_GEM_TYPES + 1)
                self.grid[r, c] = new_gem
                self._add_animation("fall", fall_duration, {"from": (-1 - (write_idx - r), c), "to": (r, c), "gem_type": new_gem})

    # --- Animation & Particle System ---

    def _add_animation(self, type, duration, data):
        self.animations.append({"type": type, "duration": duration, "progress": 0, "data": data})

    def _update_animations(self):
        dt = 1.0 / 30.0
        active_animations = []
        for anim in self.animations:
            anim["progress"] += dt
            if anim["progress"] < anim["duration"]:
                active_animations.append(anim)
            else: # Animation finished
                if anim["type"] == "swap" and anim["data"]["reverse"]:
                    # Ensure visual grid snaps back after a failed swap animation
                    self.visual_grid = np.copy(self.grid)
        self.animations = active_animations
        if not self.animations:
            self.visual_grid = np.copy(self.grid)


    def _create_particles(self, pos, color):
        r, c = pos
        center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.rng.random() * 0.5 + 0.3
            self.particles.append({"pos": [center_x, center_y], "vel": vel, "lifespan": lifespan, "max_life": lifespan, "color": color})
    
    def _update_particles(self):
        dt = 1.0 / 30.0
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 2.0 # gravity
            p["lifespan"] -= dt
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    # --- Rendering ---

    def _render_grid_lines(self):
        for i in range(self.GRID_DIM + 1):
            start_x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLORS["grid"], (start_x, self.GRID_OFFSET_Y), (start_x, self.GRID_OFFSET_Y + self.GRID_AREA_WIDTH))
            start_y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLORS["grid"], (self.GRID_OFFSET_X, start_y), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, start_y))

    def _render_gems(self):
        gem_size = int(self.CELL_SIZE * 0.8)
        offset = (self.CELL_SIZE - gem_size) // 2
        
        drawn_this_frame = set()

        for anim in self.animations:
            progress = min(1.0, anim["progress"] / anim["duration"])
            
            if anim["type"] == "swap":
                r1, c1 = anim["data"]["pos1"]
                r2, c2 = anim["data"]["pos2"]
                
                interp_progress = progress
                if anim["data"]["reverse"]:
                    interp_progress = (0.5 - abs(progress - 0.5)) * 2

                pos1_interp = (r1 + (r2-r1)*interp_progress, c1 + (c2-c1)*interp_progress)
                pos2_interp = (r2 + (r1-r2)*interp_progress, c2 + (c1-c2)*interp_progress)
                
                self._draw_gem(pos1_interp, self.visual_grid[r2, c2], gem_size, offset)
                self._draw_gem(pos2_interp, self.visual_grid[r1, c1], gem_size, offset)
                drawn_this_frame.add((r1, c1))
                drawn_this_frame.add((r2, c2))
            
            elif anim["type"] == "fall":
                fr, fc = anim["data"]["from"]
                tr, tc = anim["data"]["to"]
                
                interp_pos = (fr + (tr - fr) * progress, fc)
                self._draw_gem(interp_pos, anim["data"]["gem_type"], gem_size, offset)
                drawn_this_frame.add((tr, tc))
        
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if (r, c) not in drawn_this_frame and self.visual_grid[r, c] > 0:
                    self._draw_gem((r, c), self.visual_grid[r, c], gem_size, offset)
    
    def _draw_gem(self, pos, gem_type, size, offset):
        r, c = pos
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE + offset
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + offset
        
        color = self.COLORS.get(gem_type, (255, 255, 255))
        rect = pygame.Rect(int(x), int(y), size, size)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        
        if self.rng.random() < 0.005:
            sparkle_x = x + size * self.rng.random()
            sparkle_y = y + size * self.rng.random()
            pygame.draw.circle(self.screen, (255,255,255), (int(sparkle_x), int(sparkle_y)), 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / p["max_life"]))))
            color = p["color"]
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, (*color, alpha))
            except TypeError: # Color might not have alpha
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, color)
    
    def _render_cursor(self):
        c, r = self.cursor_pos
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        color = self.COLORS["cursor_selected"] if self.selected_gem else self.COLORS["cursor"]
        pygame.draw.rect(self.screen, color, rect, width=3)
        
        if self.selected_gem:
            sr, sc = self.selected_gem
            sx = self.GRID_OFFSET_X + sc * self.CELL_SIZE
            sy = self.GRID_OFFSET_Y + sr * self.CELL_SIZE
            s_rect = pygame.Rect(sx, sy, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLORS["cursor_selected"], s_rect, width=4, border_radius=4)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLORS["text"])
        self.screen.blit(score_text, (15, 10))

        timer_bar_width = 200
        timer_bar_height = 20
        time_ratio = max(0, self.time_left / self.TIME_LIMIT_SECS)
        
        bg_rect = pygame.Rect(self.WIDTH - timer_bar_width - 15, 10, timer_bar_width, timer_bar_height)
        pygame.draw.rect(self.screen, self.COLORS["grid"], bg_rect)
        
        fg_color = (255, 255, 0) if time_ratio > 0.5 else (255, 165, 0) if time_ratio > 0.2 else (255, 0, 0)
        fg_rect = pygame.Rect(bg_rect.x, bg_rect.y, int(timer_bar_width * time_ratio), timer_bar_height)
        pygame.draw.rect(self.screen, fg_color, fg_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        is_win = self.score >= self.TARGET_SCORE
        text = "YOU WIN!" if is_win else "GAME OVER"
        color = self.COLORS["win"] if is_win else self.COLORS["lose"]
        
        rendered_text = self.font_large.render(text, True, color)
        text_rect = rendered_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(rendered_text, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a screen, so it will not run in a headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Gem Matcher")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    running = True
    clock = pygame.time.Clock()
    
    # Track key presses to implement press-once behavior
    last_keys = pygame.key.get_pressed()

    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()

        # Simplified movement for human play
        if keys[pygame.K_UP] and not last_keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN] and not last_keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT] and not last_keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT] and not last_keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
        
        last_keys = keys

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            time.sleep(3)
            obs, info = env.reset()
        
        clock.tick(30) # Control the frame rate for human play

    env.close()