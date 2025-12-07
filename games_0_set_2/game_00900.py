
# Generated: 2025-08-27T15:08:33.590898
# Source Brief: brief_00900.md
# Brief Index: 900

        
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
        "Controls: Arrow keys to move cursor. Press space to select a gem, "
        "then use an arrow key to choose a swap direction."
    )

    game_description = (
        "A colorful match-3 puzzle game. Swap adjacent gems to create lines of "
        "3 or more. Collect 50 gems before the 60-second timer runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_GEM_TYPES = 6
        self.GEM_SIZE = 40
        self.GRID_X = (self.WIDTH - self.GRID_COLS * self.GEM_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_ROWS * self.GEM_SIZE) // 2 + 20
        self.GEM_GOAL = 50
        self.MAX_TIME = 60 * 30  # 60 seconds at 30 FPS
        self.MAX_STEPS = 1000  # Fallback termination

        # Animation timings (in frames)
        self.SWAP_DURATION = 8
        self.FALL_DURATION = 10
        self.DESTROY_DURATION = 12
        self.REFILL_DURATION = 10

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_SELECTED = (0, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TIMER_BAR_GOOD = (0, 200, 100)
        self.COLOR_TIMER_BAR_WARN = (255, 180, 0)
        self.COLOR_TIMER_BAR_BAD = (220, 50, 50)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 200, 50),   # Green
            (80, 150, 255),  # Blue
            (255, 180, 0),   # Yellow
            (200, 80, 255),  # Purple
            (255, 100, 0),   # Orange
        ]

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
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.selection_state = None
        self.selected_gem_coord = None
        self.score = None
        self.gems_collected = None
        self.time_left = None
        self.game_over = None
        self.last_space_held = None
        self.steps = None
        self.animations = []
        self.particles = []
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Use a default generator if no seed is provided
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.time_left = self.MAX_TIME
        self.game_over = False
        self.last_space_held = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selection_state = "NEUTRAL"
        self.selected_gem_coord = None
        self.animations = []
        self.particles = []

        self._initialize_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0

        self._update_animations()
        self._update_particles()
        
        self.time_left = max(0, self.time_left - 1)

        if not self.animations and not self.game_over:
            reward += self._handle_input(action)
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.gems_collected >= self.GEM_GOAL:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        reward = 0

        # --- State: NEUTRAL ---
        if self.selection_state == "NEUTRAL":
            if space_pressed:
                # Select a gem
                self.selection_state = "SELECTED"
                self.selected_gem_coord = tuple(self.cursor_pos)
                # sfx: select_gem.wav
            else:
                # Move cursor
                dx, dy = 0, 0
                if movement == 1: dy = -1  # Up
                elif movement == 2: dy = 1   # Down
                elif movement == 3: dx = -1  # Left
                elif movement == 4: dx = 1   # Right
                
                if dx != 0 or dy != 0:
                    self.cursor_pos[0] = (self.cursor_pos[0] + dx + self.GRID_COLS) % self.GRID_COLS
                    self.cursor_pos[1] = (self.cursor_pos[1] + dy + self.GRID_ROWS) % self.GRID_ROWS
                    # sfx: cursor_move.wav
        
        # --- State: SELECTED ---
        elif self.selection_state == "SELECTED":
            if space_pressed:
                # Cancel selection
                self.selection_state = "NEUTRAL"
                self.selected_gem_coord = None
                # sfx: cancel_selection.wav
            else:
                swap_dir = None
                if movement == 1: swap_dir = (0, -1) # Up
                elif movement == 2: swap_dir = (0, 1)  # Down
                elif movement == 3: swap_dir = (-1, 0) # Left
                elif movement == 4: swap_dir = (1, 0)  # Right

                if swap_dir:
                    pos1 = self.selected_gem_coord
                    pos2 = (pos1[0] + swap_dir[0], pos1[1] + swap_dir[1])
                    
                    if 0 <= pos2[0] < self.GRID_COLS and 0 <= pos2[1] < self.GRID_ROWS:
                        reward += self._attempt_swap(pos1, pos2)
                    
                    self.selection_state = "NEUTRAL"
                    self.selected_gem_coord = None
        
        return reward

    def _attempt_swap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        # Temporarily swap on the grid to check for matches
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
        
        matches = self._find_all_matches()

        # Add swap animation regardless of match
        self.animations.append({
            "type": "SWAP", "pos1": pos1, "pos2": pos2,
            "progress": 0, "duration": self.SWAP_DURATION,
            "is_match": bool(matches)
        })

        if not matches:
            # No match, swap back logically. The animation will play out the swap and back.
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            # sfx: invalid_swap.wav
            return -0.1  # Penalty for invalid swap
        else:
            # Match found! The animation will handle the visual swap, then trigger the cascade.
            # sfx: valid_swap.wav
            return 0 # Reward is handled by the cascade logic

    def _process_matches(self):
        reward = 0
        total_gems_in_cascade = 0
        
        while True:
            matches = self._find_all_matches()
            if not matches:
                break

            gems_in_this_pass = len(matches)
            total_gems_in_cascade += gems_in_this_pass
            
            # Combo bonus for chains
            if total_gems_in_cascade > 3:
                reward += 5 
                # sfx: combo_bonus.wav
            
            # Standard reward per gem
            reward += gems_in_this_pass
            self.gems_collected += gems_in_this_pass
            self.score += gems_in_this_pass * 10 * (total_gems_in_cascade // 3)

            # Animate destruction and create particles
            for x, y in matches:
                gem_type = self.grid[y, x]
                self.grid[y, x] = -1  # Mark for removal
                self.animations.append({
                    "type": "DESTROY", "pos": (x, y), "gem_type": gem_type,
                    "progress": 0, "duration": self.DESTROY_DURATION
                })
                self._create_particles(x, y, gem_type)
            # sfx: match_destroy.wav
            
            self._apply_gravity()
            self._refill_grid()
        
        return reward

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem = self.grid[r, c]
                if gem == -1: continue
                # Horizontal
                if c < self.GRID_COLS - 2 and self.grid[r, c+1] == gem and self.grid[r, c+2] == gem:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                # Vertical
                if r < self.GRID_ROWS - 2 and self.grid[r+1, c] == gem and self.grid[r+2, c] == gem:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                        self.animations.append({
                            "type": "FALL", "pos_start": (c, r), "pos_end": (c, empty_row),
                            "gem_type": self.grid[empty_row, c],
                            "progress": 0, "duration": self.FALL_DURATION
                        })
                    empty_row -= 1

    def _refill_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.rng.integers(0, self.NUM_GEM_TYPES)
                    self.animations.append({
                        "type": "REFILL", "pos": (c, r), "gem_type": self.grid[r, c],
                        "progress": 0, "duration": self.REFILL_DURATION
                    })

    def _initialize_grid(self):
        self.grid = self.rng.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_matches():
            matches = self._find_all_matches()
            for c, r in matches:
                self.grid[r, c] = self.rng.integers(0, self.NUM_GEM_TYPES)

    def _update_animations(self):
        if not self.animations:
            return

        completed_animations = []
        for anim in self.animations:
            anim["progress"] += 1
            if anim["progress"] >= anim["duration"]:
                completed_animations.append(anim)
        
        for anim in completed_animations:
            self.animations.remove(anim)
            if anim["type"] == "SWAP":
                if anim["is_match"]:
                    # Swap was successful, now process the resulting matches
                    self.score += self._process_matches()
                else:
                    # Swap was invalid, animate it back
                    self.animations.append({
                        "type": "SWAP", "pos1": anim["pos2"], "pos2": anim["pos1"],
                        "progress": 0, "duration": self.SWAP_DURATION,
                        "is_match": False # This is just a visual swap back
                    })

    def _create_particles(self, grid_x, grid_y, gem_type):
        px = self.GRID_X + grid_x * self.GEM_SIZE + self.GEM_SIZE // 2
        py = self.GRID_Y + grid_y * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                "x": px, "y": py, "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "color": color, "life": 20
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1 # gravity
            p["life"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = (self.GRID_X + c * self.GEM_SIZE, self.GRID_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw static gems
        rendered_gems = set()
        for anim in self.animations:
            if anim["type"] == "SWAP":
                rendered_gems.add(anim["pos1"])
                rendered_gems.add(anim["pos2"])
            elif anim["type"] in ["FALL", "DESTROY", "REFILL"]:
                if "pos" in anim: rendered_gems.add(anim["pos"])
                if "pos_start" in anim: rendered_gems.add(anim["pos_start"])
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != -1 and (c, r) not in rendered_gems:
                    self._render_gem(c, r, self.grid[r, c])

        # Draw animated gems
        for anim in self.animations:
            progress_ratio = anim["progress"] / anim["duration"]
            if anim["type"] == "SWAP":
                x1, y1 = anim["pos1"]
                x2, y2 = anim["pos2"]
                g1_type = self.grid[y2, x2] if anim["is_match"] else self.grid[y1, x1]
                g2_type = self.grid[y1, x1] if anim["is_match"] else self.grid[y2, x2]

                ix1 = x1 + (x2 - x1) * progress_ratio
                iy1 = y1 + (y2 - y1) * progress_ratio
                ix2 = x2 + (x1 - x2) * progress_ratio
                iy2 = y2 + (y1 - y2) * progress_ratio
                self._render_gem(ix1, iy1, g1_type)
                self._render_gem(ix2, iy2, g2_type)
            
            elif anim["type"] == "FALL":
                x, y_start = anim["pos_start"]
                _, y_end = anim["pos_end"]
                iy = y_start + (y_end - y_start) * progress_ratio
                self._render_gem(x, iy, anim["gem_type"])

            elif anim["type"] == "DESTROY":
                scale = 1.0 - math.sin(progress_ratio * math.pi)
                self._render_gem(anim["pos"][0], anim["pos"][1], anim["gem_type"], scale)
            
            elif anim["type"] == "REFILL":
                scale = progress_ratio
                self._render_gem(anim["pos"][0], anim["pos"][1], anim["gem_type"], scale)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color = (*p["color"], alpha)
            radius = max(0, p["life"] / 5)
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(radius), color)
        
        # Draw cursor
        if not self.game_over:
            cursor_color = self.COLOR_CURSOR_SELECTED if self.selection_state == "SELECTED" else self.COLOR_CURSOR
            rect = (self.GRID_X + self.cursor_pos[0] * self.GEM_SIZE, 
                    self.GRID_Y + self.cursor_pos[1] * self.GEM_SIZE, 
                    self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, cursor_color, rect, 3)

    def _render_gem(self, c, r, gem_type, scale=1.0):
        if gem_type < 0 or gem_type >= self.NUM_GEM_TYPES: return
        
        center_x = self.GRID_X + c * self.GEM_SIZE + self.GEM_SIZE / 2
        center_y = self.GRID_Y + r * self.GEM_SIZE + self.GEM_SIZE / 2
        radius = int(self.GEM_SIZE * 0.4 * scale)
        if radius <= 0: return

        color = self.GEM_COLORS[gem_type]
        highlight = tuple(min(255, val + 60) for val in color)
        
        pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, color)
        
        # Highlight
        offset = radius * 0.3
        pygame.gfxdraw.filled_circle(self.screen, int(center_x - offset), int(center_y - offset), int(radius * 0.3), highlight)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Gems collected
        gem_text = self.font_main.render(f"Gems: {self.gems_collected} / {self.GEM_GOAL}", True, self.COLOR_TEXT)
        text_rect = gem_text.get_rect(centerx=self.WIDTH / 2)
        text_rect.top = 10
        self.screen.blit(gem_text, text_rect)
        
        # Timer Bar
        timer_ratio = self.time_left / self.MAX_TIME
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        if timer_ratio > 0.5: color = self.COLOR_TIMER_BAR_GOOD
        elif timer_ratio > 0.2: color = self.COLOR_TIMER_BAR_WARN
        else: color = self.COLOR_TIMER_BAR_BAD
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_status = "YOU WIN!" if self.gems_collected >= self.GEM_GOAL else "TIME'S UP!"
            end_text = self.font_main.render(win_status, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            text_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gems_collected": self.gems_collected}

    def _check_termination(self):
        return self.time_left <= 0 or self.gems_collected >= self.GEM_GOAL or self.steps >= self.MAX_STEPS

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame window setup for human play ---
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    # --- Action mapping for keyboard ---
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print(GameEnv.user_guide)
    
    while not done:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Get keyboard state ---
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # No-op
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break # only one movement at a time
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Gems: {info['gems_collected']}")

        # --- Render to screen ---
        # The observation is (H, W, C), but pygame needs (W, H, C)
        # We also need to convert from numpy to a pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control FPS ---
        clock.tick(30)
        
    env.close()
    print("Game Over!")