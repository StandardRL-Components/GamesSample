import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:35:21.736253
# Source Brief: brief_01773.md
# Brief Index: 1773
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Cursed Farm is a puzzle game with stealth elements. The player must match
    harvested crops to cleanse a spreading curse from their farmland. By making
    large combo matches and strategically using a scarecrow for temporary
    stealth, the player can push back the blight and attempt to reach its
    source to win the game. The game ends if the curse overwhelms the player,
    if they successfully cleanse the blight's source, or if they run out of time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Cleanse a spreading curse from your farm by matching groups of three or more crops. "
        "Use a scarecrow for temporary protection and reach the source of the blight to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to match crops and "
        "press shift to activate the scarecrow."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 16
        self.GRID_ROWS = 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 15, 10)
        self.COLOR_GRID = (40, 30, 20)
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_CURSE = (100, 20, 140)
        self.COLOR_SCARECROW = (255, 190, 0)
        self.COLOR_SCARECROW_ACTIVE = (255, 255, 0)
        self.COLOR_BLIGHT = (50, 0, 80)
        self.CROP_COLORS = {
            1: (255, 140, 0),  # Pumpkin Orange
            2: (220, 20, 60),  # Radish Red
            3: (255, 215, 0),  # Corn Yellow
            4: (60, 179, 113)  # Cabbage Green
        }
        self.NUM_CROP_TYPES = len(self.CROP_COLORS)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.combo_font = pygame.font.SysFont("impact", 24)

        # --- Game Parameters ---
        self.initial_curse_rate = 0.05
        self.curse_rate_increase_interval = 200
        self.curse_rate_increase_amount = 0.01
        self.scarecrow_duration = 150  # steps (5 seconds at 30fps)
        self.scarecrow_cooldown = 300  # steps (10 seconds)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.farm_grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0]
        self.curse_level = 0.0
        self.curse_spread_rate = 0.0
        self.scarecrow_active_timer = 0
        self.scarecrow_cooldown_timer = 0
        self.scarecrow_pos = [self.GRID_COLS // 4, self.GRID_ROWS // 2]
        self.blight_source_pos = [self.GRID_COLS - 2, self.GRID_ROWS // 2]
        self.particles = []
        self.combo_texts = []
        self.last_space_state = 0
        self.last_shift_state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.visual_cursor_pos = [float(self.cursor_pos[0] * self.CELL_SIZE), float(self.cursor_pos[1] * self.CELL_SIZE)]

        self.curse_level = 10.0
        self.curse_spread_rate = self.initial_curse_rate

        self.scarecrow_active_timer = 0
        self.scarecrow_cooldown_timer = 0

        self.particles = []
        self.combo_texts = []

        self.last_space_state = 0
        self.last_shift_state = 0

        self._generate_grid()
        self._ensure_matches_exist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        reward += self._handle_input(action)
        self._update_game_state()

        # Continuous penalty for curse presence
        if self.scarecrow_active_timer <= 0:
            reward -= 0.1

        self.score += reward

        terminated = False
        truncated = False
        terminal_reward = 0
        if self.curse_level >= 100.0:
            terminal_reward = -100
            terminated = True
        elif self.cursor_pos[0] == self.blight_source_pos[0] and self.cursor_pos[1] == self.blight_source_pos[1]:
            terminal_reward = 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True
            reward += terminal_reward
            self.score += terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        reward = 0

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        if space_val == 1 and self.last_space_state == 0:
            reward += self._process_match()
        self.last_space_state = space_val

        if shift_val == 1 and self.last_shift_state == 0:
            reward += self._activate_scarecrow()
        self.last_shift_state = shift_val

        return reward

    def _update_game_state(self):
        if self.scarecrow_active_timer <= 0:
            self.curse_level = min(100.0, self.curse_level + self.curse_spread_rate)
        if self.steps > 0 and self.steps % self.curse_rate_increase_interval == 0:
            self.curse_spread_rate += self.curse_rate_increase_amount
        if self.scarecrow_active_timer > 0: self.scarecrow_active_timer -= 1
        if self.scarecrow_cooldown_timer > 0: self.scarecrow_cooldown_timer -= 1

        self._update_particles()
        self._update_combo_texts()
        
        target_x = self.cursor_pos[0] * self.CELL_SIZE
        target_y = self.cursor_pos[1] * self.CELL_SIZE
        self.visual_cursor_pos[0] += (target_x - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (target_y - self.visual_cursor_pos[1]) * 0.4

    def _generate_grid(self):
        self.farm_grid = self.np_random.integers(1, self.NUM_CROP_TYPES + 1, size=(self.GRID_COLS, self.GRID_ROWS))

    def _find_potential_matches(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if len(self._find_connected_crops(x, y)) >= 3:
                    return True
        return False

    def _ensure_matches_exist(self):
        while not self._find_potential_matches():
            self._generate_grid()

    def _find_connected_crops(self, start_x, start_y):
        crop_type = self.farm_grid[start_x, start_y]
        if crop_type == 0: return []
        q, visited, match_group = [(start_x, start_y)], set([(start_x, start_y)]), []
        while q:
            x, y = q.pop(0)
            match_group.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                    if self.farm_grid[nx, ny] == crop_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return match_group

    def _process_match(self):
        match_cells = self._find_connected_crops(self.cursor_pos[0], self.cursor_pos[1])
        if len(match_cells) < 3: return 0

        # Sound: "match_success.wav"
        reward = 1.0
        crop_type = self.farm_grid[self.cursor_pos[0], self.cursor_pos[1]]
        if len(match_cells) > 2:
            reward += 10.0 # Combo reward
            avg_x = sum(c[0] for c in match_cells) / len(match_cells) * self.CELL_SIZE
            avg_y = sum(c[1] for c in match_cells) / len(match_cells) * self.CELL_SIZE
            self.combo_texts.append({"text": f"COMBO x{len(match_cells)}!", "pos": [avg_x, avg_y], "timer": 60, "color": self.CROP_COLORS[crop_type]})

        self.curse_level = max(0, self.curse_level - len(match_cells) * 2.0)
        for x, y in match_cells:
            self.farm_grid[x, y] = 0
            self._spawn_particles(x, y, self.CROP_COLORS[crop_type])
        self._respawn_crops()
        self._ensure_matches_exist()
        return reward

    def _respawn_crops(self):
        for x in range(self.GRID_COLS):
            empty_count = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.farm_grid[x, y] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.farm_grid[x, y + empty_count] = self.farm_grid[x, y]
                    self.farm_grid[x, y] = 0
            for y in range(empty_count):
                self.farm_grid[x, y] = self.np_random.integers(1, self.NUM_CROP_TYPES + 1)

    def _activate_scarecrow(self):
        if self.scarecrow_cooldown_timer <= 0:
            # Sound: "scarecrow_activate.wav"
            self.scarecrow_active_timer = self.scarecrow_duration
            self.scarecrow_cooldown_timer = self.scarecrow_cooldown + self.scarecrow_duration
            return 5.0 if self.curse_level > 50 else 2.0
        return 0

    def _spawn_particles(self, grid_x, grid_y, color):
        px, py = (grid_x + 0.5) * self.CELL_SIZE, (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({"pos": [px, py], "vel": [math.cos(angle) * speed, math.sin(angle) * speed], "timer": self.np_random.integers(20, 40), "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95; p["vel"][1] *= 0.95
            p["timer"] -= 1
        self.particles = [p for p in self.particles if p["timer"] > 0]

    def _update_combo_texts(self):
        for ct in self.combo_texts:
            ct['timer'] -= 1; ct['pos'][1] -= 0.5
        self.combo_texts = [ct for ct in self.combo_texts if ct['timer'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_curse_overlay()
        self._render_blight_source()
        self._render_scarecrow()
        self._render_crops()
        self._render_cursor()
        self._render_particles()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_curse_overlay(self):
        if self.curse_level > 0:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(min(200, self.curse_level * 1.8))
            overlay.fill((*self.COLOR_CURSE, alpha))
            self.screen.blit(overlay, (0, 0))

    def _render_blight_source(self):
        x, y = (self.blight_source_pos[0] + 0.5) * self.CELL_SIZE, (self.blight_source_pos[1] + 0.5) * self.CELL_SIZE
        radius, pulse = self.CELL_SIZE * 0.4, math.sin(self.steps * 0.1) * 3
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(radius + pulse), self.COLOR_BLIGHT)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(radius + pulse), (255, 255, 255, 100))

    def _render_scarecrow(self):
        x, y = (self.scarecrow_pos[0] + 0.5) * self.CELL_SIZE, (self.scarecrow_pos[1] + 0.5) * self.CELL_SIZE
        w, h = self.CELL_SIZE * 0.8, self.CELL_SIZE * 0.8
        rect = pygame.Rect(x - w / 2, y - h / 2, w, h)
        pygame.draw.rect(self.screen, self.COLOR_SCARECROW, rect, border_radius=4)
        if self.scarecrow_active_timer > 0:
            glow_surface = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
            alpha = 150 * (self.scarecrow_active_timer / self.scarecrow_duration)
            pygame.draw.circle(glow_surface, (*self.COLOR_SCARECROW_ACTIVE, alpha), (w, h), w)
            self.screen.blit(glow_surface, (rect.centerx - w, rect.centery - h), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_crops(self):
        size = self.CELL_SIZE * 0.75
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                crop_type = self.farm_grid[x, y]
                if crop_type > 0:
                    px, py, color = x * self.CELL_SIZE + self.CELL_SIZE / 2, y * self.CELL_SIZE + self.CELL_SIZE / 2, self.CROP_COLORS[crop_type]
                    if crop_type == 1: pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(size/2), color)
                    elif crop_type == 2: pygame.gfxdraw.filled_polygon(self.screen, [(px, py - size/2), (px - size/2, py + size/2), (px + size/2, py + size/2)], color)
                    elif crop_type == 3: pygame.draw.rect(self.screen, color, pygame.Rect(px - size/2, py - size/2, size, size), border_radius=4)
                    elif crop_type == 4: pygame.gfxdraw.filled_polygon(self.screen, [(px, py - size/2), (px + size/2, py), (px, py + size/2), (px - size/2, py)], color)

    def _render_cursor(self):
        size = self.CELL_SIZE
        rect = pygame.Rect(self.visual_cursor_pos[0], self.visual_cursor_pos[1], size, size)
        alpha = 128 + math.sin(self.steps * 0.2) * 127
        cursor_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), width=3, border_radius=5)
        self.screen.blit(cursor_surface, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], max(0, int(p["timer"] / 8)))

    def _render_ui(self):
        self.screen.blit(self.ui_font.render(f"SCORE: {int(self.score)}", True, (255, 255, 255)), (10, 10))
        bar_w, bar_h, bar_x = 200, 20, self.SCREEN_WIDTH / 2 - 100
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_CURSE, (bar_x, 10, (self.curse_level / 100.0) * bar_w, bar_h))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, 10, bar_w, bar_h), 2)
        self.screen.blit(self.ui_font.render("CURSE", True, (255, 255, 255)), (bar_x - 70, 12))
        
        cd_text, color = "READY", (0, 255, 0)
        if self.scarecrow_active_timer > 0: cd_text, color = f"ACTIVE: {self.scarecrow_active_timer // 30 + 1}s", self.COLOR_SCARECROW_ACTIVE
        elif self.scarecrow_cooldown_timer > self.scarecrow_duration: cd_text, color = f"COOLDOWN: {(self.scarecrow_cooldown_timer - self.scarecrow_duration) // 30 + 1}s", (255, 100, 100)
        self.screen.blit(self.ui_font.render(f"SCARECROW: {cd_text}", True, color), (self.SCREEN_WIDTH - 250, 12))

        for ct in self.combo_texts:
            alpha = max(0, min(255, ct['timer'] * 5))
            text_surf = self.combo_font.render(ct['text'], True, ct['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, text_surf.get_rect(center=ct['pos']))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "curse_level": self.curse_level}

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    # Test implementation (optional)
    try:
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        print("✓ Implementation validated successfully")
    except Exception as e:
        print(f"✗ Implementation validation failed: {e}")


    # Manual play (requires 'keyboard' library)
    try:
        import keyboard
        print("\n--- Manual Control ---")
        print("Arrows: Move cursor | Space: Match crops | Shift: Activate scarecrow | Q: Quit")
        
        # Un-dummy the video driver for manual play
        os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
        pygame.display.init()
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Cursed Farm")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            movement, space, shift = 0, 0, 0
            if keyboard.is_pressed('up arrow'): movement = 1
            elif keyboard.is_pressed('down arrow'): movement = 2
            elif keyboard.is_pressed('left arrow'): movement = 3
            elif keyboard.is_pressed('right arrow'): movement = 4
            if keyboard.is_pressed('space'): space = 1
            if keyboard.is_pressed('shift'): shift = 1
            if keyboard.is_pressed('q'): break

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            env.clock.tick(30)
            
        print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")

    except ImportError:
        print("\n'keyboard' library not found. Running random agent test for 1000 steps.")
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()

    env.close()