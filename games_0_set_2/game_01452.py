
# Generated: 2025-08-27T17:11:02.198843
# Source Brief: brief_01452.md
# Brief Index: 1452

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to select column. Press space to drop a crystal."
    )

    game_description = (
        "Navigate a crystal cavern, strategically dropping crystals to align five in a row before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 7
    GRID_HEIGHT = 8
    WIN_LENGTH = 5
    MAX_TIME = 60  # seconds
    FPS = 30

    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_GRID = (40, 30, 70)
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]
    COLOR_WHITE = (240, 240, 240)
    COLOR_GRAY = (150, 150, 150)
    
    # Rendering parameters
    ISO_TILE_WIDTH = 48
    ISO_TILE_HEIGHT = 24
    CRYSTAL_HEIGHT = 36

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Calculate grid rendering offsets
        self.iso_offset_x = self.screen.get_width() // 2
        self.iso_offset_y = self.screen.get_height() - 10

        # Initialize state variables
        self.grid = None
        self.falling_crystals = None
        self.particles = None
        self.selector_pos = None
        self.next_crystal_color_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.input_cooldown = None
        self.drop_cooldown = None
        self.last_space_held = None
        self.alignment_highlights = None
        self.rng = None
        
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.falling_crystals = []
        self.particles = []
        self.alignment_highlights = collections.deque(maxlen=50)

        self.selector_pos = self.GRID_WIDTH // 2
        self.next_crystal_color_idx = self.rng.integers(0, len(self.CRYSTAL_COLORS))
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME * self.FPS
        
        self.input_cooldown = 0
        self.drop_cooldown = 0
        self.last_space_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Timers & Cooldowns ---
        self.steps += 1
        self.timer -= 1
        self.input_cooldown = max(0, self.input_cooldown - 1)
        self.drop_cooldown = max(0, self.drop_cooldown - 1)

        # --- Handle Actions ---
        movement, space_action, _ = action
        space_pressed = space_action == 1 and not self.last_space_held
        self.last_space_held = space_action == 1

        if self.input_cooldown == 0:
            if movement == 3:  # Left
                self.selector_pos = max(0, self.selector_pos - 1)
                self.input_cooldown = 5
            elif movement == 4:  # Right
                self.selector_pos = min(self.GRID_WIDTH - 1, self.selector_pos + 1)
                self.input_cooldown = 5

        if space_pressed and self.drop_cooldown == 0 and self.grid[0, self.selector_pos] == 0:
            # Sound: Drop crystal
            self._drop_crystal()
            self.drop_cooldown = 15 # Prevent spamming

        # --- Update Game Logic ---
        reward = 0
        
        # Update falling crystals
        newly_landed = self._update_falling_crystals()

        # Check for alignments from newly landed crystals
        if newly_landed:
            landed_x, landed_y, landed_color_idx = newly_landed
            alignment_reward, is_win = self._check_and_score_alignments(landed_x, landed_y, landed_color_idx)
            reward += alignment_reward
            if is_win:
                self.game_over = True
                reward = 100 # Terminal win reward
        
        # Update particles
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self.game_over or self.timer <= 0
        if terminated and not self.game_over: # Timeout
            self.game_over = True
            reward = -10 # Terminal loss reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _drop_crystal(self):
        start_x, start_y = self._grid_to_iso(self.selector_pos, -2)
        self.falling_crystals.append({
            "pos": pygame.Vector2(start_x, start_y),
            "col": self.selector_pos,
            "color_idx": self.next_crystal_color_idx,
            "velocity": 2.0
        })
        self.next_crystal_color_idx = self.rng.integers(0, len(self.CRYSTAL_COLORS))

    def _update_falling_crystals(self):
        for crystal in self.falling_crystals[:]:
            # Find target Y
            target_y = self.GRID_HEIGHT - 1
            while target_y >= 0 and self.grid[target_y, crystal["col"]] != 0:
                target_y -= 1
            
            _, iso_target_y = self._grid_to_iso(crystal["col"], target_y)

            # Gravity
            crystal["velocity"] += 0.2
            crystal["pos"].y += crystal["velocity"]

            if crystal["pos"].y >= iso_target_y:
                # Land the crystal
                # Sound: Crystal land
                self.grid[target_y, crystal["col"]] = crystal["color_idx"] + 1
                self.falling_crystals.remove(crystal)
                self._create_particles(crystal["pos"].x, crystal["pos"].y, self.CRYSTAL_COLORS[crystal["color_idx"]])
                return crystal["col"], target_y, crystal["color_idx"] + 1
        return None
    
    def _check_and_score_alignments(self, x, y, color_idx):
        if color_idx == 0: return 0, False

        is_win = False
        total_reward = 0
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # H, V, Diag, Anti-Diag
        
        for dx, dy in directions:
            line = [(x, y)]
            # Check positive direction
            for i in range(1, self.WIN_LENGTH):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == color_idx:
                    line.append((nx, ny))
                else:
                    break
            # Check negative direction
            for i in range(1, self.WIN_LENGTH):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == color_idx:
                    line.append((nx, ny))
                else:
                    break
            
            if len(line) >= 3:
                for pos in line:
                    self.alignment_highlights.append((*pos, self.steps))
                if len(line) == 3:
                    self.score += 10
                    total_reward += 0.1
                elif len(line) == 4:
                    self.score += 50
                    total_reward += 0.5
                elif len(line) >= self.WIN_LENGTH:
                    self.score += 1000
                    total_reward += 10
                    is_win = True
        
        return total_reward, is_win

    def _grid_to_iso(self, x, y):
        iso_x = (x - y) * self.ISO_TILE_WIDTH / 2 + self.iso_offset_x
        iso_y = (x + y) * self.ISO_TILE_HEIGHT / 2 + self.iso_offset_y - (self.GRID_HEIGHT * self.ISO_TILE_HEIGHT)
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = self._grid_to_iso(0, y)
            end_pos = self._grid_to_iso(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for x in range(self.GRID_WIDTH + 1):
            start_pos = self._grid_to_iso(x, 0)
            end_pos = self._grid_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Render selector
        if not self.game_over:
            sx, sy = self._grid_to_iso(self.selector_pos, -0.5)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            color = self.CRYSTAL_COLORS[self.next_crystal_color_idx]
            glow_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.circle(self.screen, glow_color, (sx, sy), 8 + pulse * 4, 2)

        # Render placed crystals
        highlights = {pos for pos in self.alignment_highlights}
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx > 0:
                    is_highlighted = any(hx==x and hy==y for hx, hy, ht in highlights)
                    self._draw_crystal(x, y, color_idx - 1, is_highlighted)

        # Render falling crystals
        for crystal in self.falling_crystals:
            self._draw_crystal_at_pos(crystal["pos"].x, crystal["pos"].y, crystal["color_idx"])
        
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['radius']))

    def _draw_crystal(self, grid_x, grid_y, color_idx, is_highlighted=False):
        iso_x, iso_y = self._grid_to_iso(grid_x, grid_y)
        self._draw_crystal_at_pos(iso_x, iso_y, color_idx, is_highlighted)

    def _draw_crystal_at_pos(self, x, y, color_idx, is_highlighted=False):
        base_color = self.CRYSTAL_COLORS[color_idx]
        
        if is_highlighted:
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 100
            base_color = tuple(min(255, c + pulse) for c in base_color)
        
        top_color = tuple(min(255, c + 60) for c in base_color)
        side_color_l = tuple(max(0, c - 30) for c in base_color)
        side_color_r = tuple(max(0, c - 60) for c in base_color)
        
        w = self.ISO_TILE_WIDTH
        h = self.ISO_TILE_HEIGHT
        ch = self.CRYSTAL_HEIGHT

        # Points for the isometric cube
        p_top = (x, y - ch)
        p_mid = (x, y)
        p_left = (x - w/2, y - h/2)
        p_right = (x + w/2, y - h/2)
        p_top_left = (x - w/2, y - h/2 - ch)
        p_top_right = (x + w/2, y - h/2 - ch)
        
        # Draw faces with antialiasing
        pygame.gfxdraw.filled_polygon(self.screen, [p_top, p_top_right, p_right, p_mid], side_color_r)
        pygame.gfxdraw.aapolygon(self.screen, [p_top, p_top_right, p_right, p_mid], side_color_r)
        
        pygame.gfxdraw.filled_polygon(self.screen, [p_top, p_top_left, p_left, p_mid], side_color_l)
        pygame.gfxdraw.aapolygon(self.screen, [p_top, p_top_left, p_left, p_mid], side_color_l)

        pygame.gfxdraw.filled_polygon(self.screen, [p_top_left, p_top, p_top_right, (x, y - h - ch)], top_color)
        pygame.gfxdraw.aapolygon(self.screen, [p_top_left, p_top, p_top_right, (x, y - h - ch)], top_color)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Timer
        secs = int(self.timer / self.FPS)
        mins = secs // 60
        secs %= 60
        timer_color = self.COLOR_WHITE if secs > 10 or (secs % 2 == 0) else self.CRYSTAL_COLORS[0]
        timer_text = self.font_medium.render(f"{mins:02}:{secs:02}", True, timer_color)
        self.screen.blit(timer_text, (self.screen.get_width() - timer_text.get_width() - 10, 10))

        # Next crystal preview
        preview_text = self.font_medium.render("Next:", True, self.COLOR_GRAY)
        self.screen.blit(preview_text, (self.screen.get_width() - 100, 50))
        self._draw_crystal_at_pos(self.screen.get_width() - 50, 80, self.next_crystal_color_idx)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.timer > 0 else "TIME'S UP"
            color = self.CRYSTAL_COLORS[1] if self.timer > 0 else self.CRYSTAL_COLORS[0]
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, x, y, color):
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed - 2),
                'radius': self.rng.random() * 2 + 2,
                'color': color,
                'life': self.rng.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # gravity
            p['life'] -= 1
            p['radius'] -= 0.05
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": max(0, self.timer / self.FPS)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Crystal Caverns")
    
    running = True
    total_reward = 0
    
    # Map pygame keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }

    while running:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move in key_map.items():
            if keys[key]:
                movement_action = move
                break # Prioritize first key found
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}, Score: {info['score']}")
            # Wait for 'R' to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        wait_for_reset = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("--- RESET ---")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
        
        env.clock.tick(env.FPS)
        
    env.close()