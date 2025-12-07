
# Generated: 2025-08-27T18:08:58.305762
# Source Brief: brief_01744.md
# Brief Index: 1744

        
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
        "Controls: Arrows to move cursor. Space to place a cyan crystal (reflects like \\). "
        "Shift to place a magenta crystal (reflects like /)."
    )

    game_description = (
        "Navigate a crystal cavern and place reflective crystals to guide a light beam. "
        "Illuminate all targets before time runs out or you run out of crystals."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.MAX_STEPS = 1800  # 60 seconds * 30 FPS
        self.INITIAL_CRYSTALS = 15
        self.NUM_TARGETS = 4
        self.NUM_WALL_CLUSTERS = 8
        self.WALL_CLUSTER_SIZE = (4, 4)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_WALL = (40, 48, 70)
        self.COLOR_WALL_TOP = (60, 70, 95)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_TARGET_OFF = (80, 80, 90)
        self.COLOR_TARGET_ON = (255, 255, 255)
        self.COLOR_TARGET_GLOW = (200, 200, 255)
        self.COLOR_BEAM_OUTER = (255, 230, 150)
        self.COLOR_BEAM_INNER = (255, 255, 220)
        self.COLOR_CRYSTAL_1 = (0, 255, 255)
        self.COLOR_CRYSTAL_2 = (255, 0, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (40, 48, 70, 180)
        self.COLOR_TIMER_BAR = (0, 255, 255)
        self.COLOR_TIMER_BAR_BG = (60, 70, 95)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        self.iso_tile_width = self.SCREEN_WIDTH / (self.GRID_WIDTH + 2)
        self.iso_tile_height = self.iso_tile_width * 0.5
        self.origin_x = self.SCREEN_WIDTH / 2
        self.origin_y = self.SCREEN_HEIGHT * 0.2

        # --- State Variables ---
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.crystals_remaining = 0
        self.cursor_pos = None
        self.crystals = []
        self.light_source_pos = None
        self.light_source_dir = None
        self.light_path = []
        self.targets = []
        self.particles = []
        self.previous_action = np.array([0, 0, 0])
        self.illuminated_targets_last_step = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.crystals_remaining = self.INITIAL_CRYSTALS
        self.cursor_pos = np.array([self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2])
        self.crystals = []
        self.light_path = []
        self.targets = []
        self.particles = []
        self.previous_action = np.array([0, 0, 0])
        self.illuminated_targets_last_step = 0

        self._generate_level()
        self._update_light_path_and_targets()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.time_remaining -= 1
        self.steps += 1
        reward = 0

        crystal_placed = self._handle_input(action)
        if crystal_placed:
            reward -= 0.01 # Cost for placing a crystal
            self._update_light_path_and_targets()
            # SFX: Crystal placement sound

        self._update_particles()
        
        num_lit_now = sum(1 for t in self.targets if t['lit'])
        newly_lit = num_lit_now - self.illuminated_targets_last_step
        if newly_lit > 0:
            reward += newly_lit * 5.0
            # SFX: Target illuminated sound

        reward += num_lit_now * 0.01
        self.score += reward
        self.illuminated_targets_last_step = num_lit_now

        terminated = False
        win_condition = num_lit_now == len(self.targets) and len(self.targets) > 0
        
        if win_condition:
            terminated = True
            reward += 50
            # SFX: Level complete fanfare
        else:
            time_out = self.time_remaining <= 0
            # Lose if out of crystals and no more can be placed with the current action
            crystals_out = self.crystals_remaining <= 0 and not (action[1] == 1 or action[2] == 1)
            
            if time_out or crystals_out:
                terminated = True
                reward -= 50
                # SFX: Failure buzzer
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "crystals": self.crystals_remaining}

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.iso_tile_width / 2
        screen_y = self.origin_y + (x + y) * self.iso_tile_height / 2
        return int(screen_x), int(screen_y)

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Create border walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Place wall clusters
        for _ in range(self.NUM_WALL_CLUSTERS):
            cx = self.np_random.integers(2, self.GRID_WIDTH - self.WALL_CLUSTER_SIZE[0] - 2)
            cy = self.np_random.integers(2, self.GRID_HEIGHT - self.WALL_CLUSTER_SIZE[1] - 2)
            for _ in range(self.np_random.integers(3, 8)):
                rx, ry = self.np_random.integers(0, self.WALL_CLUSTER_SIZE[0]), self.np_random.integers(0, self.WALL_CLUSTER_SIZE[1])
                if 1 < cx + rx < self.GRID_WIDTH - 2 and 1 < cy + ry < self.GRID_HEIGHT - 2:
                    self.grid[cx + rx, cy + ry] = 1
        
        # Light source
        self.light_source_pos = np.array([1, self.np_random.integers(3, self.GRID_HEIGHT - 3)])
        self.light_source_dir = np.array([1, 0])
        self.grid[self.light_source_pos[0], self.light_source_pos[1]] = 0 # Ensure source is not blocked

        # Place targets
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            while True:
                pos = np.array([self.np_random.integers(2, self.GRID_WIDTH - 2), self.np_random.integers(2, self.GRID_HEIGHT - 2)])
                if self.grid[pos[0], pos[1]] == 0 and not any(np.array_equal(pos, t['pos']) for t in self.targets):
                    self.targets.append({'pos': pos, 'lit': False})
                    break

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Crystal Placement (on press, not hold) ---
        crystal_placed = False
        is_space_press = space_held and not self.previous_action[1]
        is_shift_press = shift_held and not self.previous_action[2]
        
        if (is_space_press or is_shift_press) and self.crystals_remaining > 0:
            pos = self.cursor_pos
            is_valid_pos = self.grid[pos[0], pos[1]] == 0
            is_occupied = any(np.array_equal(pos, c['pos']) for c in self.crystals)
            
            if is_valid_pos and not is_occupied:
                crystal_type = 1 if is_space_press else 2
                self.crystals.append({'pos': self.cursor_pos.copy(), 'type': crystal_type})
                self.crystals_remaining -= 1
                crystal_placed = True
                
                # Add placement particles
                px, py = self._iso_to_screen(pos[0], pos[1])
                color = self.COLOR_CRYSTAL_1 if crystal_type == 1 else self.COLOR_CRYSTAL_2
                for _ in range(20):
                    self.particles.append(self._create_particle(px, py, color))

        self.previous_action = np.array([movement, space_held, shift_held])
        return crystal_placed

    def _update_light_path_and_targets(self):
        self.light_path = []
        for t in self.targets:
            t['lit'] = False

        pos = self.light_source_pos.copy().astype(float)
        direction = self.light_source_dir.copy().astype(float)
        
        self.light_path.append(self._iso_to_screen(pos[0], pos[1]))

        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT):
            next_pos = pos + direction
            
            # Check boundaries
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                self.light_path.append(self._iso_to_screen(next_pos[0], next_pos[1]))
                break
            
            pos = next_pos
            ipos = pos.astype(int)
            self.light_path.append(self._iso_to_screen(pos[0], pos[1]))

            # Check for walls
            if self.grid[ipos[0], ipos[1]] == 1:
                break

            # Check for targets
            for t in self.targets:
                if np.array_equal(ipos, t['pos']):
                    t['lit'] = True
            
            # Check for crystals
            reflected = False
            for c in self.crystals:
                if np.array_equal(ipos, c['pos']):
                    # Type 1: mirror '\', (dx, dy) -> (dy, dx)
                    if c['type'] == 1:
                        direction = np.array([direction[1], direction[0]])
                    # Type 2: mirror '/', (dx, dy) -> (-dy, -dx)
                    else:
                        direction = np.array([-direction[1], -direction[0]])
                    reflected = True
                    break
            if reflected:
                continue

    def _create_particle(self, x, y, color):
        return {
            'pos': [x, y],
            'vel': [(self.np_random.random() - 0.5) * 3, (self.np_random.random() - 0.5) * 3],
            'life': self.np_random.integers(10, 20),
            'color': color,
            'size': self.np_random.integers(3, 6)
        }

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_game(self):
        # Draw walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 1:
                    sx, sy = self._iso_to_screen(x, y)
                    points = [
                        (sx, sy),
                        (sx + self.iso_tile_width / 2, sy + self.iso_tile_height / 2),
                        (sx, sy + self.iso_tile_height),
                        (sx - self.iso_tile_width / 2, sy + self.iso_tile_height / 2)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_WALL, points)
                    pygame.draw.polygon(self.screen, self.COLOR_WALL_TOP, [points[0], points[1], (points[1][0], points[1][1] - self.iso_tile_height), (points[0][0], points[0][1] - self.iso_tile_height)])
                    pygame.draw.polygon(self.screen, self.COLOR_WALL, [(points[1][0], points[1][1] - self.iso_tile_height), points[1], points[2], (points[2][0], points[2][1] - self.iso_tile_height)])

        # Draw targets
        for t in self.targets:
            sx, sy = self._iso_to_screen(t['pos'][0], t['pos'][1])
            sy += int(self.iso_tile_height/2)
            radius = int(self.iso_tile_width / 4)
            if t['lit']:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_TARGET_ON)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_TARGET_ON)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, radius + 2, self.COLOR_TARGET_GLOW)
            else:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_TARGET_OFF)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_TARGET_OFF)

        # Draw light beam
        if len(self.light_path) > 1:
            # Glow effect
            pygame.draw.lines(self.screen, self.COLOR_BEAM_OUTER, False, self.light_path, 5)
            # Core beam
            pygame.draw.lines(self.screen, self.COLOR_BEAM_INNER, False, self.light_path, 2)
        
        # Draw crystals
        for c in self.crystals:
            sx, sy = self._iso_to_screen(c['pos'][0], c['pos'][1])
            color = self.COLOR_CRYSTAL_1 if c['type'] == 1 else self.COLOR_CRYSTAL_2
            h_w, h_h = self.iso_tile_width / 2, self.iso_tile_height / 2
            points = [(sx, sy + h_h * 0.5), (sx + h_w*0.8, sy + h_h*1.3), (sx, sy + h_h*2.1), (sx - h_w*0.8, sy + h_h*1.3)]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.aalines(self.screen, (255,255,255), True, points)

        # Draw cursor
        if not self.game_over:
            sx, sy = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
            points = [
                (sx, sy),
                (sx + self.iso_tile_width / 2, sy + self.iso_tile_height / 2),
                (sx, sy + self.iso_tile_height),
                (sx - self.iso_tile_width / 2, sy + self.iso_tile_height / 2)
            ]
            pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], max(1, int(p['size'])), max(1, int(p['size']))))
    
    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(5, 5, 200, 70)
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (panel_rect.x, panel_rect.y))

        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Crystals
        crystal_text = self.font_small.render(f"Crystals: {self.crystals_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (15, 45))
        
        # Timer Bar
        timer_width = self.SCREEN_WIDTH - 10
        timer_rect_bg = pygame.Rect(5, self.SCREEN_HEIGHT - 20, timer_width, 15)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, timer_rect_bg, border_radius=4)
        
        time_ratio = self.time_remaining / self.MAX_STEPS
        timer_rect_fg = pygame.Rect(5, self.SCREEN_HEIGHT - 20, timer_width * time_ratio, 15)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, timer_rect_fg, border_radius=4)

        if self.game_over:
            num_lit = sum(1 for t in self.targets if t['lit'])
            win = num_lit == len(self.targets) and len(self.targets) > 0
            message = "LEVEL COMPLETE" if win else "GAME OVER"
            color = self.COLOR_TARGET_ON if win else (200, 50, 50)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill(self.COLOR_UI_BG)
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(30) # Control the frame rate

    env.close()