
# Generated: 2025-08-28T05:59:43.005001
# Source Brief: brief_02783.md
# Brief Index: 2783

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Hold space and press an arrow key to swap the gem "
        "under the cursor with its neighbor in that direction."
    )

    game_description = (
        "Match gems to reach a target score in this fast-paced, grid-based puzzle game. Create combos "
        "by causing chain reactions of matches to multiply your score, but watch the timer!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.FPS = 30
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.screen_width - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.screen_height - self.GRID_HEIGHT) // 2 + 20
        self.WIN_SCORE = 1000
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        # --- Animation Timings (in frames) ---
        self.SWAP_DURATION = 8
        self.MATCH_DURATION = 12
        self.FALL_DURATION = 10

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 100, 150)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
        ]

        # --- Fonts ---
        try:
            self.font_main = pygame.font.Font(None, 32)
            self.font_combo = pygame.font.Font(None, 48)
            self.font_game_over = pygame.font.Font(None, 72)
        except IOError:
            self.font_main = pygame.font.SysFont("sans-serif", 32)
            self.font_combo = pygame.font.SysFont("sans-serif", 48)
            self.font_game_over = pygame.font.SysFont("sans-serif", 72)

        # --- State Variables ---
        self.grid = None
        self.visual_gems = None
        self.cursor_pos = None
        self.visual_cursor_pos = None
        self.time_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_phase = None
        self.animation_timer = 0
        self.swap_info = None
        self.matched_gems = None
        self.combo_multiplier = 1
        self.particles = []
        self.pending_reward = 0
        self.is_player_swap = False
        self.last_combo_pos = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.time_left = float(self.GAME_DURATION_SECONDS)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        cx, cy = self._get_screen_pos(self.cursor_pos[0], self.cursor_pos[1])
        self.visual_cursor_pos = [float(cx), float(cy)]

        self.game_phase = 'IDLE'
        self.animation_timer = 0
        self.swap_info = None
        self.matched_gems = set()
        self.combo_multiplier = 1
        self.particles = []
        self.pending_reward = 0
        self.is_player_swap = False
        self.last_combo_pos = None

        self._initialize_grid()

        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.visual_gems = [[{} for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        
        while self._find_all_matches():
            matches = self._find_all_matches()
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self._init_visual_gem(r, c)

    def _init_visual_gem(self, r, c, y_offset=0):
        x, y = self._get_screen_pos(c, r)
        self.visual_gems[r][c] = {
            'type': self.grid[r, c],
            'pos': [float(x), float(y)],
            'target_pos': [float(x), float(y)],
            'scale': 1.0,
            'alpha': 255,
            'y_offset': y_offset,
        }

    def step(self, action):
        reward = self.pending_reward
        self.pending_reward = 0

        self.steps += 1
        if not self.game_over:
            self.time_left = max(0, self.time_left - 1.0 / self.FPS)

        self._update_animations_and_phases()
        
        if self.game_phase == 'IDLE' and not self.game_over:
            action_reward = self._handle_action(action)
            reward += action_reward

        terminated = (self.score >= self.WIN_SCORE or self.time_left <= 0) and not self.game_over
        if terminated:
            self.game_over = True
            reward += 100 if self.score >= self.WIN_SCORE else -10

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        target_cursor_x, target_cursor_y = self.cursor_pos
        if movement == 1: target_cursor_y = max(0, self.cursor_pos[1] - 1)  # Up
        elif movement == 2: target_cursor_y = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)  # Down
        elif movement == 3: target_cursor_x = max(0, self.cursor_pos[0] - 1)  # Left
        elif movement == 4: target_cursor_x = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)  # Right
        
        if space_held and movement != 0:
            c1_x, c1_y = self.cursor_pos
            c2_x, c2_y = target_cursor_x, target_cursor_y
            
            # Ensure it's an adjacent swap
            if abs(c1_x - c2_x) + abs(c1_y - c2_y) == 1:
                self._start_swap(c1_y, c1_x, c2_y, c2_x)
                self.is_player_swap = True
                self.cursor_pos = [c2_x, c2_y]
        else:
            self.cursor_pos = [target_cursor_x, target_cursor_y]
        
        return 0

    def _update_animations_and_phases(self):
        # Update visual cursor position
        target_x, target_y = self._get_screen_pos(self.cursor_pos[0], self.cursor_pos[1])
        self.visual_cursor_pos[0] += (target_x - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (target_y - self.visual_cursor_pos[1]) * 0.4

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
        
        # Update gem visual positions
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = self.visual_gems[r][c]
                gem['pos'][0] += (gem['target_pos'][0] - gem['pos'][0]) * 0.4
                gem['pos'][1] += (gem['target_pos'][1] - gem['pos'][1]) * 0.4

        if self.animation_timer > 0:
            self.animation_timer -= 1
            
            if self.game_phase == 'MATCHING':
                progress = 1.0 - (self.animation_timer / self.MATCH_DURATION)
                for r, c in self.matched_gems:
                    self.visual_gems[r][c]['scale'] = 1.0 + math.sin(progress * math.pi) * 0.7
                    self.visual_gems[r][c]['alpha'] = 255 * (1.0 - progress)
            
            if self.animation_timer == 0:
                self._on_animation_finish()
        
    def _on_animation_finish(self):
        if self.game_phase == 'SWAPPING':
            # Perform logical swap
            r1, c1, r2, c2 = self.swap_info
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            
            matches = self._find_all_matches()
            if matches:
                self._start_match_phase(matches)
            else: # Invalid swap
                if self.is_player_swap:
                    self.pending_reward += -0.1
                # Swap back
                self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                self._start_swap(r1, c1, r2, c2, is_player_swap=False)
        
        elif self.game_phase == 'MATCHING':
            self._process_matches()
            self._start_fall_phase()

        elif self.game_phase == 'FALLING':
            self._refill_grid()
            matches = self._find_all_matches()
            if matches:
                self.combo_multiplier += 1
                self.pending_reward += 5 # Combo bonus
                self._start_match_phase(matches)
            else:
                self.combo_multiplier = 1
                self.last_combo_pos = None
                self.game_phase = 'IDLE'

    def _start_swap(self, r1, c1, r2, c2, is_player_swap=True):
        self.game_phase = 'SWAPPING'
        self.animation_timer = self.SWAP_DURATION
        self.swap_info = (r1, c1, r2, c2)
        self.is_player_swap = is_player_swap

        self.visual_gems[r1][c1]['target_pos'] = self._get_screen_pos(c2, r2)
        self.visual_gems[r2][c2]['target_pos'] = self._get_screen_pos(c1, r1)
        self.visual_gems[r1][c1], self.visual_gems[r2][c2] = self.visual_gems[r2][c2], self.visual_gems[r1][c1]

    def _start_match_phase(self, matches):
        self.game_phase = 'MATCHING'
        self.animation_timer = self.MATCH_DURATION
        self.matched_gems = matches
        self.last_combo_pos = [self._get_screen_pos(c, r) for r, c in matches][0]
        # sfx: match_sound

    def _start_fall_phase(self):
        self.game_phase = 'FALLING'
        self.animation_timer = self.FALL_DURATION

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != -1:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != -1:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _process_matches(self):
        num_matched = len(self.matched_gems)
        self.score += num_matched * self.combo_multiplier
        self.pending_reward += num_matched * 1 # Base reward
        
        for r, c in self.matched_gems:
            self._create_particles(r, c)
            self.grid[r, c] = -1 # Mark for removal
        # sfx: score_point_sound

    def _refill_grid(self):
        for c in range(self.GRID_SIZE):
            write_r = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_r:
                        self.grid[write_r, c] = self.grid[r, c]
                        self.visual_gems[r][c]['target_pos'] = self._get_screen_pos(c, write_r)
                        self.visual_gems[write_r][c] = self.visual_gems[r][c]
                    write_r -= 1
            
            for r in range(write_r, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                y_offset = (write_r - r + 1) * self.CELL_SIZE
                self._init_visual_gem(r, c, y_offset=y_offset)
                self.visual_gems[r][c]['pos'][1] -= y_offset

    def _create_particles(self, r, c):
        gem_type = self.visual_gems[r][c]['type']
        color = self.GEM_COLORS[gem_type]
        center_x, center_y = self.visual_gems[r][c]['pos']
        
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_screen_pos(self, c, r):
        return self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for r in range(self.GRID_SIZE + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y))
        for c in range(self.GRID_SIZE + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT))

    def _render_gems(self):
        radius = int(self.CELL_SIZE * 0.4)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem = self.visual_gems[r][c]
                if gem['type'] == -1: continue

                x, y = int(gem['pos'][0]), int(gem['pos'][1])
                current_radius = int(radius * gem['scale'])
                if current_radius <= 0: continue

                color = self.GEM_COLORS[gem['type']]
                
                # Create a temporary surface for transparency
                gem_surface = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                gem_surface.set_alpha(gem['alpha'])
                
                # Main gem body
                pygame.gfxdraw.aacircle(gem_surface, current_radius, current_radius, current_radius, color)
                pygame.gfxdraw.filled_circle(gem_surface, current_radius, current_radius, current_radius, color)
                
                # Highlight
                h_color = (min(255, color[0]+60), min(255, color[1]+60), min(255, color[2]+60))
                pygame.gfxdraw.filled_circle(gem_surface, int(current_radius * 0.7), int(current_radius * 0.7), int(current_radius * 0.3), h_color)

                self.screen.blit(gem_surface, (x - current_radius, y - current_radius))

    def _render_cursor(self):
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        x = self.visual_cursor_pos[0] - self.CELL_SIZE // 2
        y = self.visual_cursor_pos[1] - self.CELL_SIZE // 2
        self.screen.blit(s, (x, y))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            size = int(max(1, 6 * (p['lifespan'] / 30.0)))
            
            p_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, color, (size, size), size)
            self.screen.blit(p_surf, (p['pos'][0] - size, p['pos'][1] - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_main.render(f"Time: {math.ceil(self.time_left)}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(time_text, time_rect)

        if self.combo_multiplier > 1 and self.last_combo_pos:
            combo_text = self.font_combo.render(f"x{self.combo_multiplier}", True, self.COLOR_UI_TEXT)
            combo_rect = combo_text.get_rect(center=self.last_combo_pos)
            self.screen.blit(combo_text, combo_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME UP"
        text = self.font_game_over.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Gem Matcher")
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()