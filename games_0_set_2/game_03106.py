
# Generated: 2025-08-27T22:23:10.914325
# Source Brief: brief_03106.md
# Brief Index: 3106

        
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
        "Controls: Use arrow keys to move the selector. "
        "Hold space and trace a path over same-colored gems to select them. "
        "Release space to attempt a match."
    )

    game_description = (
        "Fast-paced isometric puzzle game. Match falling gems to score points before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.NUM_GEM_TYPES = 5
        self.FPS = 30
        self.MAX_TIME = 60  # seconds
        self.WIN_SCORE = 500
        self.MAX_STEPS = 1000

        # Visuals
        self.COLOR_BG = (25, 30, 45)
        self.COLOR_GRID = (45, 55, 75)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.TILE_WIDTH_ISO = 40
        self.TILE_HEIGHT_ISO = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

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
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 16)
        
        # --- State Variables ---
        self.grid = None
        self.gem_y_offsets = None
        self.cursor_pos = None
        self.selection = None
        self.particles = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.gem_fall_speed = None
        self.prev_space_held = None
        self.np_random = None

        self.reset()

        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback to a default generator if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        self.gem_y_offsets = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=float)
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selection = []
        self.particles = []
        
        self.timer = self.MAX_TIME * self.FPS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.gem_fall_speed = 1.0
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Update Game State ---
        self._update_timer_and_difficulty()
        self._handle_input(action)
        reward += self._update_gems_and_matches()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win reward
            else:
                reward -= 100  # Loss reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)
        
        # Handle selection logic
        r, c = self.cursor_pos
        if space_held:
            gem_pos = (r, c)
            if gem_pos not in self.selection:
                if not self.selection:
                    self.selection.append(gem_pos)
                else:
                    last_r, last_c = self.selection[-1]
                    current_gem_type = self.grid[r, c]
                    first_gem_type = self.grid[self.selection[0][0], self.selection[0][1]]
                    is_adjacent = abs(r - last_r) + abs(c - last_c) == 1
                    
                    if current_gem_type == first_gem_type and is_adjacent:
                        self.selection.append(gem_pos)
        
        # Trigger match check on space release
        if not space_held and self.prev_space_held:
            self._check_and_process_match()
            self.selection = []
        
        self.prev_space_held = space_held

    def _update_timer_and_difficulty(self):
        self.timer = max(0, self.timer - 1)
        # Increase fall speed every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.gem_fall_speed += 0.05

    def _check_and_process_match(self):
        if not self.selection or len(self.selection) < 3:
            return 0
        
        # All gems in selection are guaranteed to be the same color by the selection logic
        num_cleared = len(self.selection)
        reward = num_cleared
        self.score += num_cleared
        
        if num_cleared == 4:
            reward += 5
            self.score += 5
        elif num_cleared >= 5:
            reward += 10
            self.score += 10
            
        gem_type = self.grid[self.selection[0][0], self.selection[0][1]]
        color = self.GEM_COLORS[gem_type - 1]
        
        for r, c in self.selection:
            self.grid[r, c] = 0  # Mark as empty
            self._create_particles(r, c, color)
            # sfx: match_gem.wav

        return reward

    def _update_gems_and_matches(self):
        # This combines the match check (on space release) and gravity updates
        reward = 0
        if not self.prev_space_held and self.selection:
             reward = self._check_and_process_match()
             self.selection = []
        
        # Apply gravity: let gems fall into empty spaces
        for c in range(self.GRID_WIDTH):
            empty_row = -1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == 0 and empty_row == -1:
                    empty_row = r
                elif self.grid[r, c] != 0 and empty_row != -1:
                    # Drop gem
                    self.grid[empty_row, c] = self.grid[r, c]
                    self.grid[r, c] = 0
                    empty_row -= 1
        
        # Spawn new gems at the top
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                    # sfx: gem_spawn.wav
        
        return reward

    def _create_particles(self, r, c, color):
        iso_x, iso_y = self._grid_to_iso(r, c)
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            size = self.np_random.random() * 3 + 2
            self.particles.append({'pos': [iso_x, iso_y], 'vel': vel, 'life': life, 'color': color, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        return self.timer <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
    
    def _calculate_reward(self):
        # Reward is calculated event-based within the logic functions
        return 0

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.timer / self.FPS,
        }

    # --- Rendering ---
    def _grid_to_iso(self, r, c):
        x = self.ORIGIN_X + (c - r) * (self.TILE_WIDTH_ISO / 2)
        y = self.ORIGIN_Y + (c + r) * (self.TILE_HEIGHT_ISO / 2)
        return int(x), int(y)

    def _render_gem(self, x, y, color, highlight=False, selected=False):
        top_color = color
        side_color_dark = tuple(max(0, val - 60) for val in color)
        side_color_light = tuple(max(0, val - 30) for val in color)
        
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        
        # Points for the isometric cube
        top_point = (x, y)
        left_point = (x - w / 2, y + h / 2)
        right_point = (x + w / 2, y + h / 2)
        bottom_point = (x, y + h)
        
        # Draw faces
        pygame.gfxdraw.filled_polygon(self.screen, [top_point, left_point, bottom_point, right_point], side_color_dark)
        pygame.gfxdraw.filled_polygon(self.screen, [top_point, left_point, (left_point[0], left_point[1] - h/2)], side_color_light)
        pygame.gfxdraw.filled_polygon(self.screen, [top_point, right_point, (right_point[0], right_point[1] - h/2)], side_color_light)
        
        # Top face (the gem itself)
        top_face_pts = [top_point, (x - w/2, y + h/4), (x, y + h/2), (x + w/2, y + h/4)]
        pygame.gfxdraw.filled_polygon(self.screen, top_face_pts, top_color)
        pygame.gfxdraw.aapolygon(self.screen, top_face_pts, (255, 255, 255, 60))
        
        if selected:
            pygame.gfxdraw.aapolygon(self.screen, top_face_pts, (255, 255, 255))
            pygame.gfxdraw.aapolygon(self.screen, [(p[0]+1, p[1]) for p in top_face_pts], (255, 255, 255))
            pygame.gfxdraw.aapolygon(self.screen, [(p[0]-1, p[1]) for p in top_face_pts], (255, 255, 255))
        elif highlight:
            pygame.gfxdraw.aapolygon(self.screen, top_face_pts, (255, 255, 0, 200))

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._grid_to_iso(r, 0)
            p2 = self._grid_to_iso(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._grid_to_iso(0, c)
            p2 = self._grid_to_iso(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
            
        # Draw gems (back to front for correct occlusion)
        selected_coords = {tuple(pos) for pos in self.selection}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    iso_x, iso_y = self._grid_to_iso(r, c)
                    color = self.GEM_COLORS[gem_type - 1]
                    is_cursor_on = self.cursor_pos == [r, c]
                    is_selected = (r, c) in selected_coords
                    self._render_gem(iso_x, iso_y, color, highlight=is_cursor_on, selected=is_selected)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['size'] * (p['life'] / 30)))
            
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Timer bar
        timer_ratio = self.timer / (self.MAX_TIME * self.FPS)
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        # Interpolate color from green to red
        timer_color = (
            min(255, 255 * (1 - timer_ratio) * 2),
            min(255, 255 * timer_ratio * 2),
            0
        )
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height))
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            win_surf = self.font_large.render(status_text, True, (255, 255, 0))
            win_rect = win_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(win_surf, win_rect)

            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, (255, 255, 255))
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # In human mode, we need a real display.
        # The environment itself is headless, so we create a window here.
        pygame.init()
        pygame.display.set_caption("Gem Matcher")
        screen = pygame.display.set_mode((640, 400))
        clock = pygame.time.Clock()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Store key states
    key_states = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
    }

    while running:
        action = [0, 0, 0] # Default no-op action

        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in key_states:
                        key_states[event.key] = True
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        total_reward = 0
                if event.type == pygame.KEYUP:
                    if event.key in key_states:
                        key_states[event.key] = False

            # Map pygame keys to the MultiDiscrete action space
            if key_states[pygame.K_UP]: action[0] = 1
            elif key_states[pygame.K_DOWN]: action[0] = 2
            elif key_states[pygame.K_LEFT]: action[0] = 3
            elif key_states[pygame.K_RIGHT]: action[0] = 4
            
            if key_states[pygame.K_SPACE]: action[1] = 1
            if key_states[pygame.K_LSHIFT]: action[2] = 1
            
        else: # Random agent for rgb_array mode
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            # The environment returns an RGB array. We need to display it.
            # Pygame uses (width, height), numpy uses (height, width)
            # The observation is (height, width, 3), so we need to transpose for pygame.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(env.FPS)
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            if render_mode == "human":
                # Wait a bit before auto-resetting in human mode
                pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()