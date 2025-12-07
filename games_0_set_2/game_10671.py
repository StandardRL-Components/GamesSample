import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:51:51.734443
# Source Brief: brief_00671.md
# Brief Index: 671
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GridCluster: A visually stimulating puzzle game where the player races against
    time to click on oscillating grid cells, forming same-colored clusters for points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race against time to click on oscillating grid cells. Lock cells of the same "
        "color together to form large clusters and score points before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to lock a cell's "
        "current color and add it to a cluster."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_WIDTH = 400
    GAME_AREA_HEIGHT = 400
    UI_AREA_WIDTH = SCREEN_WIDTH - GAME_AREA_WIDTH

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINES = (50, 60, 80)
    COLOR_CELL_BLACK = (10, 10, 10)
    COLOR_CELL_WHITE = (220, 220, 220)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_CURSOR_GLOW = (0, 150, 150)
    COLOR_CLUSTER_HIGHLIGHT = (0, 255, 100, 100) # RGBA
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIMER_BAR_BG = (60, 60, 80)
    COLOR_TIMER_BAR_FG = (255, 50, 50)
    
    # --- Game Parameters ---
    GAME_DURATION_SECONDS = 60.0
    TARGET_SCORE = 1000
    FPS = 60
    
    INITIAL_GRID_DIMS = 10
    INITIAL_OSC_FREQ = 0.5
    
    CLUSTER_MIN_SIZE = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.cluster_highlight_surface = pygame.Surface((self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT), pygame.SRCALPHA)

        # --- State Variables ---
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_state = None
        self.grid_locked = None
        self.cursor_pos = [0, 0]
        self.oscillation_freq = 0.0
        self.time_elapsed = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.clusters = []
        self.particles = []
        self.score_milestones_freq = []
        self.score_milestones_grid = []

        # self.reset() is called by the wrapper usually, but good for standalone
        # self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        
        self.grid_rows = self.INITIAL_GRID_DIMS
        self.grid_cols = self.INITIAL_GRID_DIMS
        
        # 0 for black, 1 for white
        self.grid_state = self.np_random.integers(0, 2, size=(self.grid_rows, self.grid_cols), dtype=np.uint8)
        self.grid_locked = np.full((self.grid_rows, self.grid_cols), False, dtype=bool)
        
        self.cursor_pos = [self.grid_rows // 2, self.grid_cols // 2]
        self.oscillation_freq = self.INITIAL_OSC_FREQ
        
        self.prev_space_held = False
        self.clusters = []
        self.particles = []

        # Reset progression milestones
        self.score_milestones_freq = [200, 400, 600, 800]
        self.score_milestones_grid = [400, 800]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.time_elapsed += 1.0 / self.FPS
        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        
        clicked = space_held and not self.prev_space_held
        if clicked:
            # sfx: click_sound()
            reward = self._handle_click()
            self.score += reward # The reward from _handle_click is the score change
            self._update_difficulty()

        self.prev_space_held = space_held
        
        # --- Game Logic Update ---
        self._update_oscillations()
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        terminal_reward = 0
        if self.score >= self.TARGET_SCORE:
            # sfx: win_jingle()
            terminated = True
            terminal_reward = 100.0
            self.game_over = True
        elif self.time_elapsed >= self.GAME_DURATION_SECONDS:
            # sfx: lose_buzzer()
            terminated = True
            terminal_reward = -100.0
            self.game_over = True
            
        reward += terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[0] -= 1
        elif movement == 2: # Down
            self.cursor_pos[0] += 1
        elif movement == 3: # Left
            self.cursor_pos[1] -= 1
        elif movement == 4: # Right
            self.cursor_pos[1] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_rows - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_cols - 1)

    def _handle_click(self):
        r, c = self.cursor_pos
        if self.grid_locked[r, c]:
            return -0.01 # Penalty for clicking a locked cell
        
        # Lock the cell and its current color
        self.grid_locked[r, c] = True
        cell_color = self.grid_state[r, c]
        
        # Create visual feedback
        cell_size_h = self.GAME_AREA_HEIGHT / self.grid_rows
        cell_size_w = self.GAME_AREA_WIDTH / self.grid_cols
        px = (c + 0.5) * cell_size_w
        py = (r + 0.5) * cell_size_h
        self._create_particles((px, py))

        # --- Cluster Logic ---
        neighboring_clusters = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid_locked[nr, nc] and self.grid_state[nr, nc] == cell_color:
                for i, cluster in enumerate(self.clusters):
                    if (nr, nc) in cluster:
                        neighboring_clusters.append(i)
                        break
        
        unique_neighbor_indices = sorted(list(set(neighboring_clusters)), reverse=True)

        if not unique_neighbor_indices:
            # Creating a new potential cluster
            self.clusters.append({(r, c)})
            return -0.01 # No immediate reward, slight penalty for non-connecting click
        else:
            # Merge existing clusters
            merged_cluster = {(r, c)}
            old_total_size = 0
            for index in unique_neighbor_indices:
                old_cluster = self.clusters.pop(index)
                merged_cluster.update(old_cluster)
                old_total_size += len(old_cluster)
            
            self.clusters.append(merged_cluster)
            new_total_size = len(merged_cluster)

            # --- Reward Calculation ---
            if old_total_size < self.CLUSTER_MIN_SIZE and new_total_size >= self.CLUSTER_MIN_SIZE:
                # sfx: new_cluster_formed()
                base_reward = 10
                bonus_reward = 5 * (new_total_size - self.CLUSTER_MIN_SIZE)
                return base_reward + bonus_reward
            elif old_total_size >= self.CLUSTER_MIN_SIZE:
                # sfx: cluster_expanded()
                return 0.1 # Small reward for expanding an existing cluster
            else:
                return -0.01 # Connected to small groups, but not enough to form a scoring cluster

    def _update_difficulty(self):
        if self.score_milestones_freq and self.score >= self.score_milestones_freq[0]:
            self.score_milestones_freq.pop(0)
            self.oscillation_freq += 0.05
            # sfx: level_up()

        if self.score_milestones_grid and self.score >= self.score_milestones_grid[0]:
            self.score_milestones_grid.pop(0)
            # sfx: grid_expand()
            
            old_rows, old_cols = self.grid_rows, self.grid_cols
            self.grid_rows += 2
            self.grid_cols += 2
            
            new_state = self.np_random.integers(0, 2, size=(self.grid_rows, self.grid_cols), dtype=np.uint8)
            new_locked = np.full((self.grid_rows, self.grid_cols), False, dtype=bool)
            
            # Copy old grid to the center of the new one
            row_offset, col_offset = 1, 1
            new_state[row_offset:row_offset+old_rows, col_offset:col_offset+old_cols] = self.grid_state
            new_locked[row_offset:row_offset+old_rows, col_offset:col_offset+old_cols] = self.grid_locked
            
            self.grid_state = new_state
            self.grid_locked = new_locked
            
            # Adjust cursor and clusters
            self.cursor_pos = [self.cursor_pos[0] + row_offset, self.cursor_pos[1] + col_offset]
            self.clusters = [{(r + row_offset, c + col_offset) for r, c in cluster} for cluster in self.clusters]

    def _update_oscillations(self):
        # This calculates the color for all unlocked cells for the current frame
        osc_val = math.sin(2 * math.pi * self.oscillation_freq * self.time_elapsed) > 0
        current_color = 1 if osc_val else 0
        self.grid_state[~self.grid_locked] = current_color

    def _create_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(3, 7)
            lifetime = self.np_random.uniform(0.3, 0.7)
            color = random.choice([self.COLOR_CURSOR, self.COLOR_CELL_WHITE])
            self.particles.append({'pos': list(pos), 'vel': velocity, 'radius': radius, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98 # friction
            p['vel'][1] *= 0.98
            p['life'] -= 1.0 / self.FPS
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        game_surface = self.screen.subsurface((0, 0, self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT))
        
        cell_h = self.GAME_AREA_HEIGHT / self.grid_rows
        cell_w = self.GAME_AREA_WIDTH / self.grid_cols

        # --- Draw cells ---
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                color = self.COLOR_CELL_WHITE if self.grid_state[r, c] == 1 else self.COLOR_CELL_BLACK
                pygame.draw.rect(game_surface, color, (c * cell_w, r * cell_h, cell_w, cell_h))

        # --- Draw cluster highlights ---
        self.cluster_highlight_surface.fill((0, 0, 0, 0))
        for cluster in self.clusters:
            if len(cluster) >= self.CLUSTER_MIN_SIZE:
                for r, c in cluster:
                    pygame.draw.rect(self.cluster_highlight_surface, self.COLOR_CLUSTER_HIGHLIGHT, (c * cell_w, r * cell_h, cell_w, cell_h))
        game_surface.blit(self.cluster_highlight_surface, (0, 0))

        # --- Draw grid lines ---
        for r in range(self.grid_rows + 1):
            pygame.draw.line(game_surface, self.COLOR_GRID_LINES, (0, r * cell_h), (self.GAME_AREA_WIDTH, r * cell_h))
        for c in range(self.grid_cols + 1):
            pygame.draw.line(game_surface, self.COLOR_GRID_LINES, (c * cell_w, 0), (c * cell_w, self.GAME_AREA_HEIGHT))

        # --- Draw particles ---
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            current_radius = int(p['radius'] * alpha)
            if current_radius > 0:
                color_with_alpha = (*p['color'], int(255 * alpha))
                try:
                    pygame.gfxdraw.filled_circle(game_surface, int(p['pos'][0]), int(p['pos'][1]), current_radius, color_with_alpha)
                except TypeError: # Older pygame versions might not like 4-tuple colors
                    temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color_with_alpha, (current_radius, current_radius), current_radius)
                    game_surface.blit(temp_surf, (int(p['pos'][0]) - current_radius, int(p['pos'][1]) - current_radius))


        # --- Draw cursor ---
        cur_r, cur_c = self.cursor_pos
        cursor_rect = pygame.Rect(cur_c * cell_w, cur_r * cell_h, cell_w, cell_h)
        # Glow effect
        glow_rect = cursor_rect.inflate(8, 8)
        pygame.draw.rect(game_surface, self.COLOR_CURSOR_GLOW, glow_rect, border_radius=4)
        pygame.draw.rect(game_surface, self.COLOR_CURSOR, cursor_rect, 3, border_radius=2)

    def _render_ui(self):
        ui_surface = self.screen.subsurface((self.GAME_AREA_WIDTH, 0, self.UI_AREA_WIDTH, self.SCREEN_HEIGHT))
        
        # --- Score Display ---
        score_label = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        ui_surface.blit(score_label, (20, 20))
        score_text = self.font_large.render(f"{int(self.score):04d}", True, self.COLOR_CURSOR)
        ui_surface.blit(score_text, (20, 50))
        
        # --- Timer Bar ---
        timer_label = self.font_small.render("TIME", True, self.COLOR_TEXT)
        ui_surface.blit(timer_label, (20, 120))
        
        bar_x, bar_y, bar_w, bar_h = 20, 150, self.UI_AREA_WIDTH - 40, 30
        time_ratio = max(0, 1.0 - (self.time_elapsed / self.GAME_DURATION_SECONDS))
        
        pygame.draw.rect(ui_surface, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        pygame.draw.rect(ui_surface, self.COLOR_TIMER_BAR_FG, (bar_x, bar_y, bar_w * time_ratio, bar_h), border_radius=5)
        
        # --- Help Text ---
        help_y = self.SCREEN_HEIGHT - 100
        help_text1 = self.font_small.render("Arrows: Move", True, self.COLOR_GRID_LINES)
        help_text2 = self.font_small.render("Space: Lock Cell", True, self.COLOR_GRID_LINES)
        ui_surface.blit(help_text1, (20, help_y))
        ui_surface.blit(help_text2, (20, help_y + 25))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "osc_freq": self.oscillation_freq,
            "grid_size": f"{self.grid_rows}x{self.grid_cols}"
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run main loop in headless mode. Exiting.")
        exit()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("GridCluster")
    clock = pygame.time.Clock()
    
    done = False
    while not done:
        # --- Action Mapping for Manual Play ---
        movement = 0 # None
        space_held = 0 # Released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered image, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds before closing
            
        clock.tick(env.FPS)
        
    env.close()