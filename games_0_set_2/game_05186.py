import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move your character on the isometric grid. "
        "Collect gems and avoid the red obstacles."
    )

    # Short, user-facing description of the game
    game_description = (
        "Collect 25 sparkling gems before time runs out while dodging moving obstacles in a vibrant isometric world."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game parameters
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    WIN_GEM_COUNT = 25
    GRID_WIDTH = 14
    GRID_HEIGHT = 10
    INITIAL_OBSTACLES = 3
    INITIAL_GEMS = 5

    # Visuals
    TILE_W = 48
    TILE_H = 24
    TILE_W_HALF = TILE_W // 2
    TILE_H_HALF = TILE_H // 2
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_SHADOW = (10, 15, 20, 100)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 50, 50, 50)
    GEM_COLORS = [
        (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0)
    ]
    UI_TEXT_COLOR = (240, 240, 255)
    UI_BG_COLOR = (30, 40, 60, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces as required
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.collected_gems = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_visual_pos = [0, 0]
        self.gems = []
        self.obstacles = []
        self.particles = []
        self.np_random = None

        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() is also called by the wrapper.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.collected_gems = 0
        self.game_over = False
        
        # Player state
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_visual_pos = self._iso_to_screen(*self.player_pos)
        
        # Entity state
        self.gems = []
        self.obstacles = []
        self.particles = []
        
        occupied_cells = {tuple(self.player_pos)}

        for _ in range(self.INITIAL_GEMS):
            self._spawn_gem(occupied_cells)
        
        for _ in range(self.INITIAL_OBSTACLES):
            self._spawn_obstacle(occupied_cells)

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        # --- 1. Calculate distance to nearest gem before moving ---
        dist_before = self._get_dist_to_nearest_gem()

        # --- 2. Update player position based on action ---
        if not self.game_over:
            prev_pos = list(self.player_pos)
            if movement == 1: # Up (iso up-left)
                self.player_pos[1] -= 1
            elif movement == 2: # Down (iso down-right)
                self.player_pos[1] += 1
            elif movement == 3: # Left (iso down-left)
                self.player_pos[0] -= 1
            elif movement == 4: # Right (iso up-right)
                self.player_pos[0] += 1
            
            # Boundary checks
            self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
            self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- 3. Update game state (obstacles, particles) ---
        self._update_obstacles()
        self._update_particles()
        
        # --- 4. Check for collisions and events ---
        if not self.game_over:
            # Player-Gem collision
            gem_collected = False
            for gem in self.gems[:]:
                if self.player_pos == gem['pos']:
                    # sfx: gem_collect.wav
                    self.gems.remove(gem)
                    self.collected_gems += 1
                    self.score += 10
                    reward += 10
                    occupied_cells = {tuple(o['pos'].astype(int)) for o in self.obstacles} | {tuple(g['pos']) for g in self.gems}
                    self._spawn_gem(occupied_cells)
                    self._create_particles(self._iso_to_screen(*gem['pos']), gem['color'])
                    gem_collected = True
                    break
            
            # Player-Obstacle collision
            for obstacle in self.obstacles:
                # FIX: Use np.array_equal for element-wise comparison between the player's list position
                # and the obstacle's numpy array position. The original `==` operator returned a boolean
                # array like `[True, False]`, which cannot be evaluated in an `if` statement.
                if np.array_equal(self.player_pos, obstacle['pos'].astype(int)):
                    # sfx: player_hit.wav
                    terminated = True
                    self.game_over = True
                    reward -= 5
                    self.score -= 5
                    self._create_particles(self.player_visual_pos, self.COLOR_OBSTACLE, 30)
                    break
        
        # --- 5. Calculate distance-based reward ---
        dist_after = self._get_dist_to_nearest_gem()
        if gem_collected:
            # Don't penalize for moving away from a gem that was just collected
            pass
        elif dist_after < dist_before:
            reward += 1.0
        elif dist_after > dist_before:
            reward -= 0.1
        
        # --- 6. Check for termination conditions ---
        self.steps += 1
        if not terminated:
            if self.collected_gems >= self.WIN_GEM_COUNT:
                # sfx: win_jingle.wav
                terminated = True
                self.game_over = True
                reward += 100
                self.score += 100
            elif self.steps >= self.MAX_STEPS:
                # sfx: time_up.wav
                terminated = True
                self.game_over = True
                reward -= 10
        
        # --- 7. Difficulty scaling ---
        if self.steps > 0 and self.steps % (20 * 30) == 0 and len(self.obstacles) < 10:
             occupied_cells = {tuple(o['pos'].astype(int)) for o in self.obstacles} | {tuple(g['pos']) for g in self.gems}
             self._spawn_obstacle(occupied_cells)
        
        if self.steps > 0 and self.steps % (10 * 30) == 0:
            for obs in self.obstacles:
                obs['speed'] *= 1.05

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_gems": self.collected_gems,
            "time_left": (self.MAX_STEPS - self.steps) / 30.0,
        }

    # --- Helper Methods ---

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_H_HALF
        return [int(screen_x), int(screen_y)]

    def _spawn_gem(self, occupied_cells):
        while True:
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if tuple(pos) not in occupied_cells:
                occupied_cells.add(tuple(pos))
                self.gems.append({
                    'pos': pos,
                    'color': self.GEM_COLORS[self.np_random.integers(0, len(self.GEM_COLORS))],
                    'anim_offset': self.np_random.random() * math.pi * 2
                })
                break

    def _spawn_obstacle(self, occupied_cells):
        while True:
            pos = np.array([self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)], dtype=float)
            if tuple(pos.astype(int)) not in occupied_cells:
                direction = self.np_random.choice([-1, 1], 2)
                if self.np_random.random() < 0.5: # Move along one axis
                    direction[self.np_random.choice([0, 1])] = 0
                
                self.obstacles.append({
                    'pos': pos,
                    'vel': direction * 0.05, # Grid units per step
                    'speed': 1.0
                })
                break

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'] += obs['vel'] * obs['speed']
            # Wrap around boundaries
            if obs['pos'][0] < 0: obs['pos'][0] = self.GRID_WIDTH - 0.01
            if obs['pos'][0] >= self.GRID_WIDTH: obs['pos'][0] = 0
            if obs['pos'][1] < 0: obs['pos'][1] = self.GRID_HEIGHT - 0.01
            if obs['pos'][1] >= self.GRID_HEIGHT: obs['pos'][1] = 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return float('inf')
        player_p = np.array(self.player_pos)
        gem_positions = np.array([g['pos'] for g in self.gems])
        distances = np.linalg.norm(gem_positions - player_p, axis=1)
        return np.min(distances)

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _render_game(self):
        # --- Create a sorted list of all renderable entities ---
        render_list = []
        for gem in self.gems:
            render_list.append(('gem', gem))
        for obs in self.obstacles:
            render_list.append(('obstacle', obs))
        if not self.game_over:
             render_list.append(('player', {'pos': np.array(self.player_pos)}))
        
        # Sort by isometric Y-coordinate for correct occlusion
        render_list.sort(key=lambda item: item[1]['pos'][0] + item[1]['pos'][1])
        
        # --- Render Grid ---
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # --- Render Entities in order ---
        for type, entity in render_list:
            if type == 'player':
                target_pos = self._iso_to_screen(*self.player_pos)
                self.player_visual_pos[0] += (target_pos[0] - self.player_visual_pos[0]) * 0.5
                self.player_visual_pos[1] += (target_pos[1] - self.player_visual_pos[1]) * 0.5
                pos = [int(p) for p in self.player_visual_pos]
                
                # Shadow
                shadow_rect = pygame.Rect(0, 0, 20, 10)
                shadow_rect.center = (pos[0], pos[1] + 12)
                shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
                self.screen.blit(shadow_surf, shadow_rect.topleft)

                # Glow
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_PLAYER_GLOW)
                # Body
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)

            elif type == 'gem':
                pos = self._iso_to_screen(*entity['pos'])
                anim_sin = math.sin(self.steps * 0.1 + entity['anim_offset'])
                size = int(6 + anim_sin * 2)
                glow_size = int(10 + anim_sin * 3)
                
                # Shadow
                shadow_rect = pygame.Rect(0, 0, 12, 6)
                shadow_rect.center = (pos[0], pos[1] + 8)
                shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
                self.screen.blit(shadow_surf, shadow_rect.topleft)
                
                # Glow
                glow_color = (*entity['color'], 70)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_size, glow_color)
                
                # Body (as a diamond)
                points = [
                    (pos[0], pos[1] - size), (pos[0] + size, pos[1]),
                    (pos[0], pos[1] + size), (pos[0] - size, pos[1])
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, entity['color'])
                pygame.gfxdraw.aapolygon(self.screen, points, entity['color'])

            elif type == 'obstacle':
                pos = self._iso_to_screen(*entity['pos'])
                
                # Shadow
                shadow_rect = pygame.Rect(0, 0, 22, 11)
                shadow_rect.center = (pos[0], pos[1] + 13)
                shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
                self.screen.blit(shadow_surf, shadow_rect.topleft)
                
                # Glow
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 14, self.COLOR_OBSTACLE_GLOW)
                # Body
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_OBSTACLE)

        # --- Render Particles ---
        for p in self.particles:
            p_pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            size = int(max(1, 5 * (p['life'] / 30.0)))
            # Using a Surface for alpha blending particles
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (size, size), size)
            self.screen.blit(particle_surf, (p_pos[0] - size, p_pos[1] - size))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.UI_TEXT_COLOR)
        self.screen.blit(score_text, (15, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30.0)
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.UI_TEXT_COLOR)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(time_text, time_rect)
        
        # Gem collection status
        gem_bar_bg_rect = pygame.Rect(0, 0, 300, 30)
        gem_bar_bg_rect.center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25)
        
        bg_surf = pygame.Surface(gem_bar_bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill(self.UI_BG_COLOR)
        self.screen.blit(bg_surf, gem_bar_bg_rect.topleft)

        # Gem icon
        gem_icon_pos = (gem_bar_bg_rect.left + 20, gem_bar_bg_rect.centery)
        points = [
            (gem_icon_pos[0], gem_icon_pos[1] - 8), (gem_icon_pos[0] + 8, gem_icon_pos[1]),
            (gem_icon_pos[0], gem_icon_pos[1] + 8), (gem_icon_pos[0] - 8, gem_icon_pos[1])
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.GEM_COLORS[0])
        pygame.gfxdraw.aapolygon(self.screen, points, self.GEM_COLORS[0])
        
        # Gem text
        gem_text = self.font_large.render(f"{self.collected_gems} / {self.WIN_GEM_COUNT}", True, self.UI_TEXT_COLOR)
        gem_text_rect = gem_text.get_rect(center=(gem_bar_bg_rect.centerx + 10, gem_bar_bg_rect.centery))
        self.screen.blit(gem_text, gem_text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self.collected_gems >= self.WIN_GEM_COUNT:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_PLAYER)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_OBSTACLE)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

# --- Example usage ---
if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # To use, you must unset the headless environment variable
    # e.g., `unset SDL_VIDEODRIVER` in your shell before running.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if terminated:
            # After game over, wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()