
# Generated: 2025-08-27T16:06:36.368851
# Source Brief: brief_01122.md
# Brief Index: 1122

        
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


# Set SDL to dummy driver to run headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to collect a gem."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle game. Collect 50 gems before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = (32, 20)
        self.CELL_SIZE = (self.WIDTH // self.GRID_SIZE[0], self.HEIGHT // self.GRID_SIZE[1])
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.GEM_TARGET = 50
        self.MIN_GEMS_ON_SCREEN = 10

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_GLOW = (200, 200, 255, 50)
        self.COLOR_SCORE = (220, 220, 240)
        self.COLOR_TIMER_GOOD = (0, 255, 128)
        self.COLOR_TIMER_BAD = (255, 50, 50)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.Font(None, 48)
        self.font_guide = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.cursor_pos = None
        self.gems = None
        self.score = None
        self.steps = None
        self.last_space_held = None
        self.particles = None
        self.last_dist_to_gem = None
        self.game_over = None
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()


        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.gems = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []

        for _ in range(self.MIN_GEMS_ON_SCREEN):
            self._add_gem()
            
        self.last_dist_to_gem = self._find_nearest_gem_dist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        prev_cursor_pos = list(self.cursor_pos)

        # 1. Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right

        # Clamp cursor to grid boundaries
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE[1] - 1)

        # 2. Click (Spacebar Press)
        gem_collected_this_step = False
        if space_held and not self.last_space_held:
            gem_to_remove = None
            for gem in self.gems:
                if gem['pos'] == self.cursor_pos:
                    gem_to_remove = gem
                    break
            
            if gem_to_remove:
                # Sound: gem_collect.wav
                self.gems.remove(gem_to_remove)
                self.score += 1
                reward += 1.0
                gem_collected_this_step = True
                self._create_particles(self._grid_to_pixel(gem_to_remove['pos']), gem_to_remove['color'])

        self.last_space_held = space_held
        
        # --- Reward Shaping for Movement ---
        moved = self.cursor_pos != prev_cursor_pos
        if moved:
            current_dist = self._find_nearest_gem_dist()
            if current_dist < self.last_dist_to_gem:
                reward += 0.1  # Moved closer
            else:
                reward -= 0.01 # Moved further or same
            self.last_dist_to_gem = current_dist

        # --- Game Logic ---
        # Respawn gems
        if len(self.gems) < self.MIN_GEMS_ON_SCREEN:
            self._add_gem()
        
        # Update particles
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if self.score >= self.GEM_TARGET:
            # Sound: win.wav
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # Sound: lose.wav
            reward -= 100.0
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper & Rendering Methods ---

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE[0] + self.CELL_SIZE[0] // 2
        py = grid_pos[1] * self.CELL_SIZE[1] + self.CELL_SIZE[1] // 2
        return (px, py)

    def _add_gem(self):
        if len(self.gems) >= self.GRID_SIZE[0] * self.GRID_SIZE[1]:
            return # Grid is full
        
        occupied_pos = {tuple(g['pos']) for g in self.gems}
        pos = [self.np_random.integers(0, self.GRID_SIZE[0]), self.np_random.integers(0, self.GRID_SIZE[1])]
        while tuple(pos) in occupied_pos:
            pos = [self.np_random.integers(0, self.GRID_SIZE[0]), self.np_random.integers(0, self.GRID_SIZE[1])]
        
        gem_data = {
            'pos': pos,
            'color': random.choice(self.GEM_COLORS),
            'anim_offset': self.np_random.random() * 2 * math.pi,
            'rotation': self.np_random.random() * 360
        }
        self.gems.append(gem_data)

    def _find_nearest_gem_dist(self):
        if not self.gems:
            return float('inf')
        
        cursor_arr = np.array(self.cursor_pos)
        gem_positions = np.array([g['pos'] for g in self.gems])
        distances = np.linalg.norm(gem_positions - cursor_arr, axis=1)
        return np.min(distances)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': velocity,
                'life': self.np_random.integers(15, 25),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE[0]):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE[1]):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw gems
        for gem in self.gems:
            px, py = self._grid_to_pixel(gem['pos'])
            size = self.CELL_SIZE[0] * 0.35
            
            # Bobbing animation
            bob_offset = math.sin(self.steps / 10.0 + gem['anim_offset']) * 2
            py += bob_offset
            
            # Rotation
            gem['rotation'] = (gem['rotation'] + 2) % 360
            
            # Draw diamond shape
            points = []
            for i in range(4):
                angle = math.radians(gem['rotation'] + i * 90)
                point_size = size if i % 2 == 0 else size * 0.6
                points.append((
                    int(px + point_size * math.cos(angle)),
                    int(py + point_size * math.sin(angle))
                ))

            pygame.gfxdraw.aapolygon(self.screen, points, gem['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, gem['color'])
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(size*1.2), (*gem['color'], 30))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25.0))
            color = (*p['color'], alpha)
            size = int(max(1, p['life'] / 5))
            pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)
            
        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos)
        
        # Glow
        glow_surface = pygame.Surface((self.CELL_SIZE[0]*2, self.CELL_SIZE[1]*2), pygame.SRCALPHA)
        glow_radius = int(self.CELL_SIZE[0] * (0.8 + 0.1 * math.sin(self.steps / 8.0)))
        pygame.gfxdraw.filled_circle(glow_surface, self.CELL_SIZE[0], self.CELL_SIZE[1], glow_radius, self.COLOR_CURSOR_GLOW)
        self.screen.blit(glow_surface, (cursor_px - self.CELL_SIZE[0], cursor_py - self.CELL_SIZE[1]))
        
        # Crosshair
        size = self.CELL_SIZE[0] // 2
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_px - size, cursor_py), (cursor_px + size, cursor_py), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_px, cursor_py - size), (cursor_px, cursor_py + size), 2)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cursor_px, cursor_py), 3)


    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"GEMS: {self.score}/{self.GEM_TARGET}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        time_ratio = max(0, 1 - (self.steps / self.MAX_STEPS))
        bar_width = (self.WIDTH - 20) * time_ratio
        bar_height = 10
        
        # Interpolate color from green to red
        timer_color = (
            int(self.COLOR_TIMER_BAD[0] * (1 - time_ratio) + self.COLOR_TIMER_GOOD[0] * time_ratio),
            int(self.COLOR_TIMER_BAD[1] * (1 - time_ratio) + self.COLOR_TIMER_GOOD[1] * time_ratio),
            int(self.COLOR_TIMER_BAD[2] * (1 - time_ratio) + self.COLOR_TIMER_GOOD[2] * time_ratio),
        )
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.HEIGHT - 20, self.WIDTH - 20, bar_height))
        if bar_width > 0:
            pygame.draw.rect(self.screen, timer_color, (10, self.HEIGHT - 20, bar_width, bar_height))
            
        # Show end game message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            if self.score >= self.GEM_TARGET:
                msg = "YOU WIN!"
                color = self.COLOR_TIMER_GOOD
            else:
                msg = "TIME UP!"
                color = self.COLOR_TIMER_BAD
                
            end_text = self.font_score.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # This part of the script will not run in a headless environment
    # It's for local testing with a display.
    # To run, comment out os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # To run this example, you might need to comment out the line:
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # and ensure you have a display environment.
    
    # For demonstration purposes, we'll re-enable the display if this script is run directly.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a real Pygame window to display the environment's rendering
    pygame.display.set_caption("Gem Collector")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control Loop ---
    action = env.action_space.sample() # Start with a no-op
    action[0] = 0
    action[1] = 0
    action[2] = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        # --- Map keyboard to MultiDiscrete action space ---
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Render the observation to the real screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # We need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        
        # Display info
        font = pygame.font.Font(None, 24)
        guide_text = font.render(env.user_guide, True, (255, 255, 255))
        real_screen.blit(guide_text, (10, env.HEIGHT - 40))
        reward_text = font.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
        real_screen.blit(reward_text, (env.WIDTH - reward_text.get_width() - 10, 10))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    pygame.quit()