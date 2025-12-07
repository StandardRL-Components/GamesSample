import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:00:10.961410
# Source Brief: brief_02633.md
# Brief Index: 2633
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedurally generated river, collect sets of colored gems for a speed boost, "
        "and avoid crashing into the banks or obstacles."
    )
    user_guide = "Controls: Use ← and → arrow keys to steer your ship down the river."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_Y_POS = SCREEN_HEIGHT * 2 / 3
    MAX_STEPS = 2000
    RIVER_LENGTH = 10000 # Total scroll distance to win

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_RIVER = (25, 40, 80)
    COLOR_BANK = (40, 100, 40)
    COLOR_BANK_DARK = (20, 50, 20)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_OBSTACLE_GLOW = (90, 45, 10)
    COLOR_TRAIL = (255, 255, 255)
    COLOR_FINISH_LINE = (255, 255, 255)
    GEM_COLORS = {
        "red": (255, 50, 50),
        "green": (50, 255, 50),
        "blue": (50, 50, 255)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables (these will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_speed = 6.0
        self.scroll_y = 0.0
        self.base_scroll_speed = 3.0
        self.speed_multiplier = 1.0
        self.river_path = []
        self.gems = []
        self.obstacles = []
        self.collected_gems = {}
        self.trail = deque(maxlen=15)
        self.particle_effects = []
        self.global_rotation = 0
        
        # self.reset() # reset is called by the environment runner
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.scroll_y = 0
        self.speed_multiplier = 1.0
        self.collected_gems = {"red": 0, "green": 0, "blue": 0}
        self.trail.clear()
        self.particle_effects.clear()
        
        self._procedurally_generate_world()
        
        self.player_pos = pygame.Vector2(self.river_path[0]['center'], self.PLAYER_Y_POS)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        self._handle_input(movement)

        # --- Game Logic Update ---
        reward = self._update_game_state()

        # --- Termination Check ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Max steps reached
             # No specific reward change, just end
             pass
        
        truncated = False # No truncation condition other than max steps
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_pos.x -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos.x += self.player_speed

    def _update_game_state(self):
        current_scroll_speed = self.base_scroll_speed * self.speed_multiplier
        self.scroll_y += current_scroll_speed
        
        # Base reward for survival/progress
        reward = 0.1

        # --- Collision Detection ---
        # Riverbanks
        current_segment = self._get_river_segment(self.scroll_y + self.PLAYER_Y_POS)
        bank_left = current_segment['center'] - current_segment['width'] / 2
        bank_right = current_segment['center'] + current_segment['width'] / 2
        
        player_half_width = 8
        if self.player_pos.x - player_half_width < bank_left or self.player_pos.x + player_half_width > bank_right:
            self.game_over = True
            # sfx: crash_sound
            self._create_particles(self.player_pos, (200,200,50), 30)
            return -100.0

        # Obstacles
        player_rect = pygame.Rect(self.player_pos.x - player_half_width, self.player_pos.y - player_half_width, player_half_width*2, player_half_width*2)
        for obs in self.obstacles[:]:
            obs_screen_y = obs['y'] - self.scroll_y
            if obs_screen_y < -50: # Prune off-screen obstacles
                self.obstacles.remove(obs)
                continue
            
            obs_rect = pygame.Rect(obs['x'], obs_screen_y, obs['w'], obs['h'])
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                # sfx: crash_sound
                self._create_particles(self.player_pos, self.COLOR_OBSTACLE, 30)
                return -100.0

        # --- Gem Collection ---
        for gem in self.gems[:]:
            gem_screen_y = gem['y'] - self.scroll_y
            if gem_screen_y < -20: # Prune off-screen gems
                self.gems.remove(gem)
                continue

            gem_pos = pygame.Vector2(gem['x'], gem_screen_y)
            if self.player_pos.distance_to(gem_pos) < 20:
                self.gems.remove(gem)
                self.score += 10
                reward += 1.0
                # sfx: gem_collect_sound
                self._create_particles(gem_pos, self.GEM_COLORS[gem['type']], 10)
                
                # Check for set completion
                self.collected_gems[gem['type']] += 1
                if all(count > 0 for count in self.collected_gems.values()):
                    for key in self.collected_gems:
                        self.collected_gems[key] -= 1
                    self.score += 50
                    reward += 5.0
                    self.speed_multiplier *= 1.20
                    # sfx: set_complete_sound
                    self._create_particles(self.player_pos, (255,215,0), 40, 5.0)


        # --- Win Condition ---
        if self.scroll_y >= self.RIVER_LENGTH:
            self.game_over = True
            # sfx: victory_fanfare
            return 100.0
        
        # --- Update trail and particles ---
        self.trail.append(self.player_pos.copy())
        for p in self.particle_effects[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particle_effects.remove(p)

        self.global_rotation = (self.global_rotation + 2) % 360
        
        return reward

    def _procedurally_generate_world(self):
        self.river_path = []
        self.gems = []
        self.obstacles = []
        
        # Generate River Path
        center = self.SCREEN_WIDTH / 2
        width = self.SCREEN_WIDTH * 0.6
        y = 0
        while y < self.RIVER_LENGTH + self.SCREEN_HEIGHT:
            segment_length = self.np_random.integers(200, 401)
            self.river_path.append({'y': y, 'center': center, 'width': width})
            
            # Wander
            center += self.np_random.uniform(-50, 50)
            width += self.np_random.uniform(-30, 30)
            
            # Clamp values
            center = np.clip(center, self.SCREEN_WIDTH * 0.3, self.SCREEN_WIDTH * 0.7)
            width = np.clip(width, self.SCREEN_WIDTH * 0.2, self.SCREEN_WIDTH * 0.8)
            
            y += segment_length

        # Populate with Gems and Obstacles
        obstacle_density = 0.05
        gem_density = 0.1
        
        for y in range(0, int(self.RIVER_LENGTH), 50):
            segment = self._get_river_segment(y)
            bank_left = segment['center'] - segment['width'] / 2 + 20
            bank_right = segment['center'] + segment['width'] / 2 - 20
            
            # Place obstacles
            if self.np_random.random() < obstacle_density and y > 500: # Don't spawn obstacles at start
                obs_w = self.np_random.integers(40, 101)
                obs_h = 20
                obs_x = self.np_random.uniform(bank_left, bank_right - obs_w)
                self.obstacles.append({'x': obs_x, 'y': y, 'w': obs_w, 'h': obs_h})
                obstacle_density += 0.0001 # Slowly increase density
                
            # Place gems
            if self.np_random.random() < gem_density:
                gem_type = self.np_random.choice(list(self.GEM_COLORS.keys()))
                gem_x = self.np_random.uniform(bank_left, bank_right)
                self.gems.append({'x': gem_x, 'y': y, 'type': gem_type})
    
    def _get_river_segment(self, y_pos):
        # Find the segment corresponding to a given y-coordinate
        for i in range(len(self.river_path) - 1):
            if self.river_path[i]['y'] <= y_pos < self.river_path[i+1]['y']:
                # Interpolate for smooth transitions
                p0 = self.river_path[i]
                p1 = self.river_path[i+1]
                t = (y_pos - p0['y']) / (p1['y'] - p0['y'])
                center = p0['center'] + (p1['center'] - p0['center']) * t
                width = p0['width'] + (p1['width'] - p0['width']) * t
                return {'center': center, 'width': width}
        return self.river_path[-1]

    def _create_particles(self, pos, color, count, speed=2.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, speed)
            self.particle_effects.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 31),
                'color': color,
                'size': self.np_random.integers(2, 6)
            })

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- River ---
        for y_screen in range(0, self.SCREEN_HEIGHT, 10):
            y_world = self.scroll_y + y_screen
            segment = self._get_river_segment(y_world)
            center_x = segment['center']
            width = segment['width']
            
            # River water
            pygame.draw.rect(self.screen, self.COLOR_RIVER, (center_x - width / 2, y_screen, width, 10))
            
            # Riverbanks
            pygame.draw.line(self.screen, self.COLOR_BANK_DARK, (center_x - width / 2 - 10, y_screen), (center_x - width / 2, y_screen), 5)
            pygame.draw.line(self.screen, self.COLOR_BANK, (center_x - width / 2, y_screen), (center_x - width / 2 + 10, y_screen), 5)
            pygame.draw.line(self.screen, self.COLOR_BANK_DARK, (center_x + width / 2 - 10, y_screen), (center_x + width / 2, y_screen), 5)
            pygame.draw.line(self.screen, self.COLOR_BANK, (center_x + width / 2, y_screen), (center_x + width / 2 + 10, y_screen), 5)

        # --- Finish Line ---
        finish_y_screen = self.RIVER_LENGTH - self.scroll_y
        if 0 < finish_y_screen < self.SCREEN_HEIGHT:
            segment = self._get_river_segment(self.RIVER_LENGTH)
            bank_left = segment['center'] - segment['width'] / 2
            bank_right = segment['center'] + segment['width'] / 2
            for x in range(int(bank_left), int(bank_right), 20):
                pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (x, finish_y_screen), (x + 10, finish_y_screen), 3)

        # --- Obstacles ---
        for obs in self.obstacles:
            obs_y = obs['y'] - self.scroll_y
            if -obs['h'] < obs_y < self.SCREEN_HEIGHT:
                rect = pygame.Rect(int(obs['x']), int(obs_y), int(obs['w']), int(obs['h']))
                self._draw_glow_rect(self.screen, rect, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)
        
        # --- Gems ---
        for gem in self.gems:
            gem_y = gem['y'] - self.scroll_y
            if -20 < gem_y < self.SCREEN_HEIGHT + 20:
                self._draw_rotated_rect(self.screen, (gem['x'], gem_y), (14, 14), self.GEM_COLORS[gem['type']], self.global_rotation)

        # --- Particles ---
        for p in self.particle_effects:
            alpha = int(255 * (p['life'] / 30.0))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # --- Player Trail ---
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            color = self.COLOR_TRAIL + (alpha,)
            radius = int(8 * (i / len(self.trail)))
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (int(pos.x - radius), int(self.PLAYER_Y_POS - radius)))
        
        # --- Player ---
        if not self.game_over:
            p = self.player_pos
            points = [(p.x, p.y - 12), (p.x - 8, p.y + 8), (p.x + 8, p.y + 8)]
            self._draw_glow_poly(self.screen, points, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Speed Multiplier
        speed_text = self.font_ui.render(f"SPEED: {self.speed_multiplier:.2f}x", True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 40))

        # Collected Gems UI
        for i, (gem_type, color) in enumerate(self.GEM_COLORS.items()):
            count = self.collected_gems[gem_type]
            ui_pos = (self.SCREEN_WIDTH - 120 + i * 40, 25)
            self._draw_rotated_rect(self.screen, ui_pos, (12,12), color if count == 0 else (255,255,255), 0, border_width=2)

        if self.game_over:
            outcome = "YOU WIN!" if self.scroll_y >= self.RIVER_LENGTH else "GAME OVER"
            end_text = self.font_big.render(outcome, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.scroll_y,
            "speed_multiplier": self.speed_multiplier
        }

    # --- Drawing Helpers ---
    def _draw_glow_poly(self, surface, points, color, glow_color):
        pygame.gfxdraw.aapolygon(surface, [(int(x), int(y)) for x, y in points], glow_color)
        pygame.gfxdraw.filled_polygon(surface, [(int(x), int(y)) for x, y in points], glow_color)
        pygame.gfxdraw.aapolygon(surface, [(int(x), int(y)) for x, y in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(x), int(y)) for x, y in points], color)

    def _draw_glow_rect(self, surface, rect, color, glow_color):
        glow_rect = rect.inflate(6, 6)
        pygame.draw.rect(surface, glow_color, glow_rect, border_radius=5)
        pygame.draw.rect(surface, color, rect, border_radius=3)
    
    def _draw_rotated_rect(self, surface, center, size, color, angle, border_width=0):
        w, h = size
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        corners = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                x = center[0] + (i * w/2 * cos_a - j * h/2 * sin_a)
                y = center[1] + (i * w/2 * sin_a + j * h/2 * cos_a)
                corners.append((x, y))
        
        # Reorder for drawing
        points = [corners[0], corners[1], corners[3], corners[2]]
        
        if border_width > 0:
             pygame.gfxdraw.aapolygon(surface, points, color)
        else:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch to a real display driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.quit() # Quit the dummy driver
    pygame.init() # Re-init with the real driver
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("River Racer")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Re-initialize fonts after re-init
    env.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
    env.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
    
    while not done:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()