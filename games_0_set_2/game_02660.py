
# Generated: 2025-08-27T21:03:20.033537
# Source Brief: brief_02660.md
# Brief Index: 2660

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Press space when adjacent to a critter to capture it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch 10 scurrying critters in the arena before the 60-second timer runs out. Faster critters are worth more points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # Persistent state for difficulty scaling across episodes
    _base_critter_speed_multiplier = 1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_COLS, self.GRID_ROWS = 20, 12
        self.CELL_SIZE = 30
        self.GAME_AREA_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GAME_AREA_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.X_OFFSET = (self.WIDTH - self.GAME_AREA_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GAME_AREA_HEIGHT) // 2
        self.GAME_DURATION_FRAMES = 60 * self.FPS
        self.CRITTER_COUNT = 5
        self.CAPTURE_RADIUS = 1.5 # Manhattan distance for capture

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (150, 255, 200)
        self.COLOR_TEXT = (220, 220, 240)
        self.CRITTER_COLORS = {
            "slow": (255, 220, 50),   # Yellow
            "medium": (50, 180, 255), # Blue
            "fast": (255, 80, 80),    # Red
        }
        self.CRITTER_GLOWS = {
            "slow": (255, 240, 150),
            "medium": (150, 220, 255),
            "fast": (255, 160, 160),
        }
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.critters = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.critters_caught = 0
        self.game_over = False
        self.last_space_held = False
        self.space_press_effect = 0
        
        self.np_random = None
        
        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.critters_caught = 0
        self.game_over = False
        self.last_space_held = False
        self.space_press_effect = 0
        
        self.player_pos = np.array([self.GRID_COLS // 2, self.GRID_ROWS // 2])
        
        self.critters = []
        self.particles = []
        
        occupied_positions = {tuple(self.player_pos)}
        for _ in range(self.CRITTER_COUNT):
            self._spawn_critter(occupied_positions)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief

        # --- Game Logic ---
        
        # 1. Update particles and effects
        self._update_particles()
        if self.space_press_effect > 0:
            self.space_press_effect -= 1

        # 2. Player Movement and Reward Calculation
        old_player_pos = self.player_pos.copy()
        dist_before = self._get_dist_to_nearest_critter(old_player_pos)

        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_COLS - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_ROWS - 1)
        
        dist_after = self._get_dist_to_nearest_critter(self.player_pos)

        if movement != 0:
            if dist_after < dist_before:
                reward += 0.1 # Moved closer
            elif dist_after > dist_before:
                reward -= 0.1 # Moved further

        # 3. Critter Movement
        for critter in self.critters:
            critter['move_cooldown'] -= 1
            if critter['move_cooldown'] <= 0:
                # Move to a random adjacent cell
                move = self.np_random.integers(0, 5)
                if move == 1: critter['pos'][1] -= 1
                elif move == 2: critter['pos'][1] += 1
                elif move == 3: critter['pos'][0] -= 1
                elif move == 4: critter['pos'][0] += 1
                
                critter['pos'][0] = np.clip(critter['pos'][0], 0, self.GRID_COLS - 1)
                critter['pos'][1] = np.clip(critter['pos'][1], 0, self.GRID_ROWS - 1)
                
                critter['move_cooldown'] = critter['base_cooldown']

        # 4. Capture Logic
        is_space_press = space_held and not self.last_space_held
        if is_space_press:
            self.space_press_effect = 5 # Visual effect lasts 5 frames
            # sfx: capture_attempt.wav
            
            captured_critter = None
            for critter in self.critters:
                if self._manhattan_distance(self.player_pos, critter['pos']) < self.CAPTURE_RADIUS:
                    captured_critter = critter
                    break
            
            if captured_critter:
                # sfx: capture_success.wav
                self.critters.remove(captured_critter)
                self.critters_caught += 1
                
                capture_reward = {"slow": 1, "medium": 2, "fast": 3}[captured_critter['type']]
                self.score += capture_reward
                reward += capture_reward
                
                self._create_particles(captured_critter['pos'], self.CRITTER_COLORS[captured_critter['type']])
                
                occupied_positions = {tuple(c['pos']) for c in self.critters}
                occupied_positions.add(tuple(self.player_pos))
                self._spawn_critter(occupied_positions)
        
        self.last_space_held = space_held
        
        # 5. Check Termination
        terminated = False
        if self.critters_caught >= 10:
            reward += 10 # Victory bonus
            terminated = True
            # sfx: victory.wav
            # Increase difficulty for the next game
            GameEnv._base_critter_speed_multiplier += 0.1 
        elif self.steps >= self.GAME_DURATION_FRAMES:
            terminated = True
            # sfx: game_over.wav
        
        if terminated:
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
        return {
            "score": self.score,
            "steps": self.steps,
            "critters_caught": self.critters_caught,
            "time_remaining_seconds": (self.GAME_DURATION_FRAMES - self.steps) / self.FPS,
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            px = self.X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.Y_OFFSET), (px, self.Y_OFFSET + self.GAME_AREA_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, py), (self.X_OFFSET + self.GAME_AREA_WIDTH, py))
        
        # Draw critters
        for critter in self.critters:
            px, py = self._grid_to_pixel(critter['pos'])
            bob_offset = math.sin(self.steps / 5 + critter['pos'][0]) * 2
            
            glow_radius = int(self.CELL_SIZE * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, px, int(py + bob_offset), glow_radius, (*self.CRITTER_GLOWS[critter['type']], 50))

            radius = int(self.CELL_SIZE * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, px, int(py + bob_offset), radius, self.CRITTER_COLORS[critter['type']])
            pygame.gfxdraw.aacircle(self.screen, px, int(py + bob_offset), radius, self.CRITTER_COLORS[critter['type']])

        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        
        # Spacebar press effect
        if self.space_press_effect > 0:
            alpha = int(100 * (self.space_press_effect / 5))
            radius = int(self.CELL_SIZE * (1.5 - (self.space_press_effect / 5)))
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, (*self.COLOR_PLAYER_GLOW, alpha))

        # Player glow
        glow_radius = int(self.CELL_SIZE * 0.6 + math.sin(self.steps / 6) * 3)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, (*self.COLOR_PLAYER_GLOW, 40))
        
        # Player body
        radius = int(self.CELL_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        # Timer
        time_left_sec = max(0, (self.GAME_DURATION_FRAMES - self.steps) / self.FPS)
        mins, secs = divmod(int(time_left_sec), 60)
        timer_str = f"TIME: {mins:02}:{secs:02}"
        timer_text = self.font_large.render(timer_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 15, 10))
        
        # Critters caught
        caught_str = f"CAUGHT: {self.critters_caught} / 10"
        caught_text = self.font_small.render(caught_str, True, self.COLOR_TEXT)
        self.screen.blit(caught_text, (self.WIDTH // 2 - caught_text.get_width() // 2, self.HEIGHT - caught_text.get_height() - 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.critters_caught >= 10:
                end_msg = "YOU WIN!"
            else:
                end_msg = "TIME'S UP!"
            
            end_text = self.font_large.render(end_msg, True, self.COLOR_PLAYER_GLOW)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _spawn_critter(self, occupied_positions):
        while True:
            pos = self.np_random.integers([0, 0], [self.GRID_COLS, self.GRID_ROWS], size=2)
            if tuple(pos) not in occupied_positions:
                occupied_positions.add(tuple(pos))
                break
        
        critter_type = self.np_random.choice(["slow", "medium", "fast"], p=[0.4, 0.4, 0.2])
        base_speed = {"slow": 1.0, "medium": 1.5, "fast": 2.0}[critter_type]
        
        # Higher multiplier = faster critters = lower cooldown
        cooldown = int(self.FPS / (base_speed * self._base_critter_speed_multiplier))
        
        self.critters.append({
            "pos": pos,
            "type": critter_type,
            "base_cooldown": max(1, cooldown),
            "move_cooldown": self.np_random.integers(0, cooldown + 1)
        })

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _grid_to_pixel(self, grid_pos):
        px = self.X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_dist_to_nearest_critter(self, pos):
        if not self.critters:
            return float('inf')
        return min(self._manhattan_distance(pos, c['pos']) for c in self.critters)
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")