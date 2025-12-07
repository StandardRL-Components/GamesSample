
# Generated: 2025-08-27T13:17:57.725189
# Source Brief: brief_00320.md
# Brief Index: 320

        
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
        "Controls: Arrow keys to move your square. Collect all the fruit before the timer runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you navigate a grid to collect fruit against the clock. Optimize your path for a high score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
    
    COLOR_BG = (20, 30, 50)
    COLOR_GRID = (30, 50, 80)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    FRUIT_COLORS = [
        (255, 80, 80),   # Red
        (255, 255, 100), # Yellow
        (255, 180, 100), # Orange
        (200, 100, 255)  # Purple
    ]
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_LOW = (255, 100, 100)
    
    MAX_TIME = 60.0
    NUM_FRUITS = 15
    PLAYER_MOVE_COOLDOWN_FRAMES = 4 # Time in frames between moves
    PLAYER_INTERP_SPEED = 0.4 # How fast the player visually moves to the target cell

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except Exception:
            # Fallback if system fonts are not available (e.g., in some minimal Docker containers)
            self.font_large = pygame.font.SysFont("sans", 48)
            self.font_medium = pygame.font.SysFont("sans", 32)
            self.font_small = pygame.font.SysFont("sans", 24)

        # Initialize state variables
        self.player_grid_pos = [0, 0]
        self.player_pixel_pos = [0.0, 0.0]
        self.player_move_timer = 0
        self.fruits = []
        self.particles = []
        self.timer = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_dist_to_fruit = 0
        
        # This will call reset() and initialize the game state properly
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        # Player state
        self.player_grid_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_move_timer = 0
        
        # Game elements
        self.fruits = self._generate_fruits()
        self.particles = []
        
        # Reward calculation state
        self.last_dist_to_fruit = self._find_nearest_fruit_dist()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game has ended, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Time and Cooldowns ---
        dt = self.clock.tick(30) / 1000.0
        self.timer = max(0, self.timer - dt)
        self.player_move_timer = max(0, self.player_move_timer - 1)
        
        # --- Unpack Action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Game Logic ---
        self.steps += 1
        reward = 0
        
        # Player Movement
        if movement != 0 and self.player_move_timer == 0:
            self.player_move_timer = self.PLAYER_MOVE_COOLDOWN_FRAMES
            target_pos = list(self.player_grid_pos)
            if movement == 1: target_pos[1] -= 1  # Up
            elif movement == 2: target_pos[1] += 1  # Down
            elif movement == 3: target_pos[0] -= 1  # Left
            elif movement == 4: target_pos[0] += 1  # Right
            
            # Boundary checks
            target_pos[0] = np.clip(target_pos[0], 0, self.GRID_COLS - 1)
            target_pos[1] = np.clip(target_pos[1], 0, self.GRID_ROWS - 1)
            self.player_grid_pos = target_pos

        # Interpolate visual position for smooth movement
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_pixel_pos[0] += (target_pixel_pos[0] - self.player_pixel_pos[0]) * self.PLAYER_INTERP_SPEED
        self.player_pixel_pos[1] += (target_pixel_pos[1] - self.player_pixel_pos[1]) * self.PLAYER_INTERP_SPEED

        # Fruit Collection
        collected_fruit_index = -1
        for i, fruit in enumerate(self.fruits):
            if fruit["pos"] == self.player_grid_pos:
                collected_fruit_index = i
                break
        
        if collected_fruit_index != -1:
            fruit = self.fruits.pop(collected_fruit_index)
            reward += 10
            self.score += 10
            self._create_particles(self._grid_to_pixel(fruit["pos"]), fruit["color"])
            # SFX: Fruit collect sound

        # Update dynamic elements
        self._update_particles()
        
        # --- Reward Calculation ---
        new_dist_to_fruit = self._find_nearest_fruit_dist()
        if new_dist_to_fruit is not None and self.last_dist_to_fruit is not None:
            if new_dist_to_fruit < self.last_dist_to_fruit:
                reward += 1.0  # Moved closer
            else:
                reward -= 0.2  # Moved away or stayed same distance
        self.last_dist_to_fruit = new_dist_to_fruit
        
        # --- Termination Check ---
        terminated = False
        if self.timer <= 0:
            terminated = True
            reward -= 50
            self.score = max(0, self.score - 50)
        elif not self.fruits:
            terminated = True
            reward += 50
            self.score += 50
        
        if terminated:
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_remaining": len(self.fruits),
            "time_remaining": self.timer,
        }

    # --- Helper and Rendering Methods ---

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        y = grid_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        return [x, y]

    def _generate_fruits(self):
        fruits = []
        occupied_positions = {tuple(self.player_grid_pos)}
        for _ in range(self.NUM_FRUITS):
            while True:
                pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
                if pos not in occupied_positions:
                    occupied_positions.add(pos)
                    break
            fruits.append({
                "pos": list(pos),
                "color": random.choice(self.FRUIT_COLORS)
            })
        return fruits
        
    def _find_nearest_fruit_dist(self):
        if not self.fruits:
            return None
        player_pos = np.array(self.player_grid_pos)
        fruit_positions = np.array([f["pos"] for f in self.fruits])
        distances = np.sum(np.abs(fruit_positions - player_pos), axis=1) # Manhattan distance
        return np.min(distances)
        
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw fruits
        fruit_radius = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.3)
        for fruit in self.fruits:
            pos = self._grid_to_pixel(fruit["pos"])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), fruit_radius, fruit["color"])
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), fruit_radius, fruit["color"])
            
        # Draw particles
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(life_ratio * 5)
            alpha = int(life_ratio * 255)
            color = (*p["color"], alpha)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

        # Draw player
        player_size = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.7)
        player_rect = pygame.Rect(0, 0, player_size, player_size)
        player_rect.center = (int(self.player_pixel_pos[0]), int(self.player_pixel_pos[1]))

        # Pulsing glow effect
        glow_size = player_size + 8 * (1 + math.sin(self.steps * 0.2))
        glow_alpha = 80 + 60 * (1 + math.sin(self.steps * 0.2))
        glow_rect = pygame.Rect(0, 0, int(glow_size), int(glow_size))
        glow_rect.center = player_rect.center
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, int(glow_alpha)), glow_surface.get_rect(), border_radius=4)
        self.screen.blit(glow_surface, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        timer_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_TIMER_LOW
        timer_text = self.font_medium.render(f"Time: {math.ceil(self.timer):02}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # Remaining fruit
        fruit_text = self.font_small.render(f"Fruit Remaining: {len(self.fruits)}", True, self.COLOR_TEXT)
        fruit_rect = fruit_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(fruit_text, fruit_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "LEVEL CLEAR!" if not self.fruits else "TIME UP!"
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
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
    # Note: Gymnasium's 'human' render mode is not used here; we manually handle display.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Collector")
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAYBACK MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                done = False

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()