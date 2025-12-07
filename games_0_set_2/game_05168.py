import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the slicer. Press space to slice vertically. "
        "Slice the fruit before they fall off the screen!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An arcade fruit-slicing game. Slice 30 fruits to win. "
        "If you miss 5 fruits, you lose."
    )

    # Frames auto-advance for smooth graphics and time-based gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 30
    LOSE_MISSES = 5
    MAX_STEPS = 1000
    CURSOR_SPEED = 20.0
    BASE_FRUIT_SPEED = 2.0
    FRUIT_SPAWN_PROB = 0.05

    # Colors
    COLOR_BG_TOP = (25, 20, 40)
    COLOR_BG_BOTTOM = (60, 40, 80)
    COLOR_CURSOR = (220, 220, 255)
    COLOR_SLICE_TRAIL = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    FRUIT_COLORS = [
        {'base': (220, 50, 50), 'highlight': (255, 120, 120)},  # Red
        {'base': (50, 200, 50), 'highlight': (120, 255, 120)},  # Green
        {'base': (255, 220, 50), 'highlight': (255, 255, 150)}, # Yellow
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set up headless Pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 36)
        self.font_info = pygame.font.SysFont("sans-serif", 24)
        
        # Pre-render background gradient for performance
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            pygame.draw.line(self.background, color, (0, y), (self.SCREEN_WIDTH, y))

        # Initialize state variables
        self.cursor_pos = None
        self.fruits = []
        self.particles = []
        self.slice_effects = []
        self.steps = 0
        self.score = 0
        self.fruits_sliced = 0
        self.fruits_missed = 0
        self.game_over = False
        self.last_space_press = False
        self.np_random = None

        # self.reset() is called by the environment wrapper, but we can call it for initialization
        
        # This check is for development and ensures compliance with the spec
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.fruits = []
        self.particles = []
        self.slice_effects = []
        
        self.steps = 0
        self.score = 0
        self.fruits_sliced = 0
        self.fruits_missed = 0
        self.game_over = False
        self.last_space_press = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for existing

        # --- Handle Input ---
        if movement == 1: # Up
            self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: # Down
            self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: # Left
            self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: # Right
            self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        space_pressed = space_held and not self.last_space_press
        self.last_space_press = space_held

        if space_pressed:
            # Sfx: Whoosh
            sliced_this_step = self._perform_slice()
            if sliced_this_step > 0:
                reward += sliced_this_step  # +1 for each fruit
                if sliced_this_step > 1:
                    reward += 2  # +2 bonus for multi-slice
            self.slice_effects.append({'x': self.cursor_pos[0], 'life': 10})

        # --- Update Game State ---
        self._update_fruits()
        self._update_particles()
        self._update_slice_effects()
        self._spawn_fruit()
        self.steps += 1

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.fruits_sliced >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.fruits_missed >= self.LOSE_MISSES:
            reward -= 100
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _perform_slice(self):
        sliced_count = 0
        remaining_fruits = []
        for fruit in self.fruits:
            if abs(fruit['pos'][0] - self.cursor_pos[0]) < fruit['radius']:
                # This fruit is sliced
                sliced_count += 1
                self.fruits_sliced += 1
                # Sfx: Splat
                self._create_particles(fruit['pos'], fruit['color']['base'])
            else:
                # This fruit remains
                remaining_fruits.append(fruit)
        
        self.fruits = remaining_fruits
        return sliced_count

    def _update_fruits(self):
        remaining_fruits = []
        current_speed = self.BASE_FRUIT_SPEED + (self.fruits_sliced // 10) * 0.05
        for fruit in self.fruits:
            fruit['pos'] += fruit['vel'] * current_speed
            if fruit['pos'][1] > self.SCREEN_HEIGHT + fruit['radius']:
                # This fruit is missed
                self.fruits_missed += 1
                # Sfx: Thud
            else:
                # This fruit remains
                remaining_fruits.append(fruit)
        self.fruits = remaining_fruits

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_slice_effects(self):
        for s in self.slice_effects:
            s['life'] -= 1
        self.slice_effects = [s for s in self.slice_effects if s['life'] > 0]

    def _spawn_fruit(self):
        if self.np_random.random() < self.FRUIT_SPAWN_PROB:
            radius = self.np_random.integers(20, 35)
            pos = np.array([
                self.np_random.uniform(radius, self.SCREEN_WIDTH - radius),
                -radius
            ], dtype=np.float32)
            vel = np.array([
                self.np_random.uniform(-0.5, 0.5),
                self.np_random.uniform(0.8, 1.2)
            ], dtype=np.float32)
            color_idx = self.np_random.integers(0, len(self.FRUIT_COLORS))
            color = self.FRUIT_COLORS[color_idx]
            self.fruits.append({'pos': pos, 'vel': vel, 'radius': radius, 'color': color})

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_sliced": self.fruits_sliced,
            "fruits_missed": self.fruits_missed,
        }

    def _render_game(self):
        # Render slice trails
        for s in self.slice_effects:
            alpha = int(255 * (s['life'] / 10))
            line_surface = pygame.Surface((2, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surface, self.COLOR_SLICE_TRAIL + (alpha,), (1, 0), (1, self.SCREEN_HEIGHT), 3)
            self.screen.blit(line_surface, (int(s['x'] - 1), 0))

        # Render fruits
        for fruit in self.fruits:
            x, y, r = int(fruit['pos'][0]), int(fruit['pos'][1]), fruit['radius']
            color, highlight = fruit['color']['base'], fruit['color']['highlight']
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
            # 2.5D highlight
            highlight_r = int(r * 0.5)
            highlight_pos = (x - int(r * 0.2), y - int(r * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], highlight_r, highlight)
            
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(p['life'] / 8))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        # Render cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx - 10, cy), (cx + 10, cy), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cx, cy - 10), (cx, cy + 10), 1)

    def _render_ui(self):
        # Score
        score_text = self.font_info.render(f"Sliced: {self.fruits_sliced}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_info.render(f"Missed: {self.fruits_missed}/{self.LOSE_MISSES}", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - miss_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            if self.fruits_sliced >= self.WIN_SCORE:
                msg = "YOU WIN!"
            elif self.fruits_missed >= self.LOSE_MISSES:
                msg = "GAME OVER"
            else:
                msg = "TIME'S UP!"
            
            end_text = self.font_main.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # In human mode, we need a real display.
        # If on a headless server, you might need to unset this
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array") # Still use rgb_array for obs
        
        # Setup human rendering window
        pygame.display.set_caption("Fruit Slicer")
        human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        print(GameEnv.user_guide)
        
        while not terminated and not truncated:
            # --- Human Controls ---
            movement = 0 # no-op
            space_held = 0
            shift_held = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
                
            action = [movement, space_held, shift_held]

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Frame Rate ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()

            env.clock.tick(30) # 30 FPS
            
        print(f"Game Over! Final Info: {info}")
        env.close()

    else: # For training (rgb_array)
        env = GameEnv()
        env.reset(seed=42)
        env.validate_implementation()
        obs, info = env.reset(seed=42)
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Info: {info}")
                obs, info = env.reset(seed=42)
        env.close()