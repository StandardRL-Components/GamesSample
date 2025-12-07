
# Generated: 2025-08-27T15:48:24.882431
# Source Brief: brief_01084.md
# Brief Index: 1084

        
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
        "Controls: Arrow keys to move the cursor. "
        "Press space to slice. Hold shift to use a powerful area slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit with precise timed actions to achieve a high score. "
        "Chain slices for combos and use your power slice wisely!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_MISS = (255, 80, 80)
        self.FRUIT_COLORS = {
            "apple": (220, 40, 40),
            "banana": (240, 220, 80),
            "kiwi": (100, 180, 70),
            "orange": (255, 165, 0),
            "grape": (128, 0, 128),
        }
        
        # Game constants
        self.MAX_STEPS = 1500
        self.WIN_SCORE = 1000
        self.MAX_MISSES = 5
        self.CURSOR_SPEED = 10
        self.SLICE_RADIUS = 30
        self.POWER_SLICE_RADIUS = 80
        self.POWER_SLICE_COST = 50
        
        # Initialize state variables
        self.np_random = None
        self.cursor_pos = None
        self.fruits = None
        self.particles = None
        self.score = None
        self.misses = None
        self.steps = None
        self.combo_count = None
        self.power_meter = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.base_fruit_speed = None
        self.fruit_spawn_timer = None
        self.game_over_message = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.fruits = []
        self.particles = []
        
        self.score = 0
        self.misses = 0
        self.steps = 0
        self.combo_count = 0
        self.power_meter = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.base_fruit_speed = 2.0
        self.fruit_spawn_timer = 0
        self.game_over_message = None
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for existing to encourage action
        
        # --- Handle "Just Pressed" Logic ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Update Game Logic ---
        self.steps += 1
        
        self._update_cursor(movement)
        self._update_fruits()
        self._update_particles()
        self._spawn_fruit()
        
        sliced_something = False
        
        # --- Handle Actions ---
        if space_pressed:
            # SFX: Sword_Slice.wav
            num_sliced = self._perform_slice(self.SLICE_RADIUS)
            if num_sliced > 0:
                reward += 1 * num_sliced
                if self.combo_count == num_sliced: # Started a combo
                    reward += 5
                else: # Continued a combo
                    reward += 2 * num_sliced
                sliced_something = True
            else: # Missed slice
                self.combo_count = 0
        
        if shift_pressed and self.power_meter >= self.POWER_SLICE_COST:
            # SFX: Power_Slice_Activate.wav
            self.power_meter -= self.POWER_SLICE_COST
            num_sliced = self._perform_slice(self.POWER_SLICE_RADIUS, is_power_slice=True)
            if num_sliced > 0:
                reward += 1 * num_sliced
                if self.combo_count == num_sliced:
                    reward += 5
                else:
                    reward += 2 * num_sliced
                sliced_something = True
            else: # Missed power slice
                reward -= 1
                self.combo_count = 0
        
        if not sliced_something:
            self.combo_count = 0
            
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 100 == 0:
            self.base_fruit_speed += 0.05
            
        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over_message = "YOU WIN!"
        elif self.misses >= self.MAX_MISSES:
            reward -= 100
            terminated = True
            self.game_over_message = "GAME OVER"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_message = "TIME'S UP"
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _perform_slice(self, radius, is_power_slice=False):
        sliced_fruits = []
        
        # Create slice visual effect
        effect_color = (255, 255, 100) if is_power_slice else (255, 255, 255)
        self.particles.append({
            "pos": self.cursor_pos.copy(), "type": "shockwave",
            "radius": 10, "max_radius": radius, "lifetime": 10, "color": effect_color
        })
        
        for fruit in self.fruits:
            dist = np.linalg.norm(self.cursor_pos - fruit["pos"])
            if dist <= radius:
                sliced_fruits.append(fruit)

        if not sliced_fruits:
            return 0
        
        # SFX: Fruit_Splat.wav
        for fruit in sliced_fruits:
            self.score += 10 + self.combo_count * 2
            self.power_meter = min(100, self.power_meter + 5)
            self.combo_count += 1
            
            # Create two half-fruit particles
            for _ in range(2):
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(2, 4)
                self.particles.append({
                    "pos": fruit["pos"].copy(), "vel": vel, "type": "half_fruit",
                    "lifetime": 60, "color": fruit["color"], "radius": fruit["radius"] * 0.7
                })

        # Remove sliced fruits from main list
        self.fruits = [f for f in self.fruits if f not in sliced_fruits]
        return len(sliced_fruits)
        
    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
    def _update_fruits(self):
        fruits_to_keep = []
        for fruit in self.fruits:
            fruit["pos"] += fruit["vel"]
            if fruit["pos"][1] < self.HEIGHT + fruit["radius"]:
                fruits_to_keep.append(fruit)
            else:
                # SFX: Miss.wav
                self.misses += 1
                self.combo_count = 0
        self.fruits = fruits_to_keep
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["lifetime"] -= 1
            if p["type"] == "half_fruit":
                p["pos"] += p["vel"]
                p["vel"][1] += 0.1 # Gravity
            elif p["type"] == "shockwave":
                p["radius"] += (p["max_radius"] - 10) / 10
                
    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            x_pos = self.np_random.uniform(50, self.WIDTH - 50)
            x_vel = self.np_random.uniform(-1, 1)
            y_vel = self.base_fruit_speed + self.np_random.uniform(0, 1)
            
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            radius = 15 if fruit_type != "banana" else 20
            
            self.fruits.append({
                "pos": np.array([x_pos, -radius], dtype=np.float32),
                "vel": np.array([x_vel, y_vel], dtype=np.float32),
                "type": fruit_type,
                "color": self.FRUIT_COLORS[fruit_type],
                "radius": radius
            })
            self.fruit_spawn_timer = max(15, 60 - self.steps // 20)
            
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        if self.game_over_message:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render fruits
        for fruit in self.fruits:
            pos = fruit["pos"].astype(int)
            radius = int(fruit["radius"])
            color = fruit["color"]
            
            if fruit["type"] == "banana":
                # Draw a simple curved shape for banana
                rect = pygame.Rect(pos[0] - radius, pos[1] - radius // 2, radius * 2, radius)
                pygame.draw.arc(self.screen, color, rect, 0.5, 2.8, radius // 2)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (0,0,0,50))

        # Render particles
        for p in self.particles:
            pos = p["pos"].astype(int)
            alpha = int(255 * (p["lifetime"] / 60)) if p["type"] == 'half_fruit' else int(255 * (p["lifetime"] / 10))
            if p["type"] == "half_fruit":
                radius = int(p["radius"] * (p["lifetime"] / 60))
                if radius > 1:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*p["color"], alpha))
            elif p["type"] == "shockwave":
                radius = int(p["radius"])
                if radius > 1:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*p["color"], alpha))
                    if radius > 2:
                        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius-1, (*p["color"], alpha))
        
        # Render cursor
        c_pos = self.cursor_pos.astype(int)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, c_pos, 4, 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (c_pos[0] - 8, c_pos[1]), (c_pos[0] + 8, c_pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (c_pos[0], c_pos[1] - 8), (c_pos[0], c_pos[1] + 8), 2)
        
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))
        
        # Misses
        miss_text = self.font_small.render(f"MISSES: {self.misses}/{self.MAX_MISSES}", True, self.COLOR_MISS)
        self.screen.blit(miss_text, (self.WIDTH - miss_text.get_width() - 10, 10))
        
        # Power Meter
        power_bar_width = 150
        power_bar_height = 15
        power_fill = (self.power_meter / 100) * power_bar_width
        
        power_bar_rect = pygame.Rect((self.WIDTH - power_bar_width) // 2, 10, power_bar_width, power_bar_height)
        power_fill_rect = pygame.Rect((self.WIDTH - power_bar_width) // 2, 10, power_fill, power_bar_height)
        
        pygame.draw.rect(self.screen, (50, 50, 80), power_bar_rect, border_radius=4)
        if power_fill > 0:
            color = (255, 255, 0) if self.power_meter < self.POWER_SLICE_COST else (0, 255, 255)
            pygame.draw.rect(self.screen, color, power_fill_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, power_bar_rect, 2, border_radius=4)

        # Combo
        if self.combo_count > 1:
            combo_text = self.font_large.render(f"x{self.combo_count}", True, self.COLOR_SCORE)
            pos = (self.WIDTH // 2 - combo_text.get_width() // 2, self.HEIGHT // 2)
            self.screen.blit(combo_text, pos)
    
    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        text = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "combo": self.combo_count,
            "power_meter": self.power_meter,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()

        # --- Frame rate ---
        clock.tick(30)
        
    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()