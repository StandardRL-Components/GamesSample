
# Generated: 2025-08-28T01:59:09.241284
# Source Brief: brief_01565.md
# Brief Index: 1565

        
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
    user_guide = "Controls: ↑/↓ to move the slicer. Press space to slice."

    # Must be a short, user-facing description of the game:
    game_description = "Slice falling fruit to reach a target score before missing too many. Chain slices for combo bonuses!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.max_steps = 1000
        self.win_score = 50
        self.max_lives = 5
        
        self._setup_visuals()
        
        # Initialize state variables
        self.slicer_y = 0
        self.fruits = []
        self.particles = []
        self.slice_effects = []
        self.slicer_speed = 12
        self.fruit_spawn_timer = 0
        self.fruit_spawn_rate = 60
        self.base_fruit_fall_speed = 2.0
        self.fruit_fall_speed_increase = 0.05
        
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for submission, but can be used for testing
    
    def _setup_visuals(self):
        """Initializes colors and fonts for rendering."""
        self.COLOR_BG_TOP = (135, 206, 235)
        self.COLOR_BG_BOTTOM = (25, 25, 112)
        self.COLOR_SLICER = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_COMBO = (255, 215, 0)

        self.FRUIT_TYPES = {
            "strawberry": {"color": (220, 20, 60), "radius": 15, "points": 1},
            "lime": {"color": (50, 205, 50), "radius": 18, "points": 1},
            "orange": {"color": (255, 140, 0), "radius": 20, "points": 2},
            "blueberry": {"color": (72, 61, 139), "radius": 10, "points": 1},
            "banana": {"color": (255, 223, 0), "radius": 22, "points": 3},
        }

        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 32)
        except pygame.error:
            self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.max_lives
        self.game_over = False
        self.combo = 0
        
        self.slicer_y = self.screen_height // 2
        
        self.fruits = []
        self.particles = []
        self.slice_effects = []
        
        self.fruit_spawn_timer = 0
        self.fruit_spawn_rate = 60
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        movement = action[0]
        slice_action = action[1] == 1
        
        self._handle_input(movement, slice_action)
        self._update_slice_effects()
        
        slice_reward, fruits_sliced = self._handle_slicing() if slice_action else (0, 0)
        reward += slice_reward

        if fruits_sliced > 0:
            self.combo += fruits_sliced
            reward += 0.5 * self.combo
        elif slice_action:
            self.combo = 0 # Reset combo on a missed slice

        miss_penalty = self._update_fruits()
        reward += miss_penalty
        
        self._spawn_fruit()
        self._update_particles()
        
        # Continuous penalty for fruits on screen to encourage speed
        reward -= 0.01 * len(self.fruits)
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.win_score:
                reward += 100
            elif self.lives <= 0:
                reward -= 100
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        return (
            self.score >= self.win_score
            or self.lives <= 0
            or self.steps >= self.max_steps
        )

    def _handle_input(self, movement, slice_action):
        if movement == 1: # Up
            self.slicer_y = max(0, self.slicer_y - self.slicer_speed)
        elif movement == 2: # Down
            self.slicer_y = min(self.screen_height, self.slicer_y + self.slicer_speed)
            
        if slice_action:
            # sfx: whoosh
            self.slice_effects.append({"y": self.slicer_y, "alpha": 255})

    def _handle_slicing(self):
        reward = 0
        sliced_fruits_this_frame = 0
        
        for i in range(len(self.fruits) - 1, -1, -1):
            fruit = self.fruits[i]
            if abs(fruit["pos"][1] - self.slicer_y) < fruit["type_info"]["radius"]:
                # sfx: squish
                reward += 1.0
                self.score += fruit["type_info"]["points"]
                sliced_fruits_this_frame += 1
                
                self._create_explosion(fruit["pos"], fruit["type_info"]["color"])
                del self.fruits[i]
        
        return reward, sliced_fruits_this_frame

    def _update_fruits(self):
        miss_penalty = 0
        fall_speed = self.base_fruit_fall_speed + (self.steps // 50) * self.fruit_fall_speed_increase
        
        for i in range(len(self.fruits) - 1, -1, -1):
            fruit = self.fruits[i]
            fruit["pos"][1] += fall_speed
            
            if fruit["pos"][1] > self.screen_height + fruit["type_info"]["radius"]:
                # sfx: miss_sound
                self.lives -= 1
                miss_penalty -= 1.0
                self.combo = 0
                del self.fruits[i]
                
        return miss_penalty
        
    def _spawn_fruit(self):
        self.fruit_spawn_timer += 1
        if self.fruit_spawn_timer >= self.fruit_spawn_rate:
            self.fruit_spawn_timer = 0
            self.fruit_spawn_rate = max(20, 60 - self.steps // 100)
            
            fruit_key = random.choice(list(self.FRUIT_TYPES.keys()))
            type_info = self.FRUIT_TYPES[fruit_key]
            
            self.fruits.append({
                "pos": [random.randint(type_info["radius"], self.screen_width - type_info["radius"]), float(-type_info["radius"])],
                "type": fruit_key,
                "type_info": type_info
            })
    
    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifespan": random.randint(20, 40),
                "color": color,
                "radius": random.uniform(2, 5)
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                del self.particles[i]

    def _update_slice_effects(self):
        for i in range(len(self.slice_effects) - 1, -1, -1):
            effect = self.slice_effects[i]
            effect["alpha"] -= 25
            if effect["alpha"] <= 0:
                del self.slice_effects[i]
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = [int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
            
    def _render_game(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, p["color"], pos, int(p["radius"]))
            
        for fruit in self.fruits:
            pos = (int(fruit["pos"][0]), int(fruit["pos"][1]))
            color = fruit["type_info"]["color"]
            radius = fruit["type_info"]["radius"]
            
            if fruit["type"] == "banana":
                rect = pygame.Rect(pos[0] - radius, pos[1] - radius/2, radius*2, radius)
                pygame.draw.arc(self.screen, color, rect, math.pi * 0.2, math.pi * 0.8, 8)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        for effect in self.slice_effects:
            if effect["alpha"] > 0:
                s = pygame.Surface((self.screen_width, 5), pygame.SRCALPHA)
                s.fill((255, 255, 255, effect["alpha"]))
                self.screen.blit(s, (0, effect["y"] - 2))

        pygame.draw.line(self.screen, self.COLOR_SLICER, (0, self.slicer_y), (self.screen_width, self.slicer_y), 3)

    def _render_ui(self):
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        draw_text(f"Score: {self.score}", self.font_small, self.COLOR_TEXT, (10, 10))
        
        lives_text = f"Lives: {self.lives}"
        text_width = self.font_small.size(lives_text)[0]
        draw_text(lives_text, self.font_small, self.COLOR_TEXT, (self.screen_width - text_width - 10, 10))
        
        if self.combo > 1:
            combo_text = f"x{self.combo} COMBO!"
            text_width, text_height = self.font_large.size(combo_text)
            draw_text(combo_text, self.font_large, self.COLOR_COMBO, ((self.screen_width - text_width) // 2, 50))
            
        if self.game_over:
            msg, color = ("YOU WIN!", (0, 255, 0)) if self.score >= self.win_score else ("GAME OVER", (255, 0, 0))
            text_width, text_height = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, ((self.screen_width - text_width) // 2, (self.screen_height - text_height) // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Frame rate ---
        clock.tick(30) # Match the intended FPS

    print(f"Game Over! Final Info: {info}")
    env.close()