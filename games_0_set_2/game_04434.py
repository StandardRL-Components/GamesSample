import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the slicer. Press space to slice falling fruit."
    )

    game_description = (
        "Slice falling fruit to score points. Reach 100 points to win, but miss 5 fruits and you lose."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (10, 20, 30)
    COLOR_BG_BOTTOM = (40, 50, 60)
    COLOR_SLICER_TRAIL = (255, 255, 255)
    COLOR_SLICER_FLASH = (255, 255, 150)
    
    COLOR_APPLE = (124, 252, 0)
    COLOR_ORANGE = (255, 165, 0)
    COLOR_WATERMELON = (255, 69, 90)
    COLOR_FRUIT_SHADOW = (0, 0, 0, 50)
    COLOR_FRUIT_HIGHLIGHT = (255, 255, 255, 100)

    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    COLOR_WIN = (173, 255, 47)
    COLOR_LOSE = (255, 69, 0)

    # Game parameters
    SLICER_SPEED = 12
    TRAIL_LENGTH = 15
    MAX_STEPS = 1000
    WIN_SCORE = 100
    MAX_MISSES = 5
    
    FRUIT_TYPES = {
        "apple": {"color": COLOR_APPLE, "value": 1, "radius": 15, "reward": 0.2},
        "orange": {"color": COLOR_ORANGE, "value": 2, "radius": 18, "reward": 0.5},
        "watermelon": {"color": COLOR_WATERMELON, "value": 5, "radius": 25, "reward": 1.0},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game state variables are initialized in reset()
        self.slicer_pos = None
        self.slicer_trail = None
        self.is_slicing_this_frame = None
        self.last_space_held = None
        self.fruits = None
        self.particles = None
        self.score_popups = None
        self.steps = None
        self.score = None
        self.missed_fruits = None
        self.game_over = None
        self.win_condition = None
        self.initial_fall_speed = None
        self.initial_spawn_prob = None
        self.fall_speed_multiplier = None
        self.spawn_prob_multiplier = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.slicer_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.slicer_trail = deque(maxlen=self.TRAIL_LENGTH)
        self.is_slicing_this_frame = False
        self.last_space_held = False
        
        self.fruits = []
        self.particles = []
        self.score_popups = []

        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.game_over = False
        self.win_condition = False
        
        self.initial_fall_speed = 1.0
        self.initial_spawn_prob = 1.0 / 20.0
        self.fall_speed_multiplier = 1.0
        self.spawn_prob_multiplier = 1.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.is_slicing_this_frame = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- 1. Handle Input & Update Slicer ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        moved = False
        if movement == 1: self.slicer_pos.y -= self.SLICER_SPEED; moved = True
        elif movement == 2: self.slicer_pos.y += self.SLICER_SPEED; moved = True
        elif movement == 3: self.slicer_pos.x -= self.SLICER_SPEED; moved = True
        elif movement == 4: self.slicer_pos.x += self.SLICER_SPEED; moved = True

        if moved:
            reward -= 0.01 # Small penalty for movement

        self.slicer_pos.x = np.clip(self.slicer_pos.x, 0, self.SCREEN_WIDTH)
        self.slicer_pos.y = np.clip(self.slicer_pos.y, 0, self.SCREEN_HEIGHT)
        self.slicer_trail.append(pygame.Vector2(self.slicer_pos))

        # --- 2. Handle Slicing Action ---
        is_slicing_action = space_held and not self.last_space_held
        if is_slicing_action:
            self.is_slicing_this_frame = True
            fruits_sliced_this_frame = 0
            
            for fruit in self.fruits[:]:
                dist = self.slicer_pos.distance_to(fruit["pos"])
                if dist < fruit["radius"]:
                    self.score += fruit["value"]
                    reward += 0.1 + fruit["reward"] # Base slice reward + fruit specific
                    self._create_splash(fruit)
                    self._create_score_popup(fruit)
                    self.fruits.remove(fruit)
                    fruits_sliced_this_frame += 1
            
        self.last_space_held = space_held

        # --- 3. Update Game Entities ---
        self._update_fruits()
        self._update_particles()
        self._update_score_popups()
        
        # --- 4. Update Difficulty ---
        if self.steps > 0 and self.steps % 50 == 0:
            self.fall_speed_multiplier += 0.01
        if self.steps > 0 and self.steps % 100 == 0:
            self.spawn_prob_multiplier += 0.005

        # --- 5. Check Termination Conditions ---
        terminated = False
        truncated = False
        
        # Check for missed fruits and apply penalty
        for fruit in self.fruits[:]:
            if fruit["pos"].y > self.SCREEN_HEIGHT + fruit["radius"]:
                self.missed_fruits += 1
                reward -= 1.0
                self.fruits.remove(fruit)

        if self.score >= self.WIN_SCORE:
            reward += 100
            self.win_condition = True
            terminated = True
        if self.missed_fruits >= self.MAX_MISSES:
            reward -= 100
            terminated = True
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._draw_background()
        self._draw_fruits()
        self._draw_particles()
        self._draw_slicer()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "missed": self.missed_fruits}

    # --- Update Logic ---
    def _update_fruits(self):
        # Spawn new fruit
        spawn_chance = self.initial_spawn_prob * self.spawn_prob_multiplier
        if self.np_random.random() < spawn_chance:
            fruit_type_name = self.np_random.choice(list(self.FRUIT_TYPES.keys()))
            fruit_info = self.FRUIT_TYPES[fruit_type_name]
            self.fruits.append({
                "type": fruit_type_name,
                "pos": pygame.Vector2(self.np_random.integers(50, self.SCREEN_WIDTH - 50), -30),
                "radius": fruit_info["radius"],
                "value": fruit_info["value"],
                "color": fruit_info["color"],
                "reward": fruit_info["reward"],
                "angle": self.np_random.uniform(0, 360),
                "rotation_speed": self.np_random.uniform(-2, 2)
            })
        
        # Move existing fruits
        for fruit in self.fruits:
            fall_speed = self.initial_fall_speed * self.fall_speed_multiplier
            fruit["pos"].y += fall_speed
            fruit["angle"] += fruit["rotation_speed"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1  # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)
    
    def _update_score_popups(self):
        for s in self.score_popups[:]:
            s["pos"].y -= 0.5
            s["lifetime"] -= 1
            if s["lifetime"] <= 0:
                self.score_popups.remove(s)

    # --- Helper Creation Methods ---
    def _create_splash(self, fruit):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pygame.Vector2(fruit["pos"]),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "lifetime": self.np_random.integers(20, 40),
                "color": fruit["color"],
                "radius": self.np_random.uniform(2, 5)
            })

    def _create_score_popup(self, fruit):
        self.score_popups.append({
            "pos": pygame.Vector2(fruit["pos"]),
            "text": f"+{fruit['value']}",
            "lifetime": 40,
            "color": self.COLOR_TEXT,
        })

    # --- Drawing Methods ---
    def _draw_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_fruits(self):
        for fruit in sorted(self.fruits, key=lambda f: f['pos'].y):
            pos_x, pos_y = int(fruit["pos"].x), int(fruit["pos"].y)
            radius = fruit["radius"]
            
            # Shadow
            shadow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, self.COLOR_FRUIT_SHADOW, (0, 0, radius * 2, radius))
            self.screen.blit(shadow_surface, (pos_x - radius, pos_y + radius // 2))

            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, fruit["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, fruit["color"])
            
            # Highlight
            highlight_pos_x = pos_x - radius // 3
            highlight_pos_y = pos_y - radius // 3
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos_x, highlight_pos_y, radius // 3, self.COLOR_FRUIT_HIGHLIGHT)

    def _draw_slicer(self):
        if len(self.slicer_trail) > 1:
            points = [p for p in self.slicer_trail]
            for i in range(len(points) - 1):
                start_pos = points[i]
                end_pos = points[i+1]
                
                alpha = int(255 * (i / self.TRAIL_LENGTH))
                color = (self.COLOR_SLICER_TRAIL[0], self.COLOR_SLICER_TRAIL[1], self.COLOR_SLICER_TRAIL[2], alpha)
                
                width = 5 if self.is_slicing_this_frame else 2
                
                center_x = (start_pos.x + end_pos.x) / 2
                center_y = (start_pos.y + end_pos.y) / 2
                length = start_pos.distance_to(end_pos)
                angle = math.atan2(start_pos.y - end_pos.y, start_pos.x - end_pos.x) * (180 / math.pi)
                
                if length > 0:
                    line_surf = pygame.Surface((length, width), pygame.SRCALPHA)
                    line_surf.fill(color)
                    line_surf = pygame.transform.rotate(line_surf, angle)
                    rect = line_surf.get_rect(center=(center_x, center_y))
                    self.screen.blit(line_surf, rect)

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["lifetime"] / 40))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((int(p["radius"]*2), int(p["radius"]*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))

    def _draw_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _draw_ui(self):
        # Score
        self._draw_text(f"Score: {self.score}", self.font_small, self.COLOR_TEXT, (10, 10))
        
        # Misses
        miss_text = f"Missed: {self.missed_fruits}/{self.MAX_MISSES}"
        text_width = self.font_small.size(miss_text)[0]
        self._draw_text(miss_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

        # Score popups
        for s in self.score_popups:
            alpha = max(0, 255 * (s["lifetime"] / 40))
            color = (*s["color"], alpha)
            text_surf = self.font_small.render(s["text"], True, color)
            text_rect = text_surf.get_rect(center=(int(s["pos"].x), int(s["pos"].y)))
            self.screen.blit(text_surf, text_rect)

        # Game Over message
        if self.game_over:
            if self.win_condition:
                msg, color = "YOU WIN!", self.COLOR_WIN
            else:
                msg, color = "GAME OVER", self.COLOR_LOSE
            
            text_width, text_height = self.font_large.size(msg)
            pos = ((self.SCREEN_WIDTH - text_width) / 2, (self.SCREEN_HEIGHT - text_height) / 2)
            self._draw_text(msg, self.font_large, color, pos)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This part is for manual testing and visualization.
    # It requires a display.
    render_mode = "human"
    try:
        # Re-initialize pygame with default video driver for display
        os.environ.pop("SDL_VIDEODRIVER", None)
        import pygame
        pygame.quit() # Quit the dummy driver instance
        pygame.init() # Initialize with default driver
        
        env = GameEnv()
        obs, info = env.reset()
        
        render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Fruit Slicer")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        # Restore the dummy driver setting for the env's internal pygame instance
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        while not done:
            # --- Action Mapping for Human Play ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # --- Render to Display ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Frame Rate ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(30)
            
        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    except (pygame.error, ImportError) as e:
        print(f"Pygame display error: {e}")
        print("Could not start interactive game session. This is expected in a headless environment.")
        print("The environment itself is still valid and can be used by an RL agent.")
    
    finally:
        if 'env' in locals():
            env.close()
        pygame.quit()