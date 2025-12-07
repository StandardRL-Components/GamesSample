import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:57:40.480481
# Source Brief: brief_00777.md
# Brief Index: 777
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Race against time to transform three shapes into their matching colored targets.
    
    The player controls three shapes (circle, square, triangle) and must move them
    to their corresponding colored targets. Transformation progress occurs when a shape
    is near its target and the player holds the 'transform' button (Space).
    
    Pulsating energy fields are scattered across the level. Entering a field that
    matches the shape's target color accelerates transformation speed, while other
    fields have no effect.
    
    The goal is to transform all three shapes before the 60-second timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race against time to move three shapes to their matching colored targets. "
        "Hold the transform button when near a target, and use energy fields to speed up the process."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the selected shape. Press Shift to cycle between shapes. "
        "Hold Space to transform a shape when it is near its matching target."
    )
    auto_advance = True
    
    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (35, 40, 60)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_SHAPE_INITIAL = (150, 150, 170)
    
    TARGET_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255)
    }
    
    # Game Parameters
    GAME_DURATION_SECONDS = 60
    FPS = 30
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    
    SHAPE_SIZE = 15
    SHAPE_SPEED = 5
    TARGET_PROXIMITY = 30
    
    BASE_TRANSFORM_RATE = 0.01  # Progress per step
    ENERGY_FIELD_BONUS = 0.03   # Added progress per step in matching field
    
    NUM_ENERGY_FIELDS = 5
    FIELD_MIN_RADIUS = 40
    FIELD_MAX_RADIUS = 70
    
    # Reward structure
    REWARD_WIN = 100
    REWARD_LOSS = -100
    REWARD_TRANSFORM_COMPLETE = 20
    REWARD_IN_GOOD_FIELD = 0.05
    REWARD_DISTANCE_FACTOR = 0.002 # Scaler for distance reward

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        self.render_mode = render_mode

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.shapes = []
        self.targets = []
        self.energy_fields = []
        self.selected_shape_idx = 0
        self.last_shift_press = False
        self.last_shape_distances = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS * self.FPS
        self.selected_shape_idx = 0
        self.last_shift_press = False
        
        shape_types = ['circle', 'square', 'triangle']
        color_names = list(self.TARGET_COLORS.keys())
        self.np_random.shuffle(shape_types)
        self.np_random.shuffle(color_names)

        self.shapes = []
        self.targets = []
        
        spawn_margin = 50
        
        for i in range(3):
            shape_type = shape_types[i]
            target_color_name = color_names[i]
            
            shape_pos = np.array([
                self.np_random.uniform(spawn_margin, self.SCREEN_WIDTH - spawn_margin),
                self.np_random.uniform(spawn_margin, self.SCREEN_HEIGHT - spawn_margin)
            ])
            
            target_pos = np.array([
                self.np_random.uniform(spawn_margin, self.SCREEN_WIDTH - spawn_margin),
                self.np_random.uniform(spawn_margin, self.SCREEN_HEIGHT - spawn_margin)
            ])

            self.shapes.append({
                "id": i,
                "type": shape_type,
                "pos": shape_pos,
                "color": self.COLOR_SHAPE_INITIAL,
                "target_color_name": target_color_name,
                "transform_progress": 0.0,
                "is_transformed": False,
            })
            
            self.targets.append({
                "id": i,
                "type": shape_type,
                "pos": target_pos,
                "color_name": target_color_name
            })
        
        self.last_shape_distances = {i: self._get_distance_to_target(i) for i in range(3)}

        self.energy_fields = []
        for _ in range(self.NUM_ENERGY_FIELDS):
            field_pos = np.array([
                self.np_random.uniform(spawn_margin, self.SCREEN_WIDTH - spawn_margin),
                self.np_random.uniform(spawn_margin, self.SCREEN_HEIGHT - spawn_margin)
            ])
            self.energy_fields.append({
                "pos": field_pos,
                "color_name": self.np_random.choice(list(self.TARGET_COLORS.keys())),
                "pulse_phase": self.np_random.uniform(0, 2 * math.pi)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        self.timer -= 1
        reward = 0

        # --- Handle Actions ---
        # Cycle selected shape on SHIFT PRESS (not hold)
        if shift_held and not self.last_shift_press:
            self.selected_shape_idx = (self.selected_shape_idx + 1) % len(self.shapes)
            # sfx: UI_switch.wav
        self.last_shift_press = shift_held

        # Move the selected shape
        selected_shape = self.shapes[self.selected_shape_idx]
        if not selected_shape["is_transformed"]:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            selected_shape["pos"][0] += dx * self.SHAPE_SPEED
            selected_shape["pos"][1] += dy * self.SHAPE_SPEED
            
            # Clamp position to screen bounds
            selected_shape["pos"][0] = np.clip(selected_shape["pos"][0], self.SHAPE_SIZE, self.SCREEN_WIDTH - self.SHAPE_SIZE)
            selected_shape["pos"][1] = np.clip(selected_shape["pos"][1], self.SHAPE_SIZE, self.SCREEN_HEIGHT - self.SHAPE_SIZE)

        # --- Update Game Logic & Calculate Rewards ---
        
        # Distance-based rewards for all shapes
        for i in range(len(self.shapes)):
            if not self.shapes[i]["is_transformed"]:
                current_dist = self._get_distance_to_target(i)
                dist_change = self.last_shape_distances[i] - current_dist
                reward += dist_change * self.REWARD_DISTANCE_FACTOR
                self.last_shape_distances[i] = current_dist

        # Transformation logic
        for shape in self.shapes:
            if shape["is_transformed"]:
                continue

            dist_to_target = self._get_distance_to_target(shape["id"])
            
            if space_held and dist_to_target <= self.TARGET_PROXIMITY:
                # sfx: transform_loop.wav
                transform_rate = self.BASE_TRANSFORM_RATE
                
                # Check for energy field bonus
                for field in self.energy_fields:
                    dist_to_field = np.linalg.norm(shape["pos"] - field["pos"])
                    pulse_radius = self._get_pulse_value(field["pulse_phase"], self.FIELD_MIN_RADIUS, self.FIELD_MAX_RADIUS)
                    if dist_to_field < pulse_radius and field["color_name"] == shape["target_color_name"]:
                        transform_rate += self.ENERGY_FIELD_BONUS
                        reward += self.REWARD_IN_GOOD_FIELD
                        break # Only one field can apply bonus at a time
                
                shape["transform_progress"] = min(1.0, shape["transform_progress"] + transform_rate)
                
                if shape["transform_progress"] >= 1.0:
                    shape["is_transformed"] = True
                    shape["color"] = self.TARGET_COLORS[shape["target_color_name"]]
                    reward += self.REWARD_TRANSFORM_COMPLETE
                    self.score += 100
                    # sfx: transform_complete.wav
        
        # --- Check Termination Conditions ---
        terminated = False
        all_transformed = all(s["is_transformed"] for s in self.shapes)
        
        if all_transformed:
            reward += self.REWARD_WIN
            self.score += 500 # Bonus score
            terminated = True
            self.game_over = True
            # sfx: game_win.wav
        
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            if not all_transformed:
                reward += self.REWARD_LOSS
            terminated = True
            self.game_over = True
            # sfx: game_lose.wav

        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "timer_seconds": self.timer / self.FPS,
            "shapes_transformed": sum(1 for s in self.shapes if s["is_transformed"])
        }
        
    def _get_distance_to_target(self, shape_id):
        return np.linalg.norm(self.shapes[shape_id]["pos"] - self.targets[shape_id]["pos"])

    def _lerp_color(self, c1, c2, t):
        t = np.clip(t, 0.0, 1.0)
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )
        
    def _get_pulse_value(self, phase, min_val, max_val):
        # A sine wave that moves between min and max
        return min_val + (max_val - min_val) * (0.5 * (math.sin(phase + self.steps * 0.05) + 1))

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw energy fields
        for field in self.energy_fields:
            radius = self._get_pulse_value(field["pulse_phase"], self.FIELD_MIN_RADIUS, self.FIELD_MAX_RADIUS)
            alpha = self._get_pulse_value(field["pulse_phase"] + math.pi, 20, 60)
            color = self.TARGET_COLORS[field["color_name"]]
            pygame.gfxdraw.filled_circle(self.screen, int(field["pos"][0]), int(field["pos"][1]), int(radius), (*color, int(alpha)))
            pygame.gfxdraw.aacircle(self.screen, int(field["pos"][0]), int(field["pos"][1]), int(radius), (*color, int(alpha*1.5)))

        # Draw targets
        for target in self.targets:
            color = self.TARGET_COLORS[target["color_name"]]
            self._draw_shape(self.screen, target["type"], target["pos"], self.SHAPE_SIZE, color, width=3, is_target=True)
            
        # Draw shapes
        for i, shape in enumerate(self.shapes):
            target_color = self.TARGET_COLORS[shape["target_color_name"]]
            if not shape["is_transformed"]:
                shape["color"] = self._lerp_color(self.COLOR_SHAPE_INITIAL, target_color, shape["transform_progress"])
            
            self._draw_shape(self.screen, shape["type"], shape["pos"], self.SHAPE_SIZE, shape["color"])
            
            # Draw selector outline
            if i == self.selected_shape_idx and not shape["is_transformed"]:
                self._draw_shape(self.screen, shape["type"], shape["pos"], self.SHAPE_SIZE + 4, self.COLOR_WHITE, width=2)

    def _draw_shape(self, surface, shape_type, pos, size, color, width=0, is_target=False):
        x, y = int(pos[0]), int(pos[1])
        
        if is_target:
            # Draw a faint filled version for target area visualization
            proximity_color = (*color, 20)
            pygame.gfxdraw.filled_circle(surface, x, y, self.TARGET_PROXIMITY, proximity_color)
            pygame.gfxdraw.aacircle(surface, x, y, self.TARGET_PROXIMITY, proximity_color)

        if shape_type == 'circle':
            if width == 0:
                pygame.gfxdraw.filled_circle(surface, x, y, size, color)
            pygame.gfxdraw.aacircle(surface, x, y, size, color)
            if width > 1:
                pygame.gfxdraw.aacircle(surface, x, y, size - (width-1), color)
        
        elif shape_type == 'square':
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.rect(surface, color, rect, width)
        
        elif shape_type == 'triangle':
            points = [
                (x, y - size),
                (x - size, y + size * 0.7),
                (x + size, y + size * 0.7)
            ]
            if width == 0:
                pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_ui(self):
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"{time_left:.1f}"
        timer_color = self.COLOR_UI_TEXT if time_left > 10 else self.TARGET_COLORS["red"]
        timer_surf = self.font_large.render(timer_text, True, timer_color)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 10))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Transformed count
        transformed_count = sum(1 for s in self.shapes if s["is_transformed"])
        count_text = f"TRANSFORMED: {transformed_count} / 3"
        count_surf = self.font_small.render(count_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(count_surf, (20, 35))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Transformer")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("----------------\n")
    
    while not terminated and not truncated:
        # Action defaults
        movement = 0  # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        if keys[pygame.K_r]:
            print("Resetting environment...")
            obs, info = env.reset()
            total_reward = 0
            continue

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()