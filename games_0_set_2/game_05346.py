
# Generated: 2025-08-28T04:44:51.544177
# Source Brief: brief_05346.md
# Brief Index: 5346

        
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
        "Controls: Use arrow keys (↑↓←→) to move your slicer. Press 'space' to perform a slice."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice the falling fruits to score points. Avoid the bombs! The game gets faster as you slice more fruit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_SLICER = (100, 255, 255)
    COLOR_TRAIL = (255, 255, 255)
    COLOR_BOMB = (20, 20, 20)
    COLOR_FUSE = (255, 255, 200)
    COLOR_SPARK = (255, 180, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (30, 30, 50)
    FRUIT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (255, 255, 80),  # Yellow
        (255, 150, 50),  # Orange
    ]

    # Game parameters
    SLICER_SPEED = 12
    SLICE_WIDTH = 120
    SLICE_DURATION = 4 # frames
    
    WIN_CONDITION_FRUITS = 30
    LOSE_CONDITION_BOMBS = 3
    MAX_STEPS = 1000 * FPS // 30 # Scale max steps if FPS changes

    INITIAL_SPAWN_INTERVAL = 2.0 # seconds
    INITIAL_BOMB_PROB = 0.1
    DIFFICULTY_TIER_FRUITS = 10
    SPAWN_INTERVAL_DECREMENT = 0.2 # seconds
    BOMB_PROB_INCREMENT = 0.05
    MIN_SPAWN_INTERVAL = 0.5 # seconds

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
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        
        self.np_random = None
        self.slicer_pos = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.slice_trails = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.fruits_sliced = None
        self.bombs_hit = None
        self.spawn_timer = None
        self.spawn_interval = None
        self.bomb_probability = None
        self.last_space_held = None
        
        self.reset()
        
        # This can be commented out for performance but is good for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.slicer_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.slice_trails = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fruits_sliced = 0
        self.bombs_hit = 0
        
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL * self.FPS
        self.bomb_probability = self.INITIAL_BOMB_PROB
        self.spawn_timer = self.spawn_interval
        
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1

            # 1. Handle Input & Slicer Movement
            self._move_slicer(movement)
            space_pressed = space_held and not self.last_space_held
            self.last_space_held = space_held

            # 2. Update Game Logic
            self._update_spawner()
            self._update_entities()

            # 3. Handle Slicing Action
            if space_pressed:
                slice_reward = self._perform_slice()
                reward += slice_reward
                # sfx: slice_whoosh

            # 4. Check for Termination
            win = self.fruits_sliced >= self.WIN_CONDITION_FRUITS
            lose = self.bombs_hit >= self.LOSE_CONDITION_BOMBS
            timeout = self.steps >= self.MAX_STEPS

            if win or lose or timeout:
                terminated = True
                self.game_over = True
                if win:
                    reward += 50
                if lose:
                    reward -= 50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_slicer(self, movement):
        if movement == 1: # Up
            self.slicer_pos[1] -= self.SLICER_SPEED
        elif movement == 2: # Down
            self.slicer_pos[1] += self.SLICER_SPEED
        elif movement == 3: # Left
            self.slicer_pos[0] -= self.SLICER_SPEED
        elif movement == 4: # Right
            self.slicer_pos[0] += self.SLICER_SPEED

        self.slicer_pos[0] = np.clip(self.slicer_pos[0], 0, self.SCREEN_WIDTH)
        self.slicer_pos[1] = np.clip(self.slicer_pos[1], 0, self.SCREEN_HEIGHT)

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.spawn_interval
            
            is_bomb = self.np_random.random() < self.bomb_probability
            pos = np.array([self.np_random.uniform(50, self.SCREEN_WIDTH - 50), -20.0])
            vel = np.array([self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(2.0, 4.5)])
            
            if is_bomb:
                self.bombs.append({"pos": pos, "vel": vel, "radius": 15, "fuse_timer": self.FPS / 2})
            else:
                self.fruits.append({
                    "pos": pos, 
                    "vel": vel, 
                    "radius": self.np_random.uniform(18, 25),
                    "color": random.choice(self.FRUIT_COLORS)
                })

    def _update_entities(self):
        # Update fruits
        for fruit in self.fruits:
            fruit["pos"] += fruit["vel"]
        self.fruits = [f for f in self.fruits if f["pos"][1] < self.SCREEN_HEIGHT + 50]

        # Update bombs
        for bomb in self.bombs:
            bomb["pos"] += bomb["vel"]
            bomb["fuse_timer"] -= 1
            if bomb["fuse_timer"] <= 0:
                bomb["fuse_timer"] = self.np_random.uniform(0.5, 1.5) * self.FPS
                self._create_particles(bomb["pos"] + np.array([0, -bomb["radius"]]), 1, self.COLOR_SPARK, 5, 2, 10)
        self.bombs = [b for b in self.bombs if b["pos"][1] < self.SCREEN_HEIGHT + 50]

        # Update particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

        # Update slice trails
        for trail in self.slice_trails:
            trail["lifetime"] -= 1
        self.slice_trails = [t for t in self.slice_trails if t["lifetime"] > 0]

    def _perform_slice(self):
        reward = 0
        slice_y = self.slicer_pos[1]
        slice_x1 = self.slicer_pos[0] - self.SLICE_WIDTH / 2
        slice_x2 = self.slicer_pos[0] + self.SLICE_WIDTH / 2
        
        self.slice_trails.append({"center": self.slicer_pos.copy(), "width": self.SLICE_WIDTH, "lifetime": self.SLICE_DURATION})

        # Check fruit collisions
        sliced_fruits = []
        for fruit in self.fruits:
            if abs(fruit["pos"][1] - slice_y) < fruit["radius"]:
                if fruit["pos"][0] > slice_x1 and fruit["pos"][0] < slice_x2:
                    sliced_fruits.append(fruit)
        
        for fruit in sliced_fruits:
            self.fruits.remove(fruit)
            self.fruits_sliced += 1
            self.score += 1
            reward += 1
            self._create_particles(fruit["pos"], 20, fruit["color"], 15, 3, 20)
            self._update_difficulty()
            # sfx: fruit_slice

        # Check bomb collisions
        hit_bombs = []
        for bomb in self.bombs:
            if abs(bomb["pos"][1] - slice_y) < bomb["radius"]:
                if bomb["pos"][0] > slice_x1 and bomb["pos"][0] < slice_x2:
                    hit_bombs.append(bomb)

        for bomb in hit_bombs:
            self.bombs.remove(bomb)
            self.bombs_hit += 1
            reward -= 5
            self._create_particles(bomb["pos"], 40, self.COLOR_SPARK, 25, 5, 30)
            self._create_particles(bomb["pos"], 20, self.COLOR_BOMB, 20, 3, 40)
            # sfx: bomb_explode
            
        return reward

    def _update_difficulty(self):
        if self.fruits_sliced > 0 and self.fruits_sliced % self.DIFFICULTY_TIER_FRUITS == 0:
            new_interval = self.spawn_interval - (self.SPAWN_INTERVAL_DECREMENT * self.FPS)
            self.spawn_interval = max(new_interval, self.MIN_SPAWN_INTERVAL * self.FPS)
            self.bomb_probability = min(self.bomb_probability + self.BOMB_PROB_INCREMENT, 0.5)

    def _create_particles(self, pos, count, color, lifetime_max, speed_max, size_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(10, lifetime_max),
                "max_lifetime": lifetime_max,
                "color": color,
                "size": self.np_random.uniform(2, size_max)
            })

    def _get_observation(self):
        # Render all game elements
        self._render_background()
        self._render_particles()
        self._render_objects()
        self._render_slicer_and_trails()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["lifetime"] / p["max_lifetime"]
            alpha = int(255 * life_ratio)
            color = (*p["color"], alpha)
            size = int(p["size"] * life_ratio)
            if size > 0:
                # Use a temporary surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

    def _render_objects(self):
        # Fruits
        for fruit in self.fruits:
            x, y = int(fruit["pos"][0]), int(fruit["pos"][1])
            r = int(fruit["radius"])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, fruit["color"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit["color"])
            # Highlight
            highlight_color = (min(255, fruit["color"][0] + 50), min(255, fruit["color"][1] + 50), min(255, fruit["color"][2] + 50))
            pygame.gfxdraw.aacircle(self.screen, x - r//3, y - r//3, r//3, highlight_color)
            pygame.gfxdraw.filled_circle(self.screen, x - r//3, y - r//3, r//3, highlight_color)
        
        # Bombs
        for bomb in self.bombs:
            x, y = int(bomb["pos"][0]), int(bomb["pos"][1])
            r = int(bomb["radius"])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BOMB)
            # Fuse
            fuse_x, fuse_y = x, int(y - r)
            pygame.draw.line(self.screen, self.COLOR_FUSE, (x, y), (fuse_x, fuse_y), 3)

    def _render_slicer_and_trails(self):
        # Trails
        for trail in self.slice_trails:
            life_ratio = trail["lifetime"] / self.SLICE_DURATION
            alpha = int(150 * life_ratio)
            color = (*self.COLOR_TRAIL, alpha)
            width = int(trail["width"] * (1 + (1 - life_ratio) * 0.2))
            height = int(10 * life_ratio)
            if height > 0:
                trail_surf = pygame.Surface((width, height), pygame.SRCALPHA)
                trail_surf.fill(color)
                self.screen.blit(trail_surf, (int(trail["center"][0] - width / 2), int(trail["center"][1] - height / 2)))

        # Slicer reticle
        x, y = int(self.slicer_pos[0]), int(self.slicer_pos[1])
        pygame.draw.line(self.screen, self.COLOR_SLICER, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, self.COLOR_SLICER, (x, y - 10), (x, y + 10), 2)

    def _render_ui(self):
        def draw_text(text, font, x, y, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (x + 2, y + 2))
            self.screen.blit(text_surf, (x, y))

        score_text = f"Score: {self.score}"
        fruits_text = f"Fruits: {self.fruits_sliced}/{self.WIN_CONDITION_FRUITS}"
        bombs_text = f"Bombs: {self.bombs_hit}/{self.LOSE_CONDITION_BOMBS}"
        
        draw_text(score_text, self.font_medium, 10, 10)
        draw_text(fruits_text, self.font_medium, self.SCREEN_WIDTH - 180, 10)
        draw_text(bombs_text, self.font_medium, self.SCREEN_WIDTH - 180, 40)

        if self.game_over:
            win = self.fruits_sliced >= self.WIN_CONDITION_FRUITS
            end_text = "YOU WIN!" if win else "GAME OVER"
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            shadow_surf = self.font_large.render(end_text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_sliced": self.fruits_sliced,
            "bombs_hit": self.bombs_hit,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set up a window to display the game
    pygame.display.set_caption("Fruit Slicer")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {total_reward}, Info: {info}")
            running = False # Or wait for reset key

    env.close()