
# Generated: 2025-08-27T21:55:56.710601
# Source Brief: brief_02952.md
# Brief Index: 2952

        
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
        "Controls: ↑ and ↓ to move your car vertically. Avoid red obstacles and collect blue boosts."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced retro racer. Dodge obstacles and collect boosts to complete three laps as fast as you can."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TRACK_LENGTH = SCREEN_WIDTH * 6  # Length of one lap
    LAPS_TO_WIN = 3
    MAX_STEPS = 3000

    # Colors
    COLOR_BG = (30, 40, 50)         # Dark blue-gray
    COLOR_TRACK = (40, 120, 70)     # Muted Green
    COLOR_CAR = (255, 255, 255)     # White
    COLOR_CAR_ACCENT = (220, 30, 30) # Red
    COLOR_OBSTACLE = (220, 30, 30)  # Bright Red
    COLOR_BOOST = (50, 150, 255)    # Bright Blue
    COLOR_FINISH_LINE_1 = (255, 200, 0) # Yellow
    COLOR_FINISH_LINE_2 = (20, 20, 20)  # Dark Gray
    COLOR_UI_TEXT = (240, 240, 240)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # State variables
        self.car = None
        self.obstacles = []
        self.boosts = []
        self.particles = []
        self.world_scroll_x = 0
        self.lap_time_steps = 0
        self.current_lap = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.rng = None
        
        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional: Call to verify setup

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Car state
        self.car = {
            "rect": pygame.Rect(100, self.SCREEN_HEIGHT / 2 - 10, 40, 20),
            "y_vel": 0.0,
            "base_speed": 5.0,
            "current_speed": 5.0,
            "boost_timer": 0,
        }
        
        # World state
        self.world_scroll_x = 0
        self.obstacles = []
        self.boosts = []
        self.particles = []
        self.lap_time_steps = 0
        self.current_lap = 1
        self._generate_track_features(self.TRACK_LENGTH * self.LAPS_TO_WIN)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Small reward for surviving each frame

        # 1. Unpack and handle actions
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # 2. Update Car Physics
        if movement == 1:  # Up
            self.car["y_vel"] -= 0.8
        elif movement == 2:  # Down
            self.car["y_vel"] += 0.8
        self.car["y_vel"] *= 0.9  # Damping for smoother control
        self.car["rect"].y += self.car["y_vel"]
        self.car["rect"].top = max(20, self.car["rect"].top)
        self.car["rect"].bottom = min(self.SCREEN_HEIGHT - 20, self.car["rect"].bottom)

        # 3. Update Boost Effect
        if self.car["boost_timer"] > 0:
            self.car["boost_timer"] -= 1
            self.car["current_speed"] = self.car["base_speed"] * 1.8
            if self.steps % 2 == 0:
                self._spawn_particle(
                    self.car["rect"].left + self.world_scroll_x, self.car["rect"].centery
                )
        else:
            self.car["current_speed"] = self.car["base_speed"]

        # 4. Update World Scroll and Timers
        difficulty_modifier = 1.0 + (self.current_lap - 1) * 0.1
        self.world_scroll_x += self.car["current_speed"] * difficulty_modifier
        self.lap_time_steps += 1

        # 5. Update Game Entities
        self._update_obstacles()
        self._update_particles()

        # 6. Handle Collisions and Interactions
        car_rect_on_screen = self.car["rect"]
        
        # Obstacle collisions
        for obs in self.obstacles:
            obs_rect_on_screen = obs["rect"].move(-self.world_scroll_x, 0)
            if car_rect_on_screen.colliderect(obs_rect_on_screen):
                self.game_over = True
                reward = -100.0
                # sfx: explosion
                break
        
        # Boost collection
        if not self.game_over:
            for boost in self.boosts[:]:
                boost_x_on_screen = boost["pos"][0] - self.world_scroll_x
                dist_sq = (car_rect_on_screen.centerx - boost_x_on_screen)**2 + (car_rect_on_screen.centery - boost["pos"][1])**2
                if dist_sq < (car_rect_on_screen.width / 2 + boost["radius"])**2:
                    self.boosts.remove(boost)
                    self.car["boost_timer"] = 90  # 3 seconds at 30fps
                    reward += 5.0
                    self.score += 50
                    # sfx: boost_pickup
                    break

        # 7. Check Lap Completion
        if not self.game_over:
            lap_marker_x = self.current_lap * self.TRACK_LENGTH
            if self.world_scroll_x >= lap_marker_x:
                self.current_lap += 1
                self.lap_time_steps = 0
                if self.current_lap > self.LAPS_TO_WIN:
                    self.game_over = True
                    reward += 100.0  # Win game
                    self.score += 1000
                else:
                    reward += 20.0  # Lap complete
                    self.score += 200
                    # sfx: lap_complete

        # 8. Finalize Step
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lap": self.current_lap}

    def _generate_track_features(self, total_length):
        obstacle_spacing = self.rng.integers(300, 600, size=total_length // 300)
        boost_spacing = self.rng.integers(400, 800, size=total_length // 400)
        
        x = self.SCREEN_WIDTH
        for spacing in obstacle_spacing:
            x += spacing
            if x > total_length - self.SCREEN_WIDTH: break
            self._spawn_obstacle(x)

        x = self.SCREEN_WIDTH // 2
        for spacing in boost_spacing:
            x += spacing
            if x > total_length - self.SCREEN_WIDTH: break
            self._spawn_boost(x)

    def _spawn_obstacle(self, x_pos):
        obs_height = self.rng.integers(25, 40)
        obstacle = {
            "rect": pygame.Rect(x_pos, 0, 60, obs_height),
            "base_y": self.rng.integers(20 + obs_height, self.SCREEN_HEIGHT - 20 - obs_height),
            "amplitude": self.rng.integers(30, 100),
            "frequency": self.rng.uniform(0.005, 0.01),
            "phase": self.rng.uniform(0, 2 * math.pi),
        }
        self.obstacles.append(obstacle)

    def _spawn_boost(self, x_pos):
        boost = {
            "pos": (x_pos, self.rng.integers(50, self.SCREEN_HEIGHT - 50)),
            "radius": 12,
            "pulse_timer": self.rng.uniform(0, math.pi * 2),
        }
        self.boosts.append(boost)

    def _update_obstacles(self):
        for obs in self.obstacles[:]:
            if obs["rect"].right - self.world_scroll_x < 0:
                self.obstacles.remove(obs)
            else:
                obs["rect"].y = obs["base_y"] + obs["amplitude"] * math.sin(
                    obs["frequency"] * self.world_scroll_x + obs["phase"]
                )

    def _spawn_particle(self, world_x, world_y):
        self.particles.append({
            "pos": [world_x, world_y],
            "vel": [self.rng.uniform(-2, -0.5), self.rng.uniform(-1, 1)],
            "life": self.rng.integers(15, 30),
            "color": self.rng.choice([(255, 255, 100), (255, 150, 50)]),
            "size": self.rng.uniform(3, 6),
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["size"] -= 0.1
            if p["life"] <= 0 or p["size"] <= 0:
                self.particles.remove(p)

    def _render_game(self):
        # Track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, 20, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 40))
        for y in range(30, self.SCREEN_HEIGHT - 20, 50):
            start_x = int(-self.world_scroll_x % 80)
            for i in range(self.SCREEN_WIDTH // 80 + 2):
                pygame.draw.line(self.screen, self.COLOR_BG, (start_x + i * 80, y), (start_x + i * 80 + 40, y), 2)
        
        # Finish Lines
        for lap_n in range(1, self.LAPS_TO_WIN + 2):
            finish_x = lap_n * self.TRACK_LENGTH - self.world_scroll_x
            if -40 < finish_x < self.SCREEN_WIDTH:
                for i in range(20):
                    color = self.COLOR_FINISH_LINE_1 if i % 2 == 0 else self.COLOR_FINISH_LINE_2
                    pygame.draw.rect(self.screen, color, (finish_x, 20 + i * ((self.SCREEN_HEIGHT - 40) / 20), 40, (self.SCREEN_HEIGHT - 40) / 20 + 1))

        # Boosts
        for boost in self.boosts:
            x = boost["pos"][0] - self.world_scroll_x
            if 0 < x < self.SCREEN_WIDTH:
                boost["pulse_timer"] += 0.1
                pulse_radius = boost["radius"] + 2 * math.sin(boost["pulse_timer"])
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(boost["pos"][1]), int(pulse_radius), self.COLOR_BOOST)
                pygame.gfxdraw.aacircle(self.screen, int(x), int(boost["pos"][1]), int(pulse_radius), self.COLOR_BOOST)

        # Obstacles
        for obs in self.obstacles:
            rect_on_screen = obs["rect"].move(-self.world_scroll_x, 0)
            if self.screen.get_rect().colliderect(rect_on_screen):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect_on_screen, border_radius=3)
        
        # Particles
        for p in self.particles:
            screen_x = p["pos"][0] - self.world_scroll_x
            if 0 < screen_x < self.SCREEN_WIDTH:
                size = int(max(0, p["size"]))
                alpha = max(0, 255 * (p["life"] / 30.0))
                color = (*p["color"], alpha)
                part_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                part_surf.fill(color)
                self.screen.blit(part_surf, (screen_x - size/2, p["pos"][1] - size/2), special_flags=pygame.BLEND_RGBA_ADD)

        # Car
        car_rect = self.car["rect"]
        if self.car["boost_timer"] > 0:
            glow_size = car_rect.inflate(20, 20)
            glow_surf = pygame.Surface(glow_size.size, pygame.SRCALPHA)
            alpha = 50 + 50 * math.sin(self.steps * 0.5)
            pygame.draw.ellipse(glow_surf, (*self.COLOR_BOOST, alpha), glow_surf.get_rect())
            self.screen.blit(glow_surf, glow_size.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_CAR, car_rect, border_radius=4)
        cockpit_rect = pygame.Rect(car_rect.x + 20, car_rect.y + 5, 15, 10)
        pygame.draw.rect(self.screen, self.COLOR_CAR_ACCENT, cockpit_rect, border_radius=2)

    def _render_ui(self):
        lap_text = f"LAP: {min(self.current_lap, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}"
        lap_surf = self.font.render(lap_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_surf, (10, 10))

        time_seconds = self.lap_time_steps / 30.0
        time_text = f"TIME: {time_seconds:.2f}"
        time_surf = self.font.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 40))

        score_text = f"SCORE: {self.score}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = self.current_lap > self.LAPS_TO_WIN
            end_text = "FINISH!" if win_condition else "CRASHED!"
            color = self.COLOR_FINISH_LINE_1 if win_condition else self.COLOR_OBSTACLE
            end_surf = self.font.render(end_text, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

    def validate_implementation(self):
        print("🔬 Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Observation shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Observation dtype is {test_obs.dtype}"
        
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
        
        print("✅ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Optional: Run validation
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        # Left/Right, Space, Shift are not used in this game's mechanics
        
        action = [movement, 0, 0] # [movement, space, shift]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), but numpy uses (height, width)
        # So we need to transpose the observation back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
        
        # --- Game over check ---
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(30) # Run at 30 FPS

    pygame.quit()