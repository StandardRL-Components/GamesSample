import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:56:35.543495
# Source Brief: brief_02009.md
# Brief Index: 2009
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a spaceship through obstacle-laden
    energy fields. The goal is to collect fuel from pulsating green fields to
    survive and reach the finish line, while jumping over red obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a spaceship through obstacle-laden energy fields. Collect fuel from green fields to "
        "survive and jump over red obstacles to reach the finish line."
    )
    user_guide = (
        "Controls: Use ←→ to move left and right, and ↑ to jump. Collect green fields for fuel "
        "and avoid red obstacles."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    LEVEL_LENGTH = 5000  # World units to travel to win

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 220, 255)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_OBSTACLE_GLOW = (255, 120, 120)
    COLOR_FUEL_FIELD = (80, 255, 80)
    COLOR_FUEL_FIELD_GLOW = (150, 255, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_FUEL_BAR = (80, 220, 80)
    COLOR_FUEL_BAR_BG = (50, 50, 50)
    COLOR_FINISH_LINE = (255, 255, 255)

    # Player Physics
    PLAYER_SPEED = 4.0
    PLAYER_FRICTION = 0.90
    GRAVITY = 0.3
    JUMP_STRENGTH = 8.0
    PLAYER_SIZE = 15
    GROUND_Y = 350

    # Fuel Mechanics
    MAX_FUEL = 100.0
    FUEL_DEPLETION_RATE = 1.0 / FPS  # 1 unit per second
    FUEL_COLLECT_RATE = 20.0 / FPS # 20 units per second
    FUEL_PENALTY_THRESHOLD = 75.0
    OBSTACLE_COLLISION_FUEL_LOSS = 25.0

    # Level Generation
    INITIAL_OBSTACLE_DENSITY = 2  # Per screen width
    ENERGY_FIELD_DENSITY = 1 # Per screen width
    LEVEL_START_BUFFER = 800 # Safe zone at start

    # Episode
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.fuel = None
        self.score = None
        self.steps = None
        self.level = None
        self.obstacles = None
        self.energy_fields = None
        self.stars = None
        self.last_obstacle_hit_time = -100 # Cooldown for obstacle hit penalty


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize player state
        self.player_pos = pygame.math.Vector2(self.WIDTH / 4, self.GROUND_Y)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = True

        # Initialize game state
        self.fuel = self.MAX_FUEL
        self.score = 0
        self.steps = 0
        self.level = 1 # Always start at level 1 on reset
        self.last_obstacle_hit_time = -100

        # Procedurally generate the level
        self._generate_level()
        self._generate_stars()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.obstacles = []
        self.energy_fields = []
        
        num_obstacles = int((self.LEVEL_LENGTH / self.WIDTH) * self.INITIAL_OBSTACLE_DENSITY)
        num_fields = int((self.LEVEL_LENGTH / self.WIDTH) * self.ENERGY_FIELD_DENSITY)

        for _ in range(num_obstacles):
            w = self.np_random.integers(30, 80)
            h = self.np_random.integers(50, 150)
            x = self.np_random.uniform(self.LEVEL_START_BUFFER, self.LEVEL_LENGTH - w)
            # Obstacles are on the ground
            y = self.GROUND_Y - h + 5 
            self.obstacles.append(pygame.Rect(x, y, w, h))

        for _ in range(num_fields):
            w = self.np_random.integers(100, 200)
            h = self.np_random.integers(80, 120)
            x = self.np_random.uniform(self.LEVEL_START_BUFFER, self.LEVEL_LENGTH - w)
            y = self.np_random.uniform(self.GROUND_Y - 250, self.GROUND_Y - h - 50)
            # Pulse offset makes fields pulse at different times
            pulse_offset = self.np_random.uniform(0, 2 * math.pi)
            self.energy_fields.append({"rect": pygame.Rect(x, y, w, h), "pulse_offset": pulse_offset})
            
    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.uniform(0, self.WIDTH)
            y = self.np_random.uniform(0, self.HEIGHT)
            # Depth determines parallax speed and size
            depth = self.np_random.uniform(0.1, 0.6)
            self.stars.append({"pos": pygame.math.Vector2(x, y), "depth": depth})

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # --- 1. Handle Input & Update Player State ---
        self._handle_input(action)
        self._update_player_physics()

        # --- 2. Update Game World & Handle Interactions ---
        fuel_collected = self._handle_interactions()
        
        # --- 3. Calculate Reward ---
        # Survival reward
        reward += 0.1

        # Fuel collection reward
        if fuel_collected:
            reward += 5.0
            # sfx: collect_fuel

        # Penalty for low fuel
        if self.fuel < self.FUEL_PENALTY_THRESHOLD:
            reward -= 1.0

        # --- 4. Check for Termination ---
        terminated = False
        if self.fuel <= 0:
            reward = -100.0
            terminated = True
            # sfx: game_over_fuel
        
        if self.player_pos.x >= self.LEVEL_LENGTH:
            reward = 100.0
            terminated = True
            # sfx: victory
            
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium v1.0, truncated environments are also terminated

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, _, _ = action # space and shift are unused

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_SPEED

        # Vertical movement (jump)
        if movement == 1 and self.on_ground:
            self.player_vel.y = -self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

    def _update_player_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        # Update position
        self.player_pos += self.player_vel

        # Ground collision
        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            self.on_ground = True
            
        # Prevent moving backwards past start
        if self.player_pos.x < self.PLAYER_SIZE:
            self.player_pos.x = self.PLAYER_SIZE
            self.player_vel.x = 0

    def _handle_interactions(self):
        # Deplete fuel over time
        self.fuel -= self.FUEL_DEPLETION_RATE
        self.fuel = max(0, self.fuel)

        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )

        # Check for obstacle collisions
        # Add a cooldown to prevent rapid fuel loss from a single obstacle
        if self.steps > self.last_obstacle_hit_time + self.FPS / 2:
            for obs_rect in self.obstacles:
                if player_rect.colliderect(obs_rect):
                    self.fuel -= self.OBSTACLE_COLLISION_FUEL_LOSS
                    self.last_obstacle_hit_time = self.steps
                    self.player_vel.x *= 0.5 # Slow down on hit
                    # sfx: obstacle_hit
                    break # Only hit one obstacle per frame

        # Check for fuel collection
        fuel_collected_this_frame = False
        for field in self.energy_fields:
            # Check if field is "active" based on its pulse
            pulse_phase = (self.steps / self.FPS * 2 + field["pulse_offset"]) % (2 * math.pi)
            if math.sin(pulse_phase) > 0: # Active for half the cycle
                if player_rect.colliderect(field["rect"]):
                    self.fuel += self.FUEL_COLLECT_RATE
                    self.fuel = min(self.MAX_FUEL, self.fuel)
                    fuel_collected_this_frame = True
        
        return fuel_collected_this_frame

    def _get_observation(self):
        # The camera keeps the player centered horizontally
        camera_offset_x = self.player_pos.x - self.WIDTH / 2

        # --- RENDER ---
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_stars(camera_offset_x)

        # Finish line
        finish_x = self.LEVEL_LENGTH - camera_offset_x
        if 0 < finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_x, 0), (finish_x, self.HEIGHT), 3)

        # Game objects
        self._render_game_objects(camera_offset_x)
        self._render_player()

        # UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self, camera_offset_x):
        for star in self.stars:
            # Parallax effect
            star_x = (star["pos"].x - camera_offset_x * star["depth"]) % self.WIDTH
            star_y = star["pos"].y
            size = int(star["depth"] * 3)
            color_val = int(50 + star["depth"] * 100)
            color = (color_val, color_val, color_val + 20)
            self.screen.fill(color, (star_x, star_y, max(1, size), max(1, size)))

    def _render_game_objects(self, camera_offset_x):
        # Obstacles
        for obs_rect in self.obstacles:
            screen_rect = obs_rect.move(-camera_offset_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            # Glow effect
            glow_rect = screen_rect.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_OBSTACLE_GLOW, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            # Main shape
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect, border_radius=3)

        # Energy Fields
        for field in self.energy_fields:
            screen_rect = field["rect"].move(-camera_offset_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            
            # Pulsing effect
            pulse_phase = (self.steps / self.FPS * 2 + field["pulse_offset"]) % (2 * math.pi)
            alpha = 100 + 90 * math.sin(pulse_phase)
            if alpha > 100: # Only draw when "active"
                field_color = (*self.COLOR_FUEL_FIELD, alpha)
                glow_color = (*self.COLOR_FUEL_FIELD_GLOW, alpha / 2)

                # Glow
                glow_rect = screen_rect.inflate(12, 12)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
                self.screen.blit(s, glow_rect.topleft)
                
                # Main shape
                s = pygame.Surface(screen_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, field_color, s.get_rect(), border_radius=5)
                self.screen.blit(s, screen_rect.topleft)

    def _render_player(self):
        # Player is always in the middle of the screen horizontally
        screen_x = self.WIDTH / 2
        screen_y = self.player_pos.y

        # Points for an upward-facing triangle
        p1 = (screen_x, screen_y - self.PLAYER_SIZE)
        p2 = (screen_x - self.PLAYER_SIZE / 2, screen_y)
        p3 = (screen_x + self.PLAYER_SIZE / 2, screen_y)
        points = [p1, p2, p3]

        # Glow effect
        for i in range(5, 0, -1):
            glow_alpha = 150 - i * 30
            glow_size = self.PLAYER_SIZE + i * 2
            gp1 = (screen_x, screen_y - glow_size)
            gp2 = (screen_x - glow_size / 2, screen_y)
            gp3 = (screen_x + glow_size / 2, screen_y)
            glow_points = [(int(p[0]), int(p[1])) for p in [gp1, gp2, gp3]]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, (*self.COLOR_PLAYER_GLOW, glow_alpha))

        # Main player shape
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Fuel bar
        bar_width = 200
        bar_height = 20
        fuel_ratio = self.fuel / self.MAX_FUEL
        current_bar_width = int(bar_width * fuel_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BAR_BG, (10, 10, bar_width, bar_height))
        if current_bar_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_FUEL_BAR, (10, 10, current_bar_width, bar_height))
        
        fuel_text = self.font_small.render(f"FUEL: {int(self.fuel)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (220, 12))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(centerx=self.WIDTH/2, top=10)
        self.screen.blit(score_text, score_rect)

        # Progress
        progress = (self.player_pos.x / self.LEVEL_LENGTH * 100) if self.LEVEL_LENGTH > 0 else 0
        progress_text = self.font_small.render(f"PROGRESS: {int(progress)}%", True, self.COLOR_UI_TEXT)
        progress_rect = progress_text.get_rect(right=self.WIDTH - 10, top=12)
        self.screen.blit(progress_text, progress_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "level": self.level,
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    # Make sure to remove the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Spaceship Fuel Run")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            terminated = False
            truncated = False

        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2 # no effect
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()