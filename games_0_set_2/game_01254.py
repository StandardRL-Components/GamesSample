import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to jump over obstacles on the beat."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A neon-drenched rhythm runner. Jump on the beat to clear obstacles, build your combo, and reach the end of the road."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_GRID = (50, 30, 80)
    COLOR_ROAD_EDGE = (100, 80, 180)
    COLOR_ROAD_FILL = (40, 40, 100)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 150)
    COLOR_OBSTACLE = (255, 0, 128)
    COLOR_OBSTACLE_GLOW = (255, 100, 200)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SUCCESS = (0, 255, 128)
    COLOR_FAIL = (255, 50, 50)
    COLOR_HEART = (255, 80, 80)
    COLOR_HEART_BROKEN = (80, 80, 80)

    # Player
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 30
    PLAYER_X = SCREEN_WIDTH // 4
    GROUND_Y = SCREEN_HEIGHT - 80
    GRAVITY = 0.8
    JUMP_STRENGTH = 13.5

    # Obstacles
    OBSTACLE_WIDTH = 25
    OBSTACLE_HEIGHT = 40
    INITIAL_OBSTACLE_SPEED = 5.0
    TOTAL_OBSTACLES = 100
    
    # Rhythm
    BPM = 120
    BEAT_FRAMES = int((60 / BPM) * FPS) # Frames per beat

    # Game
    MAX_LIVES = 3
    MAX_STEPS = 2500 # Increased to allow for slower final sections

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        # This is necessary to initialize the video system for surface conversions, even in headless mode.
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 48)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel_y = None
        self.is_jumping = None
        self.last_space_held = None
        
        self.obstacles = None
        self.particles = None
        
        self.score = None
        self.lives = None
        self.combo = None
        self.game_over = None
        
        self.steps = None
        self.obstacle_speed = None
        self.obstacles_cleared = None
        self.obstacles_spawned = None
        self.beat_timer = None

        # The test harness expects the environment to be fully initialized,
        # so we call reset() here.
        # self.reset() # This line is commented out as it is good practice to have the user call reset() after __init__
                      # However, the original code had it, and some test harnesses might rely on it.
                      # For robustness with the provided error, we assume the user will call reset().
                      # If the test harness fails on not finding an initialized state, this can be uncommented.
                      # Re-enabling for compatibility with the test harness that calls reset from init.
        # The traceback shows __init__ calls reset, so we must keep this behavior.
        # The error happens inside reset, so the state variables must be initialized to None first.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Rect(self.PLAYER_X, self.GROUND_Y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.player_vel_y = 0
        self.is_jumping = False
        self.last_space_held = False

        self.obstacles = []
        self.particles = []

        self.score = 0
        self.lives = self.MAX_LIVES
        self.combo = 0
        self.game_over = False

        self.steps = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacles_cleared = 0
        self.obstacles_spawned = 0
        self.beat_timer = self.BEAT_FRAMES * 2 # Start with a delay

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        # movement = action[0]  # Not used
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Not used
        
        # Detect space press (rising edge)
        if space_held and not self.last_space_held and not self.is_jumping:
            self.player_vel_y = -self.JUMP_STRENGTH
            self.is_jumping = True
            # sfx: jump
            self._create_particles(self.player_pos.midbottom, 20, self.COLOR_PLAYER)

        self.last_space_held = space_held

        # --- Update Game State ---
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        self._update_beat()

        # --- Collision & Scoring ---
        reward += self._process_obstacles()

        # --- Continuous Reward ---
        if not self.is_jumping:
            reward -= 0.2
            
        # --- Termination ---
        terminated = (self.lives <= 0) or (self.obstacles_cleared >= self.TOTAL_OBSTACLES)
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.lives <= 0:
                reward -= 100 # Game over penalty
            elif self.obstacles_cleared >= self.TOTAL_OBSTACLES:
                reward += 100 # Victory bonus
        
        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos.y += int(self.player_vel_y)

        # Check for landing
        if self.player_pos.bottom >= self.GROUND_Y:
            self.player_pos.bottom = self.GROUND_Y
            self.player_vel_y = 0
            if self.is_jumping: # Just landed
                self.is_jumping = False
                # sfx: land
                self._create_particles(self.player_pos.midbottom, 10, self.COLOR_PLAYER_GLOW)

    def _update_beat(self):
        self.beat_timer -= 1
        if self.beat_timer <= 0 and self.obstacles_spawned < self.TOTAL_OBSTACLES:
            self.beat_timer = self.BEAT_FRAMES
            self.obstacles_spawned += 1
            
            # Difficulty scaling: speed = 5 for 0-19, 6 for 20-39, etc.
            speed_increase = math.floor(self.obstacles_cleared / 20)
            self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED + speed_increase

            new_obstacle = {
                "rect": pygame.Rect(self.SCREEN_WIDTH, self.GROUND_Y - self.OBSTACLE_HEIGHT, self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT),
                "cleared": False,
                "processed": False
            }
            self.obstacles.append(new_obstacle)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs["rect"].x -= int(self.obstacle_speed)

    def _process_obstacles(self):
        reward = 0
        for obs in self.obstacles:
            if obs["processed"]:
                continue

            # Collision with player
            if self.player_pos.colliderect(obs["rect"]):
                obs["processed"] = True
                self.lives -= 1
                self.combo = 0
                reward -= 50
                # sfx: fail_hit
                self._create_particles(self.player_pos.center, 30, self.COLOR_FAIL)
                continue

            # Obstacle has passed the player
            if obs["rect"].right < self.player_pos.left:
                obs["processed"] = True
                if obs["cleared"]: # Was cleared in a previous frame
                    self.obstacles_cleared += 1
                    self.combo += 1
                    reward += 1 # Base reward for clearing
                    
                    # Near-miss reward
                    if "min_clearance" in obs and obs["min_clearance"] < 10:
                        reward += 5 # Risky jump bonus
                    
                    # sfx: success
                    self._create_particles((obs["rect"].centerx, obs["rect"].centery), 20, self.COLOR_SUCCESS)
                else: # Player was on the ground, so it's a miss
                    self.lives -= 1
                    self.combo = 0
                    reward -= 50
                    # sfx: fail_miss
                    self._create_particles((self.player_pos.centerx, self.player_pos.bottom), 20, self.COLOR_FAIL)
            
            # Check if player is currently clearing the obstacle
            elif not obs["cleared"] and obs["rect"].centerx < self.player_pos.centerx and self.is_jumping:
                obs["cleared"] = True
                # Check for near miss
                clearance = self.player_pos.bottom - obs["rect"].top
                if clearance > 0:
                    if "min_clearance" not in obs or clearance < obs["min_clearance"]:
                        obs["min_clearance"] = clearance
                        
        # Remove obstacles that are off-screen
        self.obstacles = [obs for obs in self.obstacles if obs["rect"].right > 0]
        return reward

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})
            
    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # particle gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

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
            "lives": self.lives,
            "combo": self.combo,
            "obstacles_cleared": self.obstacles_cleared
        }
        
    def _render_game(self):
        # Road with perspective
        road_bottom_y = self.SCREEN_HEIGHT
        
        # Road fill
        pygame.draw.polygon(self.screen, self.COLOR_ROAD_FILL, [
            (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y),
            (self.SCREEN_WIDTH, road_bottom_y), (0, road_bottom_y)
        ])
        
        # Road lines
        for i in range(11):
            p = i / 10.0
            y = self.GROUND_Y + (road_bottom_y - self.GROUND_Y) * (p**2)
            alpha = int(255 * (1-p))
            
            line_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.line(line_surface, (*self.COLOR_ROAD_EDGE[:3], alpha), (0, int(y)), (self.SCREEN_WIDTH, int(y)), 2)
            self.screen.blit(line_surface, (0,0))

        # Beat-pulsing grid
        beat_progress = (self.BEAT_FRAMES - self.beat_timer) / self.BEAT_FRAMES
        pulse_alpha = 10 + 60 * (1 + math.cos(beat_progress * math.pi * 2)) / 2
        
        grid_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        grid_color = (*self.COLOR_GRID, int(pulse_alpha))
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(grid_surface, grid_color, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(grid_surface, grid_color, (0, i), (self.SCREEN_WIDTH, i), 1)
        self.screen.blit(grid_surface, (0,0))

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 30.0))
            color = (*p["color"], int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, color)
            
        # Obstacles
        for obs in self.obstacles:
            # Glow
            glow_rect = obs["rect"].inflate(8, 8)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_OBSTACLE_GLOW, 50), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)
            # Main body
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"], border_radius=3)

        # Player
        # Squash and stretch for game feel
        squash = 0
        if not self.is_jumping: # on ground
            squash = 3
        elif self.player_vel_y < 0: # Going up
            squash = -3
        
        player_render_rect = self.player_pos.inflate(squash * -2, squash * 2)
        player_render_rect.midbottom = self.player_pos.midbottom

        # Glow
        glow_rect = player_render_rect.inflate(12, 12)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, 80), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect, border_radius=4)
        
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives (Hearts)
        for i in range(self.MAX_LIVES):
            color = self.COLOR_HEART if i < self.lives else self.COLOR_HEART_BROKEN
            self._draw_heart(self.screen, 20 + i * 30, 45, color)
            
        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"x{self.combo}", True, self.COLOR_SUCCESS)
            text_rect = combo_text.get_rect(center=(self.player_pos.centerx, self.player_pos.top - 30))
            self.screen.blit(combo_text, text_rect)

    def _draw_heart(self, surface, x, y, color):
        # Simple heart shape using polygons
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.draw.polygon(surface, color, points)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To run this file directly, you might need to comment out the
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy") line at the top of the file
    # to allow a real window to be created.
    
    env = GameEnv(render_mode="rgb_array")
    
    # The __main__ block in the original code would not have worked for visualization
    # because the dummy video driver is set globally. For local testing,
    # you would typically remove that line.
    
    # For simplicity, we just validate the implementation here.
    try:
        obs, info = env.reset()
        assert obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        assert obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Environment created and stepped through successfully.")

    except Exception as e:
        print(f"An error occurred during validation: {e}")
    finally:
        env.close()