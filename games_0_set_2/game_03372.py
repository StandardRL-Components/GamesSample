
# Generated: 2025-08-27T23:10:14.915188
# Source Brief: brief_03372.md
# Brief Index: 3372

        
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
        "Controls: Arrow keys to move your ship. Avoid the red asteroids and collect blue power-ups for temporary invulnerability."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless asteroid field for 60 seconds by dodging space rocks and collecting power-ups in this top-down arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (10, 10, 26)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_SHIELD = (51, 204, 255, 100) # RGBA
    COLOR_ASTEROID = (255, 51, 51)
    COLOR_POWERUP = (51, 204, 255)
    COLOR_TEXT = (255, 255, 255)
    
    # Player
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 10

    # Asteroids
    INITIAL_ASTEROID_COUNT = 20
    BASE_ASTEROID_SPEED = 1.0
    MIN_ASTEROID_RADIUS = 8
    MAX_ASTEROID_RADIUS = 20
    DIFFICULTY_INTERVAL = 10 * FPS # Increase difficulty every 10 seconds
    DIFFICULTY_SPEED_INCREMENT = 0.25 # Speed increase is now more noticeable

    # Power-ups
    POWERUP_DURATION = 2 * FPS # 2 seconds
    POWERUP_SPAWN_INTERVAL = 5 * FPS # 5 seconds
    POWERUP_RADIUS = 10

    # Rewards
    REWARD_SURVIVAL = 0.01 # Scaled down for better balance
    REWARD_POWERUP = 5.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -10.0

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
        self.font_ui = pygame.font.Font(None, 32)
        self.font_big = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_remaining = 0
        self.player_pos = None
        self.invuln_frames = 0
        self.asteroids = []
        self.asteroid_speed_mod = 1.0
        self.powerup = None
        self.powerup_spawn_timer = 0
        self.np_random = None

        self._generate_stars()
        
        # Initialize state variables
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.choice([1, 1, 1, 2])
            self.stars.append((x, y, size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_STEPS

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.invuln_frames = 0

        self.asteroids = []
        self.asteroid_speed_mod = 1.0
        for _ in range(self.INITIAL_ASTEROID_COUNT):
            self._spawn_asteroid()

        self.powerup = None
        self.powerup_spawn_timer = self.POWERUP_SPAWN_INTERVAL
        
        return self._get_observation(), self._get_info()
    
    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.MAX_ASTEROID_RADIUS)
        elif edge == 1: # right
            pos = pygame.Vector2(self.SCREEN_WIDTH + self.MAX_ASTEROID_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        elif edge == 2: # bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.MAX_ASTEROID_RADIUS)
        else: # left
            pos = pygame.Vector2(-self.MAX_ASTEROID_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        # Ensure asteroids don't spawn directly on the player at reset
        while self.steps < 10 and pos.distance_to(self.player_pos) < 150:
            pos.x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            pos.y = self.np_random.uniform(0, self.SCREEN_HEIGHT)

        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.8, 1.2) * self.BASE_ASTEROID_SPEED * self.asteroid_speed_mod
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        radius = self.np_random.integers(self.MIN_ASTEROID_RADIUS, self.MAX_ASTEROID_RADIUS + 1)
        
        self.asteroids.append({"pos": pos, "vel": vel, "radius": radius})

    def _spawn_powerup(self):
        self.powerup = {
            "pos": pygame.Vector2(
                self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            )
        }
        self.powerup_spawn_timer = self.POWERUP_SPAWN_INTERVAL

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        reward = self._update_game_state(movement)
        
        terminated = self.game_over or self.time_remaining <= 0
        if terminated and not self.game_over: # Win condition
            self.win = True
            reward += self.REWARD_WIN
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement):
        self.steps += 1
        self.time_remaining -= 1
        reward = self.REWARD_SURVIVAL

        # --- Player Logic ---
        self._move_player(movement)
        self._wrap_around(self.player_pos, self.PLAYER_RADIUS)
        if self.invuln_frames > 0:
            self.invuln_frames -= 1

        # --- Asteroid Logic ---
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            self._wrap_around(asteroid["pos"], asteroid["radius"])
        
        # --- Power-up Logic ---
        self.powerup_spawn_timer -= 1
        if self.powerup is None and self.powerup_spawn_timer <= 0:
            self._spawn_powerup()

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.asteroid_speed_mod += self.DIFFICULTY_SPEED_INCREMENT
            # Update existing asteroid speeds
            for asteroid in self.asteroids:
                asteroid["vel"].scale_to_length(asteroid["vel"].length() * (1 + self.DIFFICULTY_SPEED_INCREMENT / self.asteroid_speed_mod))

        # --- Collision Detection & Events ---
        # Player-Asteroid
        if self.invuln_frames <= 0:
            for asteroid in self.asteroids:
                if self.player_pos.distance_to(asteroid["pos"]) < self.PLAYER_RADIUS + asteroid["radius"]:
                    self.game_over = True
                    # sfx: player_explosion.wav
                    reward += self.REWARD_LOSS
                    break
        
        # Player-Powerup
        if self.powerup and not self.game_over:
            if self.player_pos.distance_to(self.powerup["pos"]) < self.PLAYER_RADIUS + self.POWERUP_RADIUS:
                self.invuln_frames = self.POWERUP_DURATION
                self.powerup = None
                # sfx: powerup_collect.wav
                reward += self.REWARD_POWERUP
        
        return reward

    def _move_player(self, movement):
        vel = pygame.Vector2(0, 0)
        if movement == 1: vel.y = -1 # Up
        elif movement == 2: vel.y = 1 # Down
        elif movement == 3: vel.x = -1 # Left
        elif movement == 4: vel.x = 1 # Right
        
        if vel.length() > 0:
            vel.scale_to_length(self.PLAYER_SPEED)
        self.player_pos += vel

    def _wrap_around(self, pos, radius):
        if pos.x < -radius: pos.x = self.SCREEN_WIDTH + radius
        if pos.x > self.SCREEN_WIDTH + radius: pos.x = -radius
        if pos.y < -radius: pos.y = self.SCREEN_HEIGHT + radius
        if pos.y > self.SCREEN_HEIGHT + radius: pos.y = -radius

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        if self.powerup: self._render_powerup()
        self._render_asteroids()
        self._render_player()
        self._render_ui()
        
        if self.game_over or self.win:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

    def _render_powerup(self):
        # Blinking effect
        if self.steps % 30 < 20:
            pos = (int(self.powerup["pos"].x), int(self.powerup["pos"].y))
            r = self.POWERUP_RADIUS
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r, self.COLOR_POWERUP)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], r, self.COLOR_POWERUP)
            # Add a glow
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r + 3, self.COLOR_POWERUP)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid["pos"].x), int(asteroid["pos"].y))
            radius = int(asteroid["radius"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

    def _render_player(self):
        pos = self.player_pos
        r = self.PLAYER_RADIUS
        
        # Ship body (triangle)
        points = [
            (pos.x, pos.y - r),
            (pos.x - r * 0.8, pos.y + r * 0.8),
            (pos.x + r * 0.8, pos.y + r * 0.8),
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)
        
        # Shield effect
        if self.invuln_frames > 0:
            shield_alpha = 100 + int(math.sin(self.steps * 0.3) * 50) # Pulsating alpha
            shield_radius = r + 5 + int(math.sin(self.steps * 0.2) * 3) # Pulsating radius
            shield_color = (*self.COLOR_POWERUP, shield_alpha)
            
            # Create a temporary surface for transparency
            temp_surface = pygame.Surface((shield_radius*2, shield_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surface, shield_radius, shield_radius, shield_radius-1, shield_color)
            pygame.gfxdraw.filled_circle(temp_surface, shield_radius, shield_radius, shield_radius-1, shield_color)
            self.screen.blit(temp_surface, (int(pos.x - shield_radius), int(pos.y - shield_radius)))

    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

    def _render_end_screen(self):
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_PLAYER if self.win else self.COLOR_ASTEROID
        
        end_surf = self.font_big.render(text, True, color)
        pos = (
            self.SCREEN_WIDTH / 2 - end_surf.get_width() / 2,
            self.SCREEN_HEIGHT / 2 - end_surf.get_height() / 2
        )
        self.screen.blit(end_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "invulnerable": self.invuln_frames > 0,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's auto_advance is for RL agents. For human play, we need a standard game loop.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Survival")
    clock = pygame.time.Clock()
    
    total_reward = 0.0
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping ---
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
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0.0
            
        clock.tick(env.FPS)
        
    env.close()