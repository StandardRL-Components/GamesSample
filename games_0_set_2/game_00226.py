import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to pilot your ship."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a treacherous asteroid field. Dodge incoming space rocks for 60 seconds to clear each stage. Can you survive all three?"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    PLAYER_SPEED = 6
    PLAYER_RADIUS = 10
    PLAYER_TRAIL_LENGTH = 15
    PARTICLE_LIFESPAN = 40
    MAX_STAGES = 3
    STAGE_DURATION_SECONDS = 60
    
    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = [(100, 110, 120), (120, 130, 140), (140, 150, 160)]
    COLOR_PARTICLE = (255, 150, 50)
    COLOR_TEXT = (50, 255, 50)
    COLOR_TIMER_WARN = (255, 255, 0)
    COLOR_GAMEOVER = (255, 50, 50)
    COLOR_WIN = (50, 255, 255)

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
        
        try:
            self.font_small = pygame.font.SysFont("Consolas", 20)
            self.font_large = pygame.font.SysFont("Consolas", 64)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 72)
        
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.player_trail = []
        
        self.np_random = np.random.default_rng()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.stage_timer = 0
        self.player_pos = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.stage = 1
        self.stage_timer = self.STAGE_DURATION_SECONDS * self.FPS
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_trail.clear()
        
        self.particles.clear()
        self.asteroids.clear()
        
        self._generate_stars()
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            movement = action[0]
            
            # --- 1. Update Game Logic ---
            self._update_player(movement)
            self._update_asteroids()
            self._update_particles()
            
            # --- 2. Calculate Rewards ---
            reward += 0.1  # Survival reward
            if movement == 0:
                reward -= 0.2 # Penalty for inaction

            # --- 3. Check Collisions ---
            if self._check_collisions():
                self.game_over = True
                terminated = True
                reward -= 10
                self._create_explosion(self.player_pos)
                # SFX: Play explosion sound

            # --- 4. Update Timers & Stage ---
            self.stage_timer -= 1
            if self.stage_timer <= 0:
                reward += 10 # Stage complete bonus
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    self.game_over = True
                    self.win = True
                    terminated = True
                    reward += 100 # Win bonus
                    # SFX: Play win jingle
                else:
                    self.stage_timer = self.STAGE_DURATION_SECONDS * self.FPS
                    self._setup_stage()
                    # SFX: Play stage clear sound
        
        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _setup_stage(self):
        self.asteroids.clear()
        num_asteroids = 12 + self.stage * 3
        speed_multiplier = 1.0 + (self.stage - 1) * 0.2
        
        for _ in range(num_asteroids):
            self.asteroids.append(self._create_asteroid(speed_multiplier))

    def _create_asteroid(self, speed_multiplier):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -30]
        elif edge == 1: # Bottom
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30]
        elif edge == 2: # Left
            pos = [-30, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        else: # Right
            pos = [self.SCREEN_WIDTH + 30, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            
        angle = math.atan2(self.SCREEN_HEIGHT / 2 - pos[1], self.SCREEN_WIDTH / 2 - pos[0])
        angle += self.np_random.uniform(-0.5, 0.5) # Add randomness
        
        speed = self.np_random.uniform(1, 2.5) * speed_multiplier
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        radius = self.np_random.uniform(8, 25)
        color_index = self.np_random.integers(len(self.COLOR_ASTEROID))
        color = self.COLOR_ASTEROID[color_index]
        
        return {
            "pos": np.array(pos, dtype=np.float32),
            "vel": np.array(vel, dtype=np.float32),
            "radius": radius,
            "color": color
        }

    def _generate_stars(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                "pos": [self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                "depth": self.np_random.uniform(0.1, 0.7) # For parallax
            })

    def _update_player(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > self.PLAYER_TRAIL_LENGTH:
            self.player_trail.pop(0)

    def _update_asteroids(self):
        for i, asteroid in enumerate(self.asteroids):
            asteroid["pos"] += asteroid["vel"]
            # Check if asteroid is far off-screen to respawn
            if (asteroid["pos"][0] < -50 or asteroid["pos"][0] > self.SCREEN_WIDTH + 50 or
                asteroid["pos"][1] < -50 or asteroid["pos"][1] > self.SCREEN_HEIGHT + 50):
                speed_multiplier = 1.0 + (self.stage - 1) * 0.2
                self.asteroids[i] = self._create_asteroid(speed_multiplier)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_RADIUS + asteroid["radius"]:
                return True
        return False

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32),
                "lifespan": self.np_random.integers(20, self.PARTICLE_LIFESPAN),
                "size": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render stars with parallax
        for star in self.stars:
            x = int(star["pos"][0])
            y = int(star["pos"][1])
            brightness = int(star["depth"] * 200 + 55)
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (x, y), int(star["depth"]))

        # Render asteroids
        for asteroid in self.asteroids:
            pos = (int(asteroid["pos"][0]), int(asteroid["pos"][1]))
            radius = int(asteroid["radius"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, asteroid["color"])
            # FIX: Convert the generator to a tuple for the color argument
            outline_color = tuple(max(0, c - 20) for c in asteroid["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, outline_color)

        # Render player trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / self.PLAYER_TRAIL_LENGTH))
            radius = int(self.PLAYER_RADIUS * 0.5 * (i / self.PLAYER_TRAIL_LENGTH))
            if radius > 0:
                trail_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, self.COLOR_PLAYER + (alpha,), (radius, radius), radius)
                self.screen.blit(trail_surface, (int(pos[0] - radius), int(pos[1] - radius)))

        # Render player
        if not (self.game_over and not self.win):
            p_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            points = [
                (p_pos[0], p_pos[1] - self.PLAYER_RADIUS),
                (p_pos[0] - self.PLAYER_RADIUS * 0.8, p_pos[1] + self.PLAYER_RADIUS * 0.8),
                (p_pos[0] + self.PLAYER_RADIUS * 0.8, p_pos[1] + self.PLAYER_RADIUS * 0.8)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Render particles
        for p in self.particles:
            alpha = max(0, int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN)))
            color = self.COLOR_PARTICLE + (alpha,)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(part_surf, color, (size, size), size)
            self.screen.blit(part_surf, (pos[0] - size, pos[1] - size))


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH // 2 - stage_text.get_width() // 2, 10))

        # Timer
        time_left = max(0, self.stage_timer / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": max(0, self.stage_timer / self.FPS)
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroid Dodger")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        terminated = False
        
        keys_held = {
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_SPACE: False,
            pygame.K_LSHIFT: False,
        }
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in keys_held:
                        keys_held[event.key] = True
                    if event.key == pygame.K_r: # Press R to reset
                        obs, info = env.reset()
                        terminated = False
                elif event.type == pygame.KEYUP:
                    if event.key in keys_held:
                        keys_held[event.key] = False

            movement = 0 # no-op
            if keys_held[pygame.K_UP]: movement = 1
            elif keys_held[pygame.K_DOWN]: movement = 2
            elif keys_held[pygame.K_LEFT]: movement = 3
            elif keys_held[pygame.K_RIGHT]: movement = 4
            
            space_pressed = 1 if keys_held[pygame.K_SPACE] else 0
            shift_pressed = 1 if keys_held[pygame.K_LSHIFT] else 0
            
            action = [movement, space_pressed, shift_pressed]
            
            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
            
            surf = env.screen
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(GameEnv.FPS)
            
    finally:
        env.close()
        pygame.quit()