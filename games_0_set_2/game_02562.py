
# Generated: 2025-08-27T20:45:07.735604
# Source Brief: brief_02562.md
# Brief Index: 2562

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space near an asteroid to mine it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through an asteroid field. Collect ore to increase your score, "
        "but be careful! Colliding with asteroids will cost you a life."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_ORE = 100
    MAX_LIVES = 3
    MAX_STEPS = 1000
    MAX_ASTEROIDS = 25
    PLAYER_SPEED = 6
    PLAYER_SIZE = 12
    MINING_RANGE = 40
    MINING_RATE = 2  # Ore per second
    INVULNERABILITY_FRAMES = 90  # 3 seconds

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (180, 220, 255)
    COLOR_ORE_PARTICLE = (255, 223, 0)
    COLOR_EXPLOSION = (255, 100, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    ASTEROID_COLORS = {
        1: (120, 120, 120),  # Low value
        2: (165, 125, 80),   # Mid value
        3: (218, 165, 32)    # High value
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_lives = None
        self.invulnerability_timer = None
        self.ore_collected = None
        self.asteroids = []
        self.particles = []
        self.explosions = []
        self.stars = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.mining_cooldown = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_lives = self.MAX_LIVES
        self.invulnerability_timer = self.INVULNERABILITY_FRAMES
        self.ore_collected = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.asteroids.clear()
        self.particles.clear()
        self.explosions.clear()
        
        # Generate a static starfield for the episode
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.integers(50, 150)
            )
            for _ in range(150)
        ]

        # Spawn initial asteroids
        for _ in range(10):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
            
        reward = 0
        
        if not self.game_over:
            # --- Update Game Logic ---
            mining_this_step = self._update_state(action)
            
            # --- Calculate Reward ---
            reward = self._calculate_reward(mining_this_step)
            self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
             # Apply terminal rewards only once
            if self.ore_collected >= self.WIN_ORE:
                reward += 100
                self.score += 100
            else: # Loss or timeout
                reward -= 100
                self.score -= 100
            self.game_over = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_state(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        self._handle_player_movement(movement)
        self._update_asteroids()
        self._update_particles()
        self._update_explosions()
        self._check_collisions()
        
        mining_this_step = self._handle_mining(space_held)

        if self.invulnerability_timer > 0:
            self.invulnerability_timer -= 1
        if self.mining_cooldown > 0:
            self.mining_cooldown -= 1
            
        return mining_this_step

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Toroidal world wrap
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _handle_mining(self, space_held):
        if not space_held or self.mining_cooldown > 0:
            return False

        mined_something = False
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.MINING_RANGE + asteroid["radius"]:
                # sound: mining_laser_loop.wav
                asteroid["ore"] -= 1
                mined_something = True
                self.mining_cooldown = self.FPS // self.MINING_RATE
                
                # Create a particle flying from asteroid to ship
                angle_to_ship = math.atan2(self.player_pos[1] - asteroid["pos"][1], self.player_pos[0] - asteroid["pos"][0])
                particle = {
                    "pos": asteroid["pos"] + np.array([math.cos(angle_to_ship), math.sin(angle_to_ship)]) * asteroid["radius"],
                    "vel": (self.player_pos - asteroid["pos"]) / (self.FPS / 2), # Takes 0.5s to reach
                    "life": self.FPS / 2
                }
                self.particles.append(particle)
                break # Mine one asteroid at a time
        return mined_something

    def _update_asteroids(self):
        # Spawn new asteroids
        spawn_chance = 0.01 + (self.ore_collected // 10) * 0.005
        if self.np_random.random() < spawn_chance and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

        # Update existing asteroids
        for asteroid in self.asteroids[:]:
            asteroid["pos"] += asteroid["vel"]
            asteroid["pos"][0] %= self.WIDTH
            asteroid["pos"][1] %= self.HEIGHT
            if asteroid["ore"] <= 0:
                self.score += 1.0 # Depletion bonus
                self.asteroids.remove(asteroid)

    def _spawn_asteroid(self):
        # Spawn away from player
        while True:
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            if np.linalg.norm(pos - self.player_pos) > 100:
                break
        
        value = self.np_random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        radius = value * 5 + 10
        ore = value * 10
        vel = self.np_random.uniform(-1, 1, 2) * 0.5
        
        self.asteroids.append({
            "pos": pos, "vel": vel, "radius": radius, "value": value, "ore": ore
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0 or np.linalg.norm(p["pos"] - self.player_pos) < self.PLAYER_SIZE:
                # sound: ore_collect.wav
                self.ore_collected += 1
                self.score += 0.1
                self.particles.remove(p)

    def _update_explosions(self):
        for e in self.explosions[:]:
            e["progress"] += 0.05
            if e["progress"] >= 1.0:
                self.explosions.remove(e)

    def _check_collisions(self):
        if self.invulnerability_timer > 0:
            return

        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid["pos"])
            if dist < self.PLAYER_SIZE + asteroid["radius"]:
                # sound: explosion.wav
                self.player_lives -= 1
                self.invulnerability_timer = self.INVULNERABILITY_FRAMES
                self.explosions.append({"pos": self.player_pos.copy(), "progress": 0, "radius": asteroid["radius"] * 3})
                self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
                break

    def _calculate_reward(self, mining_this_step):
        # Small penalty for existing encourages action
        if not mining_this_step:
            return -0.02
        return 0

    def _check_termination(self):
        return (
            self.ore_collected >= self.WIN_ORE
            or self.player_lives <= 0
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_asteroids()
        self._render_particles()
        self._render_explosions()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "ore_collected": self.ore_collected,
        }

    # --- RENDER METHODS ---

    def _render_background(self):
        for x, y, brightness in self.stars:
            self.screen.set_at((x, y), (brightness, brightness, brightness))

    def _render_asteroids(self):
        for a in self.asteroids:
            pos = a["pos"].astype(int)
            color = self.ASTEROID_COLORS[a["value"]]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], a["radius"], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], a["radius"], tuple(c*0.8 for c in color))

    def _render_player(self):
        pos = self.player_pos.astype(int)
        size = self.PLAYER_SIZE
        
        # Flash when invulnerable
        if self.invulnerability_timer > 0 and (self.invulnerability_timer // 3) % 2 == 0:
            return

        # Glow effect
        glow_radius = int(size * 1.8)
        glow_alpha = 80
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship triangle
        points = [
            (pos[0], pos[1] - size),
            (pos[0] - size / 1.5, pos[1] + size / 2),
            (pos[0] + size / 1.5, pos[1] + size / 2),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = p["pos"].astype(int)
            pygame.draw.rect(self.screen, self.COLOR_ORE_PARTICLE, (pos[0]-1, pos[1]-1, 3, 3))

    def _render_explosions(self):
        for e in self.explosions:
            p = e["progress"]
            current_radius = int(e["radius"] * math.sin(p * math.pi)) # Grow and shrink
            alpha = int(255 * (1 - p))
            if alpha > 0:
                pos = e["pos"].astype(int)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, self.COLOR_EXPLOSION + (alpha,))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.player_lives):
            points = [
                (self.WIDTH - 70 + i * 20, 12),
                (self.WIDTH - 75 + i * 20, 28),
                (self.WIDTH - 65 + i * 20, 28),
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Ore collected
        ore_bar_width = self.WIDTH // 2
        ore_bar_height = 20
        ore_bar_x = (self.WIDTH - ore_bar_width) // 2
        ore_bar_y = self.HEIGHT - 35
        
        fill_ratio = min(1.0, self.ore_collected / self.WIN_ORE)
        fill_width = int(ore_bar_width * fill_ratio)
        
        pygame.draw.rect(self.screen, (50, 50, 80), (ore_bar_x, ore_bar_y, ore_bar_width, ore_bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_ORE_PARTICLE, (ore_bar_x, ore_bar_y, fill_width, ore_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (ore_bar_x, ore_bar_y, ore_bar_width, ore_bar_height), 2)
        
        ore_text = self.font_ui.render(f"ORE: {self.ore_collected}/{self.WIN_ORE}", True, self.COLOR_UI_TEXT)
        text_rect = ore_text.get_rect(center=(self.WIDTH/2, ore_bar_y + ore_bar_height/2))
        self.screen.blit(ore_text, text_rect)
        
        # Game Over / Win Text
        if self.game_over:
            if self.ore_collected >= self.WIN_ORE:
                msg = "MISSION COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run headlessly if not playing directly

    play_game = True # Set to True to play with keyboard

    if play_game:
        os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'quartz'
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        pygame.display.set_caption("Asteroid Miner")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        terminated = False
        while not terminated:
            # --- Get Player Input ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            # --- Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Render to Display ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        
        env.close()
    else:
        # Standard Gym loop for verification
        env = GameEnv()
        obs, info = env.reset()
        for _ in range(2000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished. Final Info: {info}")
                obs, info = env.reset()
        env.close()