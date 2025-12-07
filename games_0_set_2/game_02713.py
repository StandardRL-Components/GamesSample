
# Generated: 2025-08-28T05:42:53.913988
# Source Brief: brief_02713.md
# Brief Index: 2713

        
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

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to fire. Dodge the grey asteroids."
    )

    game_description = (
        "Pilot a spaceship in a top-down arcade shooter, dodging asteroids and blasting aliens to survive for 60 seconds."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_BULLET = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    EXPLOSION_COLORS = [(255, 255, 255), (255, 200, 0), (255, 100, 0)]

    # Player settings
    PLAYER_SPEED = 5
    PLAYER_HEALTH_START = 3
    PLAYER_FIRE_COOLDOWN = 5  # frames
    PLAYER_RADIUS = 10
    PLAYER_STILLNESS_THRESHOLD = 5 # pixels
    PLAYER_STILLNESS_FRAMES = 10

    # Entity settings
    BULLET_SPEED = 10
    BULLET_RADIUS = 3
    ALIEN_SPEED = 2
    ALIEN_RADIUS = 10
    ASTEROID_SPEED = 3
    ASTEROID_RADIUS_BASE = 15

    # Spawning
    BASE_ALIEN_SPAWN_RATE = 25
    BASE_ASTEROID_SPAWN_RATE = 35
    DIFFICULTY_INTERVAL = 150 # steps to increase difficulty

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_timer = pygame.font.Font(None, 36)
        
        self.game_over_font = pygame.font.Font(None, 72)
        self.game_over_sub_font = pygame.font.Font(None, 36)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=float)
        self.player_health = self.PLAYER_HEALTH_START
        self.player_fire_cooldown_timer = 0
        self.last_player_pos = self.player_pos.copy()
        self.still_counter = 0

        self.bullets = []
        self.aliens = []
        self.asteroids = []
        self.particles = []

        self.alien_spawn_timer = self.BASE_ALIEN_SPAWN_RATE
        self.asteroid_spawn_timer = self.BASE_ASTEROID_SPAWN_RATE
        
        self.stars = [
            (
                self.rng.integers(0, self.SCREEN_WIDTH),
                self.rng.integers(0, self.SCREEN_HEIGHT),
                self.rng.random() * 1.5,
                self.rng.integers(50, 150)
            )
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Update Game State ---
        self._handle_input(movement, space_held)
        self._update_difficulty()
        self._update_spawners()
        
        # --- Move Entities ---
        self._move_bullets()
        self._move_aliens()
        self._move_asteroids()
        self._update_particles()
        
        # --- Handle Collisions & Rewards ---
        reward += self._handle_collisions()
        
        # --- Survival & Stillness Rewards ---
        reward += 0.1  # Survival reward

        dist_moved = np.linalg.norm(self.player_pos - self.last_player_pos)
        if dist_moved < self.PLAYER_STILLNESS_THRESHOLD:
            self.still_counter += 1
        else:
            self.still_counter = 0
        if self.still_counter > self.PLAYER_STILLNESS_FRAMES:
            reward -= 0.2
        self.last_player_pos = self.player_pos.copy()

        # --- Termination Check ---
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.win = False
            self._create_explosion(self.player_pos, 40, 150)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.win = True
            reward += 100  # Victory bonus

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Firing
        self.player_fire_cooldown_timer = max(0, self.player_fire_cooldown_timer - 1)
        if space_held and self.player_fire_cooldown_timer == 0:
            # SFX: Laser shoot
            self.bullets.append(self.player_pos.copy() - [0, self.PLAYER_RADIUS])
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_difficulty(self):
        difficulty_level = self.steps // self.DIFFICULTY_INTERVAL
        self.current_alien_spawn_rate = max(10, self.BASE_ALIEN_SPAWN_RATE - difficulty_level)
        self.current_asteroid_spawn_rate = max(15, self.BASE_ASTEROID_SPAWN_RATE - difficulty_level * 1.5)

    def _update_spawners(self):
        # Aliens
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            x_pos = self.rng.integers(20, self.SCREEN_WIDTH - 20)
            self.aliens.append(np.array([x_pos, -self.ALIEN_RADIUS], dtype=float))
            self.alien_spawn_timer = self.current_alien_spawn_rate
        
        # Asteroids
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            x_pos = self.rng.integers(0, self.SCREEN_WIDTH)
            vel_x = self.rng.uniform(-1.5, 1.5)
            vel_y = self.rng.uniform(0.8, 1.2)
            
            num_points = self.rng.integers(7, 12)
            base_radius = self.rng.uniform(0.8, 1.3) * self.ASTEROID_RADIUS_BASE
            
            points = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                radius_variation = self.rng.uniform(0.75, 1.25)
                r = base_radius * radius_variation
                points.append((math.cos(angle) * r, math.sin(angle) * r))

            self.asteroids.append({
                "pos": np.array([x_pos, -base_radius], dtype=float),
                "vel": np.array([vel_x, vel_y]) * self.ASTEROID_SPEED,
                "radius": base_radius,
                "points": points,
            })
            self.asteroid_spawn_timer = self.current_asteroid_spawn_rate

    def _move_entities(self, entities, speed):
        for entity in entities:
            entity += speed
        return [e for e in entities if -20 < e[0] < self.SCREEN_WIDTH + 20 and -20 < e[1] < self.SCREEN_HEIGHT + 20]

    def _move_bullets(self):
        self.bullets = self._move_entities(self.bullets, np.array([0, -self.BULLET_SPEED]))

    def _move_aliens(self):
        self.aliens = self._move_entities(self.aliens, np.array([0, self.ALIEN_SPEED]))

    def _move_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
        self.asteroids = [a for a in self.asteroids if -50 < a["pos"][0] < self.SCREEN_WIDTH + 50 and -50 < a["pos"][1] < self.SCREEN_HEIGHT + 50]
        
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Aliens
        for b_idx, bullet_pos in reversed(list(enumerate(self.bullets))):
            for a_idx, alien_pos in reversed(list(enumerate(self.aliens))):
                if np.linalg.norm(bullet_pos - alien_pos) < self.BULLET_RADIUS + self.ALIEN_RADIUS:
                    # SFX: Alien explosion
                    self._create_explosion(alien_pos, 15, 30)
                    self.aliens.pop(a_idx)
                    self.bullets.pop(b_idx)
                    self.score += 10
                    reward += 10
                    break
        
        # Player vs Asteroids
        for a_idx, asteroid in reversed(list(enumerate(self.asteroids))):
            if np.linalg.norm(self.player_pos - asteroid["pos"]) < self.PLAYER_RADIUS + asteroid["radius"]:
                # SFX: Player hit/shield down
                self._create_explosion(self.player_pos, 25, 60)
                self.asteroids.pop(a_idx)
                self.player_health -= 1
                self.score = max(0, self.score - 50)
                reward -= 5
                break # Only one collision per frame
                
        return reward

    def _create_explosion(self, pos, count, base_life):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "life": self.rng.integers(base_life // 2, base_life),
                "max_life": base_life,
                "color": random.choice(self.EXPLOSION_COLORS)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        
        for asteroid in self.asteroids: self._render_asteroid(asteroid)
        for alien_pos in self.aliens: self._render_alien(alien_pos)
        for bullet_pos in self.bullets: self._render_bullet(bullet_pos)
        if self.player_health > 0: self._render_player()
        
        self._render_particles()
        self._render_ui()
        
        if self.game_over: self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size, alpha in self.stars:
            pygame.draw.circle(self.screen, (alpha, alpha, alpha), (x, y), size)

    def _render_player(self):
        pos = self.player_pos.astype(int)
        points = [
            (pos[0], pos[1] - self.PLAYER_RADIUS),
            (pos[0] - self.PLAYER_RADIUS, pos[1] + self.PLAYER_RADIUS),
            (pos[0] + self.PLAYER_RADIUS, pos[1] + self.PLAYER_RADIUS)
        ]
        # Glow
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        # Ship
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_alien(self, pos):
        p = pos.astype(int)
        size = self.ALIEN_RADIUS
        points = [
            (p[0], p[1] + size),
            (p[0] - size, p[1] - size),
            (p[0] + size, p[1] - size)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
        
    def _render_asteroid(self, asteroid):
        points_abs = [(p[0] + asteroid["pos"][0], p[1] + asteroid["pos"][1]) for p in asteroid["points"]]
        pygame.gfxdraw.filled_polygon(self.screen, points_abs, self.COLOR_ASTEROID)
        pygame.gfxdraw.aapolygon(self.screen, points_abs, self.COLOR_ASTEROID)

    def _render_bullet(self, pos):
        p = pos.astype(int)
        pygame.draw.circle(self.screen, self.COLOR_BULLET, p, self.BULLET_RADIUS)
        pygame.gfxdraw.aacircle(self.screen, p[0], p[1], self.BULLET_RADIUS, self.COLOR_BULLET)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            color = p["color"]
            alpha = int(255 * life_ratio)
            radius = int(life_ratio * 4)
            if radius > 0:
                pos = p["pos"].astype(int)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        for i in range(self.player_health):
            pos = (self.SCREEN_WIDTH - 20 - i * 25, 20)
            points = [(pos[0], pos[1] - 8), (pos[0] - 8, pos[1] + 8), (pos[0] + 8, pos[1] + 8)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_timer.render(f"{time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))
        
    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        msg = "VICTORY" if self.win else "GAME OVER"
        color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
        
        main_text = self.game_over_font.render(msg, True, color)
        main_rect = main_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
        self.screen.blit(main_text, main_rect)
        
        sub_text = self.game_over_sub_font.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
        sub_rect = sub_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
        self.screen.blit(sub_text, sub_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # Game loop
    while not done:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Print info
        if env.steps % GameEnv.FPS == 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Total Reward: {total_reward:.2f}")

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the final screen visible for a few seconds
    pygame.time.wait(3000)
    
    env.close()