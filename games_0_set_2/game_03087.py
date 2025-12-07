import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Player:
    """A simple class to hold player state, including the rect for position/collision."""
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.vy = 0
        self.on_ground = False


class Enemy:
    """A simple class to hold enemy state, including the rect for position/collision."""
    def __init__(self, x, y, width, height, fire_timer_offset):
        self.rect = pygame.Rect(x, y, width, height)
        self.health = 1
        self.fire_timer = fire_timer_offset


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ← and → to move, ↑ to jump. Press space to fire your weapon."
    )

    # User-facing game description
    game_description = (
        "A fast-paced, side-scrolling shooter. Navigate a hostile landscape, blast enemies, and reach the end of the level."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.VICTORY_X = 600

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (150, 255, 200)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_ENEMY_GLOW = (255, 150, 170)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (255, 100, 50)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_GROUND = (40, 30, 60)

        # Physics & Gameplay
        self.PLAYER_SPEED = 6
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.PLAYER_FIRE_COOLDOWN = 6  # frames
        self.ENEMY_FIRE_COOLDOWN = 45  # frames
        self.GROUND_Y = self.HEIGHT - 40

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Internal State ---
        self.player = None
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.parallax_stars = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""  # "VICTORY" or "DEFEAT"

        self.player_fire_timer = 0
        self.enemy_spawn_timer = 0
        self.enemy_spawn_interval = 50
        self.enemy_projectile_speed = 4.0

        self.np_random = None
        self.player_health = 3

        # self.reset() # Called from higher level wrapper

    def _create_player(self):
        player = Player(50, self.GROUND_Y - 40, 25, 40)
        return player

    def _create_enemy(self):
        x = self.np_random.integers(self.WIDTH // 2, self.WIDTH - 30)
        y = self.np_random.integers(100, self.GROUND_Y - 30)
        fire_timer_offset = self.np_random.integers(0, self.ENEMY_FIRE_COOLDOWN)
        enemy = Enemy(x, y, 30, 30, fire_timer_offset)
        return enemy

    def _create_particle(self, pos, count, color, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(*life_range)
            self.particles.append([pygame.math.Vector2(pos), vel, life, color])

    def _create_parallax_stars(self):
        self.parallax_stars = []
        for i in range(3):  # 3 layers
            layer = []
            for _ in range(50 * (i + 1)):
                x = self.np_random.integers(0, self.WIDTH)
                y = self.np_random.integers(0, self.GROUND_Y)
                size = self.np_random.integers(1, 3)
                layer.append((x, y, size))
            self.parallax_stars.append({'layer': layer, 'speed': 0.2 * (i + 1)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.player = self._create_player()
        self.enemies = [self._create_enemy() for _ in range(10)]
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        self._create_parallax_stars()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        self.player_fire_timer = 0
        self.enemy_spawn_timer = 0
        self.enemy_spawn_interval = 50
        self.enemy_projectile_speed = 4.0

        self.player_health = 3

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        prev_player_x = self.player.rect.x

        if movement == 1 and self.player.on_ground:  # Jump
            self.player.vy = self.JUMP_STRENGTH
            self.player.on_ground = False
        if movement == 3:  # Left
            self.player.rect.x -= self.PLAYER_SPEED
        if movement == 4:  # Right
            self.player.rect.x += self.PLAYER_SPEED

        if space_held and self.player_fire_timer <= 0:
            proj = pygame.Rect(self.player.rect.right, self.player.rect.centery - 2, 12, 4)
            self.player_projectiles.append(proj)
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
            self._create_particle(proj.center, 10, self.COLOR_PLAYER_PROJ, (2, 5), (5, 10))

        # --- Update Game State ---
        self.steps += 1
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_spawn_interval = max(10, self.enemy_spawn_interval - 1)
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_projectile_speed = min(10.0, self.enemy_projectile_speed + 0.05)

        # Update Player
        self.player.vy += self.GRAVITY
        self.player.rect.y += self.player.vy
        self.player.on_ground = False
        if self.player.rect.bottom >= self.GROUND_Y:
            self.player.rect.bottom = self.GROUND_Y
            self.player.vy = 0
            self.player.on_ground = True
        self.player.rect.left = max(0, self.player.rect.left)
        self.player.rect.right = min(self.WIDTH, self.player.rect.right)

        # Update Enemies
        for enemy in self.enemies:
            # Simple AI: move towards player if far, shoot
            if abs(self.player.rect.x - enemy.rect.x) > 150:
                if self.player.rect.x > enemy.rect.x:
                    enemy.rect.x += 1
                else:
                    enemy.rect.x -= 1

            enemy.fire_timer -= 1
            if enemy.fire_timer <= 0:
                direction = pygame.math.Vector2(self.player.rect.center) - pygame.math.Vector2(enemy.rect.center)
                if direction.length() > 0:
                    direction.normalize_ip()
                proj_vel = direction * self.enemy_projectile_speed
                proj_pos = pygame.math.Vector2(enemy.rect.center)
                self.enemy_projectiles.append({'pos': proj_pos, 'vel': proj_vel})
                enemy.fire_timer = self.ENEMY_FIRE_COOLDOWN + self.np_random.integers(-10, 10)

        # Spawn new enemies
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemies.append(self._create_enemy())
            self.enemy_spawn_timer = self.enemy_spawn_interval

        # Update Projectiles
        self.player_projectiles = [p for p in self.player_projectiles if p.right < self.WIDTH]
        for p in self.player_projectiles:
            p.x += 15

        updated_enemy_proj = []
        for p in self.enemy_projectiles:
            p['pos'] += p['vel']
            if 0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT:
                updated_enemy_proj.append(p)
        self.enemy_projectiles = updated_enemy_proj

        # Update Particles
        self.particles = [[p[0] + p[1], p[1] * 0.9, p[2] - 1, p[3]] for p in self.particles if p[2] > 0]

        # --- Collision Detection ---
        # Player projectiles vs Enemies
        enemies_hit = []
        for proj in self.player_projectiles:
            for enemy in self.enemies:
                if enemy.rect.colliderect(proj):
                    enemies_hit.append(enemy)
                    self.player_projectiles.remove(proj)
                    self.score += 10
                    reward += 1
                    self._create_particle(enemy.rect.center, 30, self.COLOR_PARTICLE, (1, 4), (15, 30))
                    break
        self.enemies = [e for e in self.enemies if e not in enemies_hit]

        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles:
            proj_rect = pygame.Rect(proj['pos'].x - 3, proj['pos'].y - 3, 6, 6)
            if self.player.rect.colliderect(proj_rect):
                self.enemy_projectiles.remove(proj)
                self.player_health -= 1
                reward -= 1
                self._create_particle(self.player.rect.center, 20, self.COLOR_PLAYER, (1, 3), (10, 20))
                break

        # --- Reward Shaping ---
        if self.player.rect.x > prev_player_x:
            reward += 0.1  # Moved towards goal
        elif self.player.rect.x < prev_player_x:
            reward -= 0.02  # Moved away from goal

        # --- Check Termination Conditions ---
        truncated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            self.game_outcome = "DEFEAT"
            reward = -100
        elif self.player.rect.x >= self.VICTORY_X:
            terminated = True
            self.game_over = True
            self.game_outcome = "VICTORY"
            reward = 100
            self.score += 1000
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard is to set both true on truncation

        # self.clock.tick(self.FPS) # Not needed for headless mode

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Parallax stars
        for layer_data in self.parallax_stars:
            speed = layer_data['speed']
            # A fake camera effect for a static world
            offset_x = (self.steps * speed) % self.WIDTH
            for x, y, size in layer_data['layer']:
                px = (x - offset_x) % self.WIDTH
                color_val = int(80 * speed)
                pygame.draw.rect(self.screen, (color_val, color_val, color_val + 20), (px, y, size, size))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        # Goal line
        pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, (self.VICTORY_X, 0), (self.VICTORY_X, self.HEIGHT), 2)

    def _render_game(self):
        self._render_background()

        # Particles
        for p in self.particles:
            pos, _, life, color = p
            alpha = max(0, min(255, int(life * 15)))
            radius = max(1, int(life / 5))
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, color + (alpha,))
            except TypeError: # Handle potential color tuple issues
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (color[0], color[1], color[2], alpha))

        # Player Projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, p)
            pygame.draw.line(self.screen, self.COLOR_PARTICLE, p.center, (p.centerx - 8, p.centery), 2)

        # Enemy Projectiles
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 4, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), 4, self.COLOR_ENEMY_PROJ)

        # Enemies
        for enemy in self.enemies:
            pygame.gfxdraw.box(self.screen, enemy.rect, self.COLOR_ENEMY_GLOW + (100,))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy.rect, 2)
            eye_y = enemy.rect.y + 8
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (enemy.rect.x + 8, eye_y, 14, 4))

        # Player
        if self.player:
            glow_rect = self.player.rect.inflate(10, 10)
            pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_PLAYER_GLOW + (80,))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player.rect)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Health
        health_surf = self.font_ui.render(f"HEALTH: {self.player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_surf, (10, 10))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text = self.game_outcome
            color = self.COLOR_PLAYER if self.game_outcome == "VICTORY" else self.COLOR_ENEMY

            end_surf = self.font_game_over.render(text, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        player_pos = (self.player.rect.x, self.player.rect.y) if self.player else (0,0)
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "player_pos": player_pos,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # For interactive play
    import pygame

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Side-Scrolling Shooter")
    clock = pygame.time.Clock()

    terminated = False
    running = True

    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()

        movement = 0  # no-op
        if keys[pygame.K_UP]:
            movement = 1
        # Action 2 (down) is unused
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            obs, reward, term, trunc, info = env.step(action)
            terminated = term or trunc

        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()
    pygame.quit()