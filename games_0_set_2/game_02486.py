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
        "Controls: Arrow keys to move. Hold space to fire. "
        "Dodge the red triangles and clear the arena!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a robot in a top-down arena, blasting waves of enemies. "
        "Defeat all 25 enemies to win, but watch your health!"
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_ARENA = (40, 40, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_HEALTH_GOOD = (0, 200, 100)
    COLOR_HEALTH_BAD = (100, 20, 20)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SPAWN_POINT = (100, 0, 0)

    # Game parameters
    FPS = 30
    MAX_STEPS = 2000
    TOTAL_ENEMIES = 25
    ARENA_PADDING = 20

    # Player
    PLAYER_SIZE = 20
    PLAYER_SPEED = 5
    PLAYER_MAX_HEALTH = 100
    PLAYER_FIRE_COOLDOWN = 5  # frames

    # Enemy
    ENEMY_SIZE = 18
    ENEMY_SPEED = 1.5
    ENEMY_MAX_HEALTH = 30
    ENEMY_CONTACT_DAMAGE = 20
    INITIAL_SPAWN_RATE = 0.5  # spawns per second
    MAX_SPAWN_RATE = 2.0  # spawns per second

    # Projectile
    PROJECTILE_SPEED = 15
    PROJECTILE_DAMAGE = 10
    PROJECTILE_WIDTH = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.arena_rect = pygame.Rect(
            self.ARENA_PADDING, self.ARENA_PADDING,
            self.SCREEN_WIDTH - 2 * self.ARENA_PADDING,
            self.SCREEN_HEIGHT - 2 * self.ARENA_PADDING
        )

        self._init_spawn_points()

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_aim_direction = None
        self.player_last_fire_step = None
        self.last_space_held = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.enemies_defeated = None
        self.enemies_to_spawn = None
        self.spawn_timer = None
        self.current_spawn_rate = None
        self.np_random = None

    def _init_spawn_points(self):
        self.spawn_points = []
        pad = self.ARENA_PADDING + 10
        w, h = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        for i in range(1, 4):
            self.spawn_points.append(pygame.math.Vector2(w * i / 4, pad))
            self.spawn_points.append(pygame.math.Vector2(w * i / 4, h - pad))
        self.spawn_points.append(pygame.math.Vector2(pad, h / 3))
        self.spawn_points.append(pygame.math.Vector2(pad, h * 2 / 3))
        self.spawn_points.append(pygame.math.Vector2(w - pad, h / 3))
        self.spawn_points.append(pygame.math.Vector2(w - pad, h * 2 / 3))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_direction = pygame.math.Vector2(0, -1)  # Start aiming up
        self.player_last_fire_step = -self.PLAYER_FIRE_COOLDOWN
        self.last_space_held = False

        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.enemies_defeated = 0
        self.enemies_to_spawn = self.TOTAL_ENEMIES
        self.spawn_timer = 0
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # The episode has ended, but the user may still step.
            # Return the final state without updating the simulation.
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        player_vel = pygame.math.Vector2(0, 0)
        if movement == 1: player_vel.y = -1
        elif movement == 2: player_vel.y = 1
        elif movement == 3: player_vel.x = -1
        elif movement == 4: player_vel.x = 1

        if player_vel.length() > 0:
            player_vel.normalize_ip()
            self.player_aim_direction = player_vel.copy()

        self.player_pos += player_vel * self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.arena_rect.left, self.arena_rect.right)
        self.player_pos.y = np.clip(self.player_pos.y, self.arena_rect.top, self.arena_rect.bottom)

        if space_held and (self.steps > self.player_last_fire_step + self.PLAYER_FIRE_COOLDOWN):
            # Fire on hold, not just press, for more arcade feel
            self.player_last_fire_step = self.steps
            proj_pos = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE / 2)
            self.projectiles.append({
                "pos": proj_pos,
                "vel": self.player_aim_direction * self.PROJECTILE_SPEED
            })
            # Sound: Player shoot

        # --- Update Game State ---

        # Update spawn rate
        if self.steps > 0 and self.steps % 100 == 0:
            self.current_spawn_rate = min(self.MAX_SPAWN_RATE, self.current_spawn_rate + 0.1)

        # Spawn enemies
        self.spawn_timer += self.current_spawn_rate / self.FPS
        if self.spawn_timer >= 1 and self.enemies_to_spawn > 0 and len(self.enemies) < 15:
            self.spawn_timer -= 1
            self._spawn_enemy()

        # Update projectiles
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            if not self.arena_rect.collidepoint(proj["pos"]):
                self.projectiles.remove(proj)
                reward -= 0.02 # Miss penalty
                continue

            # Projectile trail
            if self.np_random.random() < 0.5:
                self._create_particles(proj["pos"], 1, self.COLOR_PROJECTILE, 1, 3, 0.5, 5)

        # Update enemies
        for enemy in self.enemies:
            direction = (self.player_pos - enemy["pos"])
            if direction.length() > 0:
                direction.normalize_ip()
            enemy["pos"] += direction * self.ENEMY_SPEED

        # --- Collision Detection ---

        # Projectiles vs Enemies
        for proj in self.projectiles[:]:
            for enemy in self.enemies[:]:
                if proj in self.projectiles and enemy in self.enemies:
                    if (proj["pos"] - enemy["pos"]).length() < self.ENEMY_SIZE / 2:
                        self.projectiles.remove(proj)
                        enemy["health"] -= self.PROJECTILE_DAMAGE
                        reward += 0.1 # Hit reward
                        self._create_particles(proj["pos"], 5, self.COLOR_ENEMY, 2, 5, 2, 10)
                        # Sound: Enemy hit
                        if enemy["health"] <= 0:
                            self.enemies.remove(enemy)
                            self.score += 10
                            reward += 1 # Kill reward
                            self.enemies_defeated += 1
                            self._create_particles(enemy["pos"], 30, (255, 150, 0), 3, 15, 4, 20, is_explosion=True)
                            # Sound: Explosion
                        break

        # Enemies vs Player
        for enemy in self.enemies:
            if (self.player_pos - enemy["pos"]).length() < self.PLAYER_SIZE/2 + self.ENEMY_SIZE/2:
                self.player_health -= self.ENEMY_CONTACT_DAMAGE
                reward -= 1.0 # Damage penalty
                self._create_particles(self.player_pos, 10, self.COLOR_PLAYER, 2, 5, 3, 15)
                # Sound: Player damage
                # Push enemy back slightly to prevent constant damage
                direction = (enemy["pos"] - self.player_pos)
                if direction.length() > 0:
                    enemy["pos"] += direction.normalize() * 5

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["is_explosion"]:
                p["radius"] *= 0.95
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # --- Termination Check ---
        terminated = False
        truncated = False # This env doesn't truncate
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            # Sound: Player death
        elif self.enemies_defeated >= self.TOTAL_ENEMIES:
            reward += 100
            terminated = True
            # Sound: Victory
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Using terminated for timeout, could also be truncated

        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_enemy(self):
        # FIX: np.random.choice can convert list of vectors into a 2D numpy array,
        # causing errors. Use indexing for robust selection.
        idx = self.np_random.integers(len(self.spawn_points))
        spawn_pos = self.spawn_points[idx].copy()

        self.enemies.append({
            "pos": spawn_pos,
            "health": self.ENEMY_MAX_HEALTH,
        })
        self.enemies_to_spawn -= 1

    def _create_particles(self, pos, count, color, min_radius, max_radius, speed, lifespan, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            # FIX: The original `uniform(1, speed)` failed if speed < 1.
            # `uniform(0, speed)` is robust for any non-negative speed.
            vel_speed = self.np_random.uniform(0, speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * vel_speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(min_radius, max_radius),
                "color": color,
                "lifespan": self.np_random.integers(lifespan // 2, lifespan + 1),
                "is_explosion": is_explosion
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, self.arena_rect, border_radius=10)

        # Render spawn points
        for sp in self.spawn_points:
            pygame.gfxdraw.filled_circle(self.screen, int(sp.x), int(sp.y), 5, self.COLOR_SPAWN_POINT)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20))
            if alpha > 0:
                color = (*p["color"], alpha)
                surf = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p["radius"], p["radius"]), p["radius"])
                self.screen.blit(surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Render projectiles
        for proj in self.projectiles:
            start_pos = (int(proj["pos"].x), int(proj["pos"].y))
            end_pos_vec = proj["pos"] - proj["vel"] * 0.5
            end_pos = (int(end_pos_vec.x), int(end_pos_vec.y))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, self.PROJECTILE_WIDTH)

        # Render enemies
        for enemy in self.enemies:
            # Glow effect
            glow_surf = pygame.Surface((self.ENEMY_SIZE*2.5, self.ENEMY_SIZE*2.5), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ENEMY_GLOW, (self.ENEMY_SIZE*1.25, self.ENEMY_SIZE*1.25), self.ENEMY_SIZE*0.75)
            self.screen.blit(glow_surf, (int(enemy["pos"].x - self.ENEMY_SIZE*1.25), int(enemy["pos"].y - self.ENEMY_SIZE*1.25)), special_flags=pygame.BLEND_RGBA_ADD)

            # Triangle body
            angle_target = self.player_pos
            if (angle_target - enemy["pos"]).length_squared() > 0:
                 angle = (angle_target - enemy["pos"]).angle_to(pygame.math.Vector2(1, 0))
            else:
                 angle = 0
            p1 = enemy["pos"] + pygame.math.Vector2(self.ENEMY_SIZE / 2, 0).rotate(-angle)
            p2 = enemy["pos"] + pygame.math.Vector2(-self.ENEMY_SIZE / 2, self.ENEMY_SIZE / 3).rotate(-angle)
            p3 = enemy["pos"] + pygame.math.Vector2(-self.ENEMY_SIZE / 2, -self.ENEMY_SIZE / 3).rotate(-angle)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

            # Health bar
            if enemy["health"] < self.ENEMY_MAX_HEALTH:
                bar_w = self.ENEMY_SIZE * 1.5
                bar_h = 4
                bar_x = enemy["pos"].x - bar_w / 2
                bar_y = enemy["pos"].y - self.ENEMY_SIZE
                health_pct = enemy["health"] / self.ENEMY_MAX_HEALTH
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAD, (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_GOOD, (bar_x, bar_y, bar_w * health_pct, bar_h))

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))

        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_SIZE*3, self.PLAYER_SIZE*3), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE*1.5, self.PLAYER_SIZE*1.5), self.PLAYER_SIZE)
        self.screen.blit(glow_surf, (player_rect.centerx - self.PLAYER_SIZE*1.5, player_rect.centery - self.PLAYER_SIZE*1.5), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Aim indicator
        aim_start = self.player_pos
        aim_end = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, aim_start, aim_end, 2)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Player Health
        health_text = self.font_large.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 110, 5))
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAD, (self.SCREEN_WIDTH - 105, 8, 100, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GOOD, (self.SCREEN_WIDTH - 105, 8, 100 * health_pct, 15))

        # Enemies Remaining
        enemies_rem_text = self.font_small.render(
            f"ENEMIES REMAINING: {self.TOTAL_ENEMIES - self.enemies_defeated}", True, self.COLOR_UI_TEXT
        )
        text_rect = enemies_rem_text.get_rect(centerx=self.SCREEN_WIDTH / 2, bottom=self.SCREEN_HEIGHT - 5)
        self.screen.blit(enemies_rem_text, text_rect)

        if self.game_over:
            if self.enemies_defeated >= self.TOTAL_ENEMIES:
                end_text_str = "VICTORY"
            else:
                end_text_str = "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "enemies_defeated": self.enemies_defeated,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will open a window for rendering, separate from the headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    import pygame

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Arena Shooter")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame rendering ---
        # The observation is already a rendered surface array
        # We just need to convert it back to a surface and blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()