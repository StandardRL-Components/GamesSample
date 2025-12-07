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


# A helper class for particles to create visual effects like explosions.
class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.vel *= 0.98  # friction

    def draw(self, surface):
        if self.lifetime <= 0:
            return

        # Fade out effect based on lifetime
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        alpha = max(0, min(255, alpha))

        # Shrink effect
        current_size = int(self.size * (self.lifetime / self.max_lifetime))
        if current_size <= 0:
            return

        # Use a temporary surface for alpha blending to avoid affecting other drawings
        particle_surf = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, (*self.color, alpha), (current_size, current_size), current_size)
        surface.blit(particle_surf, (int(self.pos.x - current_size), int(self.pos.y - current_size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True

    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_WALL = (80, 80, 90)
    COLOR_TEXT = (230, 230, 230)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_BULLET = (255, 255, 100)
    COLOR_AMMO = (100, 150, 255)
    COLOR_AMMO_GLOW = (150, 200, 255)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.0
    PLAYER_HEALTH_MAX = 100
    PLAYER_AMMO_MAX = 100
    PLAYER_AMMO_START = 50

    # Zombies
    ZOMBIE_SIZE = 10
    ZOMBIE_SPEED = 1.2
    ZOMBIE_HEALTH = 20
    ZOMBIE_DAMAGE = 10
    INITIAL_ZOMBIES = 10
    ZOMBIE_SPAWN_RATE = 2 * FPS  # Every 2 seconds

    # Bullets
    BULLET_SIZE = 3
    BULLET_SPEED = 10.0
    BULLET_COOLDOWN = 5  # frames between shots

    # Pickups
    AMMO_PICKUP_VALUE = 20
    AMMO_PICKUP_SPAWN_RATE = 10 * FPS  # Every 10 seconds

    # Rewards
    REWARD_SURVIVE = 0.01  # Scaled down to keep total rewards reasonable
    REWARD_KILL = 1.0
    REWARD_AMMO = 0.5
    REWARD_WIN = 50.0
    REWARD_LOSE = -50.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_health = 0
        self.player_ammo = 0
        self.player_aim_dir = pygame.math.Vector2(0, -1)
        self.shoot_cooldown = 0
        self.damage_flash_timer = 0
        self.zombies = []
        self.bullets = []
        self.ammo_pickups = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over_message = ""

        self.reset()
        # self.validate_implementation() # This is for internal testing, not part of the final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_ammo = self.PLAYER_AMMO_START
        self.player_aim_dir = pygame.math.Vector2(0, -1)

        self.shoot_cooldown = 0
        self.damage_flash_timer = 0

        self.zombies = []
        self.bullets = []
        self.ammo_pickups = []
        self.particles = []

        for _ in range(self.INITIAL_ZOMBIES):
            self._spawn_zombie()

        self.steps = 0
        self.score = 0
        self.game_over_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = self.REWARD_SURVIVE
        terminated = False

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.player_health > 0:
            self._handle_input(movement, space_held)
            reward += self._update_game_state()

        self.steps += 1

        if self.player_health <= 0 and not self.game_over_message:
            terminated = True
            reward = self.REWARD_LOSE
            self.game_over_message = "YOU DIED"
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            # sfx: player_death_explosion
        elif self.steps >= self.MAX_STEPS and not self.game_over_message:
            terminated = True
            reward = self.REWARD_WIN
            self.game_over_message = "YOU SURVIVED!"
            # sfx: victory_fanfare

        truncated = False # Game does not truncate based on time limit, it terminates with a win.
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # --- Movement and Aiming ---
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1  # Right

        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            # FIX: pygame.math.Vector2 does not have a .copy() method.
            # Create a new Vector2 object to copy the value.
            self.player_aim_dir = pygame.math.Vector2(move_vec)

        # Clamp player position to stay within walls
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.WIDTH - self.PLAYER_SIZE, self.player_pos.x))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.HEIGHT - self.PLAYER_SIZE, self.player_pos.y))

        # --- Shooting ---
        if space_held and self.player_ammo > 0 and self.shoot_cooldown == 0:
            self._shoot()

    def _shoot(self):
        self.player_ammo -= 1
        self.shoot_cooldown = self.BULLET_COOLDOWN
        bullet_pos = self.player_pos + self.player_aim_dir * (self.PLAYER_SIZE + 5)
        self.bullets.append({
            "pos": bullet_pos,
            "vel": self.player_aim_dir * self.BULLET_SPEED
        })
        # sfx: laser_shoot
        # Muzzle flash effect
        flash_pos = self.player_pos + self.player_aim_dir * self.PLAYER_SIZE
        for _ in range(5):
            vel = self.player_aim_dir.rotate(self.np_random.uniform(-45, 45)) * self.np_random.uniform(1, 3)
            self.particles.append(Particle(flash_pos, vel, self.COLOR_BULLET, self.np_random.integers(1, 4), 5))

    def _update_game_state(self):
        step_reward = 0

        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.damage_flash_timer = max(0, self.damage_flash_timer - 1)

        step_reward += self._update_bullets()
        self._update_zombies()
        step_reward += self._update_pickups()
        self._update_particles()
        self._spawn_logic()

        return step_reward

    def _update_bullets(self):
        reward = 0
        bullets_to_keep = []
        for bullet in self.bullets:
            bullet["pos"] += bullet["vel"]

            # Check wall collision
            if not (0 < bullet["pos"].x < self.WIDTH and 0 < bullet["pos"].y < self.HEIGHT):
                continue

            # Check zombie collision
            hit_zombie = False
            for zombie in self.zombies:
                if (bullet["pos"] - zombie["pos"]).length() < self.ZOMBIE_SIZE + self.BULLET_SIZE:
                    zombie["health"] -= self.ZOMBIE_HEALTH  # One-shot kill
                    if zombie["health"] <= 0:
                        self.score += 10
                        reward += self.REWARD_KILL
                        self._create_explosion(zombie["pos"], self.COLOR_ZOMBIE, 20)
                        # sfx: zombie_death
                    hit_zombie = True
                    break

            if hit_zombie:
                self.zombies = [z for z in self.zombies if z["health"] > 0]
            else:
                bullets_to_keep.append(bullet)

        self.bullets = bullets_to_keep
        return reward

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = (self.player_pos - zombie["pos"])
            if direction.length_squared() > 0:
                direction.normalize_ip()
                zombie["pos"] += direction * self.ZOMBIE_SPEED

            # Player-zombie collision
            if (self.player_pos - zombie["pos"]).length() < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                if self.damage_flash_timer == 0:
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.damage_flash_timer = 20  # frames of immunity/flashing
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER, 10)
                    # sfx: player_hit
                # Knockback zombie
                zombie["pos"] -= direction * (self.PLAYER_SIZE + self.ZOMBIE_SIZE)

    def _update_pickups(self):
        reward = 0
        pickups_to_keep = []
        for pickup in self.ammo_pickups:
            if (self.player_pos - pickup["pos"]).length() < self.PLAYER_SIZE + 10:  # 10 is pickup radius
                self.player_ammo = min(self.PLAYER_AMMO_MAX, self.player_ammo + self.AMMO_PICKUP_VALUE)
                self.score += 5
                reward += self.REWARD_AMMO
                # sfx: pickup_ammo
                self._create_explosion(pickup["pos"], self.COLOR_AMMO, 15)
            else:
                pickups_to_keep.append(pickup)
        self.ammo_pickups = pickups_to_keep
        return reward

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _spawn_logic(self):
        if self.steps > 0:
            if self.steps % self.ZOMBIE_SPAWN_RATE == 0:
                self._spawn_zombie()
            if self.steps % self.AMMO_PICKUP_SPAWN_RATE == 0:
                self._spawn_ammo_pickup()

    def _spawn_zombie(self):
        pos = self._get_random_spawn_pos()
        self.zombies.append({
            "pos": pos,
            "health": self.ZOMBIE_HEALTH
        })

    def _spawn_ammo_pickup(self):
        pos = self._get_random_spawn_pos()
        self.ammo_pickups.append({"pos": pos})

    def _get_random_spawn_pos(self):
        while True:
            pos = pygame.math.Vector2(
                self.np_random.uniform(self.ZOMBIE_SIZE, self.WIDTH - self.ZOMBIE_SIZE),
                self.np_random.uniform(self.ZOMBIE_SIZE, self.HEIGHT - self.ZOMBIE_SIZE)
            )
            if (pos - self.player_pos).length() > 100:  # Don't spawn on top of player
                return pos

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.math.Vector2(speed, 0).rotate(angle)
            lifetime = self.np_random.integers(10, 30)
            size = self.np_random.integers(2, 6)
            self.particles.append(Particle(pos, vel, color, size, lifetime))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena walls (just for visual effect)
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw ammo pickups
        for pickup in self.ammo_pickups:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
            glow_radius = int(10 + pulse * 5)
            glow_alpha = int(50 + pulse * 50)
            pygame.gfxdraw.filled_circle(self.screen, int(pickup["pos"].x), int(pickup["pos"].y), glow_radius,
                                          (*self.COLOR_AMMO_GLOW, glow_alpha))
            pygame.gfxdraw.aacircle(self.screen, int(pickup["pos"].x), int(pickup["pos"].y), 10, self.COLOR_AMMO)
            pygame.gfxdraw.filled_circle(self.screen, int(pickup["pos"].x), int(pickup["pos"].y), 10, self.COLOR_AMMO)

        # Draw zombies
        for zombie in self.zombies:
            r = pygame.Rect(zombie["pos"].x - self.ZOMBIE_SIZE, zombie["pos"].y - self.ZOMBIE_SIZE,
                            self.ZOMBIE_SIZE * 2, self.ZOMBIE_SIZE * 2)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, r)

        # Draw player
        if self.player_health > 0:
            color = self.COLOR_PLAYER
            if self.damage_flash_timer > 0 and self.steps % 4 < 2:
                color = (255, 255, 255)  # Flash white when hit

            # Player triangle
            p1 = self.player_pos + self.player_aim_dir * self.PLAYER_SIZE
            p2 = self.player_pos + self.player_aim_dir.rotate(140) * self.PLAYER_SIZE * 0.8
            p3 = self.player_pos + self.player_aim_dir.rotate(-140) * self.PLAYER_SIZE * 0.8
            points = [(int(p.x), int(p.y)) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw bullets
        for bullet in self.bullets:
            pos = (int(bullet["pos"].x), int(bullet["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BULLET_SIZE, self.COLOR_BULLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BULLET_SIZE, self.COLOR_BULLET)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Health bar
        health_pct = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, (80, 0, 0), (10, self.HEIGHT - 30, bar_width, bar_height))
        pygame.draw.rect(self.screen, (200, 0, 0), (10, self.HEIGHT - 30, int(bar_width * health_pct), bar_height))

        # Score and Ammo
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, 30))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over_message:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    # To run this, you might need to comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Forcing a graphical driver if the dummy was set
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    terminated = False

    # Use a dummy screen for display if running this file directly
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0  # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Display ---
        # The observation is (H, W, C), but pygame blit needs (W, H) surface
        # We can just re-use the internal screen from the env for display
        display_screen.blit(env.screen, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)  # Wait 2 seconds before closing
    env.close()