import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:54:04.665447
# Source Brief: brief_01259.md
# Brief Index: 1259
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque


# Helper classes for game objects
class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        if alpha > 0:
            current_color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos.x), int(self.pos.y), int(self.radius), current_color
            )


class Projectile:
    def __init__(self, pos, vel):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.trail = deque(maxlen=10)
        self.radius = 3

    def update(self, gravity, portals):
        self.trail.append(self.pos.copy())
        self.vel += gravity
        self.pos += self.vel

        for i, portal in enumerate(portals):
            if (
                portal.pos.distance_to(self.pos) < portal.radius
                and portal.cooldown == 0
            ):
                other_portal = portals[1 - i]
                if other_portal.cooldown == 0:
                    # Teleport
                    self.pos = (
                        other_portal.pos.copy()
                        + self.vel.normalize() * (portal.radius + 1)
                    )
                    portal.cooldown = 20
                    other_portal.cooldown = 20
                    # SFX: Portal travel
                    return True  # Indicate teleported
        return False

    def draw(self, surface, color):
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = int(255 * (i / len(self.trail)))
                pygame.draw.aaline(surface, color, self.trail[i], self.trail[i + 1], alpha)
        pygame.gfxdraw.filled_circle(
            surface, int(self.pos.x), int(self.pos.y), self.radius, color
        )


class Clone:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.radius = 10
        self.fire_cooldown = 0
        self.fire_rate = 30  # Ticks per shot

    def update(self, bosons, projectiles):
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        if self.fire_cooldown == 0 and bosons:
            # Find nearest boson
            nearest_boson = min(bosons, key=lambda b: self.pos.distance_to(b.pos))
            direction = (nearest_boson.pos - self.pos).normalize()
            projectiles.append(Projectile(self.pos, direction * 5))
            self.fire_cooldown = self.fire_rate
            # SFX: Clone fire

    def draw(self, surface, color):
        pygame.gfxdraw.filled_circle(
            surface, int(self.pos.x), int(self.pos.y), self.radius, color
        )
        pygame.gfxdraw.aacircle(
            surface, int(self.pos.x), int(self.pos.y), self.radius, (200, 200, 255)
        )


class Boson:
    def __init__(self, pos, target_pos, speed):
        self.pos = pygame.math.Vector2(pos)
        self.vel = (target_pos - self.pos).normalize() * speed
        self.radius = 5
        self.color = random.choice(
            [(255, 50, 50), (255, 150, 50), (255, 200, 50)]
        )

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(
            surface, int(self.pos.x), int(self.pos.y), self.radius, self.color
        )
        pygame.gfxdraw.aacircle(
            surface, int(self.pos.x), int(self.pos.y), self.radius, (255, 255, 255)
        )


class Portal:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.radius = 15
        self.cooldown = 0
        self.anim_timer = random.uniform(0, 2 * math.pi)

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        self.anim_timer += 0.1

    def draw(self, surface, color):
        pulse_radius = self.radius + 2 * math.sin(self.anim_timer)
        alpha = 100 + int(50 * math.sin(self.anim_timer))
        if self.cooldown > 0:
            alpha = 50
        pygame.gfxdraw.filled_circle(
            surface,
            int(self.pos.x),
            int(self.pos.y),
            int(pulse_radius),
            color + (alpha,),
        )
        pygame.gfxdraw.aacircle(
            surface, int(self.pos.x), int(self.pos.y), int(pulse_radius), color
        )


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Defend your nucleus from waves of incoming bosons by strategically placing defensive clones and reality-bending portals."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to place an item. Press 'shift' to cycle between placing clones and portals. Use 'shift' + 'space' to flip gravity."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_NUCLEUS = (100, 150, 255)
        self.COLOR_CLONE = (0, 200, 255)
        self.COLOR_PROJECTILE = (200, 255, 255)
        self.COLOR_PORTAL = (100, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_INDICATOR = (255, 255, 255)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.wave = 0
        self.max_steps = 2000  # Increased for longer gameplay

        # Player state
        self.indicator_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.placement_modes = ["CLONE", "PORTAL_A", "PORTAL_B"]
        self.current_placement_mode_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # Game entities
        self.nucleus_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.nucleus_health = 0
        self.nucleus_max_health = 10
        self.nucleus_radius = 25
        self.clones = []
        self.bosons = []
        self.projectiles = []
        self.portals = [
            Portal(pygame.math.Vector2(100, 100)),
            Portal(pygame.math.Vector2(540, 300)),
        ]
        self.particles = []

        # Mechanics
        self.gravity = pygame.math.Vector2(0, 0)
        self.pulse_cooldown = 0
        self.pulse_max_cooldown = 450  # 15 seconds at 30fps
        self.pulse_active_time = 0
        self.pulse_radius = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.wave = 0

        self.indicator_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.current_placement_mode_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.nucleus_health = self.nucleus_max_health
        self.clones.clear()
        self.bosons.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.portals = [
            Portal(pygame.math.Vector2(100, 100)),
            Portal(pygame.math.Vector2(540, 300)),
        ]

        self.gravity = pygame.math.Vector2(0, 0)
        self.pulse_cooldown = self.pulse_max_cooldown
        self.pulse_active_time = 0

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1

        # --- 1. HANDLE ACTIONS ---
        self._handle_input(action)

        # --- 2. UPDATE GAME LOGIC ---
        self._update_pulse()
        self._update_portals()
        self._update_projectiles()
        self._update_clones()
        self._update_bosons()
        self._update_particles()

        # --- 3. CHECK COLLISIONS & GAME STATE ---
        self._handle_collisions()

        if not self.bosons and self.nucleus_health > 0:
            # SFX: Wave complete
            self.reward_this_step += 1.0
            self.score += 1
            self._spawn_wave()

        # --- 4. CALCULATE REWARD & TERMINATION ---
        terminated = self.nucleus_health <= 0 or self.steps >= self.max_steps

        if self.nucleus_health <= 0 and not self.game_over:
            self.reward_this_step -= 100.0
            self.game_over = True
            self._create_explosion(self.nucleus_pos, 100)
            # SFX: Nucleus destroyed
        elif self.steps >= self.max_steps:
            self.reward_this_step += 100.0

        reward = self.reward_this_step

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held_int, shift_held_int = action
        space_held = space_held_int == 1
        shift_held = shift_held_int == 1

        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # Move indicator
        indicator_speed = 8
        if movement == 1:
            self.indicator_pos.y -= indicator_speed
        elif movement == 2:
            self.indicator_pos.y += indicator_speed
        elif movement == 3:
            self.indicator_pos.x -= indicator_speed
        elif movement == 4:
            self.indicator_pos.x += indicator_speed
        self.indicator_pos.x = np.clip(self.indicator_pos.x, 0, self.WIDTH)
        self.indicator_pos.y = np.clip(self.indicator_pos.y, 0, self.HEIGHT)

        # Cycle modes with Shift
        if shift_pressed:
            self.current_placement_mode_idx = (
                self.current_placement_mode_idx + 1
            ) % len(self.placement_modes)
            # SFX: UI select

        # Action with Space
        if space_pressed:
            mode = self.placement_modes[self.current_placement_mode_idx]
            if mode == "CLONE":
                if len(self.clones) < 10:  # Max 10 clones
                    self.clones.append(Clone(self.indicator_pos))
                    # SFX: Place clone
            elif mode == "PORTAL_A":
                self.portals[0].pos = self.indicator_pos.copy()
                # SFX: Place portal
            elif mode == "PORTAL_B":
                self.portals[1].pos = self.indicator_pos.copy()
                # SFX: Place portal

        # Gravity flip with Shift+Space combo (emergent action)
        if shift_held and space_pressed:
            if self.gravity.y == 0:
                self.gravity.y = 0.05
            else:
                self.gravity.y *= -1
            # SFX: Gravity flip

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_pulse(self):
        if self.pulse_cooldown > 0:
            self.pulse_cooldown -= 1
        elif self.pulse_active_time == 0:
            self.pulse_active_time = 30  # Pulse lasts for 1 second
            self.pulse_radius = 0
            # SFX: Pulse charge

        if self.pulse_active_time > 0:
            self.pulse_active_time -= 1
            self.pulse_radius += 10
            if self.pulse_active_time == 0:
                self.pulse_cooldown = self.pulse_max_cooldown
                # SFX: Pulse dissipate

    def _update_portals(self):
        for portal in self.portals:
            portal.update()

    def _update_projectiles(self):
        projectiles_to_remove = []
        for p in self.projectiles:
            p.update(self.gravity, self.portals)
            if not (0 <= p.pos.x < self.WIDTH and 0 <= p.pos.y < self.HEIGHT):
                projectiles_to_remove.append(p)
        self.projectiles = [
            p for p in self.projectiles if p not in projectiles_to_remove
        ]

    def _update_clones(self):
        for clone in self.clones:
            clone.update(self.bosons, self.projectiles)

    def _update_bosons(self):
        for boson in self.bosons:
            boson.update()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _handle_collisions(self):
        # Projectiles vs Bosons
        projectiles_to_remove = set()
        bosons_to_remove = set()
        for p in self.projectiles:
            for b in self.bosons:
                if p not in projectiles_to_remove and b not in bosons_to_remove:
                    if p.pos.distance_to(b.pos) < p.radius + b.radius:
                        projectiles_to_remove.add(p)
                        bosons_to_remove.add(b)
                        self.reward_this_step += 0.1
                        self._create_explosion(b.pos, 10, b.color)
                        # SFX: Explosion

        self.projectiles = [
            p for p in self.projectiles if p not in projectiles_to_remove
        ]
        self.bosons = [b for b in self.bosons if b not in bosons_to_remove]

        # Bosons vs Nucleus
        bosons_to_remove = set()
        for b in self.bosons:
            if b.pos.distance_to(self.nucleus_pos) < b.radius + self.nucleus_radius:
                bosons_to_remove.add(b)
                self.nucleus_health -= 1
                self._create_explosion(self.nucleus_pos, 20, (255, 100, 100))
                # SFX: Nucleus hit
        self.bosons = [b for b in self.bosons if b not in bosons_to_remove]

        # Pulse vs Bosons
        if self.pulse_active_time > 0:
            bosons_to_remove = set()
            for b in self.bosons:
                if b.pos.distance_to(self.nucleus_pos) < self.pulse_radius:
                    bosons_to_remove.add(b)
                    self.reward_this_step += 0.1
                    self._create_explosion(b.pos, 10, self.COLOR_NUCLEUS)
                    # SFX: Pulse hit
            self.bosons = [b for b in self.bosons if b not in bosons_to_remove]

    def _spawn_wave(self):
        self.wave += 1
        boson_count = 3 + self.wave
        boson_speed = 0.5 + (self.wave * 0.05)
        for _ in range(boson_count):
            edge = random.randint(0, 3)
            if edge == 0:
                x, y = random.uniform(0, self.WIDTH), -20
            elif edge == 1:
                x, y = self.WIDTH + 20, random.uniform(0, self.HEIGHT)
            elif edge == 2:
                x, y = random.uniform(0, self.WIDTH), self.HEIGHT + 20
            else:
                x, y = -20, random.uniform(0, self.HEIGHT)
            self.bosons.append(
                Boson(pygame.math.Vector2(x, y), self.nucleus_pos, boson_speed)
            )

    def _create_explosion(self, pos, num_particles, color=(255, 255, 255)):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = random.uniform(1, 3)
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gravity indicator
        if self.gravity.y != 0:
            arrow_start = (self.WIDTH - 20, self.HEIGHT // 2)
            if self.gravity.y > 0:
                arrow_end = (self.WIDTH - 20, self.HEIGHT // 2 + 15)
                poly = [
                    (arrow_end[0] - 5, arrow_end[1] - 5),
                    (arrow_end[0] + 5, arrow_end[1] - 5),
                    arrow_end,
                ]
            else:
                arrow_end = (self.WIDTH - 20, self.HEIGHT // 2 - 15)
                poly = [
                    (arrow_end[0] - 5, arrow_end[1] + 5),
                    (arrow_end[0] + 5, arrow_end[1] + 5),
                    arrow_end,
                ]
            pygame.draw.aaline(self.screen, (255, 255, 255, 100), arrow_start, arrow_end)
            pygame.gfxdraw.aapolygon(self.screen, poly, (255, 255, 255, 100))

        # Draw portals
        for portal in self.portals:
            portal.draw(self.screen, self.COLOR_PORTAL)

        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.screen, self.COLOR_PROJECTILE)

        # Draw clones
        for clone in self.clones:
            clone.draw(self.screen, self.COLOR_CLONE)

        # Draw nucleus
        if self.nucleus_health > 0:
            health_ratio = self.nucleus_health / self.nucleus_max_health
            current_color = (
                int(self.COLOR_NUCLEUS[0] * health_ratio),
                int(self.COLOR_NUCLEUS[1] * health_ratio),
                int(self.COLOR_NUCLEUS[2] * health_ratio),
            )
            for i in range(5):
                alpha = 150 - i * 30
                radius = self.nucleus_radius * (1 + i * 0.1)
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    int(self.nucleus_pos.x),
                    int(self.nucleus_pos.y),
                    int(radius),
                    current_color + (alpha,),
                )
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(self.nucleus_pos.x),
                int(self.nucleus_pos.y),
                self.nucleus_radius,
                self.COLOR_NUCLEUS,
            )
            pygame.gfxdraw.aacircle(
                self.screen,
                int(self.nucleus_pos.x),
                int(self.nucleus_pos.y),
                self.nucleus_radius,
                (200, 200, 255),
            )

        # Draw bosons
        for boson in self.bosons:
            boson.draw(self.screen)

        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Draw pulse
        if self.pulse_active_time > 0:
            alpha = int(255 * (self.pulse_active_time / 30))
            pygame.gfxdraw.aacircle(
                self.screen,
                int(self.nucleus_pos.x),
                int(self.nucleus_pos.y),
                int(self.pulse_radius),
                self.COLOR_NUCLEUS + (alpha,),
            )
            pygame.gfxdraw.aacircle(
                self.screen,
                int(self.nucleus_pos.x),
                int(self.nucleus_pos.y),
                int(self.pulse_radius - 2),
                self.COLOR_NUCLEUS + (alpha // 2,),
            )

        # Draw indicator
        mode = self.placement_modes[self.current_placement_mode_idx]
        indicator_color = self.COLOR_INDICATOR
        if mode == "CLONE":
            indicator_color = self.COLOR_CLONE
        elif mode.startswith("PORTAL"):
            indicator_color = self.COLOR_PORTAL

        x, y = int(self.indicator_pos.x), int(self.indicator_pos.y)
        pygame.draw.aaline(self.screen, indicator_color, (x - 10, y), (x + 10, y), 200)
        pygame.draw.aaline(self.screen, indicator_color, (x, y - 10), (x, y + 10), 200)

    def _render_ui(self):
        # Wave number
        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Clone count
        clone_text = self.font_large.render(
            f"CLONES: {len(self.clones)}/10", True, self.COLOR_TEXT
        )
        self.screen.blit(clone_text, (self.WIDTH - clone_text.get_width() - 10, 10))

        # Health bar
        if self.nucleus_health > 0:
            health_pct = self.nucleus_health / self.nucleus_max_health
            bar_w = 200
            bar_h = 15
            fill_w = int(bar_w * health_pct)
            bar_x = (self.WIDTH - bar_w) // 2
            bar_y = self.HEIGHT - bar_h - 10
            pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, fill_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

        # Current mode text
        mode_text = self.font_small.render(
            f"MODE: {self.placement_modes[self.current_placement_mode_idx]} (SHIFT to cycle)",
            True,
            self.COLOR_TEXT,
        )
        self.screen.blit(mode_text, (10, self.HEIGHT - mode_text.get_height() - 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.nucleus_health,
            "clones": len(self.clones),
            "bosons": len(self.bosons),
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # Manual play loop
    env = GameEnv()
    obs, info = env.reset()
    terminated = False

    # Override screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Atomic Defender")

    action = [0, 0, 0]  # no-op, released, released

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()

        # Movement
        mov = 0  # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            mov = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            mov = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            mov = 4

        # Space and Shift
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [mov, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # The _get_observation call inside step already draws everything.
        # We just need to flip the display.
        pygame.display.flip()

        env.clock.tick(30)  # Limit to 30 FPS

    env.close()