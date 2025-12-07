import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the environment to run headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np


# Helper classes for game objects to keep the main class clean
class Constellation:
    """Represents a constellation with physical properties and visual stars."""

    def __init__(self, pos, radius, seed):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.mass = radius**2  # Mass is proportional to area for simple physics
        self.local_rand = random.Random(seed)

        # Procedurally generate stars within the constellation's radius
        self.stars = []
        num_stars = int(radius / 3) + self.local_rand.randint(3, 5)
        for _ in range(num_stars):
            angle = self.local_rand.uniform(0, 2 * math.pi)
            dist = self.local_rand.uniform(0, radius)
            star_pos = pygame.math.Vector2(dist * math.cos(angle), dist * math.sin(angle))
            self.stars.append(
                {
                    "pos": star_pos,
                    "brightness": self.local_rand.uniform(0.5, 1.0),
                    "size": self.local_rand.randint(1, 3),
                }
            )


class Fleet:
    """Represents an alien fleet moving along a linear path."""

    def __init__(self, start_pos, end_pos, speed):
        self.pos = pygame.math.Vector2(start_pos)
        self.start_pos = pygame.math.Vector2(start_pos)
        self.end_pos = pygame.math.Vector2(end_pos)
        self.speed = speed
        self.direction = (
            (self.end_pos - self.start_pos).normalize()
            if self.start_pos != self.end_pos
            else pygame.math.Vector2(0, 0)
        )
        self.is_captured = False
        self.is_lost = False


class GravityWell:
    """Represents a temporary gravity well placed by the player."""

    def __init__(self, pos, strength, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.strength = strength
        self.lifetime = lifetime
        self.initial_lifetime = lifetime


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Use gravity wells to bend the paths of passing alien fleets, guiding them into your constellations for capture."
    user_guide = "Use arrow keys to move the cursor. Press space to deploy a gravity well. Press shift to restart the level."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    CURSOR_STEP = 20
    MAX_STEPS = 2000

    # --- Color Palette ---
    COLOR_BG = (10, 5, 25)
    COLOR_NEBULA_1 = (40, 20, 80, 5)
    COLOR_NEBULA_2 = (80, 40, 120, 5)
    COLOR_CURSOR = (100, 255, 100, 150)
    COLOR_CURSOR_BORDER = (200, 255, 200)
    COLOR_CONSTELLATION_STAR = (255, 220, 150)
    COLOR_CONSTELLATION_GLOW = (255, 200, 100, 30)
    COLOR_CONSTELLATION_LINE = (255, 220, 150, 50)
    COLOR_FLEET = (255, 50, 50)
    COLOR_FLEET_GLOW = (255, 50, 50, 80)
    COLOR_GRAVITY_WELL = (50, 255, 50)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_ICON = (50, 255, 50)

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        self._np_random = None

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = True  # Start in a terminal state, requiring reset()
        self.level = 1
        self.unlocked_constellation_types = [30]
        self.cursor_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.gravity_wells_remaining = 0
        self.constellations = []
        self.fleets = []
        self.active_gravity_wells = []
        self.previous_space_held = False
        self.nebula_bg = self._create_nebula_background()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._np_random is None:
            seed_ = seed if seed is not None else np.random.randint(2**31 - 1)
            self._np_random, _ = gym.utils.seeding.np_random(seed_)
            random.seed(seed_)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.previous_space_held = False

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Clear old state
        self.constellations.clear()
        self.fleets.clear()
        self.active_gravity_wells.clear()

        # Unlock new constellation sizes every 5 levels
        if self.level > 1 and self.level % 5 == 0:
            new_size = 30 + (self.level // 5) * 5
            if new_size not in self.unlocked_constellation_types:
                self.unlocked_constellation_types.append(new_size)

        # Scale difficulty with level
        num_fleets = 1 + (self.level - 1) // 3
        self.gravity_wells_remaining = num_fleets + 2

        # Place constellations, avoiding overlap
        num_constellations = num_fleets + 1
        for i in range(num_constellations):
            placed = False
            for _ in range(100):  # Max 100 attempts to prevent infinite loop
                radius = random.choice(self.unlocked_constellation_types)
                pos = pygame.math.Vector2(
                    random.uniform(radius, self.WIDTH - radius),
                    random.uniform(radius, self.HEIGHT - radius),
                )
                if not any(
                    (pos - c.pos).length() < c.radius + radius + 10
                    for c in self.constellations
                ):
                    seed = self.np_random.integers(2**31 - 1)
                    self.constellations.append(Constellation(pos, radius, seed))
                    placed = True
                    break
            if not placed:  # Fallback if no space found
                self.constellations.append(
                    Constellation((self.WIDTH / 2, self.HEIGHT / 2), 30, 999)
                )

        # Place fleets with increasingly complex paths
        for i in range(num_fleets):
            path_length_mod = 1.0 + ((self.level - 1) // 3) * 0.2
            # FIX: Increase path length to prevent fleets from flying off-screen too quickly in no-op tests
            path_length = random.uniform(500, 800) * path_length_mod

            # Start fleet from outside the screen
            start_edge = random.randint(0, 3)
            if start_edge == 0:
                start_pos = (random.uniform(0, self.WIDTH), -20)
            elif start_edge == 1:
                start_pos = (random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif start_edge == 2:
                start_pos = (-20, random.uniform(0, self.HEIGHT))
            else:
                start_pos = (self.WIDTH + 20, random.uniform(0, self.HEIGHT))

            # Aim towards a point inside the screen
            target_point = (
                random.uniform(50, self.WIDTH - 50),
                random.uniform(50, self.HEIGHT - 50),
            )
            direction = (
                pygame.math.Vector2(target_point) - pygame.math.Vector2(start_pos)
            ).normalize()
            end_pos = (
                start_pos[0] + path_length * direction.x,
                start_pos[1] + path_length * direction.y,
            )
            speed = random.uniform(0.5, 1.5)
            self.fleets.append(Fleet(start_pos, end_pos, speed))

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Player Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle cursor movement
        if movement == 1:
            self.cursor_pos.y -= self.CURSOR_STEP
        elif movement == 2:
            self.cursor_pos.y += self.CURSOR_STEP
        elif movement == 3:
            self.cursor_pos.x -= self.CURSOR_STEP
        elif movement == 4:
            self.cursor_pos.x += self.CURSOR_STEP
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Handle gravity well placement (on rising edge of space press)
        if space_held and not self.previous_space_held and self.gravity_wells_remaining > 0:
            self.active_gravity_wells.append(
                GravityWell(self.cursor_pos.copy(), strength=20000, lifetime=60)
            )
            self.gravity_wells_remaining -= 1

        self.previous_space_held = space_held

        # Handle level restart (shift press)
        if shift_held:
            reward = -20  # Penalty for giving up
            terminated = True
            self.game_over = True

        # --- 2. Update Game Physics & State ---
        if not terminated:
            # Calculate fleet distances before physics update for reward calculation
            dist_before = {
                id(f): min(
                    (f.pos - c.pos).length() for c in self.constellations
                )
                if self.constellations
                else float("inf")
                for f in self.fleets
            }

            # Update and remove expired gravity wells
            self.active_gravity_wells = [
                w for w in self.active_gravity_wells if w.lifetime > 1
            ]
            for well in self.active_gravity_wells:
                well.lifetime -= 1

            # Update constellations based on gravity
            for const in self.constellations:
                total_force = pygame.math.Vector2(0, 0)
                for well in self.active_gravity_wells:
                    vec = well.pos - const.pos
                    dist_sq = max(1, vec.length_squared())  # Avoid division by zero
                    force_mag = well.strength / dist_sq
                    total_force += vec.normalize() * force_mag

                acceleration = total_force / const.mass
                const.vel += acceleration
                const.vel *= 0.95  # Damping/friction
                const.pos += const.vel

                # Boundary collision with energy loss
                if const.pos.x < const.radius:
                    const.pos.x = const.radius
                    const.vel.x *= -0.5
                if const.pos.x > self.WIDTH - const.radius:
                    const.pos.x = self.WIDTH - const.radius
                    const.vel.x *= -0.5
                if const.pos.y < const.radius:
                    const.pos.y = const.radius
                    const.vel.y *= -0.5
                if const.pos.y > self.HEIGHT - const.radius:
                    const.pos.y = self.HEIGHT - const.radius
                    const.vel.y *= -0.5

            # Update fleets and check for capture/loss
            captured_this_step = []
            for fleet in self.fleets:
                if fleet.direction.length() > 0:
                    fleet.pos += fleet.direction * fleet.speed

                for const in self.constellations:
                    if (fleet.pos - const.pos).length() < const.radius:
                        fleet.is_captured = True
                        reward += 10
                        self.score += 10
                        captured_this_step.append(fleet)
                        break

                if not fleet.is_captured and not (
                    -50 < fleet.pos.x < self.WIDTH + 50
                    and -50 < fleet.pos.y < self.HEIGHT + 50
                ):
                    fleet.is_lost = True
                    reward -= 5
                    captured_this_step.append(fleet)

            self.fleets = [f for f in self.fleets if f not in captured_this_step]

            # Calculate continuous reward based on distance change
            for f in self.fleets:
                dist_after = (
                    min((f.pos - c.pos).length() for c in self.constellations)
                    if self.constellations
                    else float("inf")
                )
                dist_change = dist_before[id(f)] - dist_after
                reward += dist_change * 0.005

            self.steps += 1

            # --- 3. Check for Termination Conditions ---
            if not self.fleets:  # Victory
                reward += 100
                self.score += 100
                terminated = True
                self.game_over = True
                self.level += 1
            elif (
                self.gravity_wells_remaining <= 0 and not self.active_gravity_wells
            ):  # Failure
                reward -= 50
                terminated = True
                self.game_over = True
            elif self.steps >= self.MAX_STEPS:  # Failure
                reward -= 20
                terminated = True
                self.game_over = True

        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,  # Truncated is always False
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.blit(self.nebula_bg, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render gravity wells
        for well in self.active_gravity_wells:
            progress = 1.0 - (well.lifetime / well.initial_lifetime)
            radius = int(progress * 100)
            alpha = int(math.sin(progress * math.pi) * 100)
            pygame.gfxdraw.aacircle(
                self.screen,
                int(well.pos.x),
                int(well.pos.y),
                radius,
                self.COLOR_GRAVITY_WELL + (alpha,),
            )

        # Render constellations
        for const in self.constellations:
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(const.pos.x),
                int(const.pos.y),
                int(const.radius),
                self.COLOR_CONSTELLATION_GLOW,
            )
            if len(const.stars) > 1:
                for i in range(len(const.stars)):
                    p1 = const.pos + const.stars[i]["pos"]
                    p2 = const.pos + const.stars[(i + 1) % len(const.stars)]["pos"]
                    pygame.draw.aaline(
                        self.screen, self.COLOR_CONSTELLATION_LINE, (p1.x, p1.y), (p2.x, p2.y)
                    )
            for star in const.stars:
                p = const.pos + star["pos"]
                size = star["size"]
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p.x), int(p.y), size, self.COLOR_CONSTELLATION_STAR
                )
                if random.random() < 0.1:
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(p.x), int(p.y), size + 1, (255, 255, 255, 150)
                    )

        # Render fleets
        for fleet in self.fleets:
            if fleet.direction.length() > 0:
                p = fleet.pos
                size = 6
                points = [
                    p + fleet.direction * size,
                    p + fleet.direction.rotate(-140) * size * 0.8,
                    p + fleet.direction.rotate(140) * size * 0.8,
                ]
                int_points = [(int(pt.x), int(pt.y)) for pt in points]
                pygame.gfxdraw.filled_trigon(
                    self.screen, *int_points[0], *int_points[1], *int_points[2], self.COLOR_FLEET_GLOW
                )
                pygame.gfxdraw.filled_trigon(
                    self.screen, *int_points[0], *int_points[1], *int_points[2], self.COLOR_FLEET
                )
                pygame.gfxdraw.aatrigon(
                    self.screen, *int_points[0], *int_points[1], *int_points[2], self.COLOR_FLEET
                )

        # Render player cursor
        if self.gravity_wells_remaining > 0:
            pygame.draw.circle(
                self.screen,
                self.COLOR_CURSOR,
                (int(self.cursor_pos.x), int(self.cursor_pos.y)),
                10,
                0,
            )
            pygame.draw.circle(
                self.screen,
                self.COLOR_CURSOR_BORDER,
                (int(self.cursor_pos.x), int(self.cursor_pos.y)),
                10,
                2,
            )

    def _render_ui(self):
        ui_text = self.font_small.render("Gravity Wells:", True, self.COLOR_UI_TEXT)
        self.screen.blit(ui_text, (10, 10))
        for i in range(self.gravity_wells_remaining):
            pygame.draw.circle(self.screen, self.COLOR_UI_ICON, (140 + i * 20, 18), 7)
            pygame.draw.circle(self.screen, self.COLOR_BG, (140 + i * 20, 18), 4)

        fleets_text = f"Fleets Remaining: {len(self.fleets)}"
        ui_text = self.font_small.render(fleets_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(ui_text, (10, 35))

        level_text = f"Nebula: {self.level}"
        ui_text = self.font_large.render(level_text, True, self.COLOR_UI_TEXT)
        text_rect = ui_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(ui_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "fleets_remaining": len(self.fleets),
            "wells_remaining": self.gravity_wells_remaining,
        }

    def _create_nebula_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        bg.fill(self.COLOR_BG)
        for _ in range(150):
            pos = (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
            radius = random.randint(20, 150)
            color = random.choice([self.COLOR_NEBULA_1, self.COLOR_NEBULA_2])
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            bg.blit(
                temp_surf,
                (pos[0] - radius, pos[1] - radius),
                special_flags=pygame.BLEND_RGBA_ADD,
            )
        return bg

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game.
    # We unset the dummy video driver to allow for a real display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)

    running = True
    terminated = False

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cosmic Puzzle")
    clock = pygame.time.Clock()

    while running:
        if terminated:
            print(f"Game Over. Final Info: {info}. Resetting in 2 seconds...")
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        move_action = 0  # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            move_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            move_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [move_action, space_action, shift_action]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Event Loop for quitting ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()