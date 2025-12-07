import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:39:45.129094
# Source Brief: brief_00547.md
# Brief Index: 547
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    Gymnasium environment where a single-celled organism navigates a fluid
    environment to collect nutrients and reach an exit.
    """

    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a single-celled organism through a fluid environment to collect "
        "nutrients and reach the exit before time runs out."
    )
    user_guide = "Controls: Use arrow keys (↑↓←→) to apply thrust and navigate your cell. Collect nutrients to score points."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 60
    MAX_EPISODE_SECONDS = 120
    MAX_STEPS = MAX_EPISODE_SECONDS * TARGET_FPS

    # Colors
    COLOR_BG_START = (10, 5, 30)
    COLOR_BG_END = (30, 20, 70)
    COLOR_PLAYER = (100, 255, 100)
    COLOR_NUTRIENT = (255, 255, 100)
    COLOR_EXIT = (100, 150, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SPEED_BOOST = (255, 100, 100)

    # Player Physics
    PLAYER_RADIUS = 12
    THRUST_FORCE = 0.3
    DRAG_COEFFICIENT = 0.96
    MAX_SPEED = 4.0

    # Game Mechanics
    NUM_NUTRIENTS = 40
    NUTRIENT_RADIUS = 6
    EXIT_RADIUS = 25
    NUTRIENTS_FOR_BOOST = 50
    SPEED_BOOST_DURATION = 10.0  # seconds
    SPEED_BOOST_MULTIPLIER = 1.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_boost = pygame.font.SysFont("Arial", 24, bold=True)

        # Game state variables
        self.player_pos = None
        self.player_vel = None
        self.thrust_direction = None
        self.nutrients = None
        self.exit_pos = None
        self.score = None
        self.steps = None
        self.time_remaining = None
        self.game_over = None
        self.nutrients_for_boost_counter = None
        self.speed_boost_timer = None
        self.speed_multiplier = None
        self.particles = None
        self.flagellum_trail = None
        self.bg_particles = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = float(self.MAX_EPISODE_SECONDS)

        # Player
        self.player_pos = pygame.Vector2(
            self.np_random.uniform(50, 100),
            self.np_random.uniform(50, self.SCREEN_HEIGHT - 50),
        )
        self.player_vel = pygame.Vector2(0, 0)
        self.thrust_direction = pygame.Vector2(0, 0)

        # Exit
        self.exit_pos = pygame.Vector2(
            self.np_random.uniform(self.SCREEN_WIDTH - 100, self.SCREEN_WIDTH - 50),
            self.np_random.uniform(50, self.SCREEN_HEIGHT - 50),
        )

        # Nutrients
        self.nutrients = []
        while len(self.nutrients) < self.NUM_NUTRIENTS:
            pos = pygame.Vector2(
                self.np_random.uniform(20, self.SCREEN_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20),
            )
            if (
                pos.distance_to(self.player_pos) > 100
                and pos.distance_to(self.exit_pos) > 100
            ):
                self.nutrients.append(pos)

        # Boost mechanics
        self.nutrients_for_boost_counter = 0
        self.speed_boost_timer = 0.0
        self.speed_multiplier = 1.0

        # Visuals
        self.particles = []
        self.flagellum_trail = []
        self.bg_particles = [
            (
                pygame.Vector2(
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT),
                ),
                self.np_random.uniform(0.1, 0.3),  # speed
            )
            for _ in range(100)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self.reset()

        # --- Action Handling ---
        movement, _, _ = action
        self._update_thrust(movement)

        # --- Game Logic ---
        dist_before = self.player_pos.distance_to(self.exit_pos)

        self._update_physics()
        self._update_timers()

        collected_nutrient = self._handle_collisions()
        boost_activated = self._update_boost_status(collected_nutrient)

        # --- Termination ---
        dist_after = self.player_pos.distance_to(self.exit_pos)
        won = dist_after < (self.PLAYER_RADIUS + self.EXIT_RADIUS)
        timed_out = self.time_remaining <= 0 or self.steps >= self.MAX_STEPS
        terminated = won or timed_out
        self.game_over = terminated

        # --- Reward Calculation ---
        reward = self._calculate_reward(
            dist_before, dist_after, collected_nutrient, boost_activated, won, timed_out
        )

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # Truncated is always False in this version
            self._get_info(),
        )

    def _update_thrust(self, movement):
        if movement == 1:
            self.thrust_direction = pygame.Vector2(0, -1)  # Up
        elif movement == 2:
            self.thrust_direction = pygame.Vector2(0, 1)  # Down
        elif movement == 3:
            self.thrust_direction = pygame.Vector2(-1, 0)  # Left
        elif movement == 4:
            self.thrust_direction = pygame.Vector2(1, 0)  # Right
        else:
            self.thrust_direction = pygame.Vector2(0, 0)

    def _update_physics(self):
        # Apply thrust
        if self.thrust_direction.length() > 0:
            self.player_vel += self.thrust_direction.normalize() * self.THRUST_FORCE

        # Apply drag
        self.player_vel *= self.DRAG_COEFFICIENT

        # Clamp speed
        current_max_speed = self.MAX_SPEED * self.speed_multiplier
        if self.player_vel.length() > current_max_speed:
            self.player_vel.scale_to_length(current_max_speed)

        # Update position
        self.player_pos += self.player_vel

        # Boundary checks
        self.player_pos.x = np.clip(
            self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS
        )
        self.player_pos.y = np.clip(
            self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS
        )

    def _update_timers(self):
        dt = 1.0 / self.TARGET_FPS
        self.time_remaining -= dt
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= dt
        else:
            self.speed_multiplier = 1.0

    def _handle_collisions(self):
        collected_nutrient = False
        collected_indices = []
        for i, nutrient_pos in enumerate(self.nutrients):
            if (
                self.player_pos.distance_to(nutrient_pos)
                < self.PLAYER_RADIUS + self.NUTRIENT_RADIUS
            ):
                collected_indices.append(i)
                self.score += 1
                self.nutrients_for_boost_counter += 1
                collected_nutrient = True
                self._spawn_collection_particles(nutrient_pos)

        # Remove collected nutrients safely
        for i in sorted(collected_indices, reverse=True):
            del self.nutrients[i]

        return collected_nutrient

    def _update_boost_status(self, collected_nutrient):
        boost_activated = False
        if (
            collected_nutrient
            and self.nutrients_for_boost_counter >= self.NUTRIENTS_FOR_BOOST
        ):
            self.nutrients_for_boost_counter = 0
            self.speed_boost_timer = self.SPEED_BOOST_DURATION
            self.speed_multiplier = self.SPEED_BOOST_MULTIPLIER
            boost_activated = True
        return boost_activated

    def _calculate_reward(
        self, dist_before, dist_after, collected_nutrient, boost_activated, won, timed_out
    ):
        reward = 0.0

        # Goal-oriented rewards
        if won:
            return 100.0
        if timed_out:
            return -10.0

        # Continuous feedback
        distance_delta = dist_before - dist_after
        if distance_delta > 0:
            reward += 0.1  # Moved closer
        else:
            reward -= 0.01  # Moved away or stood still

        # Event-based rewards
        if collected_nutrient:
            reward += 1.0
        if boost_activated:
            reward += 5.0

        return reward

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "speed_boost_active": self.speed_boost_timer > 0,
        }

    def _render_all(self):
        self._draw_background()
        self._update_and_draw_particles()
        self._draw_glow_circle(
            self.screen, self.EXIT_RADIUS, self.COLOR_EXIT, self.exit_pos, 20
        )
        for nutrient_pos in self.nutrients:
            self._draw_glow_circle(
                self.screen, self.NUTRIENT_RADIUS, self.COLOR_NUTRIENT, nutrient_pos, 15
            )
        self._draw_player()
        self._render_ui()

    def _draw_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_player(self):
        # Flagellum trail
        self.flagellum_trail.append(self.player_pos.copy())
        if len(self.flagellum_trail) > 20:
            self.flagellum_trail.pop(0)

        if len(self.flagellum_trail) > 2:
            for i in range(len(self.flagellum_trail) - 1):
                p1 = self.flagellum_trail[i]
                p2 = self.flagellum_trail[i + 1]
                alpha = int(200 * (i / len(self.flagellum_trail)))
                color = (*self.COLOR_PLAYER[:3], alpha)
                width = int(2 + 4 * (i / len(self.flagellum_trail)))
                pygame.draw.line(self.screen, color, p1, p2, width)

        # Player body glow
        boost_color = (
            self.COLOR_SPEED_BOOST if self.speed_boost_timer > 0 else self.COLOR_PLAYER
        )
        self._draw_glow_circle(
            self.screen, self.PLAYER_RADIUS, boost_color, self.player_pos, 25
        )

    def _render_ui(self):
        # Nutrient Counter
        nutrient_text = f"Nutrients: {self.score}"
        self._draw_text(nutrient_text, self.font_ui, self.COLOR_UI_TEXT, 10, 10)

        # Timer
        time_text = f"Time: {max(0, int(self.time_remaining))}"
        self._draw_text(
            time_text,
            self.font_ui,
            self.COLOR_UI_TEXT,
            self.SCREEN_WIDTH - 10,
            10,
            align="topright",
        )

        # Speed Boost Indicator
        if self.speed_boost_timer > 0:
            boost_text = f"SPEED BOOST! {int(self.speed_boost_timer)}s"
            self._draw_text(
                boost_text,
                self.font_boost,
                self.COLOR_SPEED_BOOST,
                self.SCREEN_WIDTH // 2,
                self.SCREEN_HEIGHT - 40,
                align="midbottom",
            )

    def _draw_text(self, text, font, color, x, y, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, (x, y))
        self.screen.blit(text_surface, text_rect)

    def _draw_glow_circle(self, surface, radius, color, pos, max_alpha):
        for i in range(int(radius * 1.5), 0, -2):
            alpha = int(max_alpha * (1 - (i / (radius * 2.5))) ** 2)
            if alpha <= 0:
                continue

            temp_surf = pygame.Surface((i * 2, i * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, i, i, i, (*color[:3], alpha))
            surface.blit(
                temp_surf,
                (int(pos.x - i), int(pos.y - i)),
                special_flags=pygame.BLEND_RGBA_ADD,
            )

        # Draw solid core
        pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), int(radius), color)

    def _spawn_collection_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append(
                {
                    "pos": pos.copy(),
                    "vel": vel,
                    "lifetime": self.np_random.uniform(0.5, 1.0),
                    "color": self.COLOR_NUTRIENT,
                    "size": self.np_random.uniform(2, 4),
                }
            )

    def _update_and_draw_particles(self):
        dt = 1.0 / self.TARGET_FPS

        # Background "flow" particles
        for i, (pos, speed) in enumerate(self.bg_particles):
            pos.x -= speed
            if pos.x < 0:
                pos.x = self.SCREEN_WIDTH
                pos.y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
            pygame.gfxdraw.pixel(
                self.screen, int(pos.x), int(pos.y), (*self.COLOR_BG_END, 100)
            )

        # Effect particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= dt
            if p["lifetime"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["lifetime"] / 1.0))
                pygame.draw.circle(self.screen, (*p["color"], alpha), p["pos"], p["size"])

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False

    # Use a display for human play
    # This requires pygame to be installed with display support
    try:
        display_screen = pygame.display.set_mode(
            (GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Cellular Drift")
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        display_screen = None

    total_reward = 0

    while not terminated:
        # --- Human Input Mapping ---
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        # This part requires a display to get key presses
        if display_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        else: # If no display, just send no-ops and let it run
            pass

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering for Human ---
        if display_screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(GameEnv.TARGET_FPS)
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            # In a real scenario, you might break or reset here
            # For this test script, we'll just let it end.
            break

    env.close()