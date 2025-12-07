import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Nurture a procedural plant from a seedling to a full-grown flower. "
        "Collect resources and use fertilizer to accelerate its growth."
    )
    user_guide = (
        "Use arrow keys to move the reticle. Press space to prime the fertilizer "
        "and shift to collect resources under the reticle."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000

    # Gameplay Constants
    RETICLE_SPEED = 15
    RESOURCE_RADIUS = 12
    RETICLE_RADIUS = 18
    MAX_RESOURCES = 7
    GROWTH_PER_RESOURCE = 15.0
    FERTILIZER_MULTIPLIER = 2.0

    # --- Colors ---
    COLOR_BG_TOP = pygame.Color(135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = pygame.Color(34, 139, 34)  # Forest Green
    COLOR_RETICLE = pygame.Color(255, 255, 255)
    COLOR_RETICLE_PRIMED = pygame.Color(255, 255, 0)  # Gold
    COLOR_TEXT = pygame.Color(255, 255, 240)  # Ivory

    RESOURCE_COLORS = {
        "red": pygame.Color(255, 69, 0),
        "blue": pygame.Color(30, 144, 255),
        "yellow": pygame.Color(255, 215, 0),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.plant_growth = None
        self.fertilizer_charges = None
        self.fertilizer_primed = None
        self.reticle_pos = None
        self.on_screen_resources = None
        self.particles = None
        self.plant_structure = None  # To store branches, leaves, etc.
        self.prev_space_held = None
        self.prev_shift_held = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0

        self.plant_growth = 0.0
        self.fertilizer_charges = 1
        self.fertilizer_primed = False

        self.reticle_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)

        self.on_screen_resources = []
        self.particles = []

        # Plant structure: list of {'start', 'end', 'width', 'color'} dicts
        self.plant_structure = {"branches": [], "leaves": [], "flower": []}
        self._initialize_plant()

        while len(self.on_screen_resources) < self.MAX_RESOURCES:
            self._spawn_resource()

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        self._move_reticle(movement)

        if space_pressed:
            reward += self._handle_fertilizer_prime()

        if shift_pressed:
            collection_reward, growth_achieved = self._handle_collection()
            reward += collection_reward
            if growth_achieved > 0:
                self._grow_plant(growth_achieved)

        self._update_spawns()
        self._update_particles()

        self.steps += 1
        terminated = self.plant_growth >= 100.0 or self.steps >= self.MAX_STEPS

        if terminated and self.plant_growth >= 100.0:
            reward += 100.0  # Goal-oriented reward

        self.score += reward
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    # --- Internal Logic ---

    def _move_reticle(self, movement):
        if movement == 1:  # Up
            self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2:  # Down
            self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3:  # Left
            self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4:  # Right
            self.reticle_pos.x += self.RETICLE_SPEED

        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.SCREEN_WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.SCREEN_HEIGHT)

    def _handle_fertilizer_prime(self):
        if self.fertilizer_charges > 0 and not self.fertilizer_primed:
            self.fertilizer_primed = True
            self.fertilizer_charges -= 1
            # sfx: fertilizer prime sound
            self._spawn_particles(
                self.reticle_pos, 15, self.COLOR_RETICLE_PRIMED, count=20
            )
            return 5.0  # Reward for strategic choice
        return 0.0

    def _handle_collection(self):
        for i, res in enumerate(self.on_screen_resources):
            dist = self.reticle_pos.distance_to(res["pos"])
            if dist < self.RETICLE_RADIUS + self.RESOURCE_RADIUS:
                reward = 1.0
                growth_amount = self.GROWTH_PER_RESOURCE

                # Consume resource
                resource = self.on_screen_resources.pop(i)
                # sfx: resource collect sound
                self._spawn_particles(resource["pos"], 10, resource["color"], count=15)

                if self.fertilizer_primed:
                    growth_amount *= self.FERTILIZER_MULTIPLIER
                    reward += 5.0  # Bonus for effective use
                    self.fertilizer_primed = False
                    # sfx: fertilized collection sound
                    self._spawn_particles(
                        resource["pos"],
                        15,
                        pygame.Color("gold"),
                        count=30,
                        gravity=0.1,
                    )

                self.plant_growth = min(100.0, self.plant_growth + growth_amount)
                return reward, growth_amount
        return 0.0, 0.0

    def _update_spawns(self):
        if (
            len(self.on_screen_resources) < self.MAX_RESOURCES
            and self.np_random.random() < 0.1
        ):
            self._spawn_resource()

        # Periodically grant fertilizer
        if self.steps > 0 and self.steps % 300 == 0:
            self.fertilizer_charges = min(3, self.fertilizer_charges + 1)

    def _spawn_resource(self):
        res_type = self.np_random.choice(list(self.RESOURCE_COLORS.keys()))
        pos = pygame.Vector2(
            self.np_random.uniform(
                self.RESOURCE_RADIUS, self.SCREEN_WIDTH - self.RESOURCE_RADIUS
            ),
            self.np_random.uniform(
                self.RESOURCE_RADIUS, self.SCREEN_HEIGHT - 200
            ),  # Spawn in top half
        )
        self.on_screen_resources.append(
            {
                "pos": pos,
                "type": res_type,
                "color": self.RESOURCE_COLORS[res_type],
                "pulse_phase": self.np_random.uniform(0, 2 * math.pi),
            }
        )

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"].y += p["gravity"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] * 0.98)

    def _spawn_particles(self, pos, speed, color, count=10, gravity=0.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_val = self.np_random.uniform(speed * 0.5, speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_val
            self.particles.append(
                {
                    "pos": pygame.Vector2(pos),
                    "vel": vel,
                    "life": self.np_random.integers(15, 30),
                    "color": color,
                    "radius": self.np_random.uniform(3, 6),
                    "gravity": gravity,
                }
            )

    def _initialize_plant(self):
        start_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT)
        end_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 5)
        self.plant_structure["branches"].append(
            {
                "start": start_pos,
                "end": end_pos,
                "width": 4,
                "color": pygame.Color(139, 69, 19),
            }
        )

    def _grow_plant(self, growth_achieved):
        # sfx: plant growth sound
        if not self.plant_structure["branches"]:
            return

        num_new_segments = 1 + int(growth_achieved / 10)

        for _ in range(num_new_segments):
            if not self.plant_structure["branches"]:
                continue

            # Choose a random branch end to grow from
            parent_branch = self.np_random.choice(self.plant_structure["branches"])
            start_pos = parent_branch["end"]

            parent_vec = parent_branch["end"] - parent_branch["start"]
            if parent_vec.length() == 0:
                parent_angle = -math.pi / 2
            else:
                parent_angle = math.atan2(parent_vec.y, parent_vec.x)

            # New segment properties
            angle_variation = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            new_angle = parent_angle + angle_variation
            length = self.np_random.uniform(10, 20) * (
                1 - self.plant_growth / 200
            )  # shorter branches higher up
            length = max(5, length)
            end_pos = start_pos + pygame.Vector2(
                math.cos(new_angle), math.sin(new_angle)
            ) * length

            width = max(2, parent_branch["width"] * 0.9)

            # Color transition from brown to green
            growth_ratio = self.plant_growth / 100.0
            color = pygame.Color(139, 69, 19).lerp(
                pygame.Color(0, 128, 0), min(1.0, growth_ratio * 2.0)
            )

            new_branch = {
                "start": start_pos,
                "end": end_pos,
                "width": int(width),
                "color": color,
            }
            self.plant_structure["branches"].append(new_branch)

            # Chance to spawn a leaf
            if self.plant_growth > 20 and self.np_random.random() < 0.3:
                leaf_angle = new_angle + self.np_random.choice([-1, 1]) * math.pi / 2
                leaf_pos = start_pos + (end_pos - start_pos) * self.np_random.uniform(
                    0.2, 0.8
                )
                self.plant_structure["leaves"].append(
                    {
                        "pos": leaf_pos,
                        "angle": leaf_angle,
                        "size": self.np_random.uniform(5, 15),
                        "color": pygame.Color(50, 205, 50).lerp(
                            pygame.Color(124, 252, 0), growth_ratio
                        ),
                    }
                )

        # Add a flower at high growth
        if self.plant_growth > 90 and not self.plant_structure["flower"]:
            # Find the highest point of the plant
            highest_point = min(
                self.plant_structure["branches"], key=lambda b: b["end"].y
            )["end"]
            self.plant_structure["flower"].append(
                {
                    "pos": highest_point,
                    "size": 10,
                    "color1": self.np_random.choice(list(self.RESOURCE_COLORS.values())),
                    "color2": pygame.Color("white"),
                }
            )
            # sfx: flower bloom sound
            self._spawn_particles(
                highest_point, 5, self.plant_structure["flower"][0]["color1"], count=50
            )

    # --- Rendering ---

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = self.COLOR_BG_TOP.lerp(self.COLOR_BG_BOTTOM, ratio)
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            if p["radius"] > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"]
                )

        # Draw plant
        self._draw_plant()

        # Draw resources
        for res in self.on_screen_resources:
            pulse = (math.sin(res["pulse_phase"] + self.steps * 0.1) + 1) / 2
            radius = int(self.RESOURCE_RADIUS * (1 + 0.1 * pulse))
            color = res["color"]

            pygame.gfxdraw.filled_circle(
                self.screen, int(res["pos"].x), int(res["pos"].y), radius, color
            )
            pygame.gfxdraw.aacircle(
                self.screen,
                int(res["pos"].x),
                int(res["pos"].y),
                radius,
                color.lerp((0, 0, 0), 0.3),
            )

        # Draw reticle
        reticle_color = (
            self.COLOR_RETICLE_PRIMED if self.fertilizer_primed else self.COLOR_RETICLE
        )
        pos = (int(self.reticle_pos.x), int(self.reticle_pos.y))

        pulse_fast = abs(math.sin(self.steps * 0.3))
        glow_radius = int(
            self.RETICLE_RADIUS * (1.5 + 0.5 * pulse_fast)
            if self.fertilizer_primed
            else self.RETICLE_RADIUS * 1.2
        )
        glow_color = (*reticle_color[:3], 50)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.RETICLE_RADIUS, reticle_color)
        pygame.draw.line(
            self.screen, reticle_color, (pos[0] - 5, pos[1]), (pos[0] + 5, pos[1])
        )
        pygame.draw.line(
            self.screen, reticle_color, (pos[0], pos[1] - 5), (pos[0], pos[1] + 5)
        )

    def _draw_plant(self):
        # Draw leaves (behind branches)
        for leaf in self.plant_structure["leaves"]:
            size = leaf["size"]
            angle_rad = leaf["angle"]
            points = [
                leaf["pos"] + pygame.Vector2(size, 0).rotate_rad(angle_rad),
                leaf["pos"] + pygame.Vector2(size / 3, size / 3).rotate_rad(angle_rad),
                leaf["pos"] + pygame.Vector2(-size, 0).rotate_rad(angle_rad),
                leaf["pos"]
                + pygame.Vector2(size / 3, -size / 3).rotate_rad(angle_rad),
            ]
            points_int = [(int(p.x), int(p.y)) for p in points]
            pygame.gfxdraw.filled_polygon(self.screen, points_int, leaf["color"])
            pygame.gfxdraw.aapolygon(self.screen, points_int, leaf["color"])

        # Draw branches
        for branch in self.plant_structure["branches"]:
            pygame.draw.line(
                self.screen,
                branch["color"],
                branch["start"],
                branch["end"],
                int(branch["width"]),
            )

        # Draw flower (on top)
        for flower in self.plant_structure["flower"]:
            pos = (int(flower["pos"].x), int(flower["pos"].y))
            size = flower["size"]
            # Petals
            for i in range(5):
                angle = i * (2 * math.pi / 5) + (self.steps * 0.02)
                petal_pos = (
                    int(pos[0] + math.cos(angle) * size),
                    int(pos[1] + math.sin(angle) * size),
                )
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    petal_pos[0],
                    petal_pos[1],
                    int(size * 0.8),
                    flower["color1"],
                )
                pygame.gfxdraw.aacircle(
                    self.screen,
                    petal_pos[0],
                    petal_pos[1],
                    int(size * 0.8),
                    flower["color1"],
                )
            # Center
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], int(size * 0.9), flower["color2"]
            )

    def _render_ui(self):
        # Growth Text
        growth_text = self.font_large.render(
            f"Growth: {self.plant_growth:.1f}%", True, self.COLOR_TEXT
        )
        self.screen.blit(growth_text, (10, 10))

        # Fertilizer Text
        fert_text = self.font_small.render(
            f"Fertilizer Charges: {self.fertilizer_charges}", True, self.COLOR_TEXT
        )
        self.screen.blit(fert_text, (10, 45))

        # Primed Indicator
        if self.fertilizer_primed:
            primed_text = self.font_small.render(
                "FERTILIZER PRIMED!", True, self.COLOR_RETICLE_PRIMED
            )
            text_rect = primed_text.get_rect(center=(self.SCREEN_WIDTH / 2, 20))
            self.screen.blit(primed_text, text_rect)

        # Game Over Text
        if self.plant_growth >= 100.0:
            win_text = self.font_large.render(
                "PLANT MATURE!", True, pygame.Color("gold")
            )
            text_rect = win_text.get_rect(
                center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
            )
            self.screen.blit(win_text, text_rect)
        elif self.steps >= self.MAX_STEPS:
            lose_text = self.font_large.render(
                "TIME'S UP", True, pygame.Color("red")
            )
            text_rect = lose_text.get_rect(
                center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
            )
            self.screen.blit(lose_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "plant_growth": self.plant_growth,
            "fertilizer_charges": self.fertilizer_charges,
        }

    def close(self):
        pygame.quit()


# --- Main execution block for human play ---
if __name__ == "__main__":
    # The main loop needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Procedural Plant Grower")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    while not terminated:
        # --- Human Input to Action Mapping ---
        keys = pygame.key.get_pressed()

        movement = 0  # No-op
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {total_reward:.2f}")
            pygame.time.wait(3000)  # Pause for 3 seconds before closing

    env.close()