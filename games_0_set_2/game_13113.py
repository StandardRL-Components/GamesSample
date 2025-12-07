import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for "Ziggurat Ascent".

    The player ascends a procedurally generated ziggurat by manipulating time
    to alter gravity, which is influenced by the positions of celestial bodies.
    """

    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Ascend a procedurally generated ziggurat by manipulating time to alter gravity, "
        "which is influenced by the positions of celestial bodies."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Hold space to fast-forward time, and hold shift to rewind time."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_STARS = [(100, 100, 120), (150, 150, 180), (220, 220, 255)]
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_PLATFORM = (80, 60, 50)
    COLOR_PLATFORM_EDGE = (110, 90, 80)
    COLOR_UI_TEXT = (255, 220, 180)
    COLOR_SUN = (255, 200, 0)
    COLOR_MOON = (230, 230, 255)

    # Gravity States & Colors
    GRAVITY_NORMAL = 1
    GRAVITY_REVERSED = -1
    GRAVITY_COLORS = {
        GRAVITY_NORMAL: (0, 100, 255),  # Blue
        GRAVITY_REVERSED: (0, 255, 100),  # Green
    }

    # Time Manipulation States & Colors
    TIME_REWIND = -1
    TIME_NORMAL = 0
    TIME_FAST = 1
    TIME_MANIP_COLORS = {
        TIME_REWIND: (255, 0, 255),  # Magenta/Reddish
        TIME_NORMAL: (255, 255, 0),  # Yellow
        TIME_FAST: (255, 120, 0),  # Orange
    }

    # Game & Physics
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000
    PLAYER_SIZE = 16
    PLAYER_SPEED = 5
    PLAYER_JUMP_STRENGTH = 10
    GRAVITY_ACCEL = 0.5
    PLAYER_DRAG = 0.85
    NUM_LEVELS = 15
    LEVEL_HEIGHT = 150
    SUMMIT_Y_POS = -(NUM_LEVELS * LEVEL_HEIGHT) + SCREEN_HEIGHT / 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3),
            dtype=np.uint8,
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        self.camera_y = 0.0
        self.gravity_direction = self.GRAVITY_NORMAL
        self.time_manip_state = self.TIME_NORMAL
        self.platforms = []
        self.celestial_bodies = []
        self.particles = []
        self.stars = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_stars()
        self._generate_ziggurat()
        self._generate_celestial_bodies()

        # Player state
        base_platform_top = self.SCREEN_HEIGHT - 40
        self.player_pos = pygame.Vector2(
            self.SCREEN_WIDTH / 2, base_platform_top - self.PLAYER_SIZE / 2
        )
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True

        # World state
        self.camera_y = 0.0
        self.gravity_direction = self.GRAVITY_NORMAL
        self.time_manip_state = self.TIME_NORMAL
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, True, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        old_player_y = self.player_pos.y
        old_gravity = self.gravity_direction

        # 1. Handle Time Manipulation Input
        self._update_time_manipulation_state(space_held, shift_held)

        # 2. Update Celestial Bodies & Determine Gravity
        self._update_celestial_bodies()
        self._update_gravity()

        # Reward for using time manip to change gravity
        if self.gravity_direction != old_gravity and self.time_manip_state != self.TIME_NORMAL:
            reward += 1.0  # Correctly used time manip to change physics
            for _ in range(30):
                self._spawn_particle(
                    self.player_pos, self.GRAVITY_COLORS[self.gravity_direction], 3, 5
                )

        # 3. Handle Player Input
        self._handle_player_input(movement)

        # 4. Apply Physics & Update Player
        self._apply_physics()

        # 5. Update Camera
        self._update_camera()

        # 6. Update Particles
        self._update_particles()

        # 7. Calculate Rewards & Check Termination
        # Reward for vertical movement
        y_change = old_player_y - self.player_pos.y
        reward += y_change * 0.01  # Small reward for upward progress

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.player_pos.y <= self.SUMMIT_Y_POS:
                reward += 100.0  # Reached the summit
            else:
                reward -= 10.0  # Fell off

        self.score += reward

        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_y": self.player_pos.y,
            "gravity": self.gravity_direction,
        }

    # --- Game Logic Sub-methods ---

    def _update_time_manipulation_state(self, space_held, shift_held):
        if shift_held:
            self.time_manip_state = self.TIME_REWIND
        elif space_held:
            self.time_manip_state = self.TIME_FAST
        else:
            self.time_manip_state = self.TIME_NORMAL

    def _update_celestial_bodies(self):
        time_rate = 1.0
        if self.time_manip_state == self.TIME_FAST:
            time_rate = 3.0
        if self.time_manip_state == self.TIME_REWIND:
            time_rate = -2.0

        for body in self.celestial_bodies:
            body["angle"] += body["speed"] * time_rate * 0.01
            body["pos"].x = (
                body["path_center"].x + math.cos(body["angle"]) * body["path_radius"]
            )
            body["pos"].y = (
                body["path_center"].y + math.sin(body["angle"]) * body["path_radius"]
            )

    def _update_gravity(self):
        # Default to normal gravity
        new_gravity = self.GRAVITY_NORMAL
        for body in self.celestial_bodies:
            if body["trigger_condition"](body["pos"]):
                new_gravity = body["triggers_gravity"]
                break  # First triggered body determines gravity
        self.gravity_direction = new_gravity

    def _handle_player_input(self, movement):
        # Horizontal Movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x *= self.PLAYER_DRAG  # Apply drag if no horizontal input

        # Jump
        if movement == 1 and self.is_grounded:  # Up
            self.player_vel.y = -self.PLAYER_JUMP_STRENGTH * self.gravity_direction
            self.is_grounded = False
            for _ in range(15):
                p_color = (200, 200, 255)
                p_vel = pygame.Vector2(
                    random.uniform(-1, 1),
                    random.uniform(0.5, 2) * self.gravity_direction,
                )
                self._spawn_particle(self.player_pos, p_color, 2, 3, p_vel)

    def _apply_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY_ACCEL * self.gravity_direction

        # Update position
        self.player_pos += self.player_vel

        # Collision detection and response
        self.is_grounded = False
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )

        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check if landing on top (normal gravity) or bottom (reversed gravity)
                if self.gravity_direction == self.GRAVITY_NORMAL and self.player_vel.y > 0:
                    if player_rect.bottom - self.player_vel.y <= plat.top + 1:
                        player_rect.bottom = plat.top
                        self.player_pos.y = player_rect.centery
                        self.player_vel.y = 0
                        self.is_grounded = True
                elif (
                    self.gravity_direction == self.GRAVITY_REVERSED
                    and self.player_vel.y < 0
                ):
                    if player_rect.top - self.player_vel.y >= plat.bottom - 1:
                        player_rect.top = plat.bottom
                        self.player_pos.y = player_rect.centery
                        self.player_vel.y = 0
                        self.is_grounded = True

        # Screen boundaries (horizontal)
        if self.player_pos.x < self.PLAYER_SIZE / 2:
            self.player_pos.x = self.PLAYER_SIZE / 2
            self.player_vel.x = 0
        if self.player_pos.x > self.SCREEN_WIDTH - self.PLAYER_SIZE / 2:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_SIZE / 2
            self.player_vel.x = 0

    def _update_camera(self):
        # Smoothly follow the player vertically
        target_y = -self.player_pos.y + self.SCREEN_HEIGHT * 0.4
        self.camera_y += (target_y - self.camera_y) * 0.08

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.1)

    def _check_termination(self):
        # Fell off screen
        if self.player_pos.y > -self.camera_y + self.SCREEN_HEIGHT + 50:
            return True
        # Reached summit
        if self.player_pos.y <= self.SUMMIT_Y_POS:
            return True
        return False

    # --- Generation Methods ---

    def _generate_stars(self):
        self.stars = []
        for i in range(3):  # 3 layers for parallax
            for _ in range(50 * (i + 1)):
                self.stars.append(
                    {
                        "pos": pygame.Vector2(
                            random.randint(0, self.SCREEN_WIDTH),
                            random.randint(0, self.SCREEN_HEIGHT),
                        ),
                        "depth": i + 1,
                        "color": self.COLOR_STARS[i],
                    }
                )

    def _generate_ziggurat(self):
        self.platforms = []
        # Base platform
        self.platforms.append(
            pygame.Rect(-100, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH + 200, 100)
        )

        x_pos = self.SCREEN_WIDTH / 2
        direction = 1
        for i in range(1, self.NUM_LEVELS + 1):
            y_pos = self.SCREEN_HEIGHT - (i * self.LEVEL_HEIGHT)
            width = random.randint(150, 250)

            x_pos += direction * random.uniform(100, 180)
            if x_pos + width / 2 > self.SCREEN_WIDTH or x_pos - width / 2 < 0:
                direction *= -1
                x_pos += direction * random.uniform(200, 300)
            x_pos = np.clip(x_pos, width / 2, self.SCREEN_WIDTH - width / 2)

            self.platforms.append(pygame.Rect(x_pos - width / 2, y_pos, width, 20))

        # Summit platform
        self.platforms.append(
            pygame.Rect(0, self.SUMMIT_Y_POS - 20, self.SCREEN_WIDTH, 40)
        )

    def _generate_celestial_bodies(self):
        self.celestial_bodies = []
        # The Sun, which reverses gravity when it's on the right side of the screen
        self.celestial_bodies.append(
            {
                "pos": pygame.Vector2(0, 0),
                "path_center": pygame.Vector2(
                    self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT + 200
                ),
                "path_radius": self.SCREEN_HEIGHT + 180,
                "angle": math.radians(225),
                "speed": 0.8,
                "color": self.COLOR_SUN,
                "size": 40,
                "triggers_gravity": self.GRAVITY_REVERSED,
                "trigger_condition": lambda pos: pos.x > self.SCREEN_WIDTH / 2
                and pos.y < self.SCREEN_HEIGHT,
            }
        )
        # The Moon, which has no effect (could be used for more complex puzzles)
        self.celestial_bodies.append(
            {
                "pos": pygame.Vector2(0, 0),
                "path_center": pygame.Vector2(
                    self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT + 200
                ),
                "path_radius": self.SCREEN_HEIGHT + 220,
                "angle": math.radians(315),
                "speed": 1.0,
                "color": self.COLOR_MOON,
                "size": 25,
                "triggers_gravity": self.GRAVITY_NORMAL,  # No change
                "trigger_condition": lambda pos: False,  # Never triggers
            }
        )

    def _spawn_particle(self, pos, color, size, life, vel=None):
        if vel is None:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)

        self.particles.append(
            {
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "size": random.uniform(size * 0.5, size),
                "life": random.uniform(life * 0.5, life) * 10,
            }
        )

    # --- Rendering Methods ---

    def _render_game(self):
        # Render parallax stars
        for star in self.stars:
            x = (
                star["pos"].x - self.SCREEN_WIDTH / 2
            ) / star["depth"] + self.SCREEN_WIDTH / 2
            y = (star["pos"].y + self.camera_y / (star["depth"] * 2)) % self.SCREEN_HEIGHT
            pygame.draw.circle(
                self.screen, star["color"], (int(x), int(y)), int(star["depth"] * 0.5)
            )

        # Render celestial bodies (behind platforms)
        for body in self.celestial_bodies:
            pygame.gfxdraw.filled_circle(
                self.screen, int(body["pos"].x), int(body["pos"].y), body["size"], body["color"]
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(body["pos"].x), int(body["pos"].y), body["size"], body["color"]
            )

        # Render platforms
        for plat in self.platforms:
            p_rect = plat.move(0, self.camera_y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, p_rect, 2)

        # Render particles
        for p in self.particles:
            pos = p["pos"] + pygame.Vector2(0, self.camera_y)
            pygame.draw.circle(
                self.screen, p["color"], (int(pos.x), int(pos.y)), int(p["size"])
            )

        # Render player
        player_screen_pos = self.player_pos + pygame.Vector2(0, self.camera_y)
        px, py = int(player_screen_pos.x), int(player_screen_pos.y)
        size = int(self.PLAYER_SIZE)

        # Glow effect
        glow_size = int(size * 1.8)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf, (*self.COLOR_PLAYER_GLOW, 60), (glow_size, glow_size), glow_size
        )
        self.screen.blit(
            glow_surf, (px - glow_size, py - glow_size), special_flags=pygame.BLEND_RGBA_ADD
        )

        # Player core
        player_rect = pygame.Rect(px - size / 2, py - size / 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(
            f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font_ui.render(
            f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(
            steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10)
        )

        # Gravity Indicator
        grav_color = self.GRAVITY_COLORS[self.gravity_direction]
        if self.gravity_direction == self.GRAVITY_NORMAL:
            p1 = (self.SCREEN_WIDTH - 30, self.SCREEN_HEIGHT - 40)
            p2 = (self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 20)
            p3 = (self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 40)
        else:  # Reversed
            p1 = (self.SCREEN_WIDTH - 30, self.SCREEN_HEIGHT - 20)
            p2 = (self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 40)
            p3 = (self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 20)
        pygame.draw.polygon(self.screen, grav_color, [p1, p2, p3])

        # Time Manipulation Indicator
        if self.time_manip_state != self.TIME_NORMAL:
            time_color = self.TIME_MANIP_COLORS[self.time_manip_state]
            if self.time_manip_state == self.TIME_FAST:
                points = [
                    (10, self.SCREEN_HEIGHT - 30),
                    (25, self.SCREEN_HEIGHT - 30),
                    (25, self.SCREEN_HEIGHT - 35),
                    (40, self.SCREEN_HEIGHT - 25),
                    (25, self.SCREEN_HEIGHT - 15),
                    (25, self.SCREEN_HEIGHT - 20),
                    (10, self.SCREEN_HEIGHT - 20),
                ]
            else:  # Rewind
                points = [
                    (40, self.SCREEN_HEIGHT - 30),
                    (25, self.SCREEN_HEIGHT - 30),
                    (25, self.SCREEN_HEIGHT - 35),
                    (10, self.SCREEN_HEIGHT - 25),
                    (25, self.SCREEN_HEIGHT - 15),
                    (25, self.SCREEN_HEIGHT - 20),
                    (40, self.SCREEN_HEIGHT - 20),
                ]
            pygame.draw.polygon(self.screen, time_color, points)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # The main script needs a display, so we can't use the dummy driver here.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ziggurat Ascent - Manual Test")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0

    while running:
        movement_action = 0  # None
        space_action = 0  # Released
        shift_action = 0  # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1  # Jump
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3  # Left
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4  # Right

        if keys[pygame.K_SPACE]:
            space_action = 1  # Held
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1  # Held

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()

        if terminated or truncated:
            print(
                f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}"
            )
            total_reward = 0.0
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()