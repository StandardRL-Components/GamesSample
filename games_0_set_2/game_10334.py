import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Control a falling water droplet, collect blue droplets for points, and avoid red ones. "
        "Grab snowflakes to activate a temporary ice form that can destroy hazards."
    )
    user_guide = (
        "Use ←→ arrow keys to move horizontally and ↑↓ to adjust falling speed. "
        "Press shift to activate the ice form power-up when available."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.WIN_SCORE = 20
        self.MAX_STEPS = 90 * self.FPS  # 90 seconds

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 60, 80)
        self.COLOR_PLAYER = (100, 180, 255)
        self.COLOR_PLAYER_GLOW = (200, 220, 255)
        self.COLOR_ICE = (200, 255, 255)
        self.COLOR_ICE_GLOW = (220, 255, 255)
        self.COLOR_BLUE_DROP = (0, 150, 255)
        self.COLOR_RED_DROP = (255, 80, 80)
        self.COLOR_SNOWFLAKE = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TRAIL = (150, 200, 255)

        # Physics
        self.GRAVITY = 0.03
        self.PLAYER_FORCE = 0.15
        self.FRICTION = 0.98
        self.MAX_VEL_X = 5
        self.MAX_VEL_Y = 6
        self.MIN_VEL_Y = 0.5  # Natural descent speed

        # Game Mechanics
        self.ICE_FORM_DURATION = 5 * self.FPS  # 5 seconds
        self.SNOWFLAKE_RESPAWN_TIME = 15 * self.FPS  # 15 seconds
        self.DROPLET_SPAWN_INTERVAL = 100  # pixels scrolled

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.trails = None
        self.blue_droplets = None
        self.red_droplets = None
        self.snowflakes = None

        self.score = 0
        self.steps = 0
        self.world_scroll = 0.0
        self.next_droplet_spawn_scroll = 0

        self.can_use_ice = False
        self.ice_form_active = False
        self.ice_form_timer = 0
        self.snowflake_spawn_timer = 0
        self.prev_shift_held = False

        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 3)
        self.player_vel = pygame.Vector2(0, self.MIN_VEL_Y)

        # Game objects
        self.trails = []
        self.blue_droplets = []
        self.red_droplets = []
        self.snowflakes = []

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_scroll = 0.0
        self.next_droplet_spawn_scroll = self.DROPLET_SPAWN_INTERVAL

        # Ice form state
        self.can_use_ice = False
        self.ice_form_active = False
        self.ice_form_timer = 0
        self.snowflake_spawn_timer = self.SNOWFLAKE_RESPAWN_TIME // 2
        self.prev_shift_held = False

        # Initial object placement
        for y_pos in range(int(self.HEIGHT / 2), self.HEIGHT + 100, 80):
            self._spawn_droplet_row(y_pos, safe_start=True)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return terminal state
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        self._handle_input(action)
        self._update_physics()

        # Downward movement reward
        reward += self.player_vel.y * 0.005

        collision_reward = self._handle_collisions()
        reward += collision_reward

        self._update_game_state()
        self._spawn_objects()

        self.steps += 1
        terminated = self._check_termination()

        # Final rewards on termination
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Win bonus
            elif self.steps >= self.MAX_STEPS:
                reward -= 100.0  # Time out penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_FORCE
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_FORCE

        # Vertical movement
        if movement == 1:  # Up (slow down)
            self.player_vel.y -= self.PLAYER_FORCE * 0.5
        elif movement == 2:  # Down (speed up)
            self.player_vel.y += self.PLAYER_FORCE * 0.5

        # Ice form activation (on key press, not hold)
        if shift_held and not self.prev_shift_held and self.can_use_ice:
            self.ice_form_active = True
            self.ice_form_timer = self.ICE_FORM_DURATION
            self.can_use_ice = False
            # sfx: Ice form activate sound

        self.prev_shift_held = shift_held

    def _update_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY

        # Apply friction
        self.player_vel.x *= self.FRICTION

        # Clamp velocity
        self.player_vel.x = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel.x))
        self.player_vel.y = max(self.MIN_VEL_Y, min(self.MAX_VEL_Y, self.player_vel.y))

        # Update player position
        self.player_pos += self.player_vel

        # Screen boundaries
        self.player_pos.x = max(10, min(self.WIDTH - 10, self.player_pos.x))

        # World scrolling
        scroll_amount = self.player_vel.y
        self.world_scroll += scroll_amount

        # Update object positions based on scroll
        for obj_list in [
            self.blue_droplets,
            self.red_droplets,
            self.snowflakes,
            self.trails,
        ]:
            for obj in obj_list:
                obj["pos"].y -= scroll_amount

    def _handle_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player_pos.x - 8, self.player_pos.y - 8, 16, 16)

        # Blue droplets
        for drop in self.blue_droplets[:]:
            if player_rect.colliderect(drop["rect"]):
                self.blue_droplets.remove(drop)
                self.score += 1
                reward += 1.0
                # sfx: Collect point sound

        # Red droplets
        for drop in self.red_droplets[:]:
            if player_rect.colliderect(drop["rect"]):
                if self.ice_form_active:
                    self.red_droplets.remove(drop)
                    self.score += 1  # Bonus for clearing hazard
                    reward += 1.0
                    # sfx: Ice shatter sound
                else:
                    self.red_droplets.remove(drop)
                    self.score = max(0, self.score - 5)
                    reward -= 5.0
                    self.game_over = True
                    # sfx: Player hit/fail sound

        # Snowflakes
        for flake in self.snowflakes[:]:
            if player_rect.colliderect(flake["rect"]):
                self.snowflakes.remove(flake)
                self.can_use_ice = True
                self.snowflake_spawn_timer = self.SNOWFLAKE_RESPAWN_TIME
                # sfx: Powerup collect sound

        return reward

    def _update_game_state(self):
        # Ice form timer
        if self.ice_form_active:
            self.ice_form_timer -= 1
            if self.ice_form_timer <= 0:
                self.ice_form_active = False
                # sfx: Ice form deactivate sound

        # Snowflake respawn timer
        if not self.can_use_ice and not self.snowflakes:
            self.snowflake_spawn_timer -= 1
            if self.snowflake_spawn_timer <= 0:
                self._spawn_snowflake()

        # Trail particles
        self.trails.append(
            {"pos": self.player_pos.copy(), "radius": 6, "alpha": 200}
        )
        for trail in self.trails[:]:
            trail["radius"] -= 0.2
            trail["alpha"] -= 10
            if trail["radius"] <= 0 or trail["alpha"] <= 0:
                self.trails.remove(trail)

        # Remove off-screen objects
        for obj_list in [self.blue_droplets, self.red_droplets, self.snowflakes]:
            for obj in obj_list[:]:
                if obj["pos"].y < -20:
                    obj_list.remove(obj)

    def _spawn_objects(self):
        if self.world_scroll > self.next_droplet_spawn_scroll:
            self._spawn_droplet_row(self.HEIGHT + 50)
            self.next_droplet_spawn_scroll += self.DROPLET_SPAWN_INTERVAL

    def _spawn_droplet_row(self, y_pos, safe_start=False):
        num_droplets = self.np_random.integers(2, 5)
        positions = random.sample(range(50, self.WIDTH - 50, 70), num_droplets)
        for x_pos in positions:
            x_pos += self.np_random.uniform(-20, 20)
            is_red = self.np_random.random() < 0.3
            if safe_start and abs(x_pos - self.WIDTH / 2) < 100:
                is_red = False

            if is_red:
                self.red_droplets.append(
                    {
                        "pos": pygame.Vector2(x_pos, y_pos),
                        "rect": pygame.Rect(x_pos - 6, y_pos - 6, 12, 12),
                        "radius": 6,
                    }
                )
            else:
                self.blue_droplets.append(
                    {
                        "pos": pygame.Vector2(x_pos, y_pos),
                        "rect": pygame.Rect(x_pos - 5, y_pos - 5, 10, 10),
                        "radius": 5,
                    }
                )

    def _spawn_snowflake(self):
        x = self.np_random.uniform(50, self.WIDTH - 50)
        y = self.HEIGHT + 50
        self.snowflakes.append(
            {
                "pos": pygame.Vector2(x, y),
                "rect": pygame.Rect(x - 8, y - 8, 16, 16),
                "radius": 8,
            }
        )

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE
            or self.steps >= self.MAX_STEPS
            or self.game_over
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "ice_form_ready": self.can_use_ice,
            "ice_form_active": self.ice_form_active,
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        # Create a subtle scrolling gradient effect
        scroll_y = int(self.world_scroll % self.HEIGHT)
        for i in range(3):
            y = i * self.HEIGHT - scroll_y
            rect = pygame.Rect(0, y, self.WIDTH, self.HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_BG_TOP, rect)

    def _render_game(self):
        self._render_trails()
        self._render_objects()
        self._render_player()

    def _render_trails(self):
        for trail in self.trails:
            if trail["alpha"] > 0:
                self._draw_circle(
                    self.screen,
                    trail["pos"],
                    trail["radius"],
                    self.COLOR_TRAIL,
                    trail["alpha"],
                )

    def _render_objects(self):
        # Blue Droplets
        for drop in self.blue_droplets:
            self._draw_circle(
                self.screen, drop["pos"], drop["radius"], self.COLOR_BLUE_DROP, 255
            )

        # Red Droplets
        for drop in self.red_droplets:
            self._draw_circle(
                self.screen, drop["pos"], drop["radius"], self.COLOR_RED_DROP, 255
            )

        # Snowflakes
        for flake in self.snowflakes:
            self._draw_snowflake(flake["pos"], flake["radius"], self.COLOR_SNOWFLAKE)

    def _render_player(self):
        pos = self.player_pos

        if self.ice_form_active:
            color = self.COLOR_ICE
            glow_color = self.COLOR_ICE_GLOW
            radius = 10
        else:
            color = self.COLOR_PLAYER
            glow_color = self.COLOR_PLAYER_GLOW
            radius = 8

        # Glow effect
        for i in range(4, 0, -1):
            self._draw_circle(
                self.screen, pos, radius + i * 2, glow_color, 50 - i * 10
            )

        # Main droplet
        self._draw_circle(self.screen, pos, radius, color, 255)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:02}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 10))

        # Ice Form Indicator
        if self.can_use_ice:
            ice_indicator_text = self.font_small.render(
                "ICE READY", True, self.COLOR_ICE
            )
            self.screen.blit(
                ice_indicator_text,
                (self.WIDTH / 2 - ice_indicator_text.get_width() / 2, 10),
            )
        elif self.ice_form_active:
            ice_timer = self.ice_form_timer / self.FPS
            ice_indicator_text = self.font_small.render(
                f"ICE ACTIVE {ice_timer:.1f}s", True, self.COLOR_ICE_GLOW
            )
            self.screen.blit(
                ice_indicator_text,
                (self.WIDTH / 2 - ice_indicator_text.get_width() / 2, 10),
            )

    def _draw_circle(self, surface, pos, radius, color, alpha):
        """Draws a smooth, anti-aliased circle with alpha transparency."""
        target_rect = pygame.Rect(
            pos.x - radius, pos.y - radius, radius * 2, radius * 2
        )
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)

        # Draw the filled circle
        pygame.draw.circle(shape_surf, (*color, alpha), (radius, radius), radius)

        surface.blit(shape_surf, target_rect)

    def _draw_snowflake(self, pos, radius, color):
        """Draws a simple geometric snowflake."""
        center_x, center_y = int(pos.x), int(pos.y)
        for i in range(6):
            angle = math.radians(60 * i)
            end_x = center_x + radius * math.cos(angle)
            end_y = center_y + radius * math.sin(angle)
            pygame.draw.line(
                self.screen, color, (center_x, center_y), (int(end_x), int(end_y)), 2
            )

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block will not run in the testing environment, but is useful for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Water Droplet Descent")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    while not terminated:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0  # None
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()