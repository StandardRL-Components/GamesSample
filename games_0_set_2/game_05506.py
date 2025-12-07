import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect all 5 artifacts to reveal the exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a haunted mansion by collecting artifacts and avoiding ghosts before time runs out."
    )

    # Frames auto-advance at 60fps for smooth gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PLATFORM = (40, 50, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_GHOST = (255, 80, 80)
        self.COLOR_GHOST_GLOW = (255, 80, 80, 60)
        self.COLOR_ITEM = (255, 255, 100)
        self.COLOR_ITEM_GLOW = (255, 255, 100, 70)
        self.COLOR_EXIT = (150, 100, 255)
        self.COLOR_EXIT_GLOW = (150, 100, 255, 80)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # Physics
        self.GRAVITY = 0.3
        self.JUMP_STRENGTH = -8
        self.PLAYER_SPEED = 3.5
        self.FRICTION = 0.85

        # Game settings
        self.NUM_ITEMS = 5
        self.NUM_GHOSTS = 3

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 16, 24)
        self.is_grounded = False

        self.platforms = []
        self.items = []
        self.items_collected = []
        self.ghosts = []
        self.exit_pos = pygame.math.Vector2(0, 0)
        self.exit_rect = pygame.Rect(0, 0, 32, 48)
        self.exit_open = False

        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.exit_open = False
        self.particles = []

        self._generate_level()

        self.player_vel = pygame.math.Vector2(0, 0)
        self.is_grounded = False

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Create floor
        self.platforms = [pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)]

        # Generate platforms
        for _ in range(10):
            w = self.np_random.integers(80, 200)
            h = 20
            x = self.np_random.integers(0, self.WIDTH - w)
            y = self.np_random.integers(80, self.HEIGHT - 80)
            new_platform = pygame.Rect(x, y, w, h)

            # Ensure no overlap with existing platforms
            if not any(new_platform.colliderect(p) for p in self.platforms):
                self.platforms.append(new_platform)

        # Place player
        start_platform = self.platforms[self.np_random.integers(len(self.platforms))]
        self.player_pos = pygame.math.Vector2(
            start_platform.centerx, start_platform.top - self.player_rect.height
        )
        self.player_rect.midbottom = self.player_pos

        # Place items
        self.items = []
        self.items_collected = [False] * self.NUM_ITEMS
        while len(self.items) < self.NUM_ITEMS:
            platform = self.platforms[self.np_random.integers(len(self.platforms))]
            item_pos = pygame.math.Vector2(
                self.np_random.integers(platform.left + 10, platform.right - 10),
                platform.top - 10,
            )
            # Ensure items are not too close to each other
            if not any(item_pos.distance_to(p) < 50 for p in self.items):
                self.items.append(item_pos)

        # Place exit
        eligible_platforms = [p for p in self.platforms if p.y < self.HEIGHT - 20]
        if not eligible_platforms:
             eligible_platforms = self.platforms
        exit_platform = eligible_platforms[self.np_random.integers(len(eligible_platforms))]
        self.exit_pos = pygame.math.Vector2(exit_platform.centerx, exit_platform.top)
        self.exit_rect.midbottom = (int(self.exit_pos.x), int(self.exit_pos.y))

        # Place ghosts
        self.ghosts = []
        for _ in range(self.NUM_GHOSTS):
            platform = self.platforms[self.np_random.integers(len(self.platforms))]
            start_x = platform.left + 20
            end_x = platform.right - 20
            if start_x >= end_x:
                continue

            pos = pygame.math.Vector2(
                self.np_random.integers(start_x, end_x), platform.top - 15
            )
            self.ghosts.append(
                {
                    "pos": pos,
                    "start": start_x,
                    "end": end_x,
                    "speed": self.np_random.uniform(0.8, 1.5),
                    "dir": 1,
                    "size": 15,
                }
            )

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, _, _ = action
        reward = -0.01  # Time penalty
        self.steps += 1
        self.timer -= 1

        # --- Update Game Logic ---
        dist_before = self._get_dist_to_nearest_item()
        self._update_player(movement)
        self._update_ghosts()
        self._update_particles()
        dist_after = self._get_dist_to_nearest_item()

        # Reward for moving towards item
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 0.1

        # Check item collection
        for i, item_pos in enumerate(self.items):
            if not self.items_collected[i]:
                item_rect = pygame.Rect(item_pos.x - 5, item_pos.y - 5, 10, 10)
                if self.player_rect.colliderect(item_rect):
                    self.items_collected[i] = True
                    self.score += 10
                    reward += 10
                    # SFX: Item collect sound
                    self._create_particles(item_pos, self.COLOR_ITEM, 20)

        # Check if all items are collected
        if all(self.items_collected) and not self.exit_open:
            self.exit_open = True
            # SFX: Exit open sound
            self._create_particles(self.exit_pos, self.COLOR_EXIT, 30)

        # --- Termination Checks ---
        terminated = False
        # Ghost collision
        for ghost in self.ghosts:
            ghost_rect = pygame.Rect(
                ghost["pos"].x - ghost["size"],
                ghost["pos"].y - ghost["size"],
                ghost["size"] * 2,
                ghost["size"] * 2,
            )
            if self.player_rect.colliderect(ghost_rect):
                reward = -100
                terminated = True
                # SFX: Player death sound
                break

        # Timer out
        if self.timer <= 0:
            reward = -50
            terminated = True

        # Reached exit
        if self.exit_open and self.player_rect.colliderect(self.exit_rect):
            reward = 100
            self.score += 50  # Bonus for winning
            terminated = True
            # SFX: Level complete sound

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x *= self.FRICTION

        # Vertical movement (Jump)
        if movement == 1 and self.is_grounded:
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_grounded = False
            # SFX: Jump sound

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > 10:  # Terminal velocity
            self.player_vel.y = 10

        # Update position
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y

        # Screen bounds collision
        if self.player_pos.x < self.player_rect.width / 2:
            self.player_pos.x = self.player_rect.width / 2
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - self.player_rect.width / 2:
            self.player_pos.x = self.WIDTH - self.player_rect.width / 2
            self.player_vel.x = 0

        self.player_rect.midbottom = (int(self.player_pos.x), int(self.player_pos.y))

        # Platform collisions
        self.is_grounded = False
        for platform in self.platforms:
            if self.player_rect.colliderect(platform) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                if self.player_pos.y - self.player_vel.y <= platform.top + 1:
                    self.player_pos.y = platform.top
                    self.player_vel.y = 0
                    self.is_grounded = True
                    self.player_rect.bottom = platform.top
                    break

        self.player_rect.midbottom = (int(self.player_pos.x), int(self.player_pos.y))

    def _update_ghosts(self):
        for ghost in self.ghosts:
            ghost["pos"].x += ghost["speed"] * ghost["dir"]
            if ghost["pos"].x > ghost["end"] or ghost["pos"].x < ghost["start"]:
                ghost["dir"] *= -1

            # Pulsating effect
            ghost["size"] = 15 + math.sin(self.steps * 0.1) * 3

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(
                {
                    "pos": pos.copy(),
                    "vel": pygame.math.Vector2(
                        self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)
                    ),
                    "life": self.np_random.integers(20, 40),
                    "color": color,
                }
            )

    def _get_dist_to_nearest_item(self):
        uncollected_items = [
            self.items[i] for i, c in enumerate(self.items_collected) if not c
        ]
        if not uncollected_items:
            return None

        distances = [
            self.player_pos.distance_to(item_pos) for item_pos in uncollected_items
        ]
        return min(distances)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render platforms
        for platform in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform)

        # Render exit
        if self.exit_open:
            glow_size = int(
                self.exit_rect.width * (1.5 + 0.1 * math.sin(self.steps * 0.1))
            )
            pygame.gfxdraw.filled_circle(
                self.screen,
                self.exit_rect.centerx,
                self.exit_rect.centery,
                glow_size // 2,
                self.COLOR_EXIT_GLOW,
            )
            pygame.draw.rect(
                self.screen, self.COLOR_EXIT, self.exit_rect, 3, border_radius=5
            )

        # Render items
        for i, item_pos in enumerate(self.items):
            if not self.items_collected[i]:
                glow_size = int(12 + 3 * math.sin(self.steps * 0.2 + i))
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    int(item_pos.x),
                    int(item_pos.y),
                    glow_size,
                    self.COLOR_ITEM_GLOW,
                )
                pygame.gfxdraw.filled_circle(
                    self.screen, int(item_pos.x), int(item_pos.y), 5, self.COLOR_ITEM
                )
                pygame.gfxdraw.aacircle(
                    self.screen, int(item_pos.x), int(item_pos.y), 5, self.COLOR_ITEM
                )

        # Render ghosts
        for ghost in self.ghosts:
            size = int(ghost["size"])
            pos = (int(ghost["pos"].x), int(ghost["pos"].y))
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], int(size * 1.5), self.COLOR_GHOST_GLOW
            )
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], size, self.COLOR_GHOST
            )
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_GHOST)

        # Render player
        glow_size = int(self.player_rect.width * 1.2)
        pygame.gfxdraw.filled_circle(
            self.screen,
            self.player_rect.centerx,
            self.player_rect.centery,
            glow_size,
            self.COLOR_PLAYER_GLOW,
        )
        pygame.draw.rect(
            self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3
        )

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * 6)))
            color_with_alpha = (*p["color"], alpha)
            surf = pygame.Surface((p["life"], p["life"]), pygame.SRCALPHA)
            pygame.draw.circle(surf, color_with_alpha, (p["life"]//2, p["life"]//2), p["life"]//2)
            self.screen.blit(surf, (int(p['pos'].x - p["life"]//2), int(p['pos'].y - p["life"]//2)))


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # Score
        draw_text(f"SCORE: {self.score}", self.font_medium, self.COLOR_TEXT, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_GHOST
        timer_text = f"TIME: {time_left:.1f}"
        text_width = self.font_medium.size(timer_text)[0]
        draw_text(
            timer_text, self.font_medium, timer_color, (self.WIDTH - text_width - 10, 10)
        )

        # Item collection status
        item_icon_size = 12
        total_width = self.NUM_ITEMS * (item_icon_size + 5) - 5
        start_x = (self.WIDTH - total_width) // 2
        for i in range(self.NUM_ITEMS):
            x = start_x + i * (item_icon_size + 5)
            y = self.HEIGHT - 40
            color = self.COLOR_ITEM if self.items_collected[i] else self.COLOR_PLATFORM
            pygame.draw.rect(
                self.screen, color, (x, y, item_icon_size, item_icon_size), border_radius=3
            )
            pygame.draw.rect(
                self.screen,
                self.COLOR_TEXT,
                (x, y, item_icon_size, item_icon_size),
                1,
                border_radius=3,
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "items_collected": sum(self.items_collected),
            "time_remaining": max(0, self.timer / self.FPS),
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == "__main__":
    # To run with a display, comment out the following line
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # Un-set the dummy driver if we are running interactively
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()

    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False

    # Pygame window for human interaction
    try:
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Haunted House Escape")
        interactive = True
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        interactive = False


    while not done:
        action = [0, 0, 0] # Default no-op action
        if interactive:
            # Action mapping
            keys = pygame.key.get_pressed()
            movement = 0  # no-op
            if keys[pygame.K_UP]:
                movement = 1
            # Action 2 (Down) is unused
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
        else: # In headless mode, just take random actions
            action = env.action_space.sample()


        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if interactive:
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS) # Control speed for human play

    print(f"Game Over! Final Score: {info['score']}")
    env.close()