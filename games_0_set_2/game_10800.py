import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:58:42.770741
# Source Brief: brief_00800.md
# Brief Index: 800
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from itertools import combinations

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player maneuvers and transforms platforms
    to keep at least one afloat in a continuous chain reaction puzzle.

    **Gameplay:**
    - The player controls the horizontal movement of the central (blue) platform.
    - Platforms have two states: Solid (heavy, falls) and Transparent (light, rises).
    - Colliding with another platform or a side wall flips a platform's state.
    - The player can transform the central platform (Space) or all platforms (Shift).
    - The goal is to keep any platform above the victory line for 10 seconds.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (3=left, 4=right for the player platform; 0,1,2=none)
    - `action[1]`: Transform player platform (1=press Space)
    - `action[2]`: Transform all platforms (1=press Shift)

    **Observation Space:** `Box(shape=(400, 640, 3), dtype=np.uint8)`
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Maneuver and transform platforms in a chain reaction puzzle. Keep at least one platform "
        "above the victory line to win."
    )
    user_guide = (
        "Controls: ←→ to move the player platform. Press space to transform the player platform, "
        "and shift to transform all platforms."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30  # Used for game logic timing, not rendering lock
        self.NUM_PLATFORMS = 5
        self.PLATFORM_WIDTH = 80
        self.PLATFORM_HEIGHT = 20
        self.PLAYER_PLATFORM_IDX = 2
        self.PLAYER_SPEED_X = 6.0
        self.SOLID_SPEED_Y = 2.0  # Pixels per step, falling
        self.TRANSPARENT_SPEED_Y = -1.5  # Pixels per step, rising

        self.VICTORY_Y_COORD = 100  # y-coordinate from top
        self.REWARD_Y_COORD = 250  # y-coordinate from top
        self.VICTORY_DURATION_STEPS = 10 * self.FPS
        self.MAX_EPISODE_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 60, 120)
        self.COLOR_THRESHOLD = (255, 255, 255)
        self.PLATFORM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255),   # Blue (Player)
            (255, 255, 80),  # Yellow
            (255, 80, 255)   # Magenta
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 40)

        # --- Game State Variables ---
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.platforms.clear()
        self.particles.clear()

        start_y = self.SCREEN_HEIGHT / 2
        for i in range(self.NUM_PLATFORMS):
            start_x = (i + 1) * (self.SCREEN_WIDTH / (self.NUM_PLATFORMS + 1)) - self.PLATFORM_WIDTH / 2
            vx = self.np_random.uniform(low=-1.5, high=1.5)
            platform = {
                "rect": pygame.Rect(start_x, start_y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT),
                "color": self.PLATFORM_COLORS[i],
                "vx": vx,
                "is_solid": True
            }
            self.platforms.append(platform)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        self._update_physics()

        reward = self._calculate_reward()
        self.score += reward

        self._update_victory_condition()

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        player_platform = self.platforms[self.PLAYER_PLATFORM_IDX]
        if movement == 3:  # Left
            player_platform["rect"].x -= self.PLAYER_SPEED_X
        elif movement == 4:  # Right
            player_platform["rect"].x += self.PLAYER_SPEED_X

        player_platform["rect"].left = max(0, player_platform["rect"].left)
        player_platform["rect"].right = min(self.SCREEN_WIDTH, player_platform["rect"].right)

        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            player_platform["is_solid"] = not player_platform["is_solid"]
            # sfx: transform_single
            self._create_particles(player_platform["rect"].center, player_platform["color"], 15)

        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed:
            for p in self.platforms:
                p["is_solid"] = not p["is_solid"]
            # sfx: transform_all
            for p in self.platforms:
                self._create_particles(p["rect"].center, p["color"], 10)

    def _update_physics(self):
        # Update platform positions and handle wall collisions
        for p in self.platforms:
            p["rect"].x += p["vx"]
            p["rect"].y += self.SOLID_SPEED_Y if p["is_solid"] else self.TRANSPARENT_SPEED_Y

            p["rect"].top = max(0, p["rect"].top)
            p["rect"].bottom = min(self.SCREEN_HEIGHT, p["rect"].bottom)

            if p["rect"].left <= 0 or p["rect"].right >= self.SCREEN_WIDTH:
                p["vx"] *= -1
                p["is_solid"] = not p["is_solid"]
                # sfx: bounce
                impact_point = (0 if p["rect"].left <= 0 else self.SCREEN_WIDTH, p["rect"].centery)
                self._create_particles(impact_point, p["color"], 20)
                p["rect"].left = max(0, p["rect"].left)
                p["rect"].right = min(self.SCREEN_WIDTH, p["rect"].right)

        # Handle platform-platform collisions
        collided_pairs = set()
        for i, j in combinations(range(self.NUM_PLATFORMS), 2):
            p1, p2 = self.platforms[i], self.platforms[j]
            if p1["rect"].colliderect(p2["rect"]):
                collided_pairs.add(tuple(sorted((i, j))))

        if collided_pairs:
            for i, j in collided_pairs:
                p1, p2 = self.platforms[i], self.platforms[j]
                p1["is_solid"], p2["is_solid"] = not p1["is_solid"], not p2["is_solid"]
                # sfx: collide
                collision_center = ((p1["rect"].centerx + p2["rect"].centerx) / 2, (p1["rect"].centery + p2["rect"].centery) / 2)
                avg_color = tuple((c1 + c2) // 2 for c1, c2 in zip(p1["color"], p2["color"]))
                self._create_particles(collision_center, avg_color, 30)

                # Resolve overlap to prevent sticking
                overlap_x = (p1["rect"].width + p2["rect"].width) / 2 - abs(p1["rect"].centerx - p2["rect"].centerx)
                if overlap_x > 0:
                    dx = overlap_x / 2 + 1
                    if p1["rect"].centerx < p2["rect"].centerx:
                        p1["rect"].x -= dx
                        p2["rect"].x += dx
                    else:
                        p1["rect"].x += dx
                        p2["rect"].x -= dx

        # Update particles
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                "color": color,
                "lifetime": self.np_random.integers(15, 30),
                "max_lifetime": 30
            })

    def _calculate_reward(self):
        reward = 0
        if any(p["rect"].bottom < self.REWARD_Y_COORD for p in self.platforms):
            reward += 0.1
        if self.victory_timer > 0 and self.victory_timer % self.FPS == 0:
            reward += 1.0
        return reward

    def _update_victory_condition(self):
        if any(p["rect"].bottom < self.VICTORY_Y_COORD for p in self.platforms):
            self.victory_timer += 1
        else:
            self.victory_timer = 0

    def _check_termination(self):
        if self.victory_timer >= self.VICTORY_DURATION_STEPS:
            self.score += 100
            self.game_over = True
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "victory_progress": self.victory_timer / self.VICTORY_DURATION_STEPS
        }

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_THRESHOLD, (x, self.VICTORY_Y_COORD), (x + 10, self.VICTORY_Y_COORD), 1)

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            size = max(1, int(6 * (p["lifetime"] / p["max_lifetime"])))
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, (*p["color"], alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, (*p["color"], alpha))

        for i, p in enumerate(self.platforms):
            color = p["color"]
            if p["is_solid"]:
                pygame.draw.rect(self.screen, color, p["rect"], border_radius=3)
                highlight_color = tuple(min(255, c + 40) for c in color)
                pygame.draw.rect(self.screen, highlight_color, (p["rect"].x + 2, p["rect"].y + 2, p["rect"].width - 4, 4), border_radius=2)
            else:
                s = pygame.Surface(p["rect"].size, pygame.SRCALPHA)
                pygame.draw.rect(s, (*color, 100), s.get_rect(), border_radius=3)
                pygame.draw.rect(s, (*color, 200), s.get_rect(), width=2, border_radius=3)
                self.screen.blit(s, p["rect"].topleft)
            
            if i == self.PLAYER_PLATFORM_IDX: # Add indicator for player platform
                indicator_rect = pygame.Rect(p["rect"].centerx-4, p["rect"].y-8, 8, 4)
                pygame.draw.rect(self.screen, (255,255,255), indicator_rect, border_radius=2)


    def _render_ui(self):
        for p in self.platforms:
            height_val = self.SCREEN_HEIGHT - p["rect"].centery
            text_surf = self.font_small.render(f"{height_val:.0f}", True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, text_surf.get_rect(center=(p["rect"].centerx, p["rect"].top - 10)))

        score_surf = self.font_large.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        if self.victory_timer > 0:
            timer_text = f"Afloat: {self.victory_timer/self.FPS:.1f}s / {self.VICTORY_DURATION_STEPS/self.FPS:.1f}s"
            timer_surf = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(timer_surf, timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10)))

            progress = self.victory_timer / self.VICTORY_DURATION_STEPS
            bar_w, bar_h = 200, 10
            bar_x = self.SCREEN_WIDTH - 10 - bar_w
            bar_y = 15 + self.font_large.get_height()
            pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            pygame.draw.rect(self.screen, (100, 200, 255), (bar_x, bar_y, bar_w * progress, bar_h), border_radius=3)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # To run and play the game manually
    # Make sure to unset the dummy videodriver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Platform Ascension")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()