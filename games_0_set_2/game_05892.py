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

    # Short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the slicing point. Press space to slice."
    )

    # Short, user-facing description of the game:
    game_description = (
        "Slice falling fruit to score points while avoiding bombs in this fast-paced arcade game."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial Black", 36)
        self.font_small = pygame.font.SysFont("Consolas", 20)

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 1500
        self.VICTORY_SCORE = 100
        self.CURSOR_SPEED = 15
        self.SLICE_RADIUS = 30
        self.SPAWN_INTERVAL = 45  # frames

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 60, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOMB = (10, 10, 10)
        self.COLOR_BOMB_SKULL = (200, 200, 200)
        self.COLOR_SLICE = (255, 255, 255)
        self.FRUIT_COLORS = {
            "apple": (220, 40, 40),
            "orange": (240, 140, 20),
            "lemon": (250, 250, 50),
            "lime": (50, 220, 50),
        }

        # Initialize state variables
        self.cursor_pos = None
        self.fruits = None
        self.particles = None
        self.slice_effects = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.fall_speed = None
        self.max_bombs_per_spawn = None
        self.spawn_timer = None
        self.prev_space_held = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.fruits = []
        self.particles = []
        self.slice_effects = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.fall_speed = 3.0
        self.max_bombs_per_spawn = 1
        self.spawn_timer = self.SPAWN_INTERVAL

        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1

        reward = 0.0

        # 1. Handle player input and actions
        self._move_cursor(movement)

        slice_action = space_held and not self.prev_space_held
        if slice_action:
            # SFX: whoosh
            reward += self._perform_slice()
            self.slice_effects.append({"center": pygame.Vector2(self.cursor_pos), "radius": 0, "alpha": 255})
        self.prev_space_held = space_held

        # 2. Update game state
        self._update_objects()
        self._update_effects()

        # 3. Spawn new objects
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_objects()
            self.spawn_timer = self.SPAWN_INTERVAL

        # 4. Update difficulty
        self.fall_speed = 3.0 + self.steps / 500.0
        self.max_bombs_per_spawn = min(5, 1 + self.score // 25)

        # 5. Check for termination conditions
        terminated = self.game_over
        if self.score >= self.VICTORY_SCORE:
            reward += 100
            terminated = True
        elif self.game_over:
            reward = -100  # Override other rewards on loss

        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        # Combine termination conditions
        terminated = terminated or truncated

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos.x += self.CURSOR_SPEED

        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

    def _perform_slice(self):
        sliced_something = False

        # Prioritize slicing fruits
        fruits_to_remove = []
        for fruit in self.fruits:
            if fruit["type"] != "bomb" and fruit["pos"].distance_to(self.cursor_pos) < fruit["radius"] + self.SLICE_RADIUS:
                fruits_to_remove.append(fruit)
                sliced_something = True
                self.score += 1
                self._create_particles(fruit["pos"], fruit["color"])
                # SFX: slice_fruit

        if fruits_to_remove:
            self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
            return 1.0 * len(fruits_to_remove)

        # Check for bomb slices only if no fruit was hit
        for fruit in self.fruits:
            if fruit["type"] == "bomb" and fruit["pos"].distance_to(self.cursor_pos) < fruit["radius"] + self.SLICE_RADIUS:
                self.game_over = True
                sliced_something = True
                self._create_particles(fruit["pos"], self.COLOR_BOMB_SKULL, 50)
                # SFX: explosion
                break

        return -0.1 if not sliced_something else 0

    def _update_objects(self):
        for obj in self.fruits[:]:
            obj["pos"].y += self.fall_speed
            if obj["pos"].y > self.HEIGHT + obj["radius"]:
                self.fruits.remove(obj)

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.2  # Gravity
            p["lifespan"] -= 4
            if p["lifespan"] <= 0:
                self.particles.remove(p)

        # Update slice visual effect
        for s in self.slice_effects[:]:
            s["radius"] += 4
            s["alpha"] -= 20
            if s["alpha"] <= 0:
                self.slice_effects.remove(s)

    def _spawn_objects(self):
        # Spawn fruits
        num_fruits = self.np_random.integers(2, 5)
        for _ in range(num_fruits):
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            self.fruits.append({
                "type": "fruit",
                "pos": pygame.Vector2(self.np_random.integers(50, self.WIDTH - 50), -30),
                "radius": self.np_random.integers(20, 30),
                "color": self.FRUIT_COLORS[fruit_type],
            })

        # Spawn bombs
        if self.np_random.random() < 0.5 + self.score / 200.0:  # Chance to spawn bombs increases with score
            for _ in range(self.np_random.integers(1, self.max_bombs_per_spawn + 1)):
                self.fruits.append({
                    "type": "bomb",
                    "pos": pygame.Vector2(self.np_random.integers(50, self.WIDTH - 50), -30),
                    "radius": 25,
                    "color": self.COLOR_BOMB,
                })

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "color": color,
                "lifespan": self.np_random.integers(50, 100),
            })

    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_effects()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            # Simple gradient from top to bottom
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_objects(self):
        for obj in sorted(self.fruits, key=lambda x: 1 if x['type'] == 'bomb' else 0):
            pos_int = (int(obj["pos"].x), int(obj["pos"].y))
            radius = int(obj["radius"])

            # Draw shadow
            shadow_pos = (pos_int[0] + 3, pos_int[1] + 3)
            pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], radius, (0, 0, 0, 50))

            if obj["type"] == "bomb":
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, obj["color"])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, obj["color"])
                self._draw_skull(pos_int, radius)
            else:  # Fruit
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, obj["color"])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, obj["color"])
                # Highlight
                highlight_pos = (pos_int[0] - radius // 3, pos_int[1] - radius // 3)
                pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 4,
                                              (255, 255, 255, 100))

    def _draw_skull(self, pos, radius):
        color = self.COLOR_BOMB_SKULL
        r = radius * 0.7
        # Head
        pygame.draw.ellipse(self.screen, color, (pos[0] - r * 0.6, pos[1] - r * 0.7, r * 1.2, r * 1.4))
        # Eyes
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0] - r * 0.25), int(pos[1] - r * 0.1), int(r * 0.2),
                                      self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0] + r * 0.25), int(pos[1] - r * 0.1), int(r * 0.2),
                                      self.COLOR_BOMB)
        # Nose
        pygame.draw.polygon(self.screen, self.COLOR_BOMB, [
            (pos[0], pos[1] + r * 0.2),
            (pos[0] - r * 0.1, pos[1] + r * 0.4),
            (pos[0] + r * 0.1, pos[1] + r * 0.4),
        ])

    def _render_effects(self):
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, p["lifespan"] * 2.55))
            color = (*p["color"], alpha)
            size = max(1, int(p["lifespan"] / 20))
            temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, size, size))
            self.screen.blit(temp_surf, (int(p["pos"].x - size / 2), int(p["pos"].y - size / 2)))

        # Slice effect
        for s in self.slice_effects:
            if s["alpha"] > 0:
                color = (*self.COLOR_SLICE, s["alpha"])
                temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(s["center"].x), int(s["center"].y), int(s["radius"]), color)
                self.screen.blit(temp_surf, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Bomb spawn indicator
        bomb_icon_radius = 10
        for i in range(self.max_bombs_per_spawn):
            pos = (30 + i * (bomb_icon_radius * 2 + 5), 65)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb_icon_radius, self.COLOR_BOMB)
            self._draw_skull(pos, bomb_icon_radius)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fall_speed": self.fall_speed,
            "max_bombs": self.max_bombs_per_spawn,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # To do so, you might need to comment out:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    truncated = False

    # Main game loop
    while not terminated and not truncated:
        movement = 0  # No-op
        space_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1

        action = [movement, space_held, 0]  # Shift is unused

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']} in {info['steps']} steps.")
            # Wait a moment before closing
            pygame.time.wait(2000)

    env.close()