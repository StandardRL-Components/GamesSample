# Generated: 2025-08-28T03:16:26.019052
# Source Brief: brief_01975.md
# Brief Index: 1975


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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓ to move the blade. Avoid bombs and slice fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling fruit with your blade while dodging bombs to reach a high score in this fast-paced arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.W, self.H = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG_TOP = (10, 5, 20)
        self.COLOR_BG_BOTTOM = (40, 20, 60)
        self.COLOR_BLADE = (255, 255, 255)
        self.COLOR_BLADE_GLOW = (255, 255, 150)
        self.COLOR_APPLE = (220, 30, 30)
        self.COLOR_BANANA = (255, 225, 50)
        self.COLOR_WATERMELON_RIND = (0, 150, 0)
        self.COLOR_WATERMELON_FLESH = (250, 50, 80)
        self.COLOR_GOLDEN_APPLE = (255, 215, 0)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_FUSE = (255, 100, 0)
        self.COLOR_SCORE = (240, 240, 240)
        self.COLOR_HEART = (255, 80, 80)

        # Game parameters
        self.BLADE_SPEED = 10
        self.MAX_LIVES = 3
        self.WIN_SCORE = 500
        self.MAX_STEPS = 10000
        self.INITIAL_FALL_SPEED = 2.0
        self.SPEED_INCREASE_INTERVAL = 500
        self.SPEED_INCREASE_AMOUNT = 0.1
        self.SPAWN_INTERVAL_INITIAL = 30  # frames

        # State variables will be initialized in reset()
        self.blade_y = 0
        self.blade_trail = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.fall_speed = 0.0
        self.spawn_timer = 0
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.screen_shake = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.blade_y = self.H // 2
        self.blade_trail = []
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_timer = self.SPAWN_INTERVAL_INITIAL
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Handle player input
        self._update_player(action)

        # 2. Update game objects
        self._update_objects()

        # 3. Handle collisions
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Update particles
        self._update_particles()

        # 5. Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.blade_y -= self.BLADE_SPEED
        elif movement == 2:  # Down
            self.blade_y += self.BLADE_SPEED

        self.blade_y = np.clip(self.blade_y, 0, self.H)

        # Update blade trail for visual effect
        self.blade_trail.append(self.blade_y)
        if len(self.blade_trail) > 10:
            self.blade_trail.pop(0)

    def _update_objects(self):
        # Update fall speed
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.fall_speed += self.SPEED_INCREASE_AMOUNT

        # Move existing fruits and bombs
        for fruit in self.fruits:
            fruit['pos'][1] += self.fall_speed
        for bomb in self.bombs:
            bomb['pos'][1] += self.fall_speed

        # Remove objects that are off-screen
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.H + 50]
        self.bombs = [b for b in self.bombs if b['pos'][1] < self.H + 50]

        # Spawn new objects
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = int(max(10, self.SPAWN_INTERVAL_INITIAL - self.steps / 100))
            spawn_x = self.np_random.integers(50, self.W - 50)

            # 20% chance to spawn a bomb
            if self.np_random.random() < 0.2:
                self.bombs.append({'pos': [spawn_x, -50], 'radius': 15})
            else:
                # Spawn a fruit
                fruit_type = self.np_random.choice(['apple', 'banana', 'watermelon', 'golden_apple'],
                                                   p=[0.45, 0.35, 0.15, 0.05])
                radius = 12
                if fruit_type == 'watermelon': radius = 20
                if fruit_type == 'golden_apple': radius = 15
                self.fruits.append({'pos': [spawn_x, -50], 'type': fruit_type, 'radius': radius})

    def _handle_collisions(self):
        reward = 0
        blade_rect = pygame.Rect(0, self.blade_y - 2, self.W, 4)

        # Fruit collisions
        sliced_fruits = []
        for fruit in self.fruits:
            fruit_rect = pygame.Rect(fruit['pos'][0] - fruit['radius'], fruit['pos'][1] - fruit['radius'],
                                      fruit['radius'] * 2, fruit['radius'] * 2)
            if blade_rect.colliderect(fruit_rect):
                sliced_fruits.append(fruit)
                # sfx: fruit_slice.wav
                if fruit['type'] == 'apple':
                    reward += 1
                    self.score += 10
                    self._create_slice_particles(fruit['pos'], self.COLOR_APPLE, 20)
                elif fruit['type'] == 'banana':
                    reward += 1
                    self.score += 15
                    self._create_slice_particles(fruit['pos'], self.COLOR_BANANA, 20)
                elif fruit['type'] == 'watermelon':
                    reward += 1
                    self.score += 20
                    self._create_slice_particles(fruit['pos'], self.COLOR_WATERMELON_FLESH, 30)
                elif fruit['type'] == 'golden_apple':
                    reward += 5  # Bonus reward
                    self.score += 50
                    self._create_slice_particles(fruit['pos'], self.COLOR_GOLDEN_APPLE, 40)

        self.fruits = [f for f in self.fruits if f not in sliced_fruits]

        # Bomb collisions
        for bomb in self.bombs:
            bomb_rect = pygame.Rect(bomb['pos'][0] - bomb['radius'], bomb['pos'][1] - bomb['radius'],
                                    bomb['radius'] * 2, bomb['radius'] * 2)
            if blade_rect.colliderect(bomb_rect):
                self.bombs.remove(bomb)
                self.lives -= 1
                reward -= 5
                self.screen_shake = 10
                # sfx: explosion.wav
                self._create_explosion_particles(bomb['pos'], 50)
                break  # Only one bomb collision per frame

        return reward

    def _create_slice_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append(
                {'pos': list(pos), 'vel': vel, 'color': color, 'lifespan': lifespan, 'max_lifespan': lifespan})

    def _create_explosion_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(30, 60)
            color = self.np_random.choice([self.COLOR_FUSE, (255, 165, 0), (255, 255, 0)], p=[0.5, 0.3, 0.2])
            self.particles.append(
                {'pos': list(pos), 'vel': vel, 'color': color, 'lifespan': lifespan, 'max_lifespan': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Create a render offset for screen shake
        render_offset = [0, 0]
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset[0] = self.np_random.integers(-8, 9)
            render_offset[1] = self.np_random.integers(-8, 9)

        # Draw background gradient
        for y in range(self.H):
            ratio = y / self.H
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.W, y))

        # Render all game elements
        self._render_game(render_offset)

        # Render UI overlay (not affected by shake)
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            size = int(5 * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p['pos'][0] - size + offset[0]), int(p['pos'][1] - size + offset[1])))

        # Render fruits
        for fruit in self.fruits:
            x, y = int(fruit['pos'][0] + offset[0]), int(fruit['pos'][1] + offset[1])
            r = fruit['radius']
            if fruit['type'] == 'apple':
                pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_APPLE)
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_APPLE)
            elif fruit['type'] == 'banana':
                rect = pygame.Rect(x - r, y - r // 2, r * 2, r)
                pygame.draw.arc(self.screen, self.COLOR_BANANA, rect, 0.5, 2.5, 8)
            elif fruit['type'] == 'watermelon':
                # Create points for a semi-circle polygon to represent the slice
                arc_points = []
                num_segments = 20
                # Iterate from 180 to 360 degrees (pi to 2*pi radians)
                for i in range(num_segments + 1):
                    angle = math.pi + (i / float(num_segments)) * math.pi
                    # Calculate point on circle edge
                    # Pygame's y-axis is inverted, so use -sin for standard orientation
                    px = x + r * math.cos(angle)
                    py = y - r * math.sin(angle)
                    arc_points.append((int(px), int(py)))

                # Draw the filled polygon for the flesh. gfxdraw connects the last point
                # to the first, creating the flat top of the slice automatically.
                pygame.gfxdraw.filled_polygon(self.screen, arc_points, self.COLOR_WATERMELON_FLESH)
                # Draw the anti-aliased outline for the rind.
                pygame.gfxdraw.aapolygon(self.screen, arc_points, self.COLOR_WATERMELON_RIND)

                # Seeds
                pygame.gfxdraw.filled_circle(self.screen, int(x - r / 2), int(y + r / 3), 2, (0, 0, 0))
                pygame.gfxdraw.filled_circle(self.screen, int(x + r / 2), int(y + r / 3), 2, (0, 0, 0))
            elif fruit['type'] == 'golden_apple':
                # Glow effect
                glow_r = int(r * 1.5)
                s = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_GOLDEN_APPLE, 50), (glow_r, glow_r), glow_r)
                pygame.draw.circle(s, (*self.COLOR_GOLDEN_APPLE, 80), (glow_r, glow_r), int(glow_r * 0.7))
                self.screen.blit(s, (x - glow_r, y - glow_r))
                # Main body
                pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_GOLDEN_APPLE)
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_GOLDEN_APPLE)

        # Render bombs
        for bomb in self.bombs:
            x, y = int(bomb['pos'][0] + offset[0]), int(bomb['pos'][1] + offset[1])
            r = bomb['radius']
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BOMB)
            # Fuse
            fuse_end_x = x + 5
            fuse_end_y = y - r - 5 + (self.steps % 10 > 5) * 2  # Flicker
            pygame.draw.line(self.screen, self.COLOR_FUSE, (x, y - r), (fuse_end_x, fuse_end_y), 2)
            pygame.gfxdraw.filled_circle(self.screen, fuse_end_x, fuse_end_y, 2, self.COLOR_FUSE)

        # Render blade trail
        if len(self.blade_trail) > 1:
            for i, y_pos in enumerate(self.blade_trail):
                alpha = int(150 * (i / len(self.blade_trail)))
                pygame.draw.line(self.screen, (*self.COLOR_BLADE_GLOW, alpha), (0 + offset[0], int(y_pos + offset[1])),
                                 (self.W + offset[0], int(y_pos + offset[1])), max(1, i // 2))

        # Render blade
        blade_y_offset = int(self.blade_y + offset[1])
        # Glow
        pygame.draw.line(self.screen, self.COLOR_BLADE_GLOW, (0 + offset[0], blade_y_offset),
                         (self.W + offset[0], blade_y_offset), 7)
        # Core
        pygame.draw.line(self.screen, self.COLOR_BLADE, (0 + offset[0], blade_y_offset),
                         (self.W + offset[0], blade_y_offset), 3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))

        # Lives
        for i in range(self.lives):
            self._draw_heart(self.W - 40 - (i * 40), 30, 15)

        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            status = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_text = self.font_large.render(status, True, (255, 255, 255))
            text_rect = status_text.get_rect(center=(self.W // 2, self.H // 2))
            self.screen.blit(status_text, text_rect)

    def _draw_heart(self, x, y, size):
        # Simple heart shape using two circles and a polygon
        pygame.draw.circle(self.screen, self.COLOR_HEART, (x - size // 2, y), size // 2)
        pygame.draw.circle(self.screen, self.COLOR_HEART, (x + size // 2, y), size // 2)
        pygame.draw.polygon(self.screen, self.COLOR_HEART, [
            (x - size, y),
            (x + size, y),
            (x, y + size)
        ])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by an agent

    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Create a window to display the game
    pygame.display.set_caption("Fruit Slicer")
    screen = pygame.display.set_mode((env.W, env.H))

    running = True
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0]  # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        if keys[pygame.K_r]:  # Press R to reset
            obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)

        # Convert the observation back to a Pygame surface and draw it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before auto-resetting in playable mode
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Control the frame rate
        env.clock.tick(30)

    env.close()